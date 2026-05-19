#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    cap_model_generate,
    enable_deepseek_ocr_npu_cuda_compat,
    parse_color,
    prepare_device,
    resize_pad_image,
)


PROMPT = "<image>\n<|grounding|>OCR this image."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OCR bbox JSONL from DeepSeek-OCR <|det|> output.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--ocr-image-size", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--coord-scale", type=float, default=1000.0, help="DeepSeek det coordinate scale. Use 1000 for [[0..999]] output.")
    parser.add_argument("--default-confidence", type=float, default=1.0)
    parser.add_argument("--allow-empty", action="store_true", help="Write boxes=[] instead of failing when no det boxes are found.")
    return parser.parse_args()


def iter_manifest(path: Path, max_images: int) -> Iterable[Tuple[int, Dict[str, Any], Path]]:
    manifest_dir = path.resolve().parent
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if max_images > 0 and count >= max_images:
                break
            row = json.loads(line)
            image_path = Path(str(row["image_path"]))
            if not image_path.is_absolute():
                image_path = manifest_dir / image_path
            yield count, row, image_path.resolve()
            count += 1


def raw_result_to_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, Mapping):
        for key in ("text", "result", "markdown", "output", "prediction", "pred_text"):
            if key in result:
                text = raw_result_to_text(result[key])
                if text:
                    return text
        return "\n".join(raw_result_to_text(value) for value in result.values() if value is not None)
    if isinstance(result, (list, tuple)):
        return "\n".join(raw_result_to_text(item) for item in result if item is not None)
    return str(result)


def read_raw_saved_text(output_dir: Path) -> str:
    candidates = [output_dir / "result.mmd", output_dir / "result_ori.mmd"]
    candidates.extend(sorted(output_dir.glob("*.mmd"), key=lambda path: path.stat().st_mtime, reverse=True))
    for path in candidates:
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text:
                return text
    return ""


def parse_det_payload(payload: str) -> List[List[float]]:
    payload = payload.strip()
    boxes: List[List[float]] = []
    try:
        parsed = ast.literal_eval(payload)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple)):
        candidates = parsed
        if candidates and all(isinstance(v, (int, float)) for v in candidates) and len(candidates) == 4:
            candidates = [candidates]
        for item in candidates:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                try:
                    boxes.append([float(v) for v in item])
                except (TypeError, ValueError):
                    pass
    if boxes:
        return boxes
    for match in re.finditer(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", payload):
        boxes.append([float(match.group(i)) for i in range(1, 5)])
    return boxes


def scale_box(box: Sequence[float], coord_scale: float, image_size: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    if coord_scale > 0:
        factor = float(image_size) / float(coord_scale)
        x1, y1, x2, y2 = x1 * factor, y1 * factor, x2 * factor, y2 * factor
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0.0, min(float(image_size), x1))
    x2 = max(0.0, min(float(image_size), x2))
    y1 = max(0.0, min(float(image_size), y1))
    y2 = max(0.0, min(float(image_size), y2))
    return [x1, y1, x2, y2]


def parse_deepseek_boxes(raw_text: str, coord_scale: float, image_size: int, default_confidence: float) -> List[Dict[str, Any]]:
    boxes: List[Dict[str, Any]] = []
    pair_pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>",
        flags=re.DOTALL,
    )
    for ref_text, det_payload in pair_pattern.findall(raw_text):
        clean_ref = " ".join(ref_text.split())
        for raw_box in parse_det_payload(det_payload):
            scaled = scale_box(raw_box, coord_scale, image_size)
            if scaled[2] > scaled[0] and scaled[3] > scaled[1]:
                boxes.append({
                    "bbox": scaled,
                    "confidence": float(default_confidence),
                    "text": clean_ref,
                    "raw_bbox": raw_box,
                    "confidence_source": "default_deepseek_generate_no_confidence",
                })
    if boxes:
        return boxes
    for det_payload in re.findall(r"<\|det\|>(.*?)<\|/det\|>", raw_text, flags=re.DOTALL):
        for raw_box in parse_det_payload(det_payload):
            scaled = scale_box(raw_box, coord_scale, image_size)
            if scaled[2] > scaled[0] and scaled[3] > scaled[1]:
                boxes.append({
                    "bbox": scaled,
                    "confidence": float(default_confidence),
                    "text": "",
                    "raw_bbox": raw_box,
                    "confidence_source": "default_deepseek_generate_no_confidence",
                })
    return boxes


def load_model(args: argparse.Namespace, device: torch.device):
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "use_safetensors": True,
        "torch_dtype": dtype_map[args.dtype],
    }
    if args.attn_implementation and args.attn_implementation != "none":
        load_kwargs["_attn_implementation"] = args.attn_implementation
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(str(args.model_path), **load_kwargs)
    model = model.eval().to(device)
    if args.dtype in dtype_map:
        model = model.to(dtype_map[args.dtype])
    cap_model_generate(model, args.max_new_tokens)
    return model, tokenizer


def run_infer(model: Any, tokenizer: Any, image_path: Path, output_dir: Path, args: argparse.Namespace) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    infer_kwargs: Dict[str, Any] = {
        "prompt": args.prompt,
        "image_file": str(image_path),
        "output_path": str(output_dir),
        "base_size": args.base_size,
        "image_size": args.ocr_image_size,
        "crop_mode": False,
        "save_results": True,
        "test_compress": False,
        "eval_mode": True,
    }
    try:
        result = model.infer(tokenizer, **infer_kwargs)
    except TypeError:
        infer_kwargs.pop("save_results", None)
        infer_kwargs.pop("test_compress", None)
        result = model.infer(tokenizer, **infer_kwargs)
    raw_text = raw_result_to_text(result)
    if not raw_text:
        raw_text = read_raw_saved_text(output_dir)
    return raw_text


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        raise FileExistsError(f"Refusing to overwrite existing output_jsonl: {args.output_jsonl}")
    pad_color = parse_color(args.pad_color)
    device = prepare_device(args.device_backend, args.device_id)
    if args.device_backend == "npu":
        enable_deepseek_ocr_npu_cuda_compat()
    model, tokenizer = load_model(args, device)
    input_dir = args.output_dir / "ocr_inputs_256"
    raw_dir = args.output_dir / "raw_outputs"
    rows: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for idx, row, image_path in iter_manifest(args.manifest, args.max_images):
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        padded = resize_pad_image(Image.open(image_path).convert("RGB"), args.image_size, pad_color)
        ocr_input_path = input_dir / f"{idx:06d}.png"
        ocr_input_path.parent.mkdir(parents=True, exist_ok=True)
        padded.save(ocr_input_path)
        raw_text = run_infer(model, tokenizer, ocr_input_path, raw_dir / f"{idx:06d}", args)
        boxes = parse_deepseek_boxes(
            raw_text,
            coord_scale=args.coord_scale,
            image_size=args.image_size,
            default_confidence=args.default_confidence,
        )
        if not boxes and not args.allow_empty:
            failed.append({"index": idx, "image_path": str(image_path), "reason": "no_det_boxes", "raw_text": raw_text[:1000]})
            print(f"[ERROR] index={idx} no boxes: {image_path}", flush=True)
            break
        out_row = {
            "image_path": str(image_path),
            "boxes": boxes,
            "ocr_text_raw": raw_text,
            "source_manifest": str(args.manifest),
            "manifest_index": idx,
            "coord_scale": args.coord_scale,
            "bbox_coord_space": f"{args.image_size}x{args.image_size}_resize_pad",
            "prompt": args.prompt,
        }
        rows.append(out_row)
        print(f"[INFO] {idx + 1}: boxes={len(boxes)} image={image_path.name}", flush=True)

    if failed:
        failure_path = args.output_dir / "ocr_bbox_failures.json"
        failure_path.write_text(json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"Failed to extract boxes for {len(failed)} image(s). See {failure_path}")

    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "status": "ok",
        "manifest": str(args.manifest),
        "model_path": str(args.model_path),
        "output_jsonl": str(args.output_jsonl),
        "num_rows": len(rows),
        "total_boxes": sum(len(row["boxes"]) for row in rows),
        "empty_rows": sum(1 for row in rows if not row["boxes"]),
        "coord_scale": args.coord_scale,
        "image_size": args.image_size,
        "default_confidence": args.default_confidence,
    }
    summary_path = args.output_dir / "ocr_bbox_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.output_jsonl}")
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] rows={summary['num_rows']} total_boxes={summary['total_boxes']} empty_rows={summary['empty_rows']}")


if __name__ == "__main__":
    main()
