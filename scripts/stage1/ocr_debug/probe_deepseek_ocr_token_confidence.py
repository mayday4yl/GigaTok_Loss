#!/usr/bin/env python3
import argparse
import ast
import json
import math
import re
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    enable_deepseek_ocr_npu_cuda_compat,
    parse_color,
    prepare_device,
    resize_pad_image,
)


PROMPT = "<image>\n<|grounding|>OCR this image."
CONF_SOURCE = "deepseek_token_logprob_proxy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe DeepSeek-OCR token logprob proxy confidence for OCR boxes.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-bbox-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=1, help="Use 1 for smoke, 0 for all manifest rows.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--base-size", type=int, default=1024)
    parser.add_argument("--ocr-image-size", type=int, default=640)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--coord-scale", type=float, default=1000.0)
    parser.add_argument("--require-source-count-match", action="store_true", default=True)
    parser.add_argument("--allow-overwrite", action="store_true")
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


def canonical_path(path: Any) -> str:
    return str(Path(str(path)).expanduser().resolve())


def load_source_bbox_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = canonical_path(row["image_path"])
            if image_path in rows:
                raise ValueError(f"{path}:{line_no}: duplicate image_path={image_path}")
            boxes = row.get("boxes")
            if not isinstance(boxes, list):
                raise ValueError(f"{path}:{line_no}: missing list field boxes")
            rows[image_path] = row
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


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


def build_token_offsets(tokenizer: Any, token_ids: Sequence[int]) -> Tuple[str, List[Tuple[int, int]]]:
    offsets: List[Tuple[int, int]] = []
    previous = ""
    ids_so_far: List[int] = []
    for token_id in token_ids:
        ids_so_far.append(int(token_id))
        current = tokenizer.decode(ids_so_far, skip_special_tokens=False)
        if len(current) < len(previous):
            offsets.append((len(previous), len(previous)))
        else:
            offsets.append((len(previous), len(current)))
        previous = current
    return previous, offsets


def char_span_to_token_span(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> Optional[Tuple[int, int]]:
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if token_start is None and tok_end > start:
            token_start = idx
        if tok_start < end and tok_end > start:
            token_end = idx + 1
    if token_start is None or token_end is None or token_end <= token_start:
        return None
    return token_start, token_end


def parse_scored_items(
        raw_text: str,
        offsets: Sequence[Tuple[int, int]],
        token_logprobs: Sequence[float],
        coord_scale: float,
        image_size: int) -> Tuple[List[Dict[str, Any]], int]:
    items: List[Dict[str, Any]] = []
    failed_alignment_count = 0
    pair_pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>",
        flags=re.DOTALL,
    )
    for match in pair_pattern.finditer(raw_text):
        ref_text = match.group(1)
        det_payload = match.group(2)
        clean_ref = " ".join(ref_text.split())
        span = char_span_to_token_span(offsets, match.start(1), match.end(1))
        span_logprobs: List[float] = []
        if span is None:
            failed_alignment_count += 1
        else:
            token_start, token_end = span
            token_end = min(token_end, len(token_logprobs))
            if token_end <= token_start:
                failed_alignment_count += 1
                span = None
            else:
                span_logprobs = [float(v) for v in token_logprobs[token_start:token_end]]
        mean_logprob = float(mean(span_logprobs)) if span_logprobs else None
        mean_nll = float(-mean_logprob) if mean_logprob is not None else None
        token_span = [int(span[0]), int(span[1])] if span is not None else None
        for raw_box in parse_det_payload(det_payload):
            scaled = scale_box(raw_box, coord_scale, image_size)
            if scaled[2] <= scaled[0] or scaled[3] <= scaled[1]:
                continue
            items.append({
                "bbox": scaled,
                "raw_bbox": raw_box,
                "text": clean_ref,
                "mean_logprob": mean_logprob,
                "mean_nll": mean_nll,
                "token_span": token_span,
                "confidence_source": CONF_SOURCE,
            })
    return items, failed_alignment_count


class GenerateCapture:
    def __init__(self, model: Any, max_new_tokens: int):
        self.model = model
        self.max_new_tokens = int(max_new_tokens)
        self.original_generate = model.generate
        self.output: Optional[Any] = None
        self.prefix_len: Optional[int] = None
        self.generated_token_ids: List[int] = []
        self.step_logprobs: List[float] = []
        self.scores_shapes: List[List[int]] = []

    def __enter__(self) -> "GenerateCapture":
        capture = self

        def generate_with_capture(*args: Any, **kwargs: Any) -> torch.Tensor:
            input_ids = kwargs.get("input_ids", None)
            if input_ids is None and args:
                input_ids = args[0]
            current = kwargs.get("max_new_tokens")
            kwargs["max_new_tokens"] = capture.max_new_tokens if current is None else min(int(current), capture.max_new_tokens)
            kwargs["output_scores"] = True
            kwargs["return_dict_in_generate"] = True
            kwargs["do_sample"] = False
            output = capture.original_generate(*args, **kwargs)
            capture.output = output
            sequences = output.sequences
            if input_ids is not None:
                capture.prefix_len = int(input_ids.shape[-1])
            else:
                capture.prefix_len = int(sequences.shape[-1] - len(output.scores))
            generated = sequences[0, capture.prefix_len:capture.prefix_len + len(output.scores)].detach().cpu().tolist()
            capture.generated_token_ids = [int(v) for v in generated]
            capture.scores_shapes = [list(score.shape) for score in output.scores]
            logprobs: List[float] = []
            for step_idx, token_id in enumerate(capture.generated_token_ids):
                logits = output.scores[step_idx][0].float()
                value = torch.log_softmax(logits, dim=-1)[int(token_id)].detach().cpu().item()
                logprobs.append(float(value))
            capture.step_logprobs = logprobs
            return sequences

        self.model.generate = generate_with_capture
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.model.generate = self.original_generate


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
    return model, tokenizer


def run_infer_with_scores(
        model: Any,
        tokenizer: Any,
        image_path: Path,
        output_dir: Path,
        args: argparse.Namespace) -> Tuple[str, GenerateCapture]:
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
    with GenerateCapture(model, args.max_new_tokens) as capture:
        try:
            result = model.infer(tokenizer, **infer_kwargs)
        except TypeError:
            infer_kwargs.pop("save_results", None)
            infer_kwargs.pop("test_compress", None)
            result = model.infer(tokenizer, **infer_kwargs)
    raw_text = raw_result_to_text(result)
    if not raw_text:
        raw_text = read_raw_saved_text(output_dir)
    if capture.output is None:
        raise RuntimeError("DeepSeek-OCR infer did not call generate; cannot capture output_scores.")
    return raw_text, capture


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def normalize_confidences(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    logprobs: List[float] = []
    for row in rows:
        for box in row["boxes"]:
            value = box.get("mean_logprob")
            if value is not None:
                logprobs.append(float(value))
    if not logprobs:
        return {"token_score_proxy_feasible": False, "recommended_v2_action": "skip", "reason": "no_aligned_logprobs"}
    lo = min(logprobs)
    hi = max(logprobs)
    denom = hi - lo
    for row in rows:
        for box in row["boxes"]:
            value = box.get("mean_logprob")
            if value is None:
                box["confidence"] = 0.0
            elif denom <= 1e-8:
                box["confidence"] = 1.0
            else:
                box["confidence"] = max(0.0, min(1.0, (float(value) - lo) / denom))
            box["confidence_source"] = CONF_SOURCE
    confidences = [float(box["confidence"]) for row in rows for box in row["boxes"]]
    return {
        "token_score_proxy_feasible": True,
        "recommended_v2_action": "use_proxy_confidence",
        "logprob_min": lo,
        "logprob_max": hi,
        "confidence_min": min(confidences),
        "confidence_p10": percentile(confidences, 0.10),
        "confidence_p25": percentile(confidences, 0.25),
        "confidence_mean": float(mean(confidences)),
        "confidence_median": float(median(confidences)),
        "confidence_p75": percentile(confidences, 0.75),
        "confidence_p90": percentile(confidences, 0.90),
        "confidence_max": max(confidences),
        "confidence_unique_count": len(set(round(v, 8) for v in confidences)),
    }


def save_smoke_debug(
        output_dir: Path,
        tokenizer: Any,
        capture: GenerateCapture,
        raw_text: str,
        parsed_items: List[Dict[str, Any]],
        row_debug: Dict[str, Any]) -> None:
    smoke_dir = output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "raw_output.txt").write_text(raw_text, encoding="utf-8")
    (smoke_dir / "generated_token_ids.json").write_text(
        json.dumps(capture.generated_token_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (smoke_dir / "scores_shapes.json").write_text(
        json.dumps(capture.scores_shapes, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (smoke_dir / "decoded_generated.txt").write_text(
        tokenizer.decode(capture.generated_token_ids, skip_special_tokens=False),
        encoding="utf-8",
    )
    (smoke_dir / "parsed_items.json").write_text(
        json.dumps(parsed_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (smoke_dir / "row_debug.json").write_text(
        json.dumps(row_debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output_jsonl: {args.output_jsonl}")

    pad_color = parse_color(args.pad_color)
    device = prepare_device(args.device_backend, args.device_id)
    if args.device_backend == "npu":
        enable_deepseek_ocr_npu_cuda_compat()

    source_rows = load_source_bbox_rows(args.input_bbox_jsonl)
    model, tokenizer = load_model(args, device)
    input_dir = args.output_dir / "ocr_inputs_256"
    raw_dir = args.output_dir / "raw_outputs"
    out_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    failed_alignment_count = 0
    count_mismatch_count = 0
    total_generated_boxes = 0
    total_source_boxes = 0

    for idx, manifest_row, image_path in iter_manifest(args.manifest, args.max_images):
        image_key = canonical_path(image_path)
        source_row = source_rows.get(image_key)
        if source_row is None:
            raise ValueError(f"image_path not found in source bbox jsonl: {image_key}")
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        padded = resize_pad_image(Image.open(image_path).convert("RGB"), args.image_size, pad_color)
        ocr_input_path = input_dir / f"{idx:06d}.png"
        ocr_input_path.parent.mkdir(parents=True, exist_ok=True)
        padded.save(ocr_input_path)

        raw_text, capture = run_infer_with_scores(model, tokenizer, ocr_input_path, raw_dir / f"{idx:06d}", args)
        token_count = min(len(capture.generated_token_ids), len(capture.step_logprobs))
        decoded_generated, offsets = build_token_offsets(tokenizer, capture.generated_token_ids[:token_count])
        parsed_items, row_failed_alignments = parse_scored_items(
            decoded_generated,
            offsets,
            capture.step_logprobs[:token_count],
            coord_scale=args.coord_scale,
            image_size=args.image_size,
        )
        if not parsed_items and raw_text and raw_text != decoded_generated:
            parsed_items, row_failed_alignments = parse_scored_items(
                raw_text,
                offsets,
                capture.step_logprobs[:token_count],
                coord_scale=args.coord_scale,
                image_size=args.image_size,
            )

        source_boxes = source_row.get("boxes", [])
        total_generated_boxes += len(parsed_items)
        total_source_boxes += len(source_boxes)
        failed_alignment_count += row_failed_alignments
        row_debug = {
            "manifest_index": idx,
            "image_path": image_key,
            "prefix_len": capture.prefix_len,
            "generated_token_count": len(capture.generated_token_ids),
            "scores_steps": len(capture.scores_shapes),
            "first_score_shape": capture.scores_shapes[0] if capture.scores_shapes else None,
            "decoded_raw_output_chars": len(decoded_generated),
            "parsed_items": len(parsed_items),
            "source_boxes": len(source_boxes),
            "failed_alignment_count": row_failed_alignments,
        }
        if idx == 0:
            save_smoke_debug(args.output_dir, tokenizer, capture, decoded_generated, parsed_items, row_debug)

        if len(parsed_items) != len(source_boxes):
            count_mismatch_count += 1
            failures.append({
                "manifest_index": idx,
                "image_path": image_key,
                "source_boxes": len(source_boxes),
                "generated_boxes": len(parsed_items),
                "reason": "box_count_mismatch",
            })
            if args.require_source_count_match:
                print(f"[ERROR] {idx}: source_boxes={len(source_boxes)} generated_boxes={len(parsed_items)}", flush=True)
                break

        merged_boxes: List[Dict[str, Any]] = []
        for box_idx, source_box in enumerate(source_boxes):
            merged = dict(source_box)
            if box_idx < len(parsed_items):
                scored = parsed_items[box_idx]
                merged.update({
                    "confidence_source": CONF_SOURCE,
                    "mean_logprob": scored["mean_logprob"],
                    "mean_nll": scored["mean_nll"],
                    "token_span": scored["token_span"],
                    "proxy_ref_text": scored["text"],
                })
            else:
                merged.update({
                    "confidence_source": CONF_SOURCE,
                    "mean_logprob": None,
                    "mean_nll": None,
                    "token_span": None,
                    "proxy_ref_text": "",
                })
            merged_boxes.append(merged)

        out_row = dict(source_row)
        out_row["boxes"] = merged_boxes
        out_row["ocr_text_raw_token_score_probe"] = decoded_generated
        out_row["confidence_source"] = CONF_SOURCE
        out_row["token_score_probe"] = row_debug
        out_rows.append(out_row)
        print(
            f"[INFO] {idx + 1}: source_boxes={len(source_boxes)} generated_boxes={len(parsed_items)} "
            f"failed_align={row_failed_alignments} image={image_path.name}",
            flush=True,
        )

    feasible = not failures
    norm_stats: Dict[str, Any]
    if feasible:
        norm_stats = normalize_confidences(out_rows)
        feasible = bool(norm_stats.get("token_score_proxy_feasible"))
    else:
        norm_stats = {
            "token_score_proxy_feasible": False,
            "recommended_v2_action": "skip",
            "reason": failures[0]["reason"] if failures else "unknown_failure",
        }

    summary = {
        "status": "ok" if feasible else "failed",
        "token_score_proxy_feasible": feasible,
        "recommended_v2_action": norm_stats.get("recommended_v2_action", "skip"),
        "manifest": str(args.manifest),
        "input_bbox_jsonl": str(args.input_bbox_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "model_path": str(args.model_path),
        "rows_processed": len(out_rows),
        "source_boxes": total_source_boxes,
        "generated_boxes": total_generated_boxes,
        "boxes": sum(len(row["boxes"]) for row in out_rows),
        "failed_alignment_count": failed_alignment_count,
        "box_count_mismatch_images": count_mismatch_count,
        **norm_stats,
    }
    summary_path = args.output_dir / "token_confidence_summary.json"
    failures_path = args.output_dir / "token_confidence_failures.json"

    if failures:
        failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    if feasible:
        with args.output_jsonl.open("w", encoding="utf-8") as handle:
            for row in out_rows:
                handle.write(json.dumps(row, ensure_ascii=False))
                handle.write("\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"token_score_proxy_feasible={str(feasible).lower()}")
    print(f"recommended_v2_action={summary['recommended_v2_action']}")
    print(f"boxes={summary['boxes']}")
    print(f"failed_alignment_count={failed_alignment_count}")
    if feasible:
        print(f"confidence_min={summary['confidence_min']}")
        print(f"confidence_p10={summary['confidence_p10']}")
        print(f"confidence_p25={summary['confidence_p25']}")
        print(f"confidence_mean={summary['confidence_mean']}")
        print(f"confidence_median={summary['confidence_median']}")
        print(f"confidence_p75={summary['confidence_p75']}")
        print(f"confidence_p90={summary['confidence_p90']}")
        print(f"confidence_max={summary['confidence_max']}")
        print(f"confidence_unique_count={summary['confidence_unique_count']}")
        print(f"[OK] wrote {args.output_jsonl}")
    else:
        print(f"[ERROR] token score proxy is not feasible; see {summary_path}")
        if failures:
            print(f"[ERROR] failures: {failures_path}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
