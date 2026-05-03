#!/usr/bin/env python3
"""Probe frozen DeepSeek-OCR vision feature loss on tokenizer reconstructions.

This is an offline mechanism check. It does not train GigaTok and does not call
DeepSeek-OCR text generation. The only gradient check is whether a frozen OCR
vision feature loss can backpropagate to the reconstructed image tensor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    MetricAccumulator,
    TextContext,
    autocast_context,
    dtype_from_mixed_precision,
    iter_jsonl,
    load_tokenizer_model,
    metric_values,
    parse_color,
    parse_run,
    pil_to_tensor,
    prepare_device,
    reconstruct_batch,
    resize_pad_image,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME:CONFIG:CKPT. Can be passed multiple times.")
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepseek-ocr-model", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--ocr-dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--ocr-image-size", type=int, default=512)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--text-input-mode", choices=("correct", "empty"), default="correct", help="Only used for text-conditioned runs.")
    parser.add_argument("--text-layer-pair-index", type=int, default=0)
    parser.add_argument("--attn-implementation", default="eager")
    return parser.parse_args()


def load_manifest_rows(path: Path, max_images: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    manifest_dir = path.resolve().parent
    for row in iter_jsonl(path):
        image_path = row.get("image_path")
        if not image_path:
            continue
        path_obj = Path(str(image_path))
        if not path_obj.is_absolute():
            path_obj = manifest_dir / path_obj
        row = dict(row)
        row["image_path"] = str(path_obj)
        rows.append(row)
        if max_images > 0 and len(rows) >= max_images:
            break
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    return rows


def load_deepseek_ocr_model(model_path: Path, device: torch.device, dtype_name: str, attn_implementation: str) -> torch.nn.Module:
    from transformers import AutoModel  # type: ignore

    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_safetensors": True,
    }
    if attn_implementation and attn_implementation != "none":
        kwargs["_attn_implementation"] = attn_implementation
    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    dtype = dtype_map[dtype_name]
    if dtype_name != "fp32":
        kwargs["torch_dtype"] = dtype
    model = AutoModel.from_pretrained(str(model_path), **kwargs)
    model = model.eval().to(device)
    if dtype_name != "fp32":
        model = model.to(dtype)
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def preprocess_for_ocr(images: torch.Tensor, ocr_image_size: int) -> torch.Tensor:
    """Convert GigaTok [-1, 1] images to DeepSeek-OCR normalized input."""
    images = images.clamp(-1, 1)
    if images.shape[-2:] != (ocr_image_size, ocr_image_size):
        images = F.interpolate(images, size=(ocr_image_size, ocr_image_size), mode="bicubic", align_corners=False)
    return images.clamp(-1, 1)


def extract_deepseek_ocr_features(ocr_model: torch.nn.Module, images_norm: torch.Tensor) -> torch.Tensor:
    """Extract projected OCR visual tokens without calling generation.

    Mirrors the no-crop path in DeepSeek-OCR infer(): SAM features + CLIP features
    are concatenated and projected to the OCR LM hidden width.
    """
    core = ocr_model.get_model() if hasattr(ocr_model, "get_model") else ocr_model.model
    sam_model = core.sam_model
    vision_model = core.vision_model
    projector = core.projector
    sam_features = sam_model(images_norm)
    vision_features = vision_model(images_norm, sam_features)
    features = torch.cat((vision_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)), dim=-1)
    return projector(features)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_flat = F.normalize(a.float().flatten(1), dim=1)
    b_flat = F.normalize(b.float().flatten(1), dim=1)
    return 1.0 - (a_flat * b_flat).sum(dim=1)


def make_texts(rows: Sequence[Mapping[str, Any]], mode: str) -> List[str]:
    if mode == "empty":
        return [""] * len(rows)
    return [str(row.get("text", "")) for row in rows]


def evaluate_run(
    *,
    run_name: str,
    model: torch.nn.Module,
    text_context: Optional[TextContext],
    rows: Sequence[Mapping[str, Any]],
    ocr_model: torch.nn.Module,
    device: torch.device,
    device_backend: str,
    mixed_precision: str,
    batch_size: int,
    image_size: int,
    ocr_image_size: int,
    pad_color: Tuple[int, int, int],
    text_input_mode: str,
) -> Dict[str, Any]:
    acc = MetricAccumulator()
    rows_out: List[Dict[str, Any]] = []
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pil_images = [resize_pad_image(Image.open(str(row["image_path"])), image_size, pad_color) for row in batch_rows]
        gt = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        texts = make_texts(batch_rows, text_input_mode)
        rec = reconstruct_batch(
            model,
            gt,
            texts if text_context is not None else None,
            text_context,
            device_backend=device_backend,
            mixed_precision=mixed_precision,
        )

        with torch.no_grad():
            gt_ocr = preprocess_for_ocr(gt, ocr_image_size)
            gt_feat = extract_deepseek_ocr_features(ocr_model, gt_ocr).detach()

        rec_leaf = rec.detach().clone().requires_grad_(True)
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            rec_ocr = preprocess_for_ocr(rec_leaf, ocr_image_size)
            rec_feat = extract_deepseek_ocr_features(ocr_model, rec_ocr)
            feature_mse = F.mse_loss(rec_feat.float(), gt_feat.float(), reduction="none").flatten(1).mean(dim=1)
            feature_cos = cosine_distance(rec_feat, gt_feat)
            loss = feature_mse.mean()
        loss.backward()
        grad = rec_leaf.grad
        grad_norm = grad.flatten(1).float().norm(dim=1) if grad is not None else torch.zeros(len(batch_rows), device=device)

        gt_uint8 = ((gt.detach().clamp(-1, 1) + 1.0) * 127.5).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        rec_uint8 = ((rec.detach().clamp(-1, 1) + 1.0) * 127.5).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        feature_mse_cpu = feature_mse.detach().float().cpu()
        feature_cos_cpu = feature_cos.detach().float().cpu()
        grad_norm_cpu = grad_norm.detach().float().cpu()

        for i, row in enumerate(batch_rows):
            values = metric_values(gt_uint8[i], rec_uint8[i])
            values.update(
                {
                    "ocr_feature_mse": float(feature_mse_cpu[i]),
                    "ocr_feature_cosine_distance": float(feature_cos_cpu[i]),
                    "ocr_feature_grad_norm": float(grad_norm_cpu[i]),
                }
            )
            acc.update(values)
            rows_out.append(
                {
                    "run": run_name,
                    "index": start + i,
                    "image_path": str(row["image_path"]),
                    "text": str(row.get("text", "")),
                    "metrics": values,
                }
            )

    return {"overall": acc.mean(), "samples": rows_out}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = prepare_device(args.device_backend, args.device_id)
    pad_color = parse_color(args.pad_color)
    rows = load_manifest_rows(args.manifest_jsonl, args.max_images)

    ocr_model = load_deepseek_ocr_model(
        args.deepseek_ocr_model,
        device,
        dtype_name=args.ocr_dtype,
        attn_implementation=args.attn_implementation,
    )

    results: Dict[str, Any] = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "num_images": len(rows),
        "deepseek_ocr_model": str(args.deepseek_ocr_model),
        "device_backend": args.device_backend,
        "mixed_precision": args.mixed_precision,
        "ocr_dtype": args.ocr_dtype,
        "ocr_image_size": args.ocr_image_size,
        "text_input_mode": args.text_input_mode,
        "runs": {},
    }

    for run_name, config_path, ckpt_path in args.run:
        model, _, text_context = load_tokenizer_model(
            config_path,
            ckpt_path,
            device,
            text_layer_pair_index=args.text_layer_pair_index,
        )
        result = evaluate_run(
            run_name=run_name,
            model=model,
            text_context=text_context,
            rows=rows,
            ocr_model=ocr_model,
            device=device,
            device_backend=args.device_backend,
            mixed_precision=args.mixed_precision,
            batch_size=args.batch_size,
            image_size=args.image_size,
            ocr_image_size=args.ocr_image_size,
            pad_color=pad_color,
            text_input_mode=args.text_input_mode,
        )
        results["runs"][run_name] = result["overall"]
        write_json(output_dir / f"{run_name}_samples.json", result["samples"])
        del model
        if args.device_backend == "npu":
            torch.npu.empty_cache()
        elif args.device_backend == "cuda":
            torch.cuda.empty_cache()

    write_json(output_dir / "metrics.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
