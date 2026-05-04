#!/usr/bin/env python3
"""Check whether current materialized GT images remain OCR-readable.

This diagnostic feeds the manifest images themselves, after the same
resize-pad preprocessing used by training, into an OCR backend. It does not
load or run the tokenizer/reconstruction model.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    OCRReader,
    append_jsonl,
    load_manifest_metadata,
    parse_color,
    prepare_device,
    resize_pad_image,
    text_metric_values,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OCR readability diagnostic for materialized GT images."
    )
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=100, help="0 means all rows.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--ocr-backend",
        choices=("pytesseract", "paddleocr", "easyocr", "deepseek_ocr"),
        default="deepseek_ocr",
    )
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--deepseek-ocr-model",
        default=None,
        help="Local DeepSeek-OCR path. Defaults to DEEPSEEK_OCR_MODEL or HF id.",
    )
    parser.add_argument(
        "--deepseek-ocr-prompt",
        default="<image>\n<|grounding|>OCR this image.",
    )
    parser.add_argument("--deepseek-ocr-base-size", type=int, default=1024)
    parser.add_argument("--deepseek-ocr-image-size", type=int, default=640)
    parser.add_argument("--deepseek-ocr-crop-mode", action="store_true")
    parser.add_argument("--deepseek-ocr-save-results", action="store_true")
    parser.add_argument("--deepseek-ocr-test-compress", action="store_true")
    parser.add_argument("--deepseek-ocr-attn-implementation", default="eager")
    parser.add_argument("--deepseek-ocr-dtype", choices=("auto", "fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--deepseek-ocr-max-new-tokens", type=int, default=1024)
    parser.add_argument("--strip-punctuation", action="store_true")
    parser.add_argument("--remove-spaces", action="store_true")
    parser.add_argument("--save-worst", type=int, default=20)
    return parser.parse_args()


class MeanAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums: Dict[str, float] = defaultdict(float)

    def update(self, values: Mapping[str, float]) -> None:
        self.count += 1
        for key, value in values.items():
            if not math.isnan(float(value)):
                self.sums[key] += float(value)

    def mean(self) -> Dict[str, float]:
        result: Dict[str, float] = {"count": float(self.count)}
        for key, value in sorted(self.sums.items()):
            result[key] = value / max(1, self.count)
        return result


def iter_limited(paths: Iterable[Path], max_images: int) -> Iterable[tuple[int, Path]]:
    for idx, path in enumerate(paths):
        if max_images > 0 and idx >= max_images:
            break
        yield idx, path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr_tmp_dir = output_dir / "deepseek_ocr_tmp"
    rows_jsonl = output_dir / "gt_ocr_predictions.jsonl"
    if rows_jsonl.exists():
        rows_jsonl.unlink()

    image_paths, metadata_by_path = load_manifest_metadata(args.manifest_jsonl)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit("no images to evaluate")

    missing = [str(path) for path in image_paths[:1000] if not path.exists()]
    if missing:
        raise SystemExit(f"missing image files, first missing: {missing[0]}")

    device = prepare_device(args.device_backend, args.device_id)
    pad_color = parse_color(args.pad_color)
    reader = OCRReader(
        backend=args.ocr_backend,
        lang=args.ocr_lang,
        min_confidence=args.ocr_min_confidence,
        device_backend=args.device_backend,
        device=device,
        deepseek_ocr_model=args.deepseek_ocr_model,
        deepseek_ocr_prompt=args.deepseek_ocr_prompt,
        deepseek_ocr_base_size=args.deepseek_ocr_base_size,
        deepseek_ocr_image_size=args.deepseek_ocr_image_size,
        deepseek_ocr_crop_mode=args.deepseek_ocr_crop_mode,
        deepseek_ocr_save_results=args.deepseek_ocr_save_results,
        deepseek_ocr_test_compress=args.deepseek_ocr_test_compress,
        deepseek_ocr_attn_implementation=args.deepseek_ocr_attn_implementation,
        deepseek_ocr_dtype=args.deepseek_ocr_dtype,
        deepseek_ocr_output_dir=ocr_tmp_dir,
        deepseek_ocr_max_new_tokens=args.deepseek_ocr_max_new_tokens,
    )

    overall = MeanAccumulator()
    by_subset: Dict[str, MeanAccumulator] = defaultdict(MeanAccumulator)
    worst_rows: list[Dict[str, Any]] = []
    original_sizes: Dict[str, int] = defaultdict(int)

    for idx, path in iter_limited(image_paths, args.max_images):
        meta = metadata_by_path.get(str(path.resolve()), {})
        subset = str(meta.get("subset") or "unknown")
        gt_text = str(meta.get("text") or "")
        pil = Image.open(path).convert("RGB")
        original_sizes[f"{pil.size[0]}x{pil.size[1]}"] += 1
        gt = resize_pad_image(pil, args.image_size, pad_color)
        pred_text = reader.recognize(np.asarray(gt, dtype=np.uint8))
        metrics = text_metric_values(
            gt_text,
            pred_text,
            strip_punctuation=args.strip_punctuation,
            remove_spaces=args.remove_spaces,
        )
        overall.update(metrics)
        by_subset[subset].update(metrics)
        row = {
            "index": idx,
            "image_path": str(path),
            "source_size": list(pil.size),
            "ocr_input_size": [args.image_size, args.image_size],
            "subset": subset,
            "source_key": meta.get("source_key", ""),
            "gt_text": gt_text,
            "pred_text": pred_text,
            "metrics": metrics,
        }
        append_jsonl(rows_jsonl, [row])
        worst_rows.append(row)
        if (idx + 1) % 10 == 0 or idx + 1 == len(image_paths):
            print(f"evaluated {idx + 1}/{len(image_paths)} images", flush=True)

    worst_rows.sort(key=lambda item: float(item["metrics"].get("ocr_cer", 0.0)), reverse=True)
    worst_rows = worst_rows[: max(0, args.save_worst)]
    summary = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "num_images": len(image_paths),
        "image_size": args.image_size,
        "pad_color": list(pad_color),
        "source_image_sizes": dict(sorted(original_sizes.items())),
        "ocr_backend": args.ocr_backend,
        "deepseek_ocr_model": args.deepseek_ocr_model,
        "deepseek_ocr_prompt": args.deepseek_ocr_prompt,
        "deepseek_ocr_base_size": args.deepseek_ocr_base_size,
        "deepseek_ocr_image_size": args.deepseek_ocr_image_size,
        "deepseek_ocr_crop_mode": args.deepseek_ocr_crop_mode,
        "deepseek_ocr_dtype": args.deepseek_ocr_dtype,
        "deepseek_ocr_max_new_tokens": args.deepseek_ocr_max_new_tokens,
        "overall": overall.mean(),
        "by_subset": {subset: acc.mean() for subset, acc in sorted(by_subset.items())},
        "worst_by_cer": worst_rows,
        "predictions_jsonl": str(rows_jsonl),
    }
    write_json(output_dir / "gt_ocr_summary.json", summary)

    overall_metrics = summary["overall"]
    print("\n===== GT OCR Readability =====")
    print(f"images: {len(image_paths)}")
    print(f"source sizes: {summary['source_image_sizes']}")
    print(f"exact_acc_ci: {overall_metrics.get('ocr_exact_acc_ci', float('nan')):.4f}")
    print(f"cer: {overall_metrics.get('ocr_cer', float('nan')):.4f}")
    print(f"ned_similarity: {overall_metrics.get('ocr_ned_similarity', float('nan')):.4f}")
    print(f"summary: {output_dir / 'gt_ocr_summary.json'}")
    print(f"predictions: {rows_jsonl}")


if __name__ == "__main__":
    main()
