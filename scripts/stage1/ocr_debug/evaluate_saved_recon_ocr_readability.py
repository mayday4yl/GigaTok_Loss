#!/usr/bin/env python3
"""OCR readability eval for saved unified holdout reconstructions.

This second-pass evaluator does not run the tokenizer. It reads the fixed
256x256 GT images and saved reconstruction PNGs from a completed unified eval
directory, runs an OCR backend on those images, and compares OCR text against
the manifest text with the same text metrics used by the stage-1 eval helpers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    OCRReader,
    append_jsonl,
    iter_jsonl,
    prepare_device,
    text_metric_values,
    write_json,
)


TEXT_FIELDS = (
    "text",
    "caption",
    "prompt",
    "ocr_text",
    "gt_text",
    "clean_text",
    "text_v2",
    "ocr",
    "label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OCR readability on saved 256x256 GT/reconstruction PNGs."
    )
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--per-sample-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", default=["main_best", "main_last", "baseline_last"])
    parser.add_argument("--include-gt", action="store_true")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all selected rows.")
    parser.add_argument("--sample-seed", type=int, default=None, help="If set, fixed random sample before sharding.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--ocr-backend",
        choices=("pytesseract", "paddleocr", "easyocr", "deepseek_ocr"),
        default="deepseek_ocr",
    )
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.0)
    parser.add_argument("--deepseek-ocr-model", default=None)
    parser.add_argument("--deepseek-ocr-prompt", default="<image>\n<|grounding|>OCR this image.")
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
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def metric_mean(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        for key, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(number) or math.isinf(number):
                continue
            sums[key] += number
            counts[key] += 1
    out: Dict[str, float] = {"count": float(len(rows))}
    for key in sorted(sums):
        out[key] = sums[key] / max(1, counts[key])
    return out


def choose_text(row: Mapping[str, Any]) -> tuple[str, str]:
    for field in TEXT_FIELDS:
        value = row.get(field)
        if value is None:
            continue
        text = str(value)
        if text.strip():
            return text, field
    return "", ""


def load_manifest_texts(path: Path) -> Dict[str, Dict[str, Any]]:
    metadata: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        image_path = row.get("image_path")
        if not image_path:
            continue
        text, field = choose_text(row)
        metadata[str(Path(str(image_path)).resolve())] = {
            "text": text,
            "text_field_used": field,
            "source": row.get("source") or row.get("subset") or row.get("source_key") or "",
            "bucket": row.get("bucket") or row.get("coverage_bucket") or "",
        }
    return metadata


def parse_eval_index(path_text: str) -> int:
    name = Path(path_text).name
    try:
        return int(name.split("_", 1)[0])
    except Exception:
        return 10**12


def load_samples(eval_dir: Path, per_sample_jsonl: Path, runs: Sequence[str], manifest_meta: Mapping[str, Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_image: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(per_sample_jsonl):
        run = str(row.get("run") or "")
        if run not in runs:
            continue
        image_path = str(Path(str(row.get("image_path", ""))).resolve())
        if not image_path:
            continue
        source_rec_path = Path(str(row.get("reconstruction_path", "")))
        rec_path = eval_dir / "reconstructions" / run / source_rec_path.name
        if not rec_path.exists():
            continue
        sample = by_image.setdefault(
            image_path,
            {
                "image_path": image_path,
                "eval_index": parse_eval_index(str(source_rec_path)),
                "filename": rec_path.name,
                "source": row.get("source") or manifest_meta.get(image_path, {}).get("source", ""),
                "bucket": row.get("bucket") or manifest_meta.get(image_path, {}).get("bucket", ""),
                "text_mask_coverage": row.get("text_mask_coverage"),
                "box_count": row.get("box_count"),
                "empty_box": row.get("empty_box"),
                "runs": {},
            },
        )
        sample["runs"][run] = str(rec_path)
    samples = []
    for image_path, sample in by_image.items():
        if not all(run in sample["runs"] for run in runs):
            continue
        meta = manifest_meta.get(image_path, {})
        sample["gt_text"] = str(meta.get("text") or "")
        sample["text_field_used"] = str(meta.get("text_field_used") or "")
        sample["gt_256_path"] = str(eval_dir / "gt_256" / sample["filename"])
        samples.append(sample)
    return sorted(samples, key=lambda item: int(item["eval_index"]))


def select_samples(samples: List[Dict[str, Any]], max_images: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    if max_images <= 0 or max_images >= len(samples):
        selected = list(samples)
    elif seed is None:
        selected = list(samples[:max_images])
    else:
        rng = random.Random(seed)
        selected = rng.sample(samples, max_images)
    return sorted(selected, key=lambda item: int(item["eval_index"]))


def shard_samples(samples: List[Dict[str, Any]], shard_index: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return samples
    return [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_index]


def recognize_path(reader: OCRReader, path: Path) -> str:
    image = Image.open(path).convert("RGB")
    return reader.recognize(np.asarray(image, dtype=np.uint8))


def run_eval(args: argparse.Namespace) -> None:
    eval_dir = args.eval_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_jsonl = args.per_sample_jsonl or eval_dir / "per_sample_metrics.shard00.jsonl"
    manifest_meta = load_manifest_texts(args.manifest_jsonl)
    samples = load_samples(eval_dir, per_sample_jsonl, args.runs, manifest_meta)
    selected = select_samples(samples, args.max_images, args.sample_seed)
    selected = shard_samples(selected, args.shard_index, args.num_shards)
    if not selected:
        raise SystemExit("no samples selected for this shard")

    predictions_path = output_dir / f"ocr_readability_predictions.shard{args.shard_index:02d}.jsonl"
    if predictions_path.exists():
        predictions_path.unlink()

    missing = []
    for sample in selected:
        for target_path in [Path(str(sample["gt_256_path"]))] + [Path(path) for path in sample["runs"].values()]:
            if not target_path.exists():
                missing.append(str(target_path))
    if missing:
        raise SystemExit(f"missing saved eval image: {missing[0]}")

    device = prepare_device(args.device_backend, args.device_id)
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
        deepseek_ocr_output_dir=output_dir / f"deepseek_ocr_tmp_shard{args.shard_index:02d}",
        deepseek_ocr_max_new_tokens=args.deepseek_ocr_max_new_tokens,
    )

    target_specs = [("gt_256", "gt_256_path")] if args.include_gt else []
    target_specs.extend((run, f"run:{run}") for run in args.runs)

    for sample_offset, sample in enumerate(selected):
        rows = []
        for target, key in target_specs:
            path = Path(str(sample["gt_256_path"])) if key == "gt_256_path" else Path(str(sample["runs"][target]))
            pred_text = recognize_path(reader, path)
            metrics = text_metric_values(
                str(sample.get("gt_text") or ""),
                pred_text,
                strip_punctuation=args.strip_punctuation,
                remove_spaces=args.remove_spaces,
            )
            rows.append(
                {
                    "target": target,
                    "eval_index": sample["eval_index"],
                    "filename": sample["filename"],
                    "image_path": sample["image_path"],
                    "image_file": str(path),
                    "source": sample.get("source", ""),
                    "bucket": sample.get("bucket", ""),
                    "text_mask_coverage": sample.get("text_mask_coverage"),
                    "box_count": sample.get("box_count"),
                    "empty_box": sample.get("empty_box"),
                    "text_field_used": sample.get("text_field_used", ""),
                    "gt_text": sample.get("gt_text", ""),
                    "pred_text": pred_text,
                    "metrics": metrics,
                }
            )
        append_jsonl(predictions_path, rows)
        if (sample_offset + 1) % 5 == 0 or sample_offset + 1 == len(selected):
            print(f"shard {args.shard_index:02d}: evaluated {sample_offset + 1}/{len(selected)} samples", flush=True)

    shard_summary = {
        "eval_dir": str(eval_dir),
        "manifest_jsonl": str(args.manifest_jsonl),
        "per_sample_jsonl": str(per_sample_jsonl),
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_path),
        "runs": list(args.runs),
        "include_gt": args.include_gt,
        "num_samples": len(selected),
        "num_rows": len(selected) * len(target_specs),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "device_backend": args.device_backend,
        "device_id": args.device_id,
        "ocr_backend": args.ocr_backend,
        "deepseek_ocr_model": args.deepseek_ocr_model,
    }
    write_json(output_dir / f"ocr_readability_summary.shard{args.shard_index:02d}.json", shard_summary)


def summarize(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("ocr_readability_predictions.shard*.jsonl")):
        rows.extend(read_jsonl(path))
    if not rows:
        raise SystemExit(f"no ocr_readability_predictions.shard*.jsonl in {output_dir}")

    by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_source_target: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    by_bucket_target: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = str(row.get("target") or "unknown")
        source = str(row.get("source") or "unknown")
        bucket = str(row.get("bucket") or "unknown")
        by_target[target].append(row)
        by_source_target[(source, target)].append(row)
        by_bucket_target[(bucket, target)].append(row)

    target_summary = {target: metric_mean(target_rows) for target, target_rows in sorted(by_target.items())}
    source_target_summary = {
        f"{source}/{target}": metric_mean(target_rows)
        for (source, target), target_rows in sorted(by_source_target.items())
    }
    bucket_target_summary = {
        f"{bucket}/{target}": metric_mean(target_rows)
        for (bucket, target), target_rows in sorted(by_bucket_target.items())
    }
    summary = {
        "eval_dir": str(args.eval_dir.resolve()),
        "manifest_jsonl": str(args.manifest_jsonl),
        "output_dir": str(output_dir),
        "num_prediction_rows": len(rows),
        "num_targets": len(by_target),
        "targets": sorted(by_target),
        "ocr_backend": args.ocr_backend,
        "metric_reference": "manifest_text",
        "target_summary": target_summary,
        "source_target_summary": source_target_summary,
        "bucket_target_summary": bucket_target_summary,
        "predictions_jsonl_glob": str(output_dir / "ocr_readability_predictions.shard*.jsonl"),
    }
    write_json(output_dir / "ocr_readability_summary.json", summary)

    table_csv = output_dir / "ocr_readability_table.csv"
    with table_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "count", "exact_ci", "CER", "NED", "edit_distance", "gt_chars", "pred_chars"])
        for target, metrics in sorted(target_summary.items()):
            writer.writerow(
                [
                    target,
                    int(metrics.get("count", 0)),
                    f"{metrics.get('ocr_exact_acc_ci', float('nan')):.6f}",
                    f"{metrics.get('ocr_cer', float('nan')):.6f}",
                    f"{metrics.get('ocr_ned_similarity', float('nan')):.6f}",
                    f"{metrics.get('ocr_edit_distance', float('nan')):.6f}",
                    f"{metrics.get('ocr_gt_chars', float('nan')):.2f}",
                    f"{metrics.get('ocr_pred_chars', float('nan')):.2f}",
                ]
            )

    table_md = output_dir / "ocr_readability_table.md"
    lines = [
        "# OCR Readability on Holdout 200",
        "",
        "Reference text is the fixed manifest text. Images are the saved 256x256 GT/reconstruction PNGs from the unified eval directory.",
        "",
        "| target | count | exact_ci | CER ↓ | NED ↑ | edit_distance ↓ | gt_chars | pred_chars |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, metrics in sorted(target_summary.items()):
        lines.append(
            f"| {target} | {int(metrics.get('count', 0))} | "
            f"{metrics.get('ocr_exact_acc_ci', float('nan')):.4f} | "
            f"{metrics.get('ocr_cer', float('nan')):.4f} | "
            f"{metrics.get('ocr_ned_similarity', float('nan')):.4f} | "
            f"{metrics.get('ocr_edit_distance', float('nan')):.2f} | "
            f"{metrics.get('ocr_gt_chars', float('nan')):.2f} | "
            f"{metrics.get('ocr_pred_chars', float('nan')):.2f} |"
        )
    table_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    eval_table = args.eval_dir.resolve() / "eval_metrics_table.md"
    final_report = args.eval_dir.resolve() / "final_eval_report.md"
    eval_text = eval_table.read_text(encoding="utf-8") if eval_table.exists() else "# Holdout 200 Unified Eval\n"
    final_lines = [
        eval_text.rstrip(),
        "",
        "## OCR Readability",
        "",
        table_md.read_text(encoding="utf-8").strip(),
        "",
        "## Notes",
        "",
        "- Reconstruction metrics are computed on holdout_200 and are the final comparison basis.",
        "- OCR readability is a second pass over saved reconstruction images; it does not rerun training or reconstruction.",
        "- Baseline best checkpoint was not available on the new eval server, so baseline_best is not included.",
        "- `main_best` and `main_last` are different checkpoints; `baseline_last` is the GigaTok-only 10500-step checkpoint.",
    ]
    final_report.write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    print(f"summary: {output_dir / 'ocr_readability_summary.json'}")
    print(f"table: {table_md}")
    print(f"final_report: {final_report}")


def main() -> None:
    args = parse_args()
    if args.summarize_only:
        summarize(args)
    else:
        run_eval(args)


if __name__ == "__main__":
    main()
