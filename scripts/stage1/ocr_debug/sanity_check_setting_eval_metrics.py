#!/usr/bin/env python3
"""Check setting-eval image metrics against training validation metrics.

This script recomputes reconstruction MSE/PSNR in the same tensor space used by
``vq_train.py`` validation:

    diff = (reconstruction - gt) * 0.5
    mse = mean(diff ** 2)
    psnr = -10 * log10(mse)

It is intended for debugging metric drift only. It does not retrain and does not
run OCR.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    autocast_context,
    dtype_from_mixed_precision,
    load_manifest_metadata,
    load_tokenizer_model,
    make_text_inputs_for_mode,
    parse_color,
    pil_to_tensor,
    prepare_device,
    reconstruct_batch,
    resize_pad_image,
)
from scripts.stage1.ocr_debug.evaluate_setting_metrics import (  # noqa: E402
    PAPER_FIELDS,
    csv_value,
    fmt,
    load_runs,
    parse_float,
    paper_rows,
    write_csv,
    write_markdown_table,
)


RUN_ORDER = ["orig", "noocr", "w001", "w002", "w003", "w005"]
SANITY_FIELDS = [
    "run",
    "checkpoint_weight_key",
    "num_images",
    "train_final_val_mse",
    "train_final_val_psnr",
    "train_final_val_ssim",
    "setting_eval_tensor_mse",
    "setting_eval_psnr_from_mse",
    "ratio",
    "psnr_gap",
    "gt_tensor_min",
    "gt_tensor_max",
    "gt_tensor_mean",
    "recon_tensor_min",
    "recon_tensor_max",
    "recon_tensor_mean",
    "gt_metric_image_min",
    "gt_metric_image_max",
    "gt_metric_image_mean",
    "recon_metric_image_min",
    "recon_metric_image_max",
    "recon_metric_image_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--setting-metrics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all rows in manifest.")
    parser.add_argument("--text-input-mode", choices=("correct", "empty", "shuffled", "wrong", "length_matched_wrong"), default="correct")
    parser.add_argument("--wrong-text-seed", type=int, default=0)
    parser.add_argument("--fixed-wrong-text", default="THIS IS A FIXED WRONG TEXT 0123456789")
    parser.add_argument("--checkpoint-weight-key", choices=("model", "ema", "auto", "state_dict"), default="model")
    parser.add_argument("--text-layer-pair-index", type=int, default=0)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_sanity_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SANITY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in SANITY_FIELDS})


def final_val_metrics(run_dir: Path) -> dict[str, float | None]:
    candidates = [
        run_dir / "val.csv",
        run_dir / "metrics" / "val_metrics.csv",
    ]
    rows: list[dict[str, str]] = []
    for path in candidates:
        rows = read_csv_rows(path)
        if rows:
            break
    if not rows:
        return {"val_mse": None, "val_psnr": None, "val_ssim": None}
    final = rows[-1]
    return {
        "val_mse": parse_float(final.get("val_mse")),
        "val_psnr": parse_float(final.get("val_psnr")),
        "val_ssim": parse_float(final.get("val_ssim")),
    }


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.detach().float()
    return {
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
    }


def evaluate_tensor_metrics(
    *,
    run_name: str,
    spec: Mapping[str, Any],
    image_paths: Sequence[Path],
    metadata_by_path: Mapping[str, Mapping[str, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model, _, text_context = load_tokenizer_model(
        Path(str(spec["config"])),
        Path(str(spec["ckpt"])),
        device,
        text_layer_pair_index=args.text_layer_pair_index,
        checkpoint_weight_key=args.checkpoint_weight_key,
    )
    pad_color = parse_color(args.pad_color)
    correct_texts = [
        metadata_by_path.get(str(path.resolve()), {}).get("text", "")
        for path in image_paths
    ]
    text_inputs_for_model = make_text_inputs_for_mode(
        correct_texts,
        args.text_input_mode,
        args.wrong_text_seed,
        args.fixed_wrong_text,
    )

    mse_sum = 0.0
    elem_count = 0
    range_stats: dict[str, float] | None = None

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start:start + args.batch_size]
        pil_images = [resize_pad_image(Image.open(path), args.image_size, pad_color) for path in batch_paths]
        batch = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        batch_texts = text_inputs_for_model[start:start + len(batch_paths)]
        rec = reconstruct_batch(
            model,
            batch,
            batch_texts if text_context is not None else None,
            text_context,
            device_backend=args.device_backend,
            mixed_precision=args.mixed_precision,
        )
        diff = (rec.float() - batch.float()) * 0.5
        mse_sum += float(diff.pow(2).sum().item())
        elem_count += int(diff.numel())

        if range_stats is None:
            gt_tensor = tensor_stats(batch)
            rec_tensor = tensor_stats(rec)
            gt_metric = tensor_stats(torch.clamp((batch.float() + 1.0) * 0.5, 0.0, 1.0))
            rec_metric = tensor_stats(torch.clamp((rec.float() + 1.0) * 0.5, 0.0, 1.0))
            range_stats = {
                "gt_tensor_min": gt_tensor["min"],
                "gt_tensor_max": gt_tensor["max"],
                "gt_tensor_mean": gt_tensor["mean"],
                "recon_tensor_min": rec_tensor["min"],
                "recon_tensor_max": rec_tensor["max"],
                "recon_tensor_mean": rec_tensor["mean"],
                "gt_metric_image_min": gt_metric["min"],
                "gt_metric_image_max": gt_metric["max"],
                "gt_metric_image_mean": gt_metric["mean"],
                "recon_metric_image_min": rec_metric["min"],
                "recon_metric_image_max": rec_metric["max"],
                "recon_metric_image_mean": rec_metric["mean"],
            }

    del model
    if args.device_backend == "cuda":
        torch.cuda.empty_cache()
    elif args.device_backend == "npu":
        torch.npu.empty_cache()

    mse = mse_sum / max(1, elem_count)
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    out = {
        "run": run_name,
        "checkpoint_weight_key": args.checkpoint_weight_key,
        "num_images": len(image_paths),
        "setting_eval_tensor_mse": mse,
        "setting_eval_psnr_from_mse": psnr,
    }
    if range_stats:
        out.update(range_stats)
    return out


def read_setting_metrics(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row.get("run", ""): row for row in read_csv_rows(path)}


def write_sanity_md(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Setting Eval Metric Sanity Check",
        "",
        "| run | train val_mse | tensor mse | ratio | train PSNR | tensor PSNR | gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {train_mse} | {tensor_mse} | {ratio} | {train_psnr} | {tensor_psnr} | {gap} |".format(
                run=row.get("run", ""),
                train_mse=fmt(row.get("train_final_val_mse"), 6),
                tensor_mse=fmt(row.get("setting_eval_tensor_mse"), 6),
                ratio=fmt(row.get("ratio"), 3),
                train_psnr=fmt(row.get("train_final_val_psnr"), 4),
                tensor_psnr=fmt(row.get("setting_eval_psnr_from_mse"), 4),
                gap=fmt(row.get("psnr_gap"), 4),
            )
        )
    lines.extend([
        "",
        "The tensor metric uses `diff = (reconstruction - gt) * 0.5`, matching `vq_train.py` validation.",
        "If the ratio is still far from 1, inspect checkpoint weight key, preprocessing, and text-conditioning path.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_valpsnr_tables(
    *,
    output_dir: Path,
    runs: Mapping[str, Mapping[str, Any]],
    sanity_rows: Sequence[Mapping[str, Any]],
    setting_metrics: Mapping[str, Mapping[str, str]],
) -> None:
    by_run = {str(row.get("run")): row for row in sanity_rows}
    rows = []
    for run, spec in runs.items():
        train = by_run.get(run, {})
        setting = setting_metrics.get(run, {})
        rows.append({
            "Setting": run,
            "Text Injection": "yes" if spec.get("text_injection") else "no",
            "OCR Loss": "yes" if spec.get("ocr_loss") else "no",
            "OCR Weight": "-" if spec.get("ocr_weight") is None else spec.get("ocr_weight"),
            "PSNR↑": fmt(train.get("train_final_val_psnr"), 3),
            "SSIM↑": fmt(train.get("train_final_val_ssim"), 4),
            "LPIPS↓": "",
            "Third-party OCR CER↓": fmt(setting.get("thirdparty_ocr_cer"), 4),
            "Third-party OCR NED↑": fmt(setting.get("thirdparty_ocr_ned"), 4),
            "DeepSeek OCR CER↓": fmt(setting.get("deepseek_ocr_cer"), 4),
            "Best Setting Note": "image metrics from training val.csv; OCR metrics from setting_eval",
        })
    write_csv(output_dir / "paper_table_valpsnr.csv", PAPER_FIELDS, rows)
    write_markdown_table(
        output_dir / "paper_table_valpsnr.md",
        PAPER_FIELDS,
        rows,
        [
            "Image quality columns use each run's training `val.csv` final `val_psnr` / `val_ssim`.",
            "OCR columns are copied from `setting_eval/setting_metrics.csv`.",
            "Use this table until setting-eval image metric alignment is confirmed.",
        ],
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs_json)
    image_paths, metadata_by_path = load_manifest_metadata(args.manifest_jsonl)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise SystemExit("no images to evaluate")

    device = prepare_device(args.device_backend, args.device_id)
    rows = []
    for run, spec in runs.items():
        print(f"== sanity {run} ({args.checkpoint_weight_key}) ==", flush=True)
        train = final_val_metrics(Path(str(spec["run_dir"])))
        row = evaluate_tensor_metrics(
            run_name=run,
            spec=spec,
            image_paths=image_paths,
            metadata_by_path=metadata_by_path,
            device=device,
            args=args,
        )
        row["train_final_val_mse"] = train["val_mse"]
        row["train_final_val_psnr"] = train["val_psnr"]
        row["train_final_val_ssim"] = train["val_ssim"]
        if train["val_mse"] is not None:
            row["ratio"] = row["setting_eval_tensor_mse"] / train["val_mse"]
        if train["val_psnr"] is not None:
            row["psnr_gap"] = row["setting_eval_psnr_from_mse"] - train["val_psnr"]
        rows.append(row)

    write_sanity_csv(args.output_dir / "metric_sanity.csv", rows)
    write_sanity_md(args.output_dir / "metric_sanity.md", rows)
    setting_metrics = read_setting_metrics(args.setting_metrics_csv)
    write_valpsnr_tables(
        output_dir=args.output_dir,
        runs=runs,
        sanity_rows=rows,
        setting_metrics=setting_metrics,
    )
    print(f"sanity csv: {args.output_dir / 'metric_sanity.csv'}")
    print(f"sanity md: {args.output_dir / 'metric_sanity.md'}")
    print(f"valpsnr table: {args.output_dir / 'paper_table_valpsnr.md'}")


if __name__ == "__main__":
    main()
