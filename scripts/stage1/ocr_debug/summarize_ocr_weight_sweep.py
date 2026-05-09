#!/usr/bin/env python3
"""Summarize OCR-weight sweep logs and evaluation metrics into one CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


VAL_RE = re.compile(
    r"\(step=(?P<step>\d+)\)\s+Val MSE:\s+(?P<mse>[0-9.eE+-]+),\s+"
    r"Val MAE:\s+(?P<mae>[0-9.eE+-]+),\s+Val PSNR:\s+(?P<psnr>[0-9.eE+-]+)"
)
FIELD_RE = re.compile(r"([A-Za-z0-9_]+):\s+([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def weight_tag(value: str) -> str:
    return f"{float(value):.6g}".replace(".", "p").replace("-", "m")


def read_json(path: Path) -> Optional[Mapping[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_get(mapping: Optional[Mapping[str, Any]], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def parse_train_log(log_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not log_path.exists():
        return result
    val_rows = []
    last_generator: Dict[str, float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            val_match = VAL_RE.search(line)
            if val_match:
                val_rows.append({
                    "step": int(val_match.group("step")),
                    "mse": float(val_match.group("mse")),
                    "mae": float(val_match.group("mae")),
                    "psnr": float(val_match.group("psnr")),
                })
            if "(Generator)" in line:
                fields = {}
                for key, value in FIELD_RE.findall(line):
                    try:
                        fields[key] = float(value)
                    except ValueError:
                        continue
                if fields:
                    last_generator = fields
    if val_rows:
        best = min(val_rows, key=lambda row: row["mse"])
        last = val_rows[-1]
        result.update({
            "best_val_step": best["step"],
            "best_val_mse": best["mse"],
            "best_val_mae": best["mae"],
            "best_val_psnr": best["psnr"],
            "last_val_step": last["step"],
            "last_val_mse": last["mse"],
            "last_val_mae": last["mae"],
            "last_val_psnr": last["psnr"],
        })
    for key in (
        "ocr_tf_ce_loss",
        "weighted_ocr_tf_loss",
        "ocr_tf_loss_weight",
        "residual_cross_attn_scale_mean",
        "residual_cross_attn_proj_norm_mean",
        "residual_cross_attn_raw_proj_norm_mean",
        "residual_cross_attn_attn_top1_mean",
        "residual_cross_attn_attn_entropy_norm_mean",
        "visual_memory_mask_ratio",
    ):
        if key in last_generator:
            result[f"last_{key}"] = last_generator[key]
    return result


def flatten(prefix: str, mapping: Optional[Mapping[str, Any]], out: Dict[str, Any]) -> None:
    if not isinstance(mapping, Mapping):
        return
    for key, value in mapping.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, Mapping):
            flatten(name, value, out)
        elif isinstance(value, (int, float, str)) or value is None:
            out[name] = value


def add_recon_metrics(row: Dict[str, Any], save_root: Path, run_name: str, mode: str) -> None:
    metrics = read_json(save_root / "recon_eval" / run_name / mode / "metrics.json")
    glyph = safe_get(metrics, "runs", run_name, "overall")
    baseline = safe_get(metrics, "runs", "matched_native", "overall")
    if isinstance(glyph, Mapping):
        for key in ("mse", "mae", "psnr", "ssim"):
            row[f"recon_{mode}_{key}"] = glyph.get(key)
    if isinstance(baseline, Mapping) and mode == "correct":
        for key in ("mse", "mae", "psnr", "ssim"):
            row[f"baseline_{key}"] = baseline.get(key)


def add_ocr_tf_metrics(row: Dict[str, Any], save_root: Path, run_name: str, mode: str) -> None:
    metrics = read_json(save_root / "ocr_tf_eval" / run_name / mode / "metrics.json")
    overall = safe_get(metrics, "runs", run_name)
    if isinstance(overall, Mapping):
        for key in (
            "ocr_tf_ce_loss",
            "ocr_tf_target_tokens_mean",
            "ocr_tf_truncated_count",
            "ocr_tf_loss",
        ):
            if key in overall:
                row[f"ocr_eval_{mode}_{key}"] = overall[key]


def add_grad_probe(row: Dict[str, Any], save_root: Path, run_name: str) -> None:
    metrics = read_json(save_root / "grad_probe" / run_name / "param_grad_metrics.json")
    if not isinstance(metrics, Mapping):
        return
    ratios = safe_get(metrics, "ratios", "weighted_ocr")
    if isinstance(ratios, Mapping):
        for group, values in ratios.items():
            if not isinstance(values, Mapping):
                continue
            row[f"grad_ratio_{group}_over_full"] = values.get("grad_over_full_with_ocr")
            row[f"grad_ratio_{group}_over_mse"] = values.get("grad_over_mse")
    cosines = safe_get(metrics, "cosines", "weighted_ocr")
    if isinstance(cosines, Mapping):
        for group, values in cosines.items():
            if not isinstance(values, Mapping):
                continue
            row[f"grad_cos_{group}_vs_mse"] = values.get("mse")
            row[f"grad_cos_{group}_vs_full"] = values.get("full_with_ocr")


def add_image_diff(row: Dict[str, Any], save_root: Path, run_name: str) -> None:
    metrics = read_json(save_root / "image_diff" / run_name / "text_mode_image_diff.json")
    diffs = safe_get(metrics, "diff_vs_correct")
    if isinstance(diffs, Mapping):
        for mode, values in diffs.items():
            if isinstance(values, Mapping):
                for key, value in values.items():
                    row[f"diff_{mode}_{key}"] = value


def add_sensitivity_gaps(row: Dict[str, Any]) -> None:
    correct = row.get("recon_correct_mse")
    if not isinstance(correct, (int, float)) or math.isnan(float(correct)):
        return
    for mode in ("empty", "shuffled", "length_matched_wrong"):
        value = row.get(f"recon_{mode}_mse")
        if isinstance(value, (int, float)):
            row[f"mse_gap_{mode}_minus_correct"] = float(value) - float(correct)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    for weight in args.weights:
        tag = weight_tag(weight)
        run_name = f"{args.run_prefix}_ocrw{tag}_{args.iterations}step_gbs{args.global_batch_size}_seed{args.seed}"
        row: Dict[str, Any] = {
            "weight": float(weight),
            "tag": tag,
            "run_name": run_name,
        }
        row.update(parse_train_log(args.save_root / run_name / "log.txt"))
        for mode in ("correct", "empty", "shuffled", "length_matched_wrong"):
            add_recon_metrics(row, args.save_root, run_name, mode)
            add_ocr_tf_metrics(row, args.save_root, run_name, mode)
        add_grad_probe(row, args.save_root, run_name)
        add_image_diff(row, args.save_root, run_name)
        add_sensitivity_gaps(row)
        rows.append(row)

    output = args.output or (args.save_root / "ocr_weight_sweep_summary.csv")
    write_csv(output, rows)
    print(f"wrote {output}")
    for row in rows:
        print(
            "weight={weight:g} best_mse={best} correct_mse={correct} "
            "empty_gap={empty_gap} shuffled_gap={shuf_gap} wrong_gap={wrong_gap}".format(
                weight=row["weight"],
                best=row.get("best_val_mse"),
                correct=row.get("recon_correct_mse"),
                empty_gap=row.get("mse_gap_empty_minus_correct"),
                shuf_gap=row.get("mse_gap_shuffled_minus_correct"),
                wrong_gap=row.get("mse_gap_length_matched_wrong_minus_correct"),
            )
        )


if __name__ == "__main__":
    main()
