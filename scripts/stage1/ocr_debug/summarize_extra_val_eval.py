#!/usr/bin/env python3
"""Summarize extra validation reconstruction metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


FIELDS = [
    "run",
    "num_images",
    "mse",
    "mae",
    "psnr",
    "ssim",
    "ocr_cer",
    "ocr_exact_acc_ci",
    "ocr_ned_similarity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in FIELDS})


def metric(row: Mapping[str, Any], name: str) -> str:
    value = parse_float(row.get(name))
    if value is None:
        return ""
    return f"{value:.6g}"


def write_md(path: Path, rows: Sequence[Mapping[str, Any]], metrics_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_run = {str(row["run"]): row for row in rows}
    lines = [
        "# Extra Validation Summary",
        "",
        f"Metrics source: `{metrics_path}`",
        "",
        "| run | mse ↓ | psnr ↑ | ssim ↑ | ocr_cer ↓ | ocr_ned ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {mse} | {psnr} | {ssim} | {ocr_cer} | {ocr_ned_similarity} |".format(
                run=row["run"],
                mse=metric(row, "mse"),
                psnr=metric(row, "psnr"),
                ssim=metric(row, "ssim"),
                ocr_cer=metric(row, "ocr_cer"),
                ocr_ned_similarity=metric(row, "ocr_ned_similarity"),
            )
        )

    lines.extend(["", "## Reading"])
    if {"orig", "noocr"}.issubset(by_run):
        orig_mse = parse_float(by_run["orig"].get("mse"))
        noocr_mse = parse_float(by_run["noocr"].get("mse"))
        if orig_mse is not None and noocr_mse is not None:
            verdict = "better" if noocr_mse < orig_mse else "not better"
            lines.append(f"- noocr vs orig by MSE: noocr is {verdict}.")
    if {"noocr", "w005"}.issubset(by_run):
        noocr_mse = parse_float(by_run["noocr"].get("mse"))
        w005_mse = parse_float(by_run["w005"].get("mse"))
        if noocr_mse is not None and w005_mse is not None:
            verdict = "better" if w005_mse < noocr_mse else "not better"
            lines.append(f"- w005 vs noocr by MSE: w005 is {verdict}.")
    lines.append("- This extra validation set is held out from readable50; it is still small and diagnostic.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    data = read_json(args.metrics_json)
    rows = []
    num_images = data.get("num_images", "")
    for run, run_metrics in (data.get("runs") or {}).items():
        overall = run_metrics.get("overall") or {}
        rows.append(
            {
                "run": run,
                "num_images": num_images,
                "mse": overall.get("mse"),
                "mae": overall.get("mae"),
                "psnr": overall.get("psnr"),
                "ssim": overall.get("ssim"),
                "ocr_cer": overall.get("ocr_cer"),
                "ocr_exact_acc_ci": overall.get("ocr_exact_acc_ci"),
                "ocr_ned_similarity": overall.get("ocr_ned_similarity"),
            }
        )
    rows.sort(key=lambda row: ["orig", "noocr", "w005"].index(row["run"]) if row["run"] in {"orig", "noocr", "w005"} else 99)

    write_csv(args.output_dir / "summary.csv", rows)
    write_md(args.output_dir / "summary.md", rows, args.metrics_json)
    print(f"summary csv: {args.output_dir / 'summary.csv'}")
    print(f"summary md: {args.output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
