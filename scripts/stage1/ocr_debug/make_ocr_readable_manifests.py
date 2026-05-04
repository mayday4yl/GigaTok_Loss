#!/usr/bin/env python3
"""基于 GT OCR 可读性结果筛选机制验证 manifest。

输入来自 ``diagnose_gt_ocr_readability.py`` 生成的
``gt_ocr_predictions.jsonl``。脚本会按 CER / NED 阈值筛出当前 256x256
物化图里 OCR 仍能较好读出的样本，并写出同批 train / val / holdout
manifest。这个输出只用于排除“文字已经不可读”的混杂因素，不代表独立验证集。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--ocr-predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="cleantextsynth_ocr_readable_50")
    parser.add_argument("--max-rows", type=int, default=50, help="0 means keep all matched rows.")
    parser.add_argument("--max-cer", type=float, default=0.05)
    parser.add_argument("--min-ned", type=float, default=0.95)
    parser.add_argument(
        "--match-mode",
        choices=("any", "all"),
        default="any",
        help="any: CER 或 NED 满足即可；all: 两个阈值都必须满足。",
    )
    parser.add_argument("--min-text-chars", type=int, default=1)
    parser.add_argument("--require-existing-images", action="store_true")
    parser.add_argument(
        "--allow-fewer",
        action="store_true",
        help="不足 max-rows 时仍然写出已有样本；默认不足会报错。",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_line_no"] = line_no
            yield row


def resolve_image_path(row: Mapping[str, Any], manifest_path: Path) -> Path:
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = manifest_path.resolve().parent / image_path
    return image_path.resolve()


def normalize_path(value: Any) -> str:
    return str(Path(str(value)).expanduser().resolve())


def load_source_rows(path: Path, min_text_chars: int, require_existing_images: bool) -> Dict[str, Dict[str, Any]]:
    rows_by_path: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        image_path = resolve_image_path(row, path)
        text = str(row.get("text") or "").strip()
        if len(text) < min_text_chars:
            continue
        if require_existing_images and not image_path.exists():
            continue
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        rows_by_path[str(image_path)] = clean
    return rows_by_path


def metric_value(row: Mapping[str, Any], key: str) -> float:
    metrics = row.get("metrics") or {}
    try:
        return float(metrics[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def passes_thresholds(row: Mapping[str, Any], max_cer: float, min_ned: float, match_mode: str) -> bool:
    cer = metric_value(row, "ocr_cer")
    ned = metric_value(row, "ocr_ned_similarity")
    checks = []
    if max_cer >= 0:
        checks.append(cer <= max_cer)
    if min_ned >= 0:
        checks.append(ned >= min_ned)
    if not checks:
        raise ValueError("At least one threshold must be enabled.")
    return all(checks) if match_mode == "all" else any(checks)


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def mean_or_none(values: Sequence[float]) -> float | None:
    finite = [value for value in values if value == value]
    return mean(finite) if finite else None


def main() -> None:
    args = parse_args()
    if args.max_rows < 0:
        raise SystemExit("--max-rows must be >= 0.")
    if args.min_text_chars < 0:
        raise SystemExit("--min-text-chars must be >= 0.")

    source_rows = load_source_rows(
        args.source_manifest,
        min_text_chars=args.min_text_chars,
        require_existing_images=args.require_existing_images,
    )
    if not source_rows:
        raise SystemExit(f"No eligible source rows found in {args.source_manifest}.")

    candidates: list[tuple[Dict[str, Any], Dict[str, Any]]] = []
    for pred in iter_jsonl(args.ocr_predictions_jsonl):
        image_path = normalize_path(pred.get("image_path", ""))
        source_row = source_rows.get(image_path)
        if source_row is None:
            continue
        if not passes_thresholds(pred, args.max_cer, args.min_ned, args.match_mode):
            continue
        candidates.append((source_row, pred))

    candidates.sort(
        key=lambda item: (
            metric_value(item[1], "ocr_cer"),
            -metric_value(item[1], "ocr_ned_similarity"),
            str(item[0].get("source_key") or ""),
        )
    )
    selected_pairs = candidates if args.max_rows == 0 else candidates[: args.max_rows]

    if args.max_rows > 0 and len(selected_pairs) < args.max_rows and not args.allow_fewer:
        raise SystemExit(
            f"Only found {len(selected_pairs)} OCR-readable rows, need {args.max_rows}. "
            "Use --allow-fewer or relax --max-cer / --min-ned."
        )
    if not selected_pairs:
        raise SystemExit("No OCR-readable rows selected. Relax thresholds.")

    rows = [row for row, _ in selected_pairs]
    preds = [pred for _, pred in selected_pairs]
    output_dir = args.output_dir
    train_out = output_dir / f"{args.name}_train.jsonl"
    val_out = output_dir / f"{args.name}_val.jsonl"
    holdout_out = output_dir / f"{args.name}_holdout.jsonl"
    for path in (train_out, val_out, holdout_out):
        write_jsonl(path, rows)

    cer_values = [metric_value(pred, "ocr_cer") for pred in preds]
    ned_values = [metric_value(pred, "ocr_ned_similarity") for pred in preds]
    text_lengths = [len(str(row.get("text") or "")) for row in rows]
    examples = []
    for row, pred in selected_pairs[:5]:
        examples.append(
            {
                "source_key": row.get("source_key", ""),
                "image_path": row.get("image_path", ""),
                "text_chars": len(str(row.get("text") or "")),
                "ocr_cer": metric_value(pred, "ocr_cer"),
                "ocr_ned_similarity": metric_value(pred, "ocr_ned_similarity"),
                "gt_preview": str(pred.get("gt_text") or "")[:180],
                "pred_preview": str(pred.get("pred_text") or "")[:180],
            }
        )

    summary = {
        "name": args.name,
        "source_manifest": str(args.source_manifest),
        "ocr_predictions_jsonl": str(args.ocr_predictions_jsonl),
        "thresholds": {
            "max_cer": args.max_cer,
            "min_ned": args.min_ned,
            "match_mode": args.match_mode,
        },
        "eligible_source_rows": len(source_rows),
        "matching_candidates": len(candidates),
        "selected_rows": len(rows),
        "same_eval_as_train": True,
        "train_manifest": str(train_out),
        "val_manifest": str(val_out),
        "holdout_manifest": str(holdout_out),
        "selected_metrics": {
            "ocr_cer_mean": mean_or_none(cer_values),
            "ocr_cer_max": max(cer_values),
            "ocr_ned_similarity_mean": mean_or_none(ned_values),
            "ocr_ned_similarity_min": min(ned_values),
            "text_chars_min": min(text_lengths),
            "text_chars_mean": mean(text_lengths),
            "text_chars_max": max(text_lengths),
        },
        "examples": examples,
    }
    summary_out = output_dir / f"{args.name}_summary.json"
    write_json(summary_out, summary)

    print(f"selected rows: {len(rows)} / candidates: {len(candidates)}")
    print(f"train manifest: {train_out}")
    print(f"val manifest: {val_out}")
    print(f"holdout manifest: {holdout_out}")
    print(f"summary: {summary_out}")
    print(
        "selected OCR: "
        f"mean CER={summary['selected_metrics']['ocr_cer_mean']:.4f}, "
        f"mean NED={summary['selected_metrics']['ocr_ned_similarity_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
