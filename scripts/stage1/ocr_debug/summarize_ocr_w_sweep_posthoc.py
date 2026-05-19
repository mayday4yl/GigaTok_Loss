#!/usr/bin/env python3
"""Summarize post-hoc DeepSeek-OCR eval for ocr_w_sweep checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SUMMARY_FIELDS = [
    "run",
    "ocr_weight",
    "num_images",
    "ocr_cer_mean",
    "ocr_exact_acc_ci_mean",
    "ocr_ned_similarity_mean",
    "ocr_edit_distance_mean",
    "empty_pred_rate",
    "ocr_pred_chars_mean",
    "ocr_gt_chars_mean",
    "final_val_mse",
    "final_val_psnr",
    "final_rec_loss",
    "final_ocr_tf_ce_loss",
    "final_weighted_ocr_tf_loss",
]

OCR_METRIC_FIELDS = [
    "ocr_cer",
    "ocr_exact_acc_ci",
    "ocr_ned_similarity",
    "ocr_edit_distance",
    "ocr_gt_chars",
    "ocr_pred_chars",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["noocr", "w002", "w005"])
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt_csv(row.get(field)) for field in fields})


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def fmt_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


def fmt_md(value: Any) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.6g}"


def mean(values: Sequence[float]) -> Optional[float]:
    finite = [value for value in values if value == value]
    return sum(finite) / len(finite) if finite else None


def load_prior_summary(save_root: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(save_root / "summary" / "summary.csv")
    return {row.get("run", ""): row for row in rows}


def flatten_ocr_rows(raw_path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in iter_jsonl(raw_path):
        metrics = raw.get("metrics") or {}
        row: Dict[str, Any] = {
            "index": raw.get("index"),
            "image_path": raw.get("image_path", ""),
            "gt_text": raw.get("gt_text", ""),
            "pred_text": raw.get("pred_text", ""),
        }
        for field in OCR_METRIC_FIELDS:
            row[field] = parse_float(metrics.get(field))
        rows.append(row)
    return rows


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pred_texts = [str(row.get("pred_text") or "") for row in rows]
    summary: Dict[str, Any] = {
        "num_images": len(rows),
        "empty_pred_rate": (sum(1 for text in pred_texts if not text.strip()) / len(rows)) if rows else None,
    }
    for field in OCR_METRIC_FIELDS:
        summary[f"{field}_mean"] = mean([value for row in rows if (value := parse_float(row.get(field))) is not None])
    return summary


def normalize_group_outputs(group_dir: Path) -> Dict[str, Any]:
    raw_path = group_dir / "ocr_predictions.raw.jsonl"
    if not raw_path.exists():
        alt = group_dir / "ocr_predictions.jsonl"
        raw_path = alt if alt.exists() else raw_path
    if not raw_path.exists():
        raise FileNotFoundError(f"OCR predictions not found in {group_dir}")

    rows = flatten_ocr_rows(raw_path)
    flat_path = group_dir / "ocr_preds.jsonl"
    append_jsonl(flat_path, rows)

    summary = summarize_rows(rows)
    metrics_path = group_dir / "metrics.json"
    paths_path = group_dir / "eval_paths.json"
    if metrics_path.exists():
        metrics = read_json(metrics_path)
        summary["metrics_json"] = str(metrics_path)
        summary["eval_overall"] = (metrics.get("runs") or {}).get(group_dir.name, {}).get("overall", {})
    if paths_path.exists():
        summary.update(read_json(paths_path))
    write_json(group_dir / "ocr_summary.json", summary)
    return summary


def build_summary_rows(
    *,
    out_dir: Path,
    groups: Sequence[str],
    prior_by_run: Mapping[str, Mapping[str, str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group in groups:
        group_dir = out_dir / group
        ocr_summary = normalize_group_outputs(group_dir)
        prior = prior_by_run.get(group, {})
        rows.append({
            "run": group,
            "ocr_weight": prior.get("ocr_weight", ""),
            "num_images": ocr_summary.get("num_images"),
            "ocr_cer_mean": ocr_summary.get("ocr_cer_mean"),
            "ocr_exact_acc_ci_mean": ocr_summary.get("ocr_exact_acc_ci_mean"),
            "ocr_ned_similarity_mean": ocr_summary.get("ocr_ned_similarity_mean"),
            "ocr_edit_distance_mean": ocr_summary.get("ocr_edit_distance_mean"),
            "empty_pred_rate": ocr_summary.get("empty_pred_rate"),
            "ocr_pred_chars_mean": ocr_summary.get("ocr_pred_chars_mean"),
            "ocr_gt_chars_mean": ocr_summary.get("ocr_gt_chars_mean"),
            "final_val_mse": prior.get("final_val_mse", ""),
            "final_val_psnr": prior.get("final_val_psnr", ""),
            "final_rec_loss": prior.get("final_rec_loss", ""),
            "final_ocr_tf_ce_loss": prior.get("final_ocr_tf_ce_loss", ""),
            "final_weighted_ocr_tf_loss": prior.get("final_weighted_ocr_tf_loss", ""),
            "config_path": ocr_summary.get("config_path", ""),
            "checkpoint_path": ocr_summary.get("checkpoint_path", ""),
        })
    return rows


def metric_delta(row_a: Mapping[str, Any], row_b: Mapping[str, Any], field: str) -> Optional[float]:
    a = parse_float(row_a.get(field))
    b = parse_float(row_b.get(field))
    if a is None or b is None:
        return None
    return b - a


def better_ocr(new: Mapping[str, Any], old: Mapping[str, Any]) -> Optional[bool]:
    cer_delta = metric_delta(old, new, "ocr_cer_mean")
    exact_delta = metric_delta(old, new, "ocr_exact_acc_ci_mean")
    ned_delta = metric_delta(old, new, "ocr_ned_similarity_mean")
    if cer_delta is None or exact_delta is None or ned_delta is None:
        return None
    return cer_delta < 0 and (exact_delta > 0 or ned_delta > 0)


def plot_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        "ocr_cer_mean",
        "ocr_exact_acc_ci_mean",
        "ocr_ned_similarity_mean",
        "empty_pred_rate",
    ]
    labels = [str(row.get("run", "")) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes_flat = list(axes.flatten())
    x = np.arange(len(labels))
    for ax, metric in zip(axes_flat, metrics):
        values = [parse_float(row.get(metric)) for row in rows]
        ax.bar(x, [value if value is not None else 0.0 for value in values], color=["#6b7280", "#2563eb", "#dc2626"][: len(rows)])
        ax.set_title(metric)
        ax.set_xticks(x, labels)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            label = "" if value is None else f"{value:.3g}"
            ax.text(idx, (value if value is not None else 0.0), label, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    by_run = {str(row.get("run")): row for row in rows}
    noocr = by_run.get("noocr", {})
    w002 = by_run.get("w002", {})
    w005 = by_run.get("w005", {})

    w002_better = better_ocr(w002, noocr) if w002 and noocr else None
    w005_better = better_ocr(w005, w002) if w005 and w002 else None
    w005_vs_w002_cer = metric_delta(w002, w005, "ocr_cer_mean") if w005 and w002 else None
    w005_vs_w002_exact = metric_delta(w002, w005, "ocr_exact_acc_ci_mean") if w005 and w002 else None
    w005_vs_w002_ned = metric_delta(w002, w005, "ocr_ned_similarity_mean") if w005 and w002 else None

    empty_rates = [parse_float(row.get("empty_pred_rate")) for row in rows]
    exacts = [parse_float(row.get("ocr_exact_acc_ci_mean")) for row in rows]
    neds = [parse_float(row.get("ocr_ned_similarity_mean")) for row in rows]
    cers = [parse_float(row.get("ocr_cer_mean")) for row in rows]
    unreadable = (
        all(value is not None and value >= 0.8 for value in empty_rates)
        or (
            all(value is not None and value <= 0.02 for value in exacts)
            and (
                all(value is not None and value >= 0.8 for value in cers)
                or all(value is not None and value <= 0.15 for value in neds)
            )
        )
    )

    lines = [
        "# Post-hoc OCR summary",
        "",
        "说明：本评估只使用 `ocr_w_sweep` 三组已有 step500 checkpoint，对 val readable50 reconstruction 做 DeepSeek-OCR 后验识别；没有重新训练，也没有改 crop、visual mask、scale 或 ranking loss。",
        "",
        "## Paths",
        "",
    ]
    for row in rows:
        lines.extend([
            f"- {row.get('run')}:",
            f"  - config: `{row.get('config_path', '')}`",
            f"  - checkpoint: `{row.get('checkpoint_path', '')}`",
        ])
    lines.extend([
        "",
        "## Summary table",
        "",
        "| run | OCR weight | CER mean | exact CI mean | NED mean | empty pred rate | val MSE | val PSNR | rec loss | OCR CE | weighted OCR |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for row in rows:
        lines.append(
            "| {run} | {w} | {cer} | {exact} | {ned} | {empty} | {mse} | {psnr} | {rec} | {ce} | {wce} |".format(
                run=row.get("run", ""),
                w=row.get("ocr_weight", ""),
                cer=fmt_md(row.get("ocr_cer_mean")),
                exact=fmt_md(row.get("ocr_exact_acc_ci_mean")),
                ned=fmt_md(row.get("ocr_ned_similarity_mean")),
                empty=fmt_md(row.get("empty_pred_rate")),
                mse=fmt_md(row.get("final_val_mse")),
                psnr=fmt_md(row.get("final_val_psnr")),
                rec=fmt_md(row.get("final_rec_loss")),
                ce=fmt_md(row.get("final_ocr_tf_ce_loss")),
                wce=fmt_md(row.get("final_weighted_ocr_tf_loss")),
            )
        )

    lines.extend([
        "",
        "## Required answers",
        "",
        "1. `w002` 的 OCR 后验指标是否比 `noocr` 好？",
        f"   - {'是' if w002_better else '否' if w002_better is False else '无法判断'}。判断口径是 CER 更低，并且 exact 或 NED 至少一项更高。",
        "2. `w005` 的 OCR 后验指标是否比 `w002` 好？",
        f"   - {'是' if w005_better else '否' if w005_better is False else '无法判断'}。w005-w002: CER delta={fmt_md(w005_vs_w002_cer)}, exact delta={fmt_md(w005_vs_w002_exact)}, NED delta={fmt_md(w005_vs_w002_ned)}。",
        "3. `w005` 虽然训练时 OCR CE 更低，是否真的转化成 OCR CER / exact / NED 改善？",
        f"   - {'是' if w005_better else '否' if w005_better is False else '无法判断'}，以表格中的后验 OCR 指标为准。",
        "4. 如果三组 OCR 都几乎读不出来：",
        f"   - {'当前 reconstruction 还没达到文字可读门槛，OCR TF loss 暂时没有转化成真实可读性提升。' if unreadable else '三组不是全部处于几乎不可读状态，需结合 CER/exact/NED 和样例 OCR 文本判断。'}",
        "5. 结合之前 reconstruction 指标，判断 OCR loss 当前的作用：",
    ])
    noocr_mse = parse_float(noocr.get("final_val_mse"))
    w002_mse = parse_float(w002.get("final_val_mse"))
    w005_mse = parse_float(w005.get("final_val_mse"))
    if noocr_mse is not None and w002_mse is not None and w005_mse is not None and noocr_mse < w002_mse < w005_mse:
        lines.append("   - 当前 OCR loss 没有帮助重建，重建指标从 noocr 到 w002/w005 变差；若后验 OCR 也没有同步改善，则更像是只优化了自己的 CE，尚未转化为可读性。")
    else:
        lines.append("   - 需要同时看 reconstruction 和 OCR 后验指标；若 OCR 指标改善但 val_mse/val_psnr 变差，说明 OCR loss 有影响但可能在伤重建。")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary_dir = args.out_dir / "summary"
    prior_by_run = load_prior_summary(args.save_root)
    rows = build_summary_rows(out_dir=args.out_dir, groups=args.groups, prior_by_run=prior_by_run)
    write_csv(summary_dir / "post_ocr_summary.csv", SUMMARY_FIELDS, rows)
    write_markdown(summary_dir / "post_ocr_summary.md", rows)
    plot_summary(summary_dir / "cmp_post_ocr.png", rows)
    print(f"summary_csv={summary_dir / 'post_ocr_summary.csv'}")
    print(f"summary_md={summary_dir / 'post_ocr_summary.md'}")
    print(f"summary_plot={summary_dir / 'cmp_post_ocr.png'}")


if __name__ == "__main__":
    main()
