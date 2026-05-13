#!/usr/bin/env python3
"""Post-process the OCR teacher-forcing reconstruction-effect sweep."""

from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TRAIN_RE = re.compile(
    r"\(step=(?P<step>\d+)\)\s+Train Loss:\s+(?P<train_loss>[-+0-9.eE]+),\s+"
    r"Train Steps/Sec:\s+(?P<steps_per_sec>[-+0-9.eE]+),\s+"
    r"LR:\s+(?P<lr>[-+0-9.eE]+),\s+LR_disc:\s+(?P<lr_disc>[-+0-9.eE]+)"
)
VAL_RE = re.compile(
    r"\(step=(?P<step>\d+)\)\s+Val MSE:\s+(?P<val_mse>[-+0-9.eE]+),\s+"
    r"Val MAE:\s+(?P<val_mae>[-+0-9.eE]+),\s+"
    r"Val PSNR:\s+(?P<val_psnr>[-+0-9.eE]+),\s+"
    r"Val SSIM:\s+(?P<val_ssim>[-+0-9.eE]+)"
)
FIELD_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s+(?P<value>[-+0-9.eE]+)")

TRAIN_FIELDS = [
    "step",
    "train_loss",
    "rec_loss",
    "direct_rec_loss",
    "perceptual_loss",
    "direct_p_loss",
    "vq_loss",
    "commit_loss",
    "feature_rec_loss",
    "proj_loss",
    "ocr_tf_ce_loss",
    "weighted_ocr_tf_loss",
    "ocr_tf_loss_weight",
]
VAL_FIELDS = [
    "step",
    "val_mse",
    "val_mae",
    "val_psnr",
    "val_ssim",
    "val_images",
    "is_best",
]
SUMMARY_FIELDS = [
    "run",
    "ocr_weight",
    "final_step",
    "best_val_mse",
    "final_val_mse",
    "best_val_psnr",
    "final_val_psnr",
    "best_val_ssim",
    "final_val_ssim",
    "final_rec_loss",
    "final_ocr_tf_ce_loss",
    "final_weighted_ocr_tf_loss",
    "final_feature_rec_loss",
    "final_proj_loss",
]

RECON_METRICS = ("rec_loss", "val_mse", "val_psnr", "val_ssim")
OCR_METRICS = ("ocr_tf_ce_loss", "weighted_ocr_tf_loss", "ocr_tf_loss_weight")
GIGA_METRICS = (
    "direct_rec_loss",
    "perceptual_loss",
    "direct_p_loss",
    "vq_loss",
    "commit_loss",
    "feature_rec_loss",
    "proj_loss",
)
ALL_LOSS_METRICS = (
    "rec_loss",
    "direct_rec_loss",
    "perceptual_loss",
    "direct_p_loss",
    "vq_loss",
    "commit_loss",
    "feature_rec_loss",
    "proj_loss",
    "weighted_ocr_tf_loss",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["noocr", "w002", "w005"])
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--val-max-images", type=int, default=50)
    parser.add_argument("--grid-steps", nargs="+", type=int, default=[50, 250, 500])
    parser.add_argument("--grid-every", type=int, default=0, help="If >0, auto-generate grids for existing checkpoints divisible by this interval.")
    parser.add_argument("--grid-samples", type=int, default=16)
    parser.add_argument("--run-grids", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    return parser.parse_args()


def parse_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def fmt_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return value


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


def parse_generator_metrics(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for match in FIELD_RE.finditer(line):
        value = parse_float(match.group("value"))
        if value is not None:
            out[match.group("key")] = value
    return out


def parse_log(log_path: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    pending_generator: Dict[str, float] = {}
    if not log_path.exists():
        return train_rows, val_rows

    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "(Generator)" in line:
                pending_generator = parse_generator_metrics(line)
                continue

            train_match = TRAIN_RE.search(line)
            if train_match:
                row: Dict[str, Any] = {
                    "step": int(train_match.group("step")),
                    "train_loss": parse_float(train_match.group("train_loss")),
                    "steps_per_sec": parse_float(train_match.group("steps_per_sec")),
                    "lr": parse_float(train_match.group("lr")),
                    "lr_disc": parse_float(train_match.group("lr_disc")),
                }
                row.update(pending_generator)
                pending_generator = {}
                train_rows.append(row)
                continue

            val_match = VAL_RE.search(line)
            if val_match:
                val_rows.append({
                    "step": int(val_match.group("step")),
                    "val_mse": parse_float(val_match.group("val_mse")),
                    "val_mae": parse_float(val_match.group("val_mae")),
                    "val_psnr": parse_float(val_match.group("val_psnr")),
                    "val_ssim": parse_float(val_match.group("val_ssim")),
                })
    return train_rows, val_rows


def standardize_val_rows(run_dir: Path, log_val_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    csv_rows = read_csv(run_dir / "metrics" / "val_metrics.csv")
    if not csv_rows:
        return [dict(row) for row in log_val_rows]
    out: List[Dict[str, Any]] = []
    for row in csv_rows:
        out.append({
            "step": int(float(row["step"])),
            "val_mse": parse_float(row.get("val_mse")),
            "val_mae": parse_float(row.get("val_mae")),
            "val_psnr": parse_float(row.get("val_psnr")),
            "val_ssim": parse_float(row.get("val_ssim")),
            "val_images": int(float(row["val_images"])) if row.get("val_images") else None,
            "is_best": int(float(row["is_best"])) if row.get("is_best") else None,
        })
    return out


def rows_for_metric(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]], metric: str) -> tuple[List[int], List[float]]:
    source = val_rows if metric.startswith("val_") else train_rows
    points = []
    for row in source:
        value = parse_float(row.get(metric))
        step = row.get("step")
        if value is not None and step is not None:
            points.append((int(step), value))
    return [step for step, _ in points], [value for _, value in points]


def final_step(train_rows: Sequence[Mapping[str, Any]], val_rows: Sequence[Mapping[str, Any]]) -> Optional[int]:
    steps: List[int] = []
    for rows in (train_rows, val_rows):
        for row in rows:
            step = row.get("step")
            if step is not None:
                steps.append(int(step))
    return max(steps) if steps else None


def import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_empty_plot(plt: Any, output_path: Path, title: str, message: str = "no data") -> None:
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_set(
        plt: Any,
        output_path: Path,
        title: str,
        train_rows: Sequence[Mapping[str, Any]],
        val_rows: Sequence[Mapping[str, Any]],
        metrics: Sequence[str],
) -> None:
    available = [metric for metric in metrics if rows_for_metric(train_rows, val_rows, metric)[0]]
    if not available:
        write_empty_plot(plt, output_path, title)
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(9, max(3, 2.4 * len(available))), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        steps, values = rows_for_metric(train_rows, val_rows, metric)
        ax.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=metric)
        if metric == "proj_loss":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_train_loss(
        plt: Any,
        output_path: Path,
        train_rows: Sequence[Mapping[str, Any]],
) -> None:
    metrics = ("train_loss", "rec_loss")
    if not any(rows_for_metric(train_rows, [], metric)[0] for metric in metrics):
        write_empty_plot(plt, output_path, "train loss")
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for metric in metrics:
        steps, values = rows_for_metric(train_rows, [], metric)
        if steps:
            ax.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=metric)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Train loss")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_val_metrics(
        plt: Any,
        output_path: Path,
        val_rows: Sequence[Mapping[str, Any]],
) -> None:
    plot_metric_set(plt, output_path, "validation metrics", [], val_rows, ("val_mse", "val_psnr", "val_ssim"))


def plot_train_val_recon(
        plt: Any,
        output_path: Path,
        train_rows: Sequence[Mapping[str, Any]],
        val_rows: Sequence[Mapping[str, Any]],
) -> None:
    plot_metric_set(
        plt,
        output_path,
        "train/val reconstruction metrics",
        train_rows,
        val_rows,
        ("rec_loss", "val_mse", "val_psnr", "val_ssim"),
    )


def plot_ocr_losses(
        plt: Any,
        output_path: Path,
        train_rows: Sequence[Mapping[str, Any]],
) -> None:
    plot_metric_set(
        plt,
        output_path,
        "OCR losses: raw CE, weighted contribution, and weight",
        train_rows,
        [],
        ("ocr_tf_ce_loss", "weighted_ocr_tf_loss", "ocr_tf_loss_weight"),
    )


def plot_compare_set(
        plt: Any,
        output_path: Path,
        title: str,
        run_data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
        metrics: Sequence[str],
) -> None:
    available = [
        metric for metric in metrics
        if any(rows_for_metric(data["train"], data["val"], metric)[0] for data in run_data.values())
    ]
    if not available:
        write_empty_plot(plt, output_path, title)
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(10, max(3, 2.5 * len(available))), sharex=True)
    if len(available) == 1:
        axes = [axes]
    for ax, metric in zip(axes, available):
        for run, data in run_data.items():
            steps, values = rows_for_metric(data["train"], data["val"], metric)
            if steps:
                ax.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=run)
        if metric == "proj_loss":
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def last_value(rows: Sequence[Mapping[str, Any]], metric: str) -> Optional[float]:
    for row in reversed(rows):
        value = parse_float(row.get(metric))
        if value is not None:
            return value
    return None


def best_value(rows: Sequence[Mapping[str, Any]], metric: str, mode: str) -> Optional[float]:
    values = [parse_float(row.get(metric)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return None
    return min(values) if mode == "min" else max(values)


def infer_ocr_weight(group: str) -> Optional[float]:
    if group == "w002":
        return 0.02
    if group == "w005":
        return 0.05
    return None


def checkpoint_path(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"{step:07d}.pt"


def existing_grid_steps(run_dir: Path, grid_every: int) -> List[int]:
    if grid_every <= 0:
        return []
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return []
    steps: List[int] = []
    for path in ckpt_dir.glob("*.pt"):
        if not path.stem.isdigit():
            continue
        step = int(path.stem)
        if step > 0 and step % grid_every == 0:
            steps.append(step)
    return sorted(set(steps))


def generate_grid(
        *,
        python_bin: str,
        save_root: Path,
        group: str,
        config: Path,
        ckpt: Path,
        val_manifest: Path,
        output_path: Path,
        device_backend: str,
        mixed_precision: str,
        eval_batch_size: int,
        val_max_images: int,
        grid_samples: int,
) -> None:
    step = int(ckpt.stem) if ckpt.stem.isdigit() else 0
    tmp_dir = output_path.parent / f"eval_{step}" if step > 0 else save_root / "grid_eval" / group / ckpt.stem
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        "scripts/stage1/evaluate_textatlas_reconstruction.py",
        "--run",
        f"{group}:{config}:{ckpt}",
        "--manifest-jsonl",
        str(val_manifest),
        "--output-dir",
        str(tmp_dir),
        "--device-backend",
        device_backend,
        "--mixed-precision",
        mixed_precision,
        "--batch-size",
        str(eval_batch_size),
        "--max-images",
        str(val_max_images),
        "--grid-samples",
        str(grid_samples),
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    grid = tmp_dir / "comparison_grid.png"
    if grid.exists():
        shutil.copyfile(grid, output_path)


def plot_loss_scale(
        plt: Any,
        output_path: Path,
        run_data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> None:
    metrics = ("rec_loss", "perceptual_loss", "feature_rec_loss", "proj_loss", "weighted_ocr_tf_loss")
    groups = list(run_data)
    width = 0.15
    x_positions = list(range(len(metrics)))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for idx, group in enumerate(groups):
        values = [last_value(run_data[group]["train"], metric) for metric in metrics]
        xs = [x + (idx - (len(groups) - 1) / 2.0) * width for x in x_positions]
        first_bar = True
        for x, value in zip(xs, values):
            if value is None:
                continue
            ax.bar(x, value, width=width, label=group if first_bar else None)
            first_bar = False
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_ylabel("raw loss value at final logged step")
    ax.set_title("Final logged major loss scale; proj_loss keeps raw signed value")
    ax.grid(True, axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_summary_rows(run_data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for group, data in run_data.items():
        train_rows = data["train"]
        val_rows = data["val"]
        rows.append({
            "run": group,
            "ocr_weight": infer_ocr_weight(group),
            "final_step": final_step(train_rows, val_rows),
            "best_val_mse": best_value(val_rows, "val_mse", "min"),
            "final_val_mse": last_value(val_rows, "val_mse"),
            "best_val_psnr": best_value(val_rows, "val_psnr", "max"),
            "final_val_psnr": last_value(val_rows, "val_psnr"),
            "best_val_ssim": best_value(val_rows, "val_ssim", "max"),
            "final_val_ssim": last_value(val_rows, "val_ssim"),
            "final_rec_loss": last_value(train_rows, "rec_loss"),
            "final_ocr_tf_ce_loss": last_value(train_rows, "ocr_tf_ce_loss"),
            "final_weighted_ocr_tf_loss": last_value(train_rows, "weighted_ocr_tf_loss"),
            "final_feature_rec_loss": last_value(train_rows, "feature_rec_loss"),
            "final_proj_loss": last_value(train_rows, "proj_loss"),
        })
    return rows


def compare_delta(rows_by_run: Mapping[str, Mapping[str, Any]], run_a: str, run_b: str, metric: str) -> Optional[float]:
    a = parse_float(rows_by_run.get(run_a, {}).get(metric))
    b = parse_float(rows_by_run.get(run_b, {}).get(metric))
    if a is None or b is None:
        return None
    return a - b


def fmt_md(value: Any, precision: int = 6) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.{precision}g}"


def write_summary_md(
        path: Path,
        summary_rows: Sequence[Mapping[str, Any]],
        missing_grids: Mapping[str, Sequence[str]],
) -> None:
    rows_by_run = {str(row["run"]): row for row in summary_rows}
    w002_mse_delta = compare_delta(rows_by_run, "w002", "noocr", "final_val_mse")
    w002_psnr_delta = compare_delta(rows_by_run, "w002", "noocr", "final_val_psnr")
    w005_mse_delta = compare_delta(rows_by_run, "w005", "w002", "final_val_mse")
    w005_psnr_delta = compare_delta(rows_by_run, "w005", "w002", "final_val_psnr")

    def trend_sentence(delta: Optional[float], better_when_lower: bool) -> str:
        if delta is None:
            return "缺少可比较字段。"
        improved = delta < 0 if better_when_lower else delta > 0
        worsened = delta > 0 if better_when_lower else delta < 0
        if improved:
            return "改善。"
        if worsened:
            return "变差。"
        return "基本持平。"

    lines = [
        "# OCR TF readable50 sweep summary",
        "",
        "说明：`readable50` 是诊断实验，只用于看机制和曲线，不代表完整 TextAtlas 泛化。",
        "缺失字段保持为空，表示该组未启用对应 loss 或日志没有产生该指标；没有用 0 填充。",
        "`proj_loss` 保留训练日志中的 raw signed value，可能为负。",
        "",
        "## Summary table",
        "",
        "| run | ocr_weight | final_step | best_val_mse | final_val_mse | best_val_psnr | final_val_psnr | best_val_ssim | final_val_ssim | final_rec_loss | final_ocr_tf_ce_loss | final_weighted_ocr_tf_loss | final_feature_rec_loss | final_proj_loss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            "| {run} | {ocr_weight} | {final_step} | {best_val_mse} | {final_val_mse} | {best_val_psnr} | "
            "{final_val_psnr} | {best_val_ssim} | {final_val_ssim} | {final_rec_loss} | {final_ocr_tf_ce_loss} | "
            "{final_weighted_ocr_tf_loss} | {final_feature_rec_loss} | {final_proj_loss} |".format(
                run=row["run"],
                ocr_weight=fmt_md(row.get("ocr_weight")),
                final_step=fmt_md(row.get("final_step"), precision=0),
                best_val_mse=fmt_md(row.get("best_val_mse")),
                final_val_mse=fmt_md(row.get("final_val_mse")),
                best_val_psnr=fmt_md(row.get("best_val_psnr")),
                final_val_psnr=fmt_md(row.get("final_val_psnr")),
                best_val_ssim=fmt_md(row.get("best_val_ssim")),
                final_val_ssim=fmt_md(row.get("final_val_ssim")),
                final_rec_loss=fmt_md(row.get("final_rec_loss")),
                final_ocr_tf_ce_loss=fmt_md(row.get("final_ocr_tf_ce_loss")),
                final_weighted_ocr_tf_loss=fmt_md(row.get("final_weighted_ocr_tf_loss")),
                final_feature_rec_loss=fmt_md(row.get("final_feature_rec_loss")),
                final_proj_loss=fmt_md(row.get("final_proj_loss")),
            )
        )

    lines.extend([
        "",
        "## Required answers",
        "",
        "1. OCR CE 是否下降。",
        "   - 看 `ocr_losses.png` 的 raw `ocr_tf_ce_loss`；它和 `weighted_ocr_tf_loss` 不是同一个量。",
        "2. weighted OCR loss 的量级有多大。",
        "   - 看 `ocr_losses.png` 和 `summary.csv` 的 `final_weighted_ocr_tf_loss`；前期受 warmup 影响，上升是正常的。",
        "3. w005 是否比 noocr 更早清晰。",
        "   - 对比每 500 step 的 grid，尤其是 `grid_500.png` 到 `grid_2000.png`。",
        "4. w005 最终 5000 step 是否比 noocr 好。",
        "   - 对比 `final_val_mse / final_val_psnr / final_val_ssim` 和 `grid_5000.png`。",
        "5. w005 是否伤重建。",
        "   - 如果 w005 的 MSE 更高、PSNR/SSIM 更低，或 grid 更差，则说明 OCR 0.05 仍在伤重建。",
        "6. 如果 OCR CE 降了但重建没变好。",
        "   - 说明 OCR 目标可能被优化了，但没有转化成 reconstruction improvement。",
        "",
        "## Legacy 500-step sweep checks",
        "",
        "1. w002 相比 noocr，重建曲线有没有改善？",
        f"   - final_val_mse 差值 w002-noocr：{fmt_md(w002_mse_delta)}，{trend_sentence(w002_mse_delta, True)}",
        f"   - final_val_psnr 差值 w002-noocr：{fmt_md(w002_psnr_delta)}，{trend_sentence(w002_psnr_delta, False)}",
        "2. w005 相比 w002，OCR 曲线有没有更明显下降？",
        "   - 以 `cmp_ocr.png` 和 `summary.csv` 的 final raw/weighted OCR loss 为准；若 raw `ocr_tf_ce_loss` 没下降，则不认为更明显。",
        "3. w005 是否开始伤重建，比如 val_mse 变差、val_psnr 下降、grid 变差？",
        f"   - final_val_mse 差值 w005-w002：{fmt_md(w005_mse_delta)}，{trend_sentence(w005_mse_delta, True)}",
        f"   - final_val_psnr 差值 w005-w002：{fmt_md(w005_psnr_delta)}，{trend_sentence(w005_psnr_delta, False)}",
        "   - grid 需要结合对应 step 的 `grid_*.png` 人工确认。",
        "4. 如果 w005 的 weighted OCR 已经很大但 raw OCR 曲线不下降，说明不要继续加 OCR weight。",
        "   - 判断时看 `cmp_ocr.png`：若 `weighted_ocr_tf_loss` 量级明显增大而 `ocr_tf_ce_loss` 不下降，停止继续加权重。",
        "",
        "## Checkpoint / grid status",
        "",
    ])
    for group, messages in missing_grids.items():
        if messages:
            lines.append(f"- {group}: " + "; ".join(messages))
        else:
            lines.append(f"- {group}: requested grids generated or already available.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    save_root = args.save_root.resolve()
    plt = import_pyplot()
    run_data: Dict[str, Dict[str, Sequence[Mapping[str, Any]]]] = {}
    missing_grids: Dict[str, List[str]] = {}

    for group in args.groups:
        run_dir = save_root / group
        train_rows, log_val_rows = parse_log(run_dir / "log.txt")
        val_rows = standardize_val_rows(run_dir, log_val_rows)
        write_csv(run_dir / "train.csv", TRAIN_FIELDS, train_rows)
        write_csv(run_dir / "val.csv", VAL_FIELDS, val_rows)
        run_data[group] = {"train": train_rows, "val": val_rows}

        plot_metric_set(plt, run_dir / "recon.png", f"{group}: reconstruction metrics", train_rows, val_rows, RECON_METRICS)
        plot_metric_set(plt, run_dir / "ocr.png", f"{group}: OCR teacher-forcing metrics", train_rows, val_rows, OCR_METRICS)
        plot_metric_set(plt, run_dir / "giga.png", f"{group}: GigaTok native losses", train_rows, val_rows, GIGA_METRICS)
        plot_metric_set(plt, run_dir / "all_loss.png", f"{group}: all monitored losses", train_rows, val_rows, ALL_LOSS_METRICS)
        plot_train_loss(plt, run_dir / "train_loss.png", train_rows)
        plot_val_metrics(plt, run_dir / "val_metrics.png", val_rows)
        plot_train_val_recon(plt, run_dir / "train_val_recon.png", train_rows, val_rows)
        plot_metric_set(plt, run_dir / "giga_losses.png", f"{group}: GigaTok native losses", train_rows, val_rows, GIGA_METRICS)
        plot_ocr_losses(plt, run_dir / "ocr_losses.png", train_rows)

        missing_grids[group] = []
        if args.run_grids:
            config = save_root / "configs" / f"{group}.yaml"
            if not config.exists():
                missing_grids[group].append(f"config missing: {config}")
            grid_steps = existing_grid_steps(run_dir, args.grid_every) if args.grid_every > 0 else list(args.grid_steps)
            for step in grid_steps:
                ckpt = checkpoint_path(run_dir, step)
                output = run_dir / f"grid_{step}.png"
                if not ckpt.exists():
                    missing_grids[group].append(f"checkpoint missing for grid_{step}: {ckpt}")
                    continue
                if not config.exists():
                    continue
                generate_grid(
                    python_bin=args.python_bin,
                    save_root=save_root,
                    group=group,
                    config=config,
                    ckpt=ckpt,
                    val_manifest=args.val_manifest,
                    output_path=output,
                    device_backend=args.device_backend,
                    mixed_precision=args.mixed_precision,
                    eval_batch_size=args.eval_batch_size,
                    val_max_images=args.val_max_images,
                    grid_samples=args.grid_samples,
                )

    summary_dir = save_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_compare_set(plt, summary_dir / "cmp_recon.png", "OCR-off vs OCR weights: reconstruction", run_data, ("val_mse", "val_psnr", "rec_loss"))
    plot_compare_set(plt, summary_dir / "cmp_ocr.png", "OCR teacher-forcing curves", run_data, OCR_METRICS)
    plot_compare_set(plt, summary_dir / "cmp_giga.png", "GigaTok native loss stability", run_data, ("feature_rec_loss", "proj_loss", "perceptual_loss", "direct_rec_loss"))
    plot_loss_scale(plt, summary_dir / "cmp_loss_scale.png", run_data)

    summary_rows = build_summary_rows(run_data)
    write_csv(summary_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_summary_md(summary_dir / "summary.md", summary_rows, missing_grids)
    print(f"wrote {summary_dir}")


if __name__ == "__main__":
    main()
