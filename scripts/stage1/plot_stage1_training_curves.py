#!/usr/bin/env python3
"""Plot stage-1 training curves from GigaTok tokenizer logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


LogSpec = Tuple[str, Path]


TRAIN_RE = re.compile(
    r"\(step=(?P<step>\d+)\)\s+Train Loss:\s+(?P<train_loss>[-+0-9.eE]+),\s+"
    r"Train Steps/Sec:\s+(?P<steps_per_sec>[-+0-9.eE]+),\s+"
    r"LR:\s+(?P<lr>[-+0-9.eE]+),\s+LR_disc:\s+(?P<lr_disc>[-+0-9.eE]+)"
)
KV_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*):\s+(?P<value>[-+0-9.eE]+)")


def parse_log_spec(value: str) -> LogSpec:
    parts = value.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("--log must be NAME:/path/to/log")
    return parts[0], Path(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", type=parse_log_spec, required=True, help="NAME:/path/to/train.log")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smooth-window", type=int, default=1, help="Moving average window for plotted curves.")
    return parser.parse_args()


def parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return math.nan


def parse_generator_metrics(line: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for match in KV_RE.finditer(line):
        metrics[match.group("key")] = parse_float(match.group("value"))
    return metrics


def parse_log(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    pending_generator: Dict[str, float] | None = None

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "(Generator)" in line:
                pending_generator = parse_generator_metrics(line)
                continue

            train_match = TRAIN_RE.search(line)
            if not train_match:
                continue

            row: Dict[str, float] = {
                "step": float(int(train_match.group("step"))),
                "train_loss": parse_float(train_match.group("train_loss")),
                "steps_per_sec": parse_float(train_match.group("steps_per_sec")),
                "lr": parse_float(train_match.group("lr")),
                "lr_disc": parse_float(train_match.group("lr_disc")),
            }
            if pending_generator:
                row.update(pending_generator)
                pending_generator = None
            rows.append(row)

    return rows


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    smoothed: List[float] = []
    running = 0.0
    queue: List[float] = []
    for value in values:
        queue.append(value)
        running += value
        if len(queue) > window:
            running -= queue.pop(0)
        smoothed.append(running / len(queue))
    return smoothed


def metric_values(rows: Sequence[Mapping[str, float]], metric: str, smooth_window: int) -> Tuple[List[float], List[float]]:
    points = [(row["step"], row[metric]) for row in rows if metric in row and not math.isnan(row[metric])]
    if not points:
        return [], []
    steps = [point[0] for point in points]
    values = moving_average([point[1] for point in points], smooth_window)
    return steps, values


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, all_rows: Mapping[str, Sequence[Mapping[str, float]]]) -> None:
    fields = {"run", "step"}
    for rows in all_rows.values():
        for row in rows:
            fields.update(row.keys())
    ordered_fields = ["run", "step"] + sorted(field for field in fields if field not in {"run", "step"})

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fields)
        writer.writeheader()
        for run_name, rows in all_rows.items():
            for row in rows:
                out = {"run": run_name, **row}
                writer.writerow(out)


def summarize(all_rows: Mapping[str, Sequence[Mapping[str, float]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for run_name, rows in all_rows.items():
        if not rows:
            summary[run_name] = {"rows": 0}
            continue
        train_rows = [row for row in rows if "train_loss" in row and not math.isnan(row["train_loss"])]
        best_train = min(train_rows, key=lambda row: row["train_loss"]) if train_rows else None
        last = rows[-1]
        summary[run_name] = {
            "rows": len(rows),
            "first_step": int(rows[0]["step"]),
            "last_step": int(last["step"]),
            "last_train_loss": last.get("train_loss"),
            "last_steps_per_sec": last.get("steps_per_sec"),
            "best_train_loss_step": int(best_train["step"]) if best_train else None,
            "best_train_loss": best_train["train_loss"] if best_train else None,
            "last_rec_loss": last.get("rec_loss"),
            "last_direct_rec_loss": last.get("direct_rec_loss"),
            "last_perceptual_loss": last.get("perceptual_loss"),
            "last_feature_rec_loss": last.get("feature_rec_loss"),
            "last_hr_loss": last.get("hr_loss"),
            "last_weighted_hr_loss": last.get("weighted_hr_loss"),
        }
    return summary


def import_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_metric_group(
    plt: Any,
    all_rows: Mapping[str, Sequence[Mapping[str, float]]],
    metrics: Sequence[str],
    output_path: Path,
    title: str,
    smooth_window: int,
    logy_metrics: Iterable[str] = (),
) -> bool:
    available_metrics = [metric for metric in metrics if any(metric in row for rows in all_rows.values() for row in rows)]
    if not available_metrics:
        return False

    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(10, max(3, 2.8 * len(available_metrics))), sharex=True)
    if len(available_metrics) == 1:
        axes = [axes]

    logy_metric_set = set(logy_metrics)
    for ax, metric in zip(axes, available_metrics):
        for run_name, rows in all_rows.items():
            steps, values = metric_values(rows, metric, smooth_window)
            if steps:
                ax.plot(steps, values, label=run_name, linewidth=1.6)
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        if metric in logy_metric_set:
            ax.set_yscale("log")
        ax.legend(loc="best")

    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: Dict[str, List[Dict[str, float]]] = {}
    for run_name, log_path in args.log:
        rows = parse_log(log_path)
        if not rows:
            raise SystemExit(f"No train-loss rows parsed from {log_path}")
        all_rows[run_name] = rows

    write_csv(output_dir / "training_curves.csv", all_rows)
    write_json(output_dir / "training_summary.json", summarize(all_rows))

    plt = import_pyplot()
    plot_metric_group(
        plt,
        all_rows,
        metrics=("train_loss",),
        output_path=output_dir / "train_loss.png",
        title="Stage-1 train loss",
        smooth_window=args.smooth_window,
    )
    plot_metric_group(
        plt,
        all_rows,
        metrics=("rec_loss", "direct_rec_loss", "perceptual_loss", "direct_p_loss", "feature_rec_loss", "codebook_usage"),
        output_path=output_dir / "reconstruction_components.png",
        title="Stage-1 reconstruction-related components",
        smooth_window=args.smooth_window,
    )
    plot_metric_group(
        plt,
        all_rows,
        metrics=("hr_loss", "weighted_hr_loss", "hr_spectrum_uniformity", "selected_layer"),
        output_path=output_dir / "hr_metrics.png",
        title="Stage-1 HR metrics",
        smooth_window=args.smooth_window,
        logy_metrics=("hr_loss", "weighted_hr_loss"),
    )

    print(f"Wrote plots and summaries to {output_dir}")


if __name__ == "__main__":
    main()
