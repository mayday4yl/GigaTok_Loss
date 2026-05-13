#!/usr/bin/env python3
"""Evaluate OCR/text setting sweep checkpoints into compact selection tables."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


RUN_ORDER = ["orig", "noocr", "w001", "w002", "w003", "w005"]
SETTING_FIELDS = [
    "run",
    "ocr_weight",
    "text_injection",
    "ocr_loss",
    "step",
    "num_images",
    "image_metric_source",
    "image_metric_mse",
    "psnr",
    "ssim",
    "lpips",
    "deepseek_ocr_cer",
    "deepseek_ocr_ned",
    "deepseek_ocr_exact",
    "thirdparty_ocr_backend",
    "thirdparty_ocr_cer",
    "thirdparty_ocr_ned",
    "thirdparty_ocr_exact",
    "final_rec_loss",
    "final_ocr_tf_ce_loss",
]
PAPER_FIELDS = [
    "Setting",
    "Text Injection",
    "OCR Loss",
    "OCR Weight",
    "PSNR↑",
    "SSIM↑",
    "LPIPS↓",
    "Third-party OCR CER↓",
    "Third-party OCR NED↑",
    "DeepSeek OCR CER↓",
    "Best Setting Note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=5000)
    parser.add_argument(
        "--image-metric-source",
        choices=("setting_eval", "train_val_csv"),
        default="setting_eval",
        help="Where to read PSNR/SSIM from. train_val_csv matches the historical training summary; setting_eval uses the freshly recomputed reconstruction metrics.",
    )
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all rows in manifest.")
    parser.add_argument("--grid-samples", type=int, default=12)
    parser.add_argument("--grid-seed", type=int, default=0)
    parser.add_argument("--checkpoint-weight-key", choices=("auto", "model", "ema", "state_dict"), default="model", help="Use model by default to match training validation metrics; auto preserves historical EMA preference.")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-deepseek-ocr", action="store_true")
    parser.add_argument("--deepseek-ocr-model", default="/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR")
    parser.add_argument("--deepseek-ocr-prompt", default="<image>\n<|grounding|>OCR this image.")
    parser.add_argument("--deepseek-ocr-base-size", type=int, default=1024)
    parser.add_argument("--deepseek-ocr-image-size", type=int, default=640)
    parser.add_argument("--deepseek-ocr-attn-implementation", default="eager")
    parser.add_argument("--deepseek-ocr-dtype", choices=("auto", "fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--deepseek-ocr-max-new-tokens", type=int, default=1024)
    parser.add_argument("--thirdparty-ocr-backend", choices=("auto", "none", "paddleocr", "easyocr"), default="auto")
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


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})


def csv_value(value: Any) -> Any:
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


def fmt(value: Any, digits: int = 4) -> str:
    number = parse_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def load_runs(path: Path) -> Dict[str, Dict[str, Any]]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object")
    runs: Dict[str, Dict[str, Any]] = {}
    for name, spec in data.items():
        if not isinstance(spec, dict):
            raise ValueError(f"run spec for {name} must be an object")
        for key in ("config", "ckpt", "run_dir"):
            value = Path(str(spec.get(key, "")))
            if not value.exists():
                raise FileNotFoundError(f"{name}.{key} not found: {value}")
        runs[str(name)] = dict(spec)
    return {name: runs[name] for name in RUN_ORDER if name in runs}


def run_specs(runs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return [f"{name}:{spec['config']}:{spec['ckpt']}" for name, spec in runs.items()]


def run_evaluate(
    *,
    args: argparse.Namespace,
    runs: Mapping[str, Mapping[str, Any]],
    output_dir: Path,
    ocr_backend: str,
    ocr_jsonl: Path | None,
    grid_samples: int,
) -> Path:
    cmd = [
        args.python_bin,
        "scripts/stage1/evaluate_textatlas_reconstruction.py",
        "--manifest-jsonl",
        str(args.manifest_jsonl),
        "--output-dir",
        str(output_dir),
        "--device-backend",
        args.device_backend,
        "--device-id",
        str(args.device_id),
        "--mixed-precision",
        args.mixed_precision,
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--max-images",
        str(args.max_images),
        "--grid-samples",
        str(grid_samples),
        "--grid-seed",
        str(args.grid_seed),
        "--checkpoint-weight-key",
        args.checkpoint_weight_key,
        "--text-input-mode",
        "correct",
        "--ocr-backend",
        ocr_backend,
    ]
    for spec in run_specs(runs):
        cmd.extend(["--run", spec])
    if ocr_jsonl is not None:
        cmd.extend(["--ocr-jsonl", str(ocr_jsonl)])
    if ocr_backend == "deepseek_ocr":
        cmd.extend(
            [
                "--deepseek-ocr-model",
                str(args.deepseek_ocr_model),
                "--deepseek-ocr-prompt",
                args.deepseek_ocr_prompt,
                "--deepseek-ocr-base-size",
                str(args.deepseek_ocr_base_size),
                "--deepseek-ocr-image-size",
                str(args.deepseek_ocr_image_size),
                "--deepseek-ocr-attn-implementation",
                args.deepseek_ocr_attn_implementation,
                "--deepseek-ocr-dtype",
                args.deepseek_ocr_dtype,
                "--deepseek-ocr-output-dir",
                str(output_dir / "deepseek_ocr_tmp"),
                "--deepseek-ocr-max-new-tokens",
                str(args.deepseek_ocr_max_new_tokens),
            ]
        )
    subprocess.run(cmd, check=True)
    return output_dir / "metrics.json"


def mean(values: Sequence[float]) -> float | None:
    finite = [value for value in values if value == value]
    return sum(finite) / len(finite) if finite else None


def normalize_ocr_predictions(raw_path: Path, flat_path: Path) -> Dict[str, Dict[str, float | None]]:
    by_run: Dict[str, Dict[str, list[float]]] = {}
    flat_rows = []
    if not raw_path.exists():
        return {}
    for row in iter_jsonl(raw_path):
        metrics = row.get("metrics") or {}
        run = str(row.get("run") or "")
        flat = {
            "run": run,
            "image_path": row.get("image_path", ""),
            "gt_text": row.get("gt_text", ""),
            "pred_text": row.get("pred_text", ""),
            "cer": parse_float(metrics.get("ocr_cer")),
            "ned": parse_float(metrics.get("ocr_ned_similarity")),
            "exact": parse_float(metrics.get("ocr_exact_acc_ci")),
        }
        flat_rows.append(flat)
        acc = by_run.setdefault(run, {"cer": [], "ned": [], "exact": []})
        for src, dst in (("cer", "cer"), ("ned", "ned"), ("exact", "exact")):
            value = parse_float(flat.get(src))
            if value is not None:
                acc[dst].append(value)
    write_jsonl(flat_path, flat_rows)
    return {
        run: {
            "cer": mean(values["cer"]),
            "ned": mean(values["ned"]),
            "exact": mean(values["exact"]),
        }
        for run, values in by_run.items()
    }


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def final_train_value(run_dir: Path, field: str) -> str:
    rows = read_csv_rows(run_dir / "train.csv")
    for row in reversed(rows):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def final_validation_metrics(run_dir: Path) -> dict[str, Any]:
    candidates = [
        run_dir / "val.csv",
        run_dir / "metrics" / "val_metrics.csv",
    ]
    for path in candidates:
        rows = read_csv_rows(path)
        if not rows:
            continue
        final = rows[-1]
        return {
            "source": str(path),
            "step": parse_float(final.get("step")),
            "val_mse": parse_float(final.get("val_mse")),
            "val_psnr": parse_float(final.get("val_psnr")),
            "val_ssim": parse_float(final.get("val_ssim")),
            "val_images": parse_float(final.get("val_images")),
        }
    return {"source": None, "step": None, "val_mse": None, "val_psnr": None, "val_ssim": None, "val_images": None}


def thirdparty_backend(requested: str) -> tuple[str | None, str]:
    if requested == "none":
        return None, "disabled by --thirdparty-ocr-backend none"
    if requested in {"paddleocr", "easyocr"}:
        module = "paddleocr" if requested == "paddleocr" else "easyocr"
        if importlib.util.find_spec(module) is None:
            return None, f"requested {requested}, but Python package {module} is unavailable"
        return requested, f"using requested backend: {requested}"
    if importlib.util.find_spec("paddleocr") is not None:
        return "paddleocr", "auto-selected PaddleOCR"
    if importlib.util.find_spec("easyocr") is not None:
        return "easyocr", "auto-selected EasyOCR"
    return None, "Third-party OCR unavailable; checked PaddleOCR and EasyOCR"


def lpips_status() -> str:
    if importlib.util.find_spec("lpips") is None:
        return "LPIPS unavailable; Python package lpips is not installed."
    return "LPIPS package is available, but this compact evaluator leaves LPIPS blank unless a dedicated LPIPS pass is added."


def build_setting_rows(
    *,
    args: argparse.Namespace,
    runs: Mapping[str, Mapping[str, Any]],
    recon_metrics: Mapping[str, Any],
    image_metric_source: str,
    deepseek_metrics: Mapping[str, Mapping[str, float | None]],
    thirdparty_metrics: Mapping[str, Mapping[str, float | None]],
    thirdparty_name: str | None,
) -> list[dict[str, Any]]:
    num_images = recon_metrics.get("num_images", "")
    rows = []
    for name, spec in runs.items():
        overall = ((recon_metrics.get("runs") or {}).get(name) or {}).get("overall") or {}
        train = final_validation_metrics(Path(str(spec["run_dir"]))) if image_metric_source == "train_val_csv" else {}
        deep = deepseek_metrics.get(name, {})
        third = thirdparty_metrics.get(name, {})
        psnr = train.get("val_psnr") if image_metric_source == "train_val_csv" else overall.get("psnr")
        ssim = train.get("val_ssim") if image_metric_source == "train_val_csv" else overall.get("ssim")
        mse = train.get("val_mse") if image_metric_source == "train_val_csv" else overall.get("mse")
        image_metric_note = "training val.csv" if image_metric_source == "train_val_csv" else "setting_eval reconstruction"
        rows.append(
            {
                "run": name,
                "ocr_weight": spec.get("ocr_weight"),
                "text_injection": bool(spec.get("text_injection")),
                "ocr_loss": bool(spec.get("ocr_loss")),
                "step": args.step,
                "num_images": num_images,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": "",
                "deepseek_ocr_cer": deep.get("cer"),
                "deepseek_ocr_ned": deep.get("ned"),
                "deepseek_ocr_exact": deep.get("exact"),
                "thirdparty_ocr_backend": thirdparty_name or "",
                "thirdparty_ocr_cer": third.get("cer"),
                "thirdparty_ocr_ned": third.get("ned"),
                "thirdparty_ocr_exact": third.get("exact"),
                "final_rec_loss": final_train_value(Path(str(spec["run_dir"])), "rec_loss"),
                "final_ocr_tf_ce_loss": final_train_value(Path(str(spec["run_dir"])), "ocr_tf_ce_loss") if spec.get("ocr_loss") else "",
                "image_metric_source": image_metric_note,
                "image_metric_mse": mse,
            }
        )
    return rows


def metric_float_by_run(rows: Sequence[Mapping[str, Any]], run: str, field: str) -> float | None:
    for row in rows:
        if row.get("run") == run:
            return parse_float(row.get(field))
    return None


def choose_best(rows: Sequence[Mapping[str, Any]], thirdparty_available: bool) -> tuple[str, dict[str, str]]:
    notes: dict[str, str] = {}
    noocr_psnr = metric_float_by_run(rows, "noocr", "psnr")
    noocr_ssim = metric_float_by_run(rows, "noocr", "ssim")
    noocr_lpips = metric_float_by_run(rows, "noocr", "lpips")
    ocr_metric_cer = "thirdparty_ocr_cer" if thirdparty_available else "deepseek_ocr_cer"
    ocr_metric_ned = "thirdparty_ocr_ned" if thirdparty_available else "deepseek_ocr_ned"
    candidates = []
    for row in rows:
        run = str(row.get("run"))
        psnr = parse_float(row.get("psnr"))
        ssim = parse_float(row.get("ssim"))
        lpips_value = parse_float(row.get("lpips"))
        cer = parse_float(row.get(ocr_metric_cer))
        ned = parse_float(row.get(ocr_metric_ned))
        passes = True
        reasons = []
        if noocr_psnr is not None and psnr is not None and noocr_psnr - psnr > 0.3:
            passes = False
            reasons.append("PSNR drop > 0.3 dB vs noocr")
        if noocr_ssim is not None and ssim is not None and noocr_ssim - ssim > 0.005:
            passes = False
            reasons.append("SSIM drop > 0.005 vs noocr")
        if noocr_lpips is not None and lpips_value is not None and lpips_value > noocr_lpips * 1.05:
            passes = False
            reasons.append("LPIPS > 1.05x noocr")
        if cer is None:
            reasons.append("OCR CER unavailable")
        if passes and cer is not None:
            candidates.append((cer, -(ned if ned is not None else -1.0), run))
        notes[run] = "; ".join(reasons) if reasons else "passes image-quality gate"

    if not candidates:
        return "noocr" if any(row.get("run") == "noocr" for row in rows) else str(rows[0].get("run")), notes
    candidates.sort()
    best = candidates[0][2]
    noocr_cer = metric_float_by_run(rows, "noocr", ocr_metric_cer)
    best_cer = metric_float_by_run(rows, best, ocr_metric_cer)
    if best != "noocr" and noocr_cer is not None and best_cer is not None and abs(noocr_cer - best_cer) < 0.005:
        notes[best] = notes.get(best, "") + "; OCR gain vs noocr is very small"
        return "noocr", notes
    return best, notes


def paper_rows(rows: Sequence[Mapping[str, Any]], best: str, notes: Mapping[str, str]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        run = str(row.get("run"))
        text_injection = "yes" if row.get("text_injection") else "no"
        ocr_loss = "yes" if row.get("ocr_loss") else "no"
        weight = row.get("ocr_weight")
        if weight is None or weight == "":
            weight_text = "-"
        else:
            weight_text = str(weight)
        note = "recommended by current rule" if run == best else notes.get(run, "")
        out.append(
            {
                "Setting": run,
                "Text Injection": text_injection,
                "OCR Loss": ocr_loss,
                "OCR Weight": weight_text,
                "PSNR↑": fmt(row.get("psnr"), 3),
                "SSIM↑": fmt(row.get("ssim"), 4),
                "LPIPS↓": fmt(row.get("lpips"), 4),
                "Third-party OCR CER↓": fmt(row.get("thirdparty_ocr_cer"), 4),
                "Third-party OCR NED↑": fmt(row.get("thirdparty_ocr_ned"), 4),
                "DeepSeek OCR CER↓": fmt(row.get("deepseek_ocr_cer"), 4),
                "Best Setting Note": note,
            }
        )
    return out


def write_markdown_table(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]], extra_lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    if extra_lines:
        lines.extend(["", *extra_lines])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_by_metric(rows: Sequence[Mapping[str, Any]], field: str, higher: bool) -> str:
    pairs = []
    for row in rows:
        value = parse_float(row.get(field))
        if value is not None:
            pairs.append((value, str(row.get("run"))))
    if not pairs:
        return "unavailable"
    return (max if higher else min)(pairs)[1]


def write_setting_md(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    best: str,
    thirdparty_available: bool,
    thirdparty_note: str,
    lpips_note: str,
    image_metric_source: str,
) -> None:
    image_metric_note = "training val.csv" if image_metric_source == "train_val_csv" else "setting_eval reconstruction"
    lines = [
        "# Setting Metrics",
        "",
        f"- Image metrics source: {image_metric_note}.",
        f"- Best image PSNR: {best_by_metric(rows, 'psnr', True)}.",
        f"- Best image SSIM: {best_by_metric(rows, 'ssim', True)}.",
        f"- Best third-party OCR CER: {best_by_metric(rows, 'thirdparty_ocr_cer', False)}.",
        f"- Best DeepSeek OCR CER: {best_by_metric(rows, 'deepseek_ocr_cer', False)}.",
        f"- Recommended setting by current rule: {best}.",
        f"- Third-party OCR: {thirdparty_note}.",
        f"- {lpips_note}",
    ]
    if not thirdparty_available:
        lines.append(
            "- Third-party OCR unavailable; current OCR conclusion only uses DeepSeek diagnostic OCR and should not be treated as final readability evidence."
        )
    lines.append("- Need independent OCR-readable holdout for final model selection before large-scale training.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    thirdparty_available: bool,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"warning: matplotlib unavailable, skip plots: {exc}", file=sys.stderr)
        return

    labels = [str(row.get("run")) for row in rows]
    psnr = [parse_float(row.get("psnr")) for row in rows]
    cer_field = "thirdparty_ocr_cer" if thirdparty_available else "deepseek_ocr_cer"
    cer = [parse_float(row.get(cer_field)) for row in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, x, y in zip(labels, psnr, cer):
        if x is None or y is None:
            continue
        ax.scatter(x, y, s=60)
        ax.text(x, y, f" {label}", va="center", fontsize=9)
    ax.set_xlabel("PSNR")
    ax.set_ylabel(("Third-party" if thirdparty_available else "DeepSeek diagnostic") + " OCR CER")
    ax.set_title("PSNR vs OCR CER" + ("" if thirdparty_available else " (diagnostic only)"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "tradeoff_psnr_vs_cer.png", dpi=160)
    plt.close(fig)

    metrics = [
        ("PSNR", "psnr"),
        ("SSIM", "ssim"),
        ("LPIPS", "lpips"),
        ("OCR CER", cer_field),
        ("OCR NED", "thirdparty_ocr_ned" if thirdparty_available else "deepseek_ocr_ned"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.5))
    for ax, (title, field) in zip(axes, metrics):
        values = [parse_float(row.get(field)) for row in rows]
        ax.bar(labels, [value if value is not None else 0.0 for value in values], color="#4b5563")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        for idx, value in enumerate(values):
            if value is not None:
                ax.text(idx, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "compare_core_metrics.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs_json)
    if not args.manifest_jsonl.exists():
        raise FileNotFoundError(f"manifest not found: {args.manifest_jsonl}")

    recon_dir = args.output_dir / "reconstruction"
    recon_metrics_path = run_evaluate(
        args=args,
        runs=runs,
        output_dir=recon_dir,
        ocr_backend="none",
        ocr_jsonl=None,
        grid_samples=args.grid_samples,
    )
    recon_metrics = read_json(recon_metrics_path)
    grid_src = recon_dir / "comparison_grid.png"
    if grid_src.exists():
        shutil.copy2(grid_src, args.output_dir / "qual_grid_5000.png")

    deepseek_metrics: Dict[str, Dict[str, float | None]] = {}
    deepseek_note = "skipped"
    deepseek_model = Path(str(args.deepseek_ocr_model))
    if args.skip_deepseek_ocr:
        deepseek_note = "disabled by --skip-deepseek-ocr"
    elif not deepseek_model.exists():
        deepseek_note = f"DeepSeek OCR model not found: {deepseek_model}"
    else:
        deepseek_raw = args.output_dir / "deepseek_eval" / "ocr_raw.jsonl"
        try:
            run_evaluate(
                args=args,
                runs=runs,
                output_dir=args.output_dir / "deepseek_eval",
                ocr_backend="deepseek_ocr",
                ocr_jsonl=deepseek_raw,
                grid_samples=0,
            )
            deepseek_metrics = normalize_ocr_predictions(deepseek_raw, args.output_dir / "ocr_predictions_deepseek.jsonl")
            deepseek_note = "DeepSeek diagnostic OCR evaluated"
        except subprocess.CalledProcessError as exc:
            deepseek_note = f"DeepSeek diagnostic OCR failed with exit code {exc.returncode}; fields left blank"

    thirdparty_name, thirdparty_note = thirdparty_backend(args.thirdparty_ocr_backend)
    thirdparty_metrics: Dict[str, Dict[str, float | None]] = {}
    if thirdparty_name is not None:
        thirdparty_raw = args.output_dir / "thirdparty_eval" / "ocr_raw.jsonl"
        try:
            run_evaluate(
                args=args,
                runs=runs,
                output_dir=args.output_dir / "thirdparty_eval",
                ocr_backend=thirdparty_name,
                ocr_jsonl=thirdparty_raw,
                grid_samples=0,
            )
            thirdparty_metrics = normalize_ocr_predictions(thirdparty_raw, args.output_dir / "ocr_predictions_thirdparty.jsonl")
        except subprocess.CalledProcessError as exc:
            thirdparty_note = (
                f"{thirdparty_name} was importable but evaluation failed with exit code {exc.returncode}; "
                "third-party OCR fields left blank"
            )
            thirdparty_name = None

    rows = build_setting_rows(
        args=args,
        runs=runs,
        recon_metrics=recon_metrics,
        image_metric_source=args.image_metric_source,
        deepseek_metrics=deepseek_metrics,
        thirdparty_metrics=thirdparty_metrics,
        thirdparty_name=thirdparty_name,
    )
    write_csv(args.output_dir / "setting_metrics.csv", SETTING_FIELDS, rows)

    thirdparty_available = thirdparty_name is not None
    best, notes = choose_best(rows, thirdparty_available=thirdparty_available)
    p_rows = paper_rows(rows, best, notes)
    write_csv(args.output_dir / "paper_table.csv", PAPER_FIELDS, p_rows)

    summary_lines = [
        f"1. Best PSNR setting: {best_by_metric(rows, 'psnr', True)}.",
        f"2. Best third-party OCR CER/NED setting: {best_by_metric(rows, 'thirdparty_ocr_cer', False)} / {best_by_metric(rows, 'thirdparty_ocr_ned', True)}.",
        f"3. DeepSeek OCR best CER setting: {best_by_metric(rows, 'deepseek_ocr_cer', False)}.",
        f"4. OCR weight benefit should be accepted only if OCR metrics improve while PSNR/SSIM stay near noocr.",
        f"5. Recommended candidate: {best}.",
    ]
    if not thirdparty_available:
        summary_lines.append(
            "Third-party OCR unavailable; current OCR conclusion only uses DeepSeek diagnostic OCR and should not be treated as final readability evidence."
        )
    summary_lines.append(f"DeepSeek status: {deepseek_note}.")
    summary_lines.append(f"Third-party OCR status: {thirdparty_note}.")
    summary_lines.append("Need independent OCR-readable holdout for final model selection before large-scale training.")

    write_markdown_table(args.output_dir / "paper_table.md", PAPER_FIELDS, p_rows, summary_lines)
    write_setting_md(
        args.output_dir / "setting_metrics.md",
        rows,
        best,
        thirdparty_available=thirdparty_available,
        thirdparty_note=thirdparty_note,
        lpips_note=lpips_status(),
        image_metric_source=args.image_metric_source,
    )
    plot_outputs(args.output_dir, rows, thirdparty_available=thirdparty_available)
    write_json(
        args.output_dir / "evaluation_manifest.json",
        {
            "runs_json": str(args.runs_json),
            "manifest_jsonl": str(args.manifest_jsonl),
            "step": args.step,
            "image_metric_source": args.image_metric_source,
            "deepseek_note": deepseek_note,
            "thirdparty_note": thirdparty_note,
            "recommended_setting": best,
        },
    )

    print(f"setting_metrics: {args.output_dir / 'setting_metrics.csv'}")
    print(f"paper_table: {args.output_dir / 'paper_table.md'}")
    print(f"qual_grid: {args.output_dir / 'qual_grid_5000.png'}")
    print(f"recommended_setting: {best}")


if __name__ == "__main__":
    main()
