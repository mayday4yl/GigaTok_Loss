#!/usr/bin/env python3
"""Build the compact two-ablation evaluation table and text-region metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

try:
    from skimage.metrics import structural_similarity as ssim_metric
except Exception:  # noqa: BLE001
    ssim_metric = None


PAPER_FIELDS = [
    "Setting",
    "Train Loss Type",
    "High-rank",
    "Local-sim",
    "PSNR↑",
    "SSIM↑",
    "DeepSeek OCR CER↓",
    "Text-PSNR↑",
    "Text-SSIM↑",
    "Text-LPIPS↓",
    "Text-Laplacian-MSE↓",
    "bbox_failed_rate↓",
    "Third-party OCR CER↓",
    "Third-party OCR NED↑",
    "Note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=2500)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grid-samples", type=int, default=12)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--deepseek-ocr-model", default="/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR")
    parser.add_argument("--deepseek-ocr-max-new-tokens", type=int, default=1024)
    parser.add_argument("--thirdparty-ocr-status", default="unavailable")
    parser.add_argument("--bbox-debug-samples", type=int, default=12)
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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})


def fmt(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value in (1, "1", "true", "True", "yes", "Yes"):
        return "yes"
    if value in (0, "0", "false", "False", "no", "No"):
        return "no"
    if value in (None, ""):
        return ""
    return str(value)


def resize_pad_image(img: Image.Image, image_size: int, pad_color=(255, 255, 255)) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    scale = min(image_size / width, image_size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = img.resize(resized_size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), pad_color)
    canvas.paste(resized, ((image_size - resized_size[0]) // 2, (image_size - resized_size[1]) // 2))
    return canvas


def manifest_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    for row in rows:
        image_path = Path(str(row.get("image_path") or row.get("path") or row.get("file_name") or ""))
        if not image_path.exists():
            raise FileNotFoundError(f"manifest image missing: {image_path}")
    return rows


def row_image_path(row: Mapping[str, Any]) -> Path:
    return Path(str(row.get("image_path") or row.get("path") or row.get("file_name")))


def estimate_text_bbox(gt_image: Image.Image, margin: int = 8) -> tuple[tuple[int, int, int, int], bool, float]:
    gray = np.asarray(gt_image.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    edge = max(4, min(h, w) // 32)
    border = np.concatenate([
        gray[:edge, :].reshape(-1),
        gray[-edge:, :].reshape(-1),
        gray[:, :edge].reshape(-1),
        gray[:, -edge:].reshape(-1),
    ])
    bg = float(np.median(border))
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])
    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    grad = gx + gy
    grad_thr = max(0.06, float(np.percentile(grad, 94)))
    mask = (
        (np.abs(gray - bg) > 0.10)
        | (gray < min(0.75, bg - 0.08))
        | (grad > grad_thr)
    )
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(size=7))
    mask_img = mask_img.filter(ImageFilter.MinFilter(size=3))
    mask = np.asarray(mask_img) > 0
    ys, xs = np.where(mask)
    failed = False
    if len(xs) < 20:
        failed = True
        return (0, 0, w, h), failed, 1.0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    area_ratio = ((x1 - x0) * (y1 - y0)) / float(w * h)
    if area_ratio < 0.002:
        failed = True
        return (0, 0, w, h), failed, 1.0
    return (x0, y0, x1, y1), failed, area_ratio


def psnr_from_mse(mse: float) -> float:
    if mse <= 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def laplacian(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    center = padded[1:-1, 1:-1]
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * center
    )


class OptionalLPIPS:
    def __init__(self):
        self.status = "unavailable"
        self.model = None
        try:
            import lpips  # type: ignore
            self.model = lpips.LPIPS(net="alex").eval().cpu()
            self.status = "available"
        except Exception as exc:  # noqa: BLE001
            self.status = f"unavailable: {exc!r}"
            self.model = None

    def __call__(self, gt_crop: Image.Image, rec_crop: Image.Image) -> float | None:
        if self.model is None:
            return None
        gt = gt_crop.resize((128, 128), resample=Image.Resampling.BICUBIC)
        rec = rec_crop.resize((128, 128), resample=Image.Resampling.BICUBIC)
        gt_arr = np.asarray(gt, dtype=np.float32) / 127.5 - 1.0
        rec_arr = np.asarray(rec, dtype=np.float32) / 127.5 - 1.0
        gt_tensor = torch.from_numpy(gt_arr).permute(2, 0, 1).unsqueeze(0)
        rec_tensor = torch.from_numpy(rec_arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            return float(self.model(gt_tensor, rec_tensor).item())


def compute_text_region_metrics(
        *,
        rows: Sequence[Mapping[str, Any]],
        runs: Mapping[str, Mapping[str, Any]],
        reconstruction_root: Path,
        output_dir: Path,
        image_size: int,
        debug_samples: int,
) -> dict[str, Any]:
    lpips_metric = OptionalLPIPS()
    detail_rows = []
    bbox_rows = []
    summary: dict[str, dict[str, Any]] = {}
    debug_cells = []

    for idx, row in enumerate(rows):
        gt = resize_pad_image(Image.open(row_image_path(row)), image_size)
        bbox, failed, area_ratio = estimate_text_bbox(gt)
        bbox_rows.append({
            "image_index": idx,
            "image_path": str(row_image_path(row)),
            "bbox_xyxy": bbox,
            "bbox_area_ratio": area_ratio,
            "text_bbox_failed": int(failed),
        })
        if idx < debug_samples:
            dbg = gt.copy()
            draw = ImageDraw.Draw(dbg)
            draw.rectangle(bbox, outline=(255, 0, 0), width=3)
            draw.text((4, 4), f"#{idx} failed={int(failed)}", fill=(255, 0, 0))
            debug_cells.append(dbg)
        gt_crop = gt.crop(bbox)
        gt_arr = np.asarray(gt_crop, dtype=np.float32) / 255.0
        gt_gray = np.asarray(gt_crop.convert("L"), dtype=np.float32) / 255.0
        gt_lap = laplacian(gt_gray)

        for run in runs:
            rec_path = reconstruction_root / run / f"{idx:03d}.png"
            if not rec_path.exists():
                raise FileNotFoundError(f"missing reconstruction image: {rec_path}")
            rec = Image.open(rec_path).convert("RGB").resize((image_size, image_size), Image.Resampling.BICUBIC)
            rec_crop = rec.crop(bbox)
            rec_arr = np.asarray(rec_crop, dtype=np.float32) / 255.0
            rec_gray = np.asarray(rec_crop.convert("L"), dtype=np.float32) / 255.0
            mse = float(np.mean((gt_arr - rec_arr) ** 2))
            text_psnr = psnr_from_mse(mse)
            text_ssim = None
            if ssim_metric is not None and min(gt_gray.shape) >= 7:
                text_ssim = float(ssim_metric(gt_gray, rec_gray, data_range=1.0))
            rec_lap = laplacian(rec_gray)
            lap_mse = float(np.mean((gt_lap - rec_lap) ** 2))
            text_lpips = lpips_metric(gt_crop, rec_crop)
            detail_rows.append({
                "run": run,
                "image_index": idx,
                "image_path": str(row_image_path(row)),
                "bbox_xyxy": bbox,
                "bbox_area_ratio": area_ratio,
                "text_bbox_failed": int(failed),
                "text_psnr": text_psnr,
                "text_ssim": text_ssim,
                "text_lpips": text_lpips,
                "text_laplacian_mse": lap_mse,
            })

    if debug_cells:
        cols = min(4, len(debug_cells))
        cell = image_size
        out = Image.new("RGB", (cols * cell, math.ceil(len(debug_cells) / cols) * cell), (255, 255, 255))
        for idx, cell_img in enumerate(debug_cells):
            out.paste(cell_img, ((idx % cols) * cell, (idx // cols) * cell))
        out.save(output_dir / "text_bbox_debug_grid.png")

    for run in runs:
        run_rows = [row for row in detail_rows if row["run"] == run]
        def mean_field(field: str) -> float | None:
            values = [row[field] for row in run_rows if row.get(field) is not None]
            if not values:
                return None
            return float(np.mean(values))
        summary[run] = {
            "text_psnr": mean_field("text_psnr"),
            "text_ssim": mean_field("text_ssim"),
            "text_lpips": mean_field("text_lpips"),
            "text_laplacian_mse": mean_field("text_laplacian_mse"),
            "bbox_failed_rate": float(np.mean([row["text_bbox_failed"] for row in run_rows])) if run_rows else None,
            "num_images": len(run_rows),
        }
    out_summary = {
        "lpips_status": lpips_metric.status,
        "runs": summary,
    }
    write_jsonl(output_dir / "text_region_metrics.jsonl", detail_rows)
    write_jsonl(output_dir / "text_bbox_records.jsonl", bbox_rows)
    write_json(output_dir / "text_region_metrics_summary.json", out_summary)
    return out_summary


def run_base_eval(args: argparse.Namespace) -> None:
    cmd = [
        args.python_bin,
        "scripts/stage1/ocr_debug/evaluate_setting_metrics.py",
        "--runs-json",
        str(args.runs_json),
        "--manifest-jsonl",
        str(args.manifest_jsonl),
        "--output-dir",
        str(args.output_dir),
        "--step",
        str(args.step),
        "--image-metric-source",
        "train_val_csv",
        "--device-backend",
        args.device_backend,
        "--device-id",
        str(args.device_id),
        "--mixed-precision",
        args.mixed_precision,
        "--image-size",
        str(args.image_size),
        "--batch-size",
        str(args.batch_size),
        "--grid-samples",
        str(args.grid_samples),
        "--checkpoint-weight-key",
        "model",
        "--deepseek-ocr-model",
        args.deepseek_ocr_model,
        "--deepseek-ocr-max-new-tokens",
        str(args.deepseek_ocr_max_new_tokens),
        "--thirdparty-ocr-backend",
        "none",
    ]
    subprocess.run(cmd, check=True)


def copy_reconstructions(output_dir: Path, runs: Mapping[str, Any], num_images: int) -> Path:
    source_root = output_dir / "deepseek_eval" / "deepseek_ocr_tmp"
    if not source_root.exists():
        raise FileNotFoundError(f"DeepSeek temp image directory not found: {source_root}")
    reconstruction_root = output_dir / "reconstruction"
    total = 0
    for run_index, run in enumerate(runs):
        run_dir = reconstruction_root / run
        run_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(num_images):
            src_num = run_index * num_images + idx + 1
            src = source_root / f"recon_{src_num:06d}.png"
            dst = run_dir / f"{idx:03d}.png"
            if not src.exists():
                raise FileNotFoundError(f"missing source reconstruction image: {src}")
            shutil.copy2(src, dst)
            total += 1
    if total != len(runs) * num_images:
        raise RuntimeError(f"expected {len(runs) * num_images} reconstructions, copied {total}")
    return reconstruction_root


def build_final_tables(
        *,
        args: argparse.Namespace,
        runs: Mapping[str, Mapping[str, Any]],
        text_summary: Mapping[str, Any],
) -> None:
    base_rows = read_csv_rows(args.output_dir / "setting_metrics.csv")
    rows_by_run = {row["run"]: row for row in base_rows}
    text_runs = text_summary.get("runs", {})
    out_rows = []
    setting_rows = []
    for run in runs:
        spec = runs.get(run, {})
        base = rows_by_run.get(run, {})
        text = text_runs.get(run, {})
        if spec.get("train_loss_type") is not None:
            train_loss_type = str(spec.get("train_loss_type"))
            high_rank = yes_no(spec.get("high_rank", ""))
            local_sim = yes_no(spec.get("local_sim", spec.get("local_similarity", "")))
            note = str(spec.get("note", ""))
        elif run == "w001":
            train_loss_type = "text + OCR CE(0.01)"
            high_rank = "no"
            local_sim = "no"
            note = "baseline reused at target step"
        elif run == "w001_ocrvis_only":
            train_loss_type = "OCR visual projected feature alignment"
            high_rank = "no"
            local_sim = "no"
            note = "single OCR CE -> OCR visual projected feature alignment"
        elif run == "w001_hr_only":
            train_loss_type = "text + OCR CE(0.01) + HR only"
            high_rank = "yes"
            local_sim = "no"
            note = "single high-rank ablation"
        elif run == "w001_local_only":
            train_loss_type = "text + OCR CE(0.01) + local-sim only"
            high_rank = "no"
            local_sim = "yes"
            note = "single local-sim ablation"
        elif run == "w001_hr_local":
            train_loss_type = "text + OCR CE(0.01) + HR + local-sim"
            high_rank = "yes"
            local_sim = "yes"
            note = "combined route context only"
        elif run == "ocrvis_hr_local":
            train_loss_type = "OCR visual projected alignment + HR + local-sim"
            high_rank = "yes"
            local_sim = "yes"
            note = "combined route context only; OCR CE disabled"
        else:
            train_loss_type = str(spec.get("train_loss_type", "unknown"))
            high_rank = yes_no(spec.get("high_rank", ""))
            local_sim = yes_no(spec.get("local_sim", spec.get("local_similarity", "")))
            note = str(spec.get("note", "unrecognized setting"))
        setting_row = dict(base)
        setting_row.update({
            "train_loss_type": train_loss_type,
            "high_rank": high_rank,
            "local_sim": local_sim,
            "text_psnr": text.get("text_psnr"),
            "text_ssim": text.get("text_ssim"),
            "text_lpips": text.get("text_lpips"),
            "text_laplacian_mse": text.get("text_laplacian_mse"),
            "bbox_failed_rate": text.get("bbox_failed_rate"),
        })
        setting_rows.append(setting_row)
        out_rows.append({
            "Setting": run,
            "Train Loss Type": train_loss_type,
            "High-rank": high_rank,
            "Local-sim": local_sim,
            "PSNR↑": fmt(base.get("psnr"), 3),
            "SSIM↑": fmt(base.get("ssim"), 4),
            "DeepSeek OCR CER↓": fmt(base.get("deepseek_ocr_cer"), 4),
            "Text-PSNR↑": fmt(text.get("text_psnr"), 3),
            "Text-SSIM↑": fmt(text.get("text_ssim"), 4),
            "Text-LPIPS↓": fmt(text.get("text_lpips"), 4),
            "Text-Laplacian-MSE↓": fmt(text.get("text_laplacian_mse"), 6),
            "bbox_failed_rate↓": fmt(text.get("bbox_failed_rate"), 4),
            "Third-party OCR CER↓": "" if args.thirdparty_ocr_status == "unavailable" else args.thirdparty_ocr_status,
            "Third-party OCR NED↑": "",
            "Note": note,
        })
    write_csv(args.output_dir / "setting_metrics.csv", list(setting_rows[0].keys()), setting_rows)
    write_csv(args.output_dir / "paper_table.csv", PAPER_FIELDS, out_rows)

    lines = [
        "| " + " | ".join(PAPER_FIELDS) + " |",
        "| " + " | ".join(["---"] * len(PAPER_FIELDS)) + " |",
    ]
    for row in out_rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in PAPER_FIELDS) + " |")
    lines.extend([
        "",
        f"Preliminary {args.step}-step quick screening.",
        "Do not compare directly with 5000-step final tables.",
        "PSNR/SSIM are read from training validation CSV at the target step, not recomputed post-hoc.",
        "DeepSeek OCR CER is teacher-related diagnostic and not sufficient as independent evidence.",
        "Text-region metrics use the same GT-derived bbox for all settings of the same sample.",
        "Primary local visual evidence is text-region SSIM/LPIPS/Laplacian metrics.",
        "Causal single-module comparisons are: w001 vs w001_ocrvis_only for OCR CE -> OCR visual projected alignment; w001 vs w001_hr_only for high-rank only; w001 vs w001_local_only for local-sim only.",
        "w001_hr_local and ocrvis_hr_local are combined-route context only; do not attribute their changes to a single module.",
        f"Third-party OCR status: {args.thirdparty_ocr_status}.",
        f"LPIPS status: {text_summary.get('lpips_status')}.",
    ])
    (args.output_dir / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text("\n".join(lines[3:]) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.step <= 0:
        raise SystemExit("--step must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = read_json(args.runs_json)
    rows = manifest_rows(args.manifest_jsonl)
    if len(rows) != 50:
        raise ValueError(f"expected readable50 manifest with 50 rows, got {len(rows)}")

    run_base_eval(args)
    reconstruction_root = copy_reconstructions(args.output_dir, runs, len(rows))
    text_summary = compute_text_region_metrics(
        rows=rows,
        runs=runs,
        reconstruction_root=reconstruction_root,
        output_dir=args.output_dir,
        image_size=args.image_size,
        debug_samples=args.bbox_debug_samples,
    )
    build_final_tables(args=args, runs=runs, text_summary=text_summary)
    write_json(args.output_dir / "thirdparty_ocr_summary.json", {"status": args.thirdparty_ocr_status})
    if (args.output_dir / "qual_grid_5000.png").exists():
        shutil.copy2(args.output_dir / "qual_grid_5000.png", args.output_dir / f"qual_grid_{args.step}.png")
    print(f"paper_table: {args.output_dir / 'paper_table.md'}")
    print(f"setting_metrics: {args.output_dir / 'setting_metrics.csv'}")
    print(f"reconstruction_root: {reconstruction_root}")


if __name__ == "__main__":
    main()
