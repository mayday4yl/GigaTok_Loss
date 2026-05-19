#!/usr/bin/env python3
"""Select text-rich hardmix TextAtlas manifests for OCR-box-gate training.

This script is intentionally conservative:
- all image statistics are computed on the same 256x256 resize-pad input used by
  tokenizer training;
- OCR boxes are not silently faked: the default "boxes" are explicitly marked as
  a lightweight image-component estimate;
- original GigaTok easy-sparse filtering uses the image-only VQ checkpoint/config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from skimage import filters, measure, metrics, morphology  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


REPO_HINTS = (
    Path("/home/ma-user/work/GigaTok_hr/GigaTok_Loss_two_ablation"),
    Path("/home/ma-user/work/GigaTok_hr/GigaTok_Loss"),
)
PERSIST_ROOT = Path("/home/ma-user/work/GigaTok_hr/gigatok_persist")
OUT_DIR = PERSIST_ROOT / "outputs/text_rich_hardmix_manifests"

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
TARGET_BUCKETS = {
    "train": {"local": 2000, "medium": 4000, "dense": 3000, "ultra_dense": 1000},
    "val": {"local": 200, "medium": 400, "dense": 300, "ultra_dense": 100},
    "holdout": {"local": 100, "medium": 200, "dense": 150, "ultra_dense": 50},
}
TARGET_COUNTS = {"train": 10000, "val": 1000, "holdout": 500}
RESERVE_COUNTS = {"train": 15000, "val": 1500, "holdout": 800}
CANDIDATE_CAPS = {"train": 30000, "val": 3000, "holdout": 1500}
IMAGE_SIZE = 256
SEED = 0


@dataclass
class Paths:
    repo: Path
    train_manifest: Path
    val_manifest: Path
    holdout_manifest: Path
    tokenizer: Path
    vq_config: Path
    vq_ckpt: Path


def jdump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def discover_paths() -> Paths:
    repo = next((path for path in REPO_HINTS if path.exists()), None)
    if repo is None:
        raise FileNotFoundError(f"repo not found under hints: {REPO_HINTS}")
    train = PERSIST_ROOT / "outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl"
    val = PERSIST_ROOT / "outputs/textatlas_stage1_fixed_310k/manifest/val_materialized_manifest_v2text.jsonl"
    holdout = PERSIST_ROOT / "outputs/textatlas_stage1_fixed_310k/manifest_holdout_eval/holdout_materialized_manifest_v2text.jsonl"
    tokenizer = PERSIST_ROOT / "models/google_byt5-small"
    vq_config = repo / "configs/vq/VQ_BL256_dino_disc.yaml"
    vq_ckpt = PERSIST_ROOT / "checkpoints/VQ_BL256_dino_disc.pt"
    for path in (train, val, holdout, tokenizer, vq_config, vq_ckpt):
        if not path.exists():
            raise FileNotFoundError(path)
    return Paths(repo, train, val, holdout, tokenizer, vq_config, vq_ckpt)


def choose_text_field(row: Mapping[str, Any]) -> Tuple[str, str]:
    counts = []
    for field in TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return field, value.strip()
        if value not in (None, "", []):
            text = str(value).strip()
            if text:
                return field, text
    return "", ""


def scan_manifest(path: Path, split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = []
    field_hits = Counter()
    image_missing = 0
    empty_text = 0
    for line_no, row in enumerate(iter_jsonl(path), start=1):
        text_field, text = choose_text_field(row)
        if text_field:
            field_hits[text_field] += 1
        else:
            empty_text += 1
        image_path = row.get("image_path")
        if not image_path:
            image_missing += 1
        normalized = dict(row)
        normalized["_line_no"] = line_no
        normalized["_split"] = split
        normalized["text_field_used"] = text_field
        normalized["_selected_text"] = text
        rows.append(normalized)
    stats = {
        "path": str(path),
        "rows": len(rows),
        "image_path_missing_count": image_missing,
        "empty_text_count": empty_text,
        "empty_text_rate": empty_text / max(1, len(rows)),
        "text_field_hits": {field: int(field_hits.get(field, 0)) for field in TEXT_FIELDS},
    }
    return rows, stats


def load_byt5_tokenizer(path: Path):
    try:
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to load ByT5 tokenizer from {path}: {exc}") from exc


def infer_byt5_byte_offset(rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer: Any) -> Optional[int]:
    """Infer exact ByT5 token length from UTF-8 byte length when possible.

    ByT5 tokenization is byte-level. On this local tokenizer the length is
    usually `len(text.encode("utf-8")) + 1` because EOS is appended. We still
    verify this with the actual local tokenizer before using the fast path.
    """
    samples: List[str] = []
    for rows in rows_by_split.values():
        for row in rows:
            text = str(row.get("_selected_text") or "")
            if text:
                samples.append(text)
            if len(samples) >= 256:
                break
        if len(samples) >= 256:
            break
    if not samples:
        return None
    encoded = tokenizer(samples, add_special_tokens=True, padding=False, truncation=False)
    diffs = {len(ids) - len(text.encode("utf-8")) for text, ids in zip(samples, encoded["input_ids"])}
    if len(diffs) == 1:
        offset = int(next(iter(diffs)))
        if 0 <= offset <= 4:
            return offset
    return None


def add_token_stats(rows: List[Dict[str, Any]], tokenizer: Any, fast_byte_offset: Optional[int], batch_size: int = 512) -> None:
    if fast_byte_offset is not None:
        for row in rows:
            text = str(row.get("_selected_text") or "")
            token_len = len(text.encode("utf-8")) + int(fast_byte_offset) if text else 0
            row["byt5_token_len"] = int(token_len)
            row["char_len"] = int(len(text))
            row["word_len"] = int(len(text.split()))
            row["over_1024"] = bool(token_len > 1024)
        return
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [row.get("_selected_text", "") for row in batch]
        encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False)
        for row, ids in zip(batch, encoded["input_ids"]):
            text = row.get("_selected_text", "")
            row["byt5_token_len"] = int(len(ids))
            row["char_len"] = int(len(text))
            row["word_len"] = int(len(text.split()))
            row["over_1024"] = bool(len(ids) > 1024)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def token_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    vals = [float(row.get("byt5_token_len", 0)) for row in rows]
    return {
        "p50": percentile(vals, 50),
        "p70": percentile(vals, 70),
        "p80": percentile(vals, 80),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "max": max(vals) if vals else 0,
        "over_1024_ratio": sum(1 for row in rows if row.get("over_1024")) / max(1, len(rows)),
    }


def select_token_candidates(rows: List[Dict[str, Any]], split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cap = CANDIDATE_CAPS[split]
    vals = [float(row["byt5_token_len"]) for row in rows if row.get("_selected_text")]
    p70 = percentile(vals, 70)
    eligible = [row for row in rows if row.get("_selected_text") and int(row["byt5_token_len"]) >= p70]
    sorted_rows = sorted(
        [row for row in rows if row.get("_selected_text")],
        key=lambda row: int(row["byt5_token_len"]),
        reverse=True,
    )
    if len(eligible) >= cap:
        candidates = sorted(eligible, key=lambda row: int(row["byt5_token_len"]), reverse=True)[:cap]
        policy = f"token_len>=p70 capped to top {cap}"
    else:
        candidates = sorted_rows[:cap]
        policy = f"p70 yielded {len(eligible)}; using top {cap}"
    summary = token_summary(rows)
    summary.update({
        "candidate_policy": policy,
        "candidate_count": len(candidates),
        "candidate_token_min": min((row["byt5_token_len"] for row in candidates), default=0),
    })
    return candidates, summary


def resize_pad_image(img: Image.Image, image_size: int = IMAGE_SIZE, fill: int = 255) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = img.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), color=(fill, fill, fill))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def color_entropy(arr: np.ndarray, bins: int = 8) -> float:
    flat = (arr.reshape(-1, 3).astype(np.int32) * bins // 256).clip(0, bins - 1)
    idx = flat[:, 0] * bins * bins + flat[:, 1] * bins + flat[:, 2]
    hist = np.bincount(idx, minlength=bins ** 3).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def estimate_text_boxes(img: Image.Image) -> Tuple[List[List[int]], Dict[str, Any]]:
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr_f = arr.astype(np.float32) / 255.0
    mx = arr_f.max(axis=2)
    mn = arr_f.min(axis=2)
    saturation = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    gray = (0.299 * arr_f[:, :, 0] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 2])
    white = np.all(arr > 245, axis=2)
    white_background_ratio = float(white.mean())
    mean_saturation = float(saturation.mean())
    entropy = color_entropy(arr)
    sobel = filters.sobel(gray)
    try:
        local_thr = filters.threshold_sauvola(gray, window_size=25, k=0.2)
        dark_fg = gray < (local_thr - 0.01)
    except Exception:  # noqa: BLE001
        dark_fg = gray < filters.threshold_otsu(gray)
    edge_thr = max(float(np.percentile(sobel, 82)), 0.025)
    edge_fg = sobel >= edge_thr
    non_white = ~white
    color_fg = (saturation >= max(float(np.percentile(saturation, 80)), 0.12)) & non_white
    mask = ((dark_fg | color_fg) & (edge_fg | non_white))
    mask = morphology.remove_small_objects(mask.astype(bool), min_size=4)
    mask = morphology.binary_closing(mask, footprint=np.ones((2, 3), dtype=bool))
    merged = morphology.binary_dilation(mask, footprint=np.ones((3, 9), dtype=bool))
    label = measure.label(merged)
    boxes: List[List[int]] = []
    heights: List[int] = []
    for prop in measure.regionprops(label):
        y1, x1, y2, x2 = prop.bbox
        h = int(y2 - y1)
        w = int(x2 - x1)
        area = int(prop.area)
        if h < 3 or w < 3 or area < 8:
            continue
        if h > 96 or w > IMAGE_SIZE:
            continue
        extent = area / max(1, h * w)
        if extent < 0.025:
            continue
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
        heights.append(h)
    union = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        union[y1:y2, x1:x2] = True
    coverage = float(union.mean())
    if heights:
        median_h = float(np.median(heights))
        min_h = float(np.min(heights))
        small_ratio = float(sum(h < 6 for h in heights) / len(heights))
    else:
        median_h = 0.0
        min_h = 0.0
        small_ratio = 0.0
    stats = {
        "bbox_source": "lightweight_256_component_estimator",
        "box_count": int(len(boxes)),
        "estimated_box_count": int(len(boxes)),
        "bbox_union_coverage_ratio": coverage,
        "median_box_height_256": median_h,
        "min_box_height_256": min_h,
        "small_box_ratio": small_ratio,
        "white_background_ratio": white_background_ratio,
        "mean_saturation": mean_saturation,
        "image_color_entropy": entropy,
        "visual_diversity": float(0.5 * mean_saturation + 0.3 * min(entropy / 8.0, 1.0) + 0.2 * (1.0 - white_background_ratio)),
        "doc_like": bool(white_background_ratio >= 0.72 and mean_saturation <= 0.14 and coverage >= 0.45 and entropy <= 4.0),
    }
    return boxes, stats


def add_image_stats(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        path = Path(str(row["image_path"]))
        if not path.exists():
            row["_image_exists"] = False
            continue
        img = resize_pad_image(Image.open(path), IMAGE_SIZE)
        boxes, stats = estimate_text_boxes(img)
        row["_image_exists"] = True
        row["estimated_boxes_256"] = boxes
        row.update(stats)


def coverage_bucket(cov: float) -> str:
    if cov < 0.30:
        return "local"
    if cov < 0.70:
        return "medium"
    if cov < 0.90:
        return "dense"
    return "ultra_dense"


def normalize_values(rows: Sequence[Dict[str, Any]], key: str) -> Dict[int, float]:
    vals = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
    if len(vals) == 0:
        return {}
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    denom = max(hi - lo, 1e-8)
    return {id(row): float(np.clip((float(row.get(key, 0.0)) - lo) / denom, 0.0, 1.0)) for row in rows}


def add_selection_scores(rows: List[Dict[str, Any]]) -> None:
    norm_tok = normalize_values(rows, "byt5_token_len")
    norm_h = normalize_values(rows, "median_box_height_256")
    norm_box = normalize_values(rows, "box_count")
    norm_vis = normalize_values(rows, "visual_diversity")
    for row in rows:
        row["selection_score"] = (
            0.50 * norm_tok.get(id(row), 0.0)
            + 0.20 * norm_h.get(id(row), 0.0)
            + 0.20 * norm_box.get(id(row), 0.0)
            + 0.10 * norm_vis.get(id(row), 0.0)
        )
        row["coverage_bucket"] = coverage_bucket(float(row.get("bbox_union_coverage_ratio", 0.0)))


def filter_image_candidates(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    filtered = []
    reject = Counter()
    for row in rows:
        if not row.get("_image_exists"):
            reject["missing_image"] += 1
            continue
        if int(row.get("box_count", 0)) == 0:
            reject["box_count_zero"] += 1
            continue
        if float(row.get("bbox_union_coverage_ratio", 0.0)) < 0.05:
            reject["coverage_lt_0.05"] += 1
            continue
        if float(row.get("median_box_height_256", 0.0)) < 6.0:
            reject["median_box_height_lt_6"] += 1
            continue
        filtered.append(row)
    add_selection_scores(filtered)
    return filtered, {"kept": len(filtered), "rejected": dict(reject)}


def choose_reserve(rows: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    target = RESERVE_COUNTS[split]
    selected = stratified_pick(rows, TARGET_BUCKETS[split], multiplier=math.ceil(target / max(1, TARGET_COUNTS[split])), total=target)
    if len(selected) < target:
        seen = {id(row) for row in selected}
        backfill = [row for row in sorted(rows, key=lambda r: r["selection_score"], reverse=True) if id(row) not in seen]
        selected.extend(backfill[:target - len(selected)])
    return selected[:target]


def stratified_pick(
    rows: List[Dict[str, Any]],
    bucket_targets: Mapping[str, int],
    *,
    multiplier: int = 1,
    total: Optional[int] = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(SEED)
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("easy_sparse"):
            continue
        by_bucket[str(row.get("coverage_bucket", "local"))].append(row)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=lambda row: float(row.get("selection_score", 0.0)), reverse=True)
    selected: List[Dict[str, Any]] = []
    used = set()
    shortages: Dict[str, int] = {}
    for bucket, target in bucket_targets.items():
        need = int(target) * int(multiplier)
        pool = by_bucket.get(bucket, [])
        if len(pool) < need:
            shortages[bucket] = need - len(pool)
            chosen = list(pool)
        else:
            top_pool = pool[: min(len(pool), 2 * need)]
            chosen = rng.sample(top_pool, need)
        for row in chosen:
            if id(row) not in used:
                used.add(id(row))
                selected.append(row)
    if total is not None and len(selected) > total:
        selected.sort(key=lambda row: float(row.get("selection_score", 0.0)), reverse=True)
        selected = selected[:total]
    selected_shortages = shortages
    for row in selected:
        row.setdefault("_selection_shortages", selected_shortages)
    return selected


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(127.5 * x + 128.0, 0, 255)
    return x.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def psnr_from_mse(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10(1.0 / mse))


def safe_ssim(a: np.ndarray, b: np.ndarray) -> float:
    h, w = a.shape[:2]
    if h < 7 or w < 7:
        return float("nan")
    try:
        return float(metrics.structural_similarity(a, b, channel_axis=2, data_range=255))
    except TypeError:
        return float(metrics.structural_similarity(a, b, multichannel=True, data_range=255))


def bbox_mask(boxes: Sequence[Sequence[int]]) -> np.ndarray:
    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        mask[int(y1):int(y2), int(x1):int(x2)] = True
    return mask


def reconstruction_metrics(gt: np.ndarray, rec: np.ndarray, boxes: Sequence[Sequence[int]]) -> Dict[str, float]:
    diff = (gt.astype(np.float32) - rec.astype(np.float32)) / 255.0
    full_mse = float(np.mean(diff ** 2))
    out = {
        "full_mse": full_mse,
        "full_psnr": psnr_from_mse(full_mse),
        "full_ssim": safe_ssim(gt, rec),
    }
    mask = bbox_mask(boxes)
    if not mask.any():
        out.update({"text_mse": float("nan"), "text_psnr": float("nan"), "text_ssim": float("nan"), "text_laplacian_mse": float("nan")})
        return out
    text_mse = float(np.mean(diff[mask] ** 2))
    ys, xs = np.where(mask)
    y1, y2 = max(0, ys.min() - 2), min(IMAGE_SIZE, ys.max() + 3)
    x1, x2 = max(0, xs.min() - 2), min(IMAGE_SIZE, xs.max() + 3)
    gt_crop = gt[y1:y2, x1:x2]
    rec_crop = rec[y1:y2, x1:x2]
    gt_gray = (0.299 * gt[:, :, 0] + 0.587 * gt[:, :, 1] + 0.114 * gt[:, :, 2]).astype(np.float32) / 255.0
    rec_gray = (0.299 * rec[:, :, 0] + 0.587 * rec[:, :, 1] + 0.114 * rec[:, :, 2]).astype(np.float32) / 255.0
    lap_mse = float(np.mean((filters.laplace(gt_gray)[mask] - filters.laplace(rec_gray)[mask]) ** 2))
    out.update({
        "text_mse": text_mse,
        "text_psnr": psnr_from_mse(text_mse),
        "text_ssim": safe_ssim(gt_crop, rec_crop),
        "text_laplacian_mse": lap_mse,
    })
    return out


def run_original_reconstruction(
    paths: Paths,
    rows_by_split: Mapping[str, List[Dict[str, Any]]],
    *,
    batch_size: int,
    pilot_images: int,
    max_minutes: float,
    device_backend: str,
    mixed_precision: str,
) -> Dict[str, Any]:
    sys.path.insert(0, str(paths.repo))
    from scripts.stage1.evaluate_textatlas_reconstruction import (  # type: ignore
        load_tokenizer_model,
        prepare_device,
        reconstruct_batch,
    )

    audit = {
        "config": str(paths.vq_config),
        "ckpt": str(paths.vq_ckpt),
        "confirmed_disabled": confirm_original_config(paths.vq_config),
        "device_backend": device_backend,
        "mixed_precision": mixed_precision,
        "batch_size": batch_size,
    }
    if not all(audit["confirmed_disabled"].values()):
        raise RuntimeError(f"original VQ config did not disable text/OCR/gate cleanly: {audit['confirmed_disabled']}")
    device = prepare_device(device_backend, 0)
    t0_load = time.time()
    model, _config, text_context = load_tokenizer_model(
        paths.vq_config,
        paths.vq_ckpt,
        device,
        text_layer_pair_index=0,
        checkpoint_weight_key="auto",
    )
    if text_context is not None:
        raise RuntimeError("original reconstruction unexpectedly created text_context")
    audit["model_load_seconds"] = time.time() - t0_load

    all_reserve = []
    for split, rows in rows_by_split.items():
        for row in rows:
            row["_recon_split"] = split
            all_reserve.append(row)
    pilot = []
    rng = random.Random(SEED)
    for split, rows in rows_by_split.items():
        take = min(max(1, pilot_images // 3), len(rows))
        pilot.extend(rng.sample(rows, take) if len(rows) > take else rows)
    t0 = time.time()
    _ = reconstruct_rows(model, text_context, device_backend, mixed_precision, device, pilot, batch_size)
    pilot_seconds = time.time() - t0
    per_image = pilot_seconds / max(1, len(pilot))
    estimated_seconds = per_image * len(all_reserve)
    audit.update({
        "pilot_images": len(pilot),
        "pilot_seconds": pilot_seconds,
        "pilot_seconds_per_image": per_image,
        "estimated_full_reserve_seconds": estimated_seconds,
        "estimated_full_reserve_minutes": estimated_seconds / 60.0,
    })
    if estimated_seconds / 60.0 > max_minutes:
        jdump(OUT_DIR / "reconstruction_pilot_stop_summary.json", audit)
        raise RuntimeError(
            f"original VQ reconstruction reserve estimate {estimated_seconds/60.0:.1f} min exceeds limit {max_minutes:.1f} min"
        )
    t1 = time.time()
    for split, rows in rows_by_split.items():
        _ = reconstruct_rows(model, text_context, device_backend, mixed_precision, device, rows, batch_size)
    audit["full_reconstruction_seconds"] = time.time() - t1
    return audit


def reconstruct_rows(
    model: torch.nn.Module,
    text_context: Any,
    device_backend: str,
    mixed_precision: str,
    device: torch.device,
    rows: List[Dict[str, Any]],
    batch_size: int,
) -> int:
    from scripts.stage1.evaluate_textatlas_reconstruction import reconstruct_batch  # type: ignore

    count = 0
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start:start + batch_size]
        pil_images = [resize_pad_image(Image.open(str(row["image_path"])), IMAGE_SIZE) for row in batch_rows]
        batch = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device)
        with torch.inference_mode():
            rec = reconstruct_batch(
                model,
                batch,
                None,
                text_context,
                device_backend=device_backend,
                mixed_precision=mixed_precision,
                causal_type=None,
            )
        rec_np = tensor_to_uint8(rec)
        for row, gt_img, rec_img in zip(batch_rows, pil_images, rec_np):
            gt = np.asarray(gt_img, dtype=np.uint8)
            row.update(reconstruction_metrics(gt, rec_img, row.get("estimated_boxes_256", [])))
            count += 1
    return count


def confirm_original_config(config_path: Path) -> Dict[str, bool]:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return {
        "text_conditioning_disabled": not bool((cfg.get("text_conditioning") or {}).get("enabled", False)),
        "text_recon_conditioning_disabled": not bool((cfg.get("text_recon_conditioning") or {}).get("enabled", False)),
        "ocr_teacher_forcing_loss_disabled": not bool((cfg.get("ocr_teacher_forcing_loss") or {}).get("enabled", False)),
        "ocr_box_gate_disabled": not bool((cfg.get("ocr_box_gate") or {}).get("enabled", False)),
        "text_hr_disabled": not bool((cfg.get("text_hr") or {}).get("enabled", False)),
        "local_similarity_loss_disabled": not bool((cfg.get("local_similarity_loss") or {}).get("enabled", False)),
        "ocr_visual_alignment_loss_disabled": not bool((cfg.get("ocr_visual_alignment_loss") or {}).get("enabled", False)),
    }


def add_easy_sparse_flags(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    values = {key: [float(row.get(key, np.nan)) for row in rows] for key in ("text_ssim", "text_psnr", "text_laplacian_mse", "box_count")}
    for key in values:
        values[key] = [v for v in values[key] if not math.isnan(v)]
    p75_ssim = percentile(values["text_ssim"], 75)
    p25_lap = percentile(values["text_laplacian_mse"], 25)
    p25_box = percentile(values["box_count"], 25)
    psnr_vals = np.asarray(values["text_psnr"], dtype=np.float64)
    ssim_vals = np.asarray(values["text_ssim"], dtype=np.float64)
    lap_vals = np.asarray(values["text_laplacian_mse"], dtype=np.float64)
    means = {
        "psnr": float(np.nanmean(psnr_vals)) if len(psnr_vals) else 0.0,
        "ssim": float(np.nanmean(ssim_vals)) if len(ssim_vals) else 0.0,
        "lap": float(np.nanmean(lap_vals)) if len(lap_vals) else 0.0,
    }
    stds = {
        "psnr": float(np.nanstd(psnr_vals) + 1e-8) if len(psnr_vals) else 1.0,
        "ssim": float(np.nanstd(ssim_vals) + 1e-8) if len(ssim_vals) else 1.0,
        "lap": float(np.nanstd(lap_vals) + 1e-8) if len(lap_vals) else 1.0,
    }
    easy_scores = []
    for row in rows:
        easy_score = (
            (float(row.get("text_ssim", 0.0)) - means["ssim"]) / stds["ssim"]
            + (float(row.get("text_psnr", 0.0)) - means["psnr"]) / stds["psnr"]
            - (float(row.get("text_laplacian_mse", 0.0)) - means["lap"]) / stds["lap"]
        )
        row["easy_score"] = float(easy_score)
        easy_scores.append(easy_score)
    easy_threshold = percentile(easy_scores, 75)
    rejected = []
    for row in rows:
        sparse = (
            float(row.get("bbox_union_coverage_ratio", 0.0)) < 0.15
            and float(row.get("box_count", 0)) <= max(3.0, p25_box)
        )
        good_recon = (
            float(row.get("text_ssim", 0.0)) >= p75_ssim
            or float(row.get("text_laplacian_mse", 1e9)) <= p25_lap
            or float(row.get("easy_score", -1e9)) >= easy_threshold
        )
        row["easy_sparse"] = bool(sparse and good_recon)
        if row["easy_sparse"]:
            rejected.append(row)
    return {
        "rows": len(rows),
        "p75_text_ssim": p75_ssim,
        "p25_text_laplacian_mse": p25_lap,
        "p25_box_count": p25_box,
        "easy_score_p75": easy_threshold,
        "easy_sparse_rejected_count": len(rejected),
    }


def select_final(rows: List[Dict[str, Any]], split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    targets = TARGET_BUCKETS[split]
    selected = stratified_pick(rows, targets)
    shortage = {}
    selected_by_bucket = Counter(row["coverage_bucket"] for row in selected)
    for bucket, target in targets.items():
        if selected_by_bucket[bucket] < target:
            shortage[bucket] = target - selected_by_bucket[bucket]
    if shortage:
        selected_ids = {id(row) for row in selected}
        for bucket, need in list(shortage.items()):
            candidates = [
                row for row in rows
                if id(row) not in selected_ids and not row.get("easy_sparse")
            ]
            candidates.sort(
                key=lambda row: (
                    abs(bucket_order(row["coverage_bucket"]) - bucket_order(bucket)),
                    -float(row.get("selection_score", 0.0)),
                )
            )
            for row in candidates[:need]:
                selected.append(row)
                selected_ids.add(id(row))
    selected = control_doc_like(selected, rows, targets)
    selected = selected[:TARGET_COUNTS[split]]
    summary = {
        "target": TARGET_COUNTS[split],
        "selected": len(selected),
        "bucket_counts": dict(Counter(row["coverage_bucket"] for row in selected)),
        "doc_like_ratio": sum(1 for row in selected if row.get("doc_like")) / max(1, len(selected)),
        "shortage_before_backfill": shortage,
    }
    return selected, summary


def bucket_order(bucket: str) -> int:
    return {"local": 0, "medium": 1, "dense": 2, "ultra_dense": 3}.get(bucket, 0)


def control_doc_like(selected: List[Dict[str, Any]], all_rows: List[Dict[str, Any]], targets: Mapping[str, int]) -> List[Dict[str, Any]]:
    ratio = sum(1 for row in selected if row.get("doc_like")) / max(1, len(selected))
    if ratio <= 0.40:
        return selected
    selected_ids = {id(row) for row in selected}
    selected_by_bucket = defaultdict(list)
    for row in selected:
        selected_by_bucket[row["coverage_bucket"]].append(row)
    replacements_by_bucket = defaultdict(list)
    for row in all_rows:
        if id(row) in selected_ids or row.get("easy_sparse") or row.get("doc_like"):
            continue
        replacements_by_bucket[row["coverage_bucket"]].append(row)
    for rows in replacements_by_bucket.values():
        rows.sort(key=lambda row: float(row.get("visual_diversity", 0.0)), reverse=True)
    new_selected = list(selected)
    for idx, row in enumerate(list(new_selected)):
        current_ratio = sum(1 for x in new_selected if x.get("doc_like")) / max(1, len(new_selected))
        if current_ratio <= 0.40:
            break
        if not row.get("doc_like"):
            continue
        repl_pool = replacements_by_bucket.get(row["coverage_bucket"], [])
        if not repl_pool:
            continue
        replacement = repl_pool.pop(0)
        new_selected[idx] = replacement
    return new_selected


def clean_row(row: Mapping[str, Any], scored: bool) -> Dict[str, Any]:
    out = {k: v for k, v in row.items() if not k.startswith("_")}
    if not scored:
        keep = {
            "dataset", "subset", "split", "hf_split", "hf_row_idx", "source_key", "image_path",
            "materialized", "selection_seed", "selection_strategy", "hf_image_path", "text",
            "text_source", "raw_annotation", "width", "height", "file_size", "materialize_order",
            "parquet_path", "parquet_global_start_idx", "parquet_local_row_idx", "text_extraction_status",
        }
        out = {k: v for k, v in out.items() if k in keep}
    return out


def distribution(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    vals = [float(row.get(key, 0.0)) for row in rows if row.get(key) is not None and not math.isnan(float(row.get(key, 0.0)))]
    return {
        "p10": percentile(vals, 10),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "mean": float(np.mean(vals)) if vals else 0.0,
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
    }


def save_hist(path: Path, rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]], key: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for split, rows in rows_by_split.items():
        vals = [float(row.get(key, 0.0)) for row in rows]
        plt.hist(vals, bins=40, alpha=0.45, label=split)
    plt.title(title)
    plt.xlabel(key)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def preview_grid(path: Path, selected_by_split: Mapping[str, Sequence[Dict[str, Any]]]) -> None:
    rows = []
    for bucket in ("local", "medium", "dense", "ultra_dense"):
        bucket_rows = []
        for split_rows in selected_by_split.values():
            bucket_rows.extend([row for row in split_rows if row.get("coverage_bucket") == bucket])
        bucket_rows.sort(key=lambda row: float(row.get("selection_score", 0.0)), reverse=True)
        rows.extend(bucket_rows[:10])
    cell = 256
    label_h = 58
    cols = 10
    rows_n = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * cell, rows_n * (cell + label_h)), "white")
    font = ImageFont.load_default()
    for idx, row in enumerate(rows):
        img = resize_pad_image(Image.open(str(row["image_path"])), IMAGE_SIZE)
        draw = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in row.get("estimated_boxes_256", [])[:80]:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
        x = (idx % cols) * cell
        y = (idx // cols) * (cell + label_h)
        canvas.paste(img, (x, y))
        label = (
            f"{row.get('coverage_bucket')} tok={row.get('byt5_token_len')} cov={float(row.get('bbox_union_coverage_ratio',0)):.2f}\n"
            f"box={row.get('box_count')} ssim={float(row.get('text_ssim', float('nan'))):.3f} easy={float(row.get('easy_score', 0)):.2f}"
        )
        ImageDraw.Draw(canvas).text((x + 4, y + cell + 3), label, fill=(0, 0, 0), font=font)
    canvas.save(path)


def alias_outputs() -> None:
    aliases = {
        "text_rich_10k_train.jsonl": "train_10k_text_rich_hardmix.jsonl",
        "text_rich_1k_val.jsonl": "val_1k_text_rich_hardmix.jsonl",
        "text_rich_500_holdout.jsonl": "holdout_500_text_rich_hardmix.jsonl",
        "text_rich_20_qual.jsonl": "qual_20_text_rich_hardmix.jsonl",
    }
    for src, dst in aliases.items():
        src_path = OUT_DIR / src
        if src_path.exists():
            (OUT_DIR / dst).write_text(src_path.read_text(encoding="utf-8"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-backend", choices=("npu", "cuda", "cpu"), default="npu")
    parser.add_argument("--mixed-precision", choices=("bf16", "fp16", "none"), default="bf16")
    parser.add_argument("--recon-batch-size", type=int, default=8)
    parser.add_argument("--recon-pilot-images", type=int, default=180)
    parser.add_argument("--max-reconstruction-minutes", type=float, default=180.0)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = discover_paths()
    discovery = {
        "repo_root": str(paths.repo),
        "train_manifest": str(paths.train_manifest),
        "val_manifest": str(paths.val_manifest),
        "holdout_manifest": str(paths.holdout_manifest),
        "tokenizer_path": str(paths.tokenizer),
        "original_gigatok_vq_config": str(paths.vq_config),
        "original_gigatok_vq_checkpoint": str(paths.vq_ckpt),
        "transform": "resize_pad_arr: RGB, scale=min(256/w,256/h), bicubic resize, centered on white 256x256 canvas; no crop",
        "deepseek_bbox_all_candidate_decision": "not run for all candidates; using lightweight_256_component_estimator for hardmix selection because full DeepSeek-OCR bbox/proxy is deferred to selected splits",
    }
    print(json.dumps({"path_discovery_summary": discovery}, indent=2, ensure_ascii=False), flush=True)
    jdump(OUT_DIR / "path_discovery_summary.json", discovery)

    rows_by_split = {}
    manifest_stats = {}
    for split, manifest in (("train", paths.train_manifest), ("val", paths.val_manifest), ("holdout", paths.holdout_manifest)):
        rows, stats = scan_manifest(manifest, split)
        if stats["empty_text_rate"] > 0.01:
            jdump(OUT_DIR / "stage0_failed_empty_text_summary.json", stats)
            raise RuntimeError(f"{split} empty_text_rate={stats['empty_text_rate']:.4f} > 1%")
        rows_by_split[split] = rows
        manifest_stats[split] = stats

    tokenizer = load_byt5_tokenizer(paths.tokenizer)
    fast_byte_offset = infer_byt5_byte_offset(rows_by_split, tokenizer)
    print(f"[tokenizer] byt5_fast_byte_offset={fast_byte_offset}", flush=True)
    candidate_by_split = {}
    candidate_stats = {}
    for split, rows in rows_by_split.items():
        add_token_stats(rows, tokenizer, fast_byte_offset)
        candidates, stats = select_token_candidates(rows, split)
        add_image_stats(candidates)
        filtered, filter_stats = filter_image_candidates(candidates)
        candidate_by_split[split] = filtered
        candidate_stats[split] = {**stats, "image_filter": filter_stats}
        print(f"[{split}] rows={len(rows)} candidates={len(candidates)} filtered={len(filtered)}", flush=True)

    reserve_by_split = {split: choose_reserve(rows, split) for split, rows in candidate_by_split.items()}
    recon_audit = run_original_reconstruction(
        paths,
        reserve_by_split,
        batch_size=args.recon_batch_size,
        pilot_images=args.recon_pilot_images,
        max_minutes=args.max_reconstruction_minutes,
        device_backend=args.device_backend,
        mixed_precision=args.mixed_precision,
    )

    easy_summary = {}
    rejected_easy = []
    for split, rows in reserve_by_split.items():
        easy_summary[split] = add_easy_sparse_flags(rows)
        rejected_easy.extend([clean_row(row, scored=True) for row in rows if row.get("easy_sparse")])
    write_jsonl(OUT_DIR / "rejected_easy_sparse.jsonl", rejected_easy)
    jdump(OUT_DIR / "easy_sparse_summary.json", easy_summary)

    selected_by_split = {}
    final_stats = {}
    for split, rows in reserve_by_split.items():
        selected, stats = select_final(rows, split)
        selected_by_split[split] = selected
        final_stats[split] = stats

    train_keys = {row.get("source_key") for row in selected_by_split["train"]}
    val_keys = {row.get("source_key") for row in selected_by_split["val"]}
    holdout_keys = {row.get("source_key") for row in selected_by_split["holdout"]}
    overlaps = {
        "train_val": len(train_keys & val_keys),
        "train_holdout": len(train_keys & holdout_keys),
        "val_holdout": len(val_keys & holdout_keys),
    }
    if any(overlaps.values()):
        jdump(OUT_DIR / "overlap_failure_summary.json", overlaps)
        raise RuntimeError(f"split overlap found: {overlaps}")

    qual_rows = []
    for bucket in ("local", "medium", "dense", "ultra_dense"):
        bucket_pool = [row for row in selected_by_split["holdout"] if row["coverage_bucket"] == bucket]
        bucket_pool.sort(key=lambda row: row["selection_score"], reverse=True)
        qual_rows.extend(bucket_pool[:5])
    qual_rows = qual_rows[:20]

    output_map = {
        "train": ("text_rich_10k_train.jsonl", "text_rich_10k_train.scored.jsonl"),
        "val": ("text_rich_1k_val.jsonl", "text_rich_1k_val.scored.jsonl"),
        "holdout": ("text_rich_500_holdout.jsonl", "text_rich_500_holdout.scored.jsonl"),
        "qual": ("text_rich_20_qual.jsonl", "text_rich_20_qual.scored.jsonl"),
    }
    for split in ("train", "val", "holdout"):
        write_jsonl(OUT_DIR / output_map[split][0], [clean_row(row, scored=False) for row in selected_by_split[split]])
        write_jsonl(OUT_DIR / output_map[split][1], [clean_row(row, scored=True) for row in selected_by_split[split]])
    write_jsonl(OUT_DIR / output_map["qual"][0], [clean_row(row, scored=False) for row in qual_rows])
    write_jsonl(OUT_DIR / output_map["qual"][1], [clean_row(row, scored=True) for row in qual_rows])
    alias_outputs()

    save_hist(OUT_DIR / "token_len_histogram.png", selected_by_split, "byt5_token_len", "Selected ByT5 token length")
    save_hist(OUT_DIR / "coverage_histogram.png", selected_by_split, "bbox_union_coverage_ratio", "Selected bbox coverage")
    save_hist(OUT_DIR / "box_height_histogram.png", selected_by_split, "median_box_height_256", "Selected median box height")
    save_hist(OUT_DIR / "doc_like_histogram.png", selected_by_split, "doc_like", "Selected doc-like flag")
    preview_grid(OUT_DIR / "sample_preview_grid.png", selected_by_split)

    bucket_counts = {
        split: dict(Counter(row["coverage_bucket"] for row in rows))
        for split, rows in selected_by_split.items()
    }
    jdump(OUT_DIR / "bucket_counts.json", bucket_counts)

    selected_all = [row for rows in selected_by_split.values() for row in rows]
    summary = {
        "source_manifest_paths": {split: manifest_stats[split]["path"] for split in manifest_stats},
        "tokenizer_path": str(paths.tokenizer),
        "original_gigatok_checkpoint_path": str(paths.vq_ckpt),
        "original_gigatok_reconstruction_config_used": str(paths.vq_config),
        "original_reconstruction_disabled_confirmations": recon_audit["confirmed_disabled"],
        "transform_resize_pad_method_used": discovery["transform"],
        "bbox_source": "lightweight_256_component_estimator",
        "deepseek_bbox_proxy_note": "DeepSeek-OCR bbox/proxy is intentionally deferred to selected train/val/holdout; this script only uses lightweight 256-component boxes for hardmix selection.",
        "train_val_holdout_source_counts": {split: manifest_stats[split]["rows"] for split in manifest_stats},
        "stage0_manifest_stats": manifest_stats,
        "candidate_counts": {split: candidate_stats[split]["candidate_count"] for split in candidate_stats},
        "candidate_stats": candidate_stats,
        "reserve_counts": {split: len(rows) for split, rows in reserve_by_split.items()},
        "final_counts": {split: len(rows) for split, rows in selected_by_split.items()},
        "overlap_checks": overlaps,
        "token_len_distribution": {split: distribution(rows, "byt5_token_len") for split, rows in selected_by_split.items()},
        "coverage_distribution": {split: distribution(rows, "bbox_union_coverage_ratio") for split, rows in selected_by_split.items()},
        "box_height_distribution": {split: distribution(rows, "median_box_height_256") for split, rows in selected_by_split.items()},
        "doc_like_ratio": {split: sum(1 for row in rows if row.get("doc_like")) / max(1, len(rows)) for split, rows in selected_by_split.items()},
        "easy_sparse_rejected_count": len(rejected_easy),
        "easy_sparse_summary": easy_summary,
        "selected_bucket_counts": bucket_counts,
        "selected_token_len": distribution(selected_all, "byt5_token_len"),
        "selected_coverage": distribution(selected_all, "bbox_union_coverage_ratio"),
        "selected_text_ssim": distribution(selected_all, "text_ssim"),
        "reconstruction_throughput_pilot": recon_audit,
        "warning_bucket_shortage_or_backfill": {split: final_stats[split].get("shortage_before_backfill", {}) for split in final_stats},
        "output_files": {split: [str(OUT_DIR / name) for name in output_map[split]] for split in output_map},
    }
    jdump(OUT_DIR / "selection_summary.json", summary)
    print(json.dumps({"done": True, "output_dir": str(OUT_DIR), "bucket_counts": bucket_counts}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
