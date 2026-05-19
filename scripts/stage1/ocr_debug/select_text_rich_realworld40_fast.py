#!/usr/bin/env python3
"""Fast real-world upweighted TextAtlas selector for OCR-box-gate prep.

This preparation script deliberately does not create final training OCR boxes,
proxy confidence, text-feature cache, or launch configs. The lightweight boxes
computed here are for selection and preview only.
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

THIS_FILE = Path(__file__).resolve()
REPO_IMPORT_ROOT = THIS_FILE.parents[3]
if str(REPO_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_IMPORT_ROOT))

from scripts.stage1.ocr_debug import select_text_rich_hardmix_manifests as base  # noqa: E402


PERSIST_ROOT = Path("/home/ma-user/work/GigaTok_hr/gigatok_persist")
REPO_ROOT = Path("/home/ma-user/work/GigaTok_hr/GigaTok_Loss_two_ablation")
OUT_DIR = PERSIST_ROOT / "outputs/text_rich_realworld_upweighted_manifests"
PREV_DIR = PERSIST_ROOT / "outputs/text_rich_hardmix_manifests"
IMAGE_SIZE = 256
SEED = 0

SOURCES = ("TextScenesHQ", "TextVisionBlend", "StyledTextSynth", "CleanTextSynth", "LongWordsSubset-A")
REALWORLD_SOURCES = {"TextScenesHQ", "TextVisionBlend"}
TARGETS = {
    "train": {"TextScenesHQ": 2500, "TextVisionBlend": 1500, "StyledTextSynth": 3000, "CleanTextSynth": 2800, "LongWordsSubset-A": 200},
    "val": {"TextScenesHQ": 250, "TextVisionBlend": 150, "StyledTextSynth": 300, "CleanTextSynth": 280, "LongWordsSubset-A": 20},
    "holdout": {"TextScenesHQ": 125, "TextVisionBlend": 75, "StyledTextSynth": 150, "CleanTextSynth": 140, "LongWordsSubset-A": 10},
    "qual": {"TextScenesHQ": 5, "TextVisionBlend": 3, "StyledTextSynth": 6, "CleanTextSynth": 5, "LongWordsSubset-A": 1},
}
REALWORLD_BUCKET_RATIOS = {
    "TextScenesHQ": {"local": 0.30, "medium": 0.45, "dense": 0.20, "ultra_dense": 0.05},
    "TextVisionBlend": {"local": 0.25, "medium": 0.45, "dense": 0.25, "ultra_dense": 0.05},
}
BACKFILL_ORDER = {
    "TextScenesHQ": ("TextVisionBlend", "StyledTextSynth", "CleanTextSynth", "LongWordsSubset-A"),
    "TextVisionBlend": ("TextScenesHQ", "StyledTextSynth", "CleanTextSynth", "LongWordsSubset-A"),
    "StyledTextSynth": ("TextVisionBlend", "TextScenesHQ", "CleanTextSynth", "LongWordsSubset-A"),
    "CleanTextSynth": ("StyledTextSynth", "TextVisionBlend", "TextScenesHQ", "LongWordsSubset-A"),
    "LongWordsSubset-A": ("StyledTextSynth", "CleanTextSynth", "TextVisionBlend", "TextScenesHQ"),
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def percentile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return 0.0
    return float(np.percentile(np.asarray(vals, dtype=np.float64), q))


def distribution(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, float]:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            value_f = float(value)
        except Exception:  # noqa: BLE001
            continue
        if not math.isnan(value_f):
            vals.append(value_f)
    return {
        "p10": percentile(vals, 10),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "mean": float(np.mean(vals)) if vals else 0.0,
        "min": min(vals) if vals else 0.0,
        "max": max(vals) if vals else 0.0,
    }


def clean_row(row: Mapping[str, Any], scored: bool) -> Dict[str, Any]:
    out = {k: v for k, v in row.items() if not str(k).startswith("_")}
    if scored:
        return out
    keep = {
        "dataset", "subset", "split", "hf_split", "hf_row_idx", "source_key", "image_path",
        "materialized", "selection_seed", "selection_strategy", "hf_image_path", "text",
        "text_source", "raw_annotation", "width", "height", "file_size", "materialize_order",
        "parquet_path", "parquet_global_start_idx", "parquet_local_row_idx", "text_extraction_status",
    }
    return {k: v for k, v in out.items() if k in keep}


def discover_paths() -> Dict[str, Any]:
    train = PERSIST_ROOT / "outputs/textatlas_stage1_fixed_310k/manifest/train_materialized_manifest_v2text.jsonl"
    val = PERSIST_ROOT / "outputs/textatlas_stage1_fixed_310k/manifest/val_materialized_manifest_v2text.jsonl"
    holdout = PERSIST_ROOT / "outputs/textatlas_stage1_fixed_310k/manifest_holdout_eval/holdout_materialized_manifest_v2text.jsonl"
    tokenizer = PERSIST_ROOT / "models/google_byt5-small"
    required = {
        "repo_root": REPO_ROOT,
        "train_manifest": train,
        "val_manifest": val,
        "holdout_manifest": holdout,
        "tokenizer_path": tokenizer,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError({"missing": missing})
    summary = {key: str(path) for key, path in required.items()}
    summary.update({
        "output_dir": str(OUT_DIR),
        "previous_scored_pool_dir": str(PREV_DIR),
        "transform_resize_pad_method_used": (
            "resize-pad to 256: keep aspect ratio, bicubic resize, centered on white 256x256 canvas; no crop"
        ),
        "bbox_usage": "selection_only_not_final_training_bbox",
        "bbox_source_for_selection": "lightweight_estimator",
        "deepseek_ocr_usage": "not_run_in_fast_v1; final DeepSeek-OCR bbox/proxy deferred until user confirms selected manifests",
    })
    return summary


def scan_manifest(path: Path, split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    field_hits = Counter()
    source_counts = Counter()
    image_missing = 0
    empty_text = 0
    for line_no, row in enumerate(iter_jsonl(path), 1):
        text_field, text = base.choose_text_field(row)
        if text_field:
            field_hits[text_field] += 1
        else:
            empty_text += 1
        image_path = str(row.get("image_path") or "")
        if not image_path or not Path(image_path).exists():
            image_missing += 1
        source = str(row.get("subset") or row.get("source") or "UNKNOWN")
        source_counts[source] += 1
        r = dict(row)
        r["_line_no"] = line_no
        r["_split"] = split
        r["text_field_used"] = text_field
        r["_selected_text"] = text
        r["_source"] = source
        rows.append(r)
    stats = {
        "status": "done",
        "input_paths": [str(path)],
        "output_paths": [],
        "sample_count": len(rows),
        "missing_count": image_missing,
        "warning_count": 0,
        "silent_fallback": False,
        "fallback_reason": None,
        "affects_training": "No, this is split/source/text discovery only.",
        "rows": len(rows),
        "image_path_missing_count": image_missing,
        "empty_text_count": empty_text,
        "empty_text_rate": empty_text / max(1, len(rows)),
        "text_field_hits": {field: int(field_hits.get(field, 0)) for field in base.TEXT_FIELDS},
        "source_counts": dict(source_counts),
        "source_field": "subset",
    }
    return rows, stats


def add_token_stats(rows: List[Dict[str, Any]], tokenizer: Any, fast_byte_offset: Optional[int]) -> None:
    base.add_token_stats(rows, tokenizer, fast_byte_offset)


def score_rows(rows: List[Dict[str, Any]], *, source: str, origin: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    kept: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    reject_counts = Counter()
    token_vals = [float(r.get("byt5_token_len", 0)) for r in rows]
    token_p80 = percentile(token_vals, 80)
    for idx, row in enumerate(rows, 1):
        if idx == 1 or idx % 1000 == 0 or idx == len(rows):
            print(f"[score_rows/{source}] {idx}/{len(rows)}", flush=True)
        r = dict(row)
        r["source"] = source
        r["score_origin"] = origin
        try:
            img = base.resize_pad_image(Image.open(str(r["image_path"])), IMAGE_SIZE)
            boxes, stats = base.estimate_text_boxes(img)
        except Exception as exc:  # noqa: BLE001
            reject_counts["image_open_or_estimator_error"] += 1
            r["reject_reason"] = f"image_open_or_estimator_error:{type(exc).__name__}"
            rejected.append(clean_row(r, scored=True))
            continue
        r["estimated_boxes_256"] = boxes
        r.update(stats)
        r["bbox_source_for_selection"] = "lightweight_estimator"
        r["bbox_usage"] = "selection_only_not_final_training_bbox"
        r["coverage_bucket"] = base.coverage_bucket(float(r.get("bbox_union_coverage_ratio", 0.0)))
        missed = (
            float(r.get("byt5_token_len", 0)) >= token_p80
            and (int(r.get("box_count", 0)) <= 2 or float(r.get("bbox_union_coverage_ratio", 0.0)) < 0.08)
        )
        r["possible_missed_text"] = bool(missed)
        r["doc_like_score"] = float(
            0.45 * float(r.get("white_background_ratio", 0.0))
            + 0.25 * (1.0 - min(float(r.get("mean_saturation", 0.0)) / 0.35, 1.0))
            + 0.20 * float(r.get("bbox_union_coverage_ratio", 0.0))
            + 0.10 * (1.0 - min(float(r.get("image_color_entropy", 0.0)) / 8.0, 1.0))
        )
        reason = None
        if int(r.get("box_count", 0)) == 0:
            reason = "box_count_zero"
        elif float(r.get("bbox_union_coverage_ratio", 0.0)) < 0.05:
            reason = "coverage_lt_0.05"
        elif float(r.get("median_box_height_256", 0.0)) < 6.0:
            reason = "median_box_height_lt_6"
        elif source in REALWORLD_SOURCES and r["possible_missed_text"]:
            reason = "realworld_possible_missed_text"
        elif source in REALWORLD_SOURCES and float(r.get("small_box_ratio", 0.0)) > 0.65:
            reason = "realworld_small_box_ratio_gt_0.65"
        if reason:
            reject_counts[reason] += 1
            r["reject_reason"] = reason
            rejected.append(clean_row(r, scored=True))
            continue
        kept.append(r)
    return kept, {"kept": len(kept), "rejected": dict(reject_counts)}, rejected


def load_previous_scored(split: str, source: str) -> List[Dict[str, Any]]:
    fn = {
        "train": "text_rich_10k_train.scored.jsonl",
        "val": "text_rich_1k_val.scored.jsonl",
        "holdout": "text_rich_500_holdout.scored.jsonl",
    }[split]
    path = PREV_DIR / fn
    if not path.exists():
        return []
    rows = []
    for row in iter_jsonl(path):
        if row.get("subset") == source:
            r = dict(row)
            r["_source"] = source
            r["_split"] = split
            r["_selected_text"] = str(r.get("text") or "")
            r["source"] = source
            r["score_origin"] = "reused_existing_scored_pool"
            r["bbox_source_for_selection"] = r.get("bbox_source_for_selection") or r.get("bbox_source") or "lightweight_estimator"
            r["bbox_usage"] = "selection_only_not_final_training_bbox"
            if "coverage_bucket" not in r:
                r["coverage_bucket"] = base.coverage_bucket(float(r.get("bbox_union_coverage_ratio", 0.0)))
            rows.append(r)
    return rows


def rows_for_source(rows: Sequence[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("_source") == source]


def candidate_subset(rows: List[Dict[str, Any]], target: int, *, source: str, split: str, already_keys: set[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    available = [r for r in rows if str(r.get("source_key") or r.get("image_path")) not in already_keys and r.get("_selected_text")]
    vals = [float(r.get("byt5_token_len", 0)) for r in available]
    p60 = percentile(vals, 60)
    p70 = percentile(vals, 70)
    target_multiplier = 3 if source in REALWORLD_SOURCES else 2
    cap = max(target * target_multiplier, target * 2)
    eligible = [r for r in available if float(r.get("byt5_token_len", 0)) >= p60]
    if len(eligible) < target * 2:
        eligible = sorted(available, key=lambda r: float(r.get("byt5_token_len", 0)), reverse=True)
        policy = "p60 insufficient; top token_len supplement"
    else:
        eligible = sorted(eligible, key=lambda r: float(r.get("byt5_token_len", 0)), reverse=True)
        policy = "token_len>=source_p60 then cap"
    selected = [dict(r) for r in eligible[:cap]]
    return selected, {
        "source": source,
        "split": split,
        "available": len(available),
        "target": target,
        "cap": cap,
        "p60": p60,
        "p70": p70,
        "p80": percentile(vals, 80),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "candidate_count": len(selected),
        "policy": policy,
    }


def normalize(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[int, float]:
    vals = np.asarray([float(r.get(key, 0.0) or 0.0) for r in rows], dtype=np.float64)
    if len(vals) == 0:
        return {}
    lo = float(np.percentile(vals, 5))
    hi = float(np.percentile(vals, 95))
    denom = max(hi - lo, 1e-8)
    return {id(r): float(np.clip((float(r.get(key, 0.0) or 0.0) - lo) / denom, 0.0, 1.0)) for r in rows}


def add_scores(rows: List[Dict[str, Any]]) -> None:
    nt = normalize(rows, "byt5_token_len")
    nh = normalize(rows, "median_box_height_256")
    nb = normalize(rows, "box_count")
    nv = normalize(rows, "visual_diversity")
    for row in rows:
        row["selection_score"] = float(
            0.45 * nt.get(id(row), 0.0)
            + 0.20 * nh.get(id(row), 0.0)
            + 0.20 * nb.get(id(row), 0.0)
            + 0.15 * nv.get(id(row), 0.0)
        )
        if row.get("doc_like"):
            row["selection_score"] -= 0.08
        if row.get("possible_missed_text"):
            row["selection_score"] -= 0.12
        row["coverage_bucket"] = base.coverage_bucket(float(row.get("bbox_union_coverage_ratio", 0.0)))


def bucket_targets(source: str, target: int) -> Dict[str, int]:
    ratios = REALWORLD_BUCKET_RATIOS[source]
    raw = {b: int(round(target * ratio)) for b, ratio in ratios.items()}
    diff = target - sum(raw.values())
    order = ["medium", "local", "dense", "ultra_dense"]
    idx = 0
    while diff != 0:
        b = order[idx % len(order)]
        raw[b] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        idx += 1
    return raw


def stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    value = SEED
    for ch in text:
        value = (value * 131 + ord(ch)) % 1000003
    return value


def pick_top2n_random(pool: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0 or not pool:
        return []
    pool = sorted(pool, key=lambda r: float(r.get("selection_score", 0.0)), reverse=True)
    if len(pool) <= n:
        return list(pool)
    top_pool = pool[: min(len(pool), 2 * n)]
    rng = random.Random(seed)
    return rng.sample(top_pool, n)


def select_source(pool: List[Dict[str, Any]], source: str, target: int, split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    add_scores(pool)
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    shortages: Dict[str, int] = {}
    backfill: List[Dict[str, Any]] = []
    if source in REALWORLD_SOURCES:
        by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in pool:
            by_bucket[str(row.get("coverage_bucket"))].append(row)
        for bucket, need in bucket_targets(source, target).items():
            chosen = pick_top2n_random(by_bucket.get(bucket, []), need, stable_seed(split, source, bucket))
            if len(chosen) < need:
                shortages[bucket] = need - len(chosen)
            for row in chosen:
                key = id(row)
                if key not in selected_ids:
                    selected.append(row)
                    selected_ids.add(key)
        if shortages:
            remaining = [r for r in pool if id(r) not in selected_ids]
            remaining.sort(
                key=lambda r: (
                    min(abs(bucket_index(str(r.get("coverage_bucket"))) - bucket_index(b)) for b in shortages),
                    -float(r.get("selection_score", 0.0)),
                )
            )
            need_total = target - len(selected)
            backfill = remaining[:need_total]
            selected.extend(backfill)
    else:
        selected = pick_top2n_random(pool, target, stable_seed(split, source))
        if len(selected) < target:
            shortages["source_total"] = target - len(selected)
    selected = selected[:target]
    return selected, {
        "target": target,
        "selected": len(selected),
        "bucket_counts": dict(Counter(str(r.get("coverage_bucket")) for r in selected)),
        "shortages": shortages,
        "backfill_count": len(backfill),
    }


def bucket_index(bucket: str) -> int:
    return {"local": 0, "medium": 1, "dense": 2, "ultra_dense": 3}.get(bucket, 0)


def apply_doc_like_control(rows: List[Dict[str, Any]], all_pool: List[Dict[str, Any]], max_ratio: float = 0.40) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ratio = doc_like_ratio(rows)
    if ratio <= max_ratio:
        return rows, {"attempted": False, "before": ratio, "after": ratio, "replaced": 0}
    selected_keys = {str(r.get("source_key") or r.get("image_path")) for r in rows}
    replacements_by_source_bucket: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in all_pool:
        key = str(row.get("source_key") or row.get("image_path"))
        if key in selected_keys or row.get("doc_like"):
            continue
        replacements_by_source_bucket[(str(row.get("source")), str(row.get("coverage_bucket")))].append(row)
    for repls in replacements_by_source_bucket.values():
        repls.sort(key=lambda r: float(r.get("visual_diversity", 0.0)), reverse=True)
    out = list(rows)
    replaced = 0
    for idx, row in enumerate(list(out)):
        if doc_like_ratio(out) <= max_ratio:
            break
        if not row.get("doc_like"):
            continue
        key = (str(row.get("source")), str(row.get("coverage_bucket")))
        pool = replacements_by_source_bucket.get(key, [])
        if not pool:
            continue
        out[idx] = pool.pop(0)
        replaced += 1
    return out, {"attempted": True, "before": ratio, "after": doc_like_ratio(out), "replaced": replaced}


def doc_like_ratio(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(1 for row in rows if row.get("doc_like")) / max(1, len(rows))


def duplicate_for_train(rows: List[Dict[str, Any]], source: str, need: int) -> List[Dict[str, Any]]:
    if need <= 0:
        return []
    candidates = [
        r for r in rows
        if r.get("source") == source
        and source in REALWORLD_SOURCES
        and not r.get("doc_like")
        and not r.get("possible_missed_text")
        and int(r.get("box_count", 0)) >= 3
        and float(r.get("median_box_height_256", 0.0)) >= 7.0
    ]
    candidates.sort(key=lambda r: float(r.get("selection_score", 0.0)), reverse=True)
    dupes = []
    for row in candidates[:need]:
        d = dict(row)
        key = str(row.get("source_key") or row.get("image_path"))
        d["duplicate_from"] = key
        d["repeat_index"] = 2
        d["repeat_count"] = 2
        d["duplicate_reason"] = "realworld_hard_sample_shortage_weighting"
        dupes.append(d)
    return dupes


def preview_grid(path: Path, rows: Sequence[Dict[str, Any]], *, per_source: Mapping[str, int], title: str) -> None:
    chosen: List[Dict[str, Any]] = []
    for source, count in per_source.items():
        source_rows = [r for r in rows if r.get("source") == source]
        source_rows.sort(key=lambda r: float(r.get("selection_score", 0.0)), reverse=True)
        chosen.extend(source_rows[:count])
    if not chosen:
        return
    cell = 256
    label_h = 64
    cols = 10
    rows_n = math.ceil(len(chosen) / cols)
    canvas = Image.new("RGB", (cols * cell, rows_n * (cell + label_h)), "white")
    draw_canvas = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    draw_canvas.text((8, 4), title, fill=(0, 0, 0), font=font)
    y_offset = 16
    for idx, row in enumerate(chosen):
        x = (idx % cols) * cell
        y = y_offset + (idx // cols) * (cell + label_h)
        try:
            img = base.resize_pad_image(Image.open(str(row["image_path"])), IMAGE_SIZE)
        except Exception:  # noqa: BLE001
            img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), "#eeeeee")
        d = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in row.get("estimated_boxes_256", [])[:120]:
            d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=1)
        canvas.paste(img, (x, y))
        label = (
            f"{row.get('source')} {row.get('coverage_bucket')}\n"
            f"tok={row.get('byt5_token_len')} cov={float(row.get('bbox_union_coverage_ratio', 0.0)):.2f} "
            f"box={row.get('box_count')} h={float(row.get('median_box_height_256', 0.0)):.1f}\n"
            f"small={float(row.get('small_box_ratio', 0.0)):.2f} doc={int(bool(row.get('doc_like')))}"
        )
        draw_canvas.text((x + 4, y + cell + 3), label, fill=(0, 0, 0), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_hist(path: Path, rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]], key: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    for split, rows in rows_by_split.items():
        vals = [float(row.get(key, 0.0) or 0.0) for row in rows]
        plt.hist(vals, bins=40, alpha=0.45, label=split)
    plt.title(title)
    plt.xlabel(key)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def write_discovery_md(path: Path, discovery: Mapping[str, Any]) -> None:
    lines = ["# Path Discovery", "", f"- generated_at: {now()}"]
    for key, value in discovery.items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-duplicate-ratio", type=float, default=0.10)
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state_dir = OUT_DIR / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "running").write_text(now() + "\n", encoding="utf-8")

    stage_summary: Dict[str, Any] = {}
    discovery = discover_paths()
    jdump(OUT_DIR / "path_discovery_summary.json", discovery)
    write_discovery_md(OUT_DIR / "path_discovery_summary.md", discovery)
    print(json.dumps({"stage": "path_discovery", "summary": discovery}, indent=2, ensure_ascii=False), flush=True)

    manifest_paths = {
        "train": Path(discovery["train_manifest"]),
        "val": Path(discovery["val_manifest"]),
        "holdout": Path(discovery["holdout_manifest"]),
    }
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    manifest_stats = {}
    for split, path in manifest_paths.items():
        rows, stats = scan_manifest(path, split)
        if stats["empty_text_rate"] > 0.01:
            raise RuntimeError(f"{split} empty_text_rate={stats['empty_text_rate']:.4f} > 1%")
        rows_by_split[split] = rows
        manifest_stats[split] = stats
    stage_summary["A0_split_check"] = {
        "status": "done",
        "input_paths": [str(p) for p in manifest_paths.values()],
        "output_paths": [str(OUT_DIR / "selection_summary.json")],
        "sample_count": sum(len(v) for v in rows_by_split.values()),
        "missing_count": sum(v["image_path_missing_count"] for v in manifest_stats.values()),
        "warning_count": 0,
        "silent_fallback": False,
        "fallback_reason": None,
        "affects_training": "No; confirms source/text/image availability before selection.",
    }

    tokenizer = AutoTokenizer.from_pretrained(discovery["tokenizer_path"], local_files_only=True)
    fast_offset = base.infer_byt5_byte_offset(rows_by_split, tokenizer)
    for rows in rows_by_split.values():
        add_token_stats(rows, tokenizer, fast_offset)
    source_token_summary = {}
    for split, rows in rows_by_split.items():
        source_token_summary[split] = {
            source: base.token_summary(rows_for_source(rows, source))
            for source in SOURCES
        }
    stage_summary["A1_token_stats"] = {
        "status": "done",
        "input_paths": [discovery["tokenizer_path"]],
        "output_paths": [],
        "sample_count": sum(len(v) for v in rows_by_split.values()),
        "missing_count": 0,
        "warning_count": 0,
        "silent_fallback": False,
        "fallback_reason": None,
        "affects_training": "No; token length is only for selection ranking.",
        "byt5_fast_byte_offset": fast_offset,
    }

    all_pools: Dict[str, Dict[str, List[Dict[str, Any]]]] = {split: {} for split in ("train", "val", "holdout")}
    candidate_stats: Dict[str, Any] = defaultdict(dict)
    rejected_realworld: List[Dict[str, Any]] = []
    source_processing: Dict[str, Any] = defaultdict(dict)
    for split in ("train", "val", "holdout"):
        for source in SOURCES:
            target = TARGETS[split][source]
            existing = [] if source in REALWORLD_SOURCES else load_previous_scored(split, source)
            origin = "reused_existing_scored_pool" if existing else "none"
            need_more = source in REALWORLD_SOURCES or len(existing) < max(target, int(target * 1.3))
            scored_new: List[Dict[str, Any]] = []
            score_stats: Dict[str, Any] = {"kept": 0, "rejected": {}}
            cand_stats: Dict[str, Any] = {}
            if need_more:
                already = {str(r.get("source_key") or r.get("image_path")) for r in existing}
                candidates, cand_stats = candidate_subset(rows_for_source(rows_by_split[split], source), target, source=source, split=split, already_keys=already)
                print(f"[{split}/{source}] scoring_lightweight candidates={len(candidates)} reused={len(existing)}", flush=True)
                scored_new, score_stats, rejected = score_rows(candidates, source=source, origin="newly_scored_lightweight" if source in REALWORLD_SOURCES else "lightweight_supplement_scored")
                if source in REALWORLD_SOURCES:
                    rejected_realworld.extend(rejected)
            pool = existing + scored_new
            for row in pool:
                row["source"] = source
                row["_split"] = split
                row.setdefault("bbox_usage", "selection_only_not_final_training_bbox")
                row.setdefault("bbox_source_for_selection", "lightweight_estimator")
                row.setdefault("deepseek_ocr_usage", "not_run_in_fast_v1")
            all_pools[split][source] = pool
            candidate_stats[split][source] = {
                "target": target,
                "existing_reused_count": len(existing),
                "new_candidate_stats": cand_stats,
                "new_score_stats": score_stats,
                "pool_count": len(pool),
                "bbox_source": "lightweight_estimator",
                "heavy_check": False,
            }
            source_processing[split][source] = {
                "mode": "newly_scored_lightweight" if source in REALWORLD_SOURCES else ("reused_existing_scored_pool" if not need_more else "reused_plus_lightweight_supplement"),
                "newly_scored_count": len(scored_new),
                "reused_count": len(existing),
                "heavy_check": False,
                "bbox_source": "lightweight_estimator",
            }
            print(f"[{split}/{source}] reused={len(existing)} new={len(scored_new)} pool={len(pool)}", flush=True)

    selected_by_split: Dict[str, List[Dict[str, Any]]] = {}
    source_select_stats: Dict[str, Any] = defaultdict(dict)
    shortage_backfill: Dict[str, Any] = defaultdict(dict)
    for split in ("train", "val", "holdout"):
        selected_split: List[Dict[str, Any]] = []
        selected_keys = set()
        for source in SOURCES:
            target = TARGETS[split][source]
            pool = [r for r in all_pools[split][source] if str(r.get("source_key") or r.get("image_path")) not in selected_keys]
            chosen, stats = select_source(pool, source, target, split)
            selected_keys.update(str(r.get("source_key") or r.get("image_path")) for r in chosen)
            source_select_stats[split][source] = stats
            selected_split.extend(chosen)
        for source in SOURCES:
            current = [r for r in selected_split if r.get("source") == source]
            target = TARGETS[split][source]
            if len(current) >= target:
                continue
            need = target - len(current)
            if split == "train" and source in REALWORLD_SOURCES:
                max_dupes = int(args.max_duplicate_ratio * sum(TARGETS[split].values())) - sum(1 for r in selected_split if r.get("duplicate_from"))
                dupes = duplicate_for_train(current, source, min(need, max(0, max_dupes)))
                selected_split.extend(dupes)
                need -= len(dupes)
                shortage_backfill[split][source] = {"duplicate_added": len(dupes), "remaining_need": need}
            if need > 0:
                for alt in BACKFILL_ORDER[source]:
                    alt_pool = [
                        r for r in all_pools[split][alt]
                        if str(r.get("source_key") or r.get("image_path")) not in selected_keys
                        and not r.get("doc_like")
                    ]
                    add_scores(alt_pool)
                    alt_pool.sort(key=lambda r: float(r.get("selection_score", 0.0)), reverse=True)
                    take = alt_pool[:need]
                    for row in take:
                        row["backfill_for_source"] = source
                        row["backfill_reason"] = "source_shortage_no_easy_samples"
                    selected_split.extend(take)
                    selected_keys.update(str(r.get("source_key") or r.get("image_path")) for r in take)
                    need -= len(take)
                    if need <= 0:
                        break
                shortage_backfill[split].setdefault(source, {})["remaining_after_backfill"] = need
        all_pool_flat = [row for source_rows in all_pools[split].values() for row in source_rows]
        selected_split, doc_control = apply_doc_like_control(selected_split, all_pool_flat)
        for row in selected_split:
            row["selection_seed"] = SEED
            row["selection_strategy"] = "realworld40_text_rich_fast_v1"
        selected_by_split[split] = selected_split[: sum(TARGETS[split].values())]
        source_select_stats[split]["doc_like_control"] = doc_control

    train_unique = {str(r.get("image_path")) for r in selected_by_split["train"]}
    val_unique = {str(r.get("image_path")) for r in selected_by_split["val"]}
    holdout_unique = {str(r.get("image_path")) for r in selected_by_split["holdout"]}
    overlaps = {
        "unique_train_val": len(train_unique & val_unique),
        "unique_train_holdout": len(train_unique & holdout_unique),
        "unique_val_holdout": len(val_unique & holdout_unique),
    }
    if any(overlaps.values()):
        jdump(OUT_DIR / "overlap_failure_summary.json", overlaps)
        raise RuntimeError(f"split leakage found: {overlaps}")

    qual_rows: List[Dict[str, Any]] = []
    for source, target in TARGETS["qual"].items():
        pool = [r for r in selected_by_split["holdout"] if r.get("source") == source]
        pool.sort(key=lambda r: float(r.get("selection_score", 0.0)), reverse=True)
        qual_rows.extend(pool[:target])

    outputs = {
        "train": ("train_10k_realworld40_text_rich.jsonl", "train_10k_realworld40_text_rich.scored.jsonl"),
        "val": ("val_1k_realworld40_text_rich.jsonl", "val_1k_realworld40_text_rich.scored.jsonl"),
        "holdout": ("holdout_500_realworld40_text_rich.jsonl", "holdout_500_realworld40_text_rich.scored.jsonl"),
        "qual": ("qual_20_realworld40_text_rich.jsonl", "qual_20_realworld40_text_rich.scored.jsonl"),
    }
    for split in ("train", "val", "holdout"):
        write_jsonl(OUT_DIR / outputs[split][0], [clean_row(r, scored=False) for r in selected_by_split[split]])
        write_jsonl(OUT_DIR / outputs[split][1], [clean_row(r, scored=True) for r in selected_by_split[split]])
    write_jsonl(OUT_DIR / outputs["qual"][0], [clean_row(r, scored=False) for r in qual_rows])
    write_jsonl(OUT_DIR / outputs["qual"][1], [clean_row(r, scored=True) for r in qual_rows])

    write_jsonl(OUT_DIR / "rejected_realworld_low_quality.jsonl", rejected_realworld)
    rejected_summary = {
        "rejected_realworld_low_quality_count": len(rejected_realworld),
        "reason_counts": dict(Counter(r.get("reject_reason", "unknown") for r in rejected_realworld)),
    }
    jdump(OUT_DIR / "rejected_summary.json", rejected_summary)

    save_hist(OUT_DIR / "token_len_histogram.png", selected_by_split, "byt5_token_len", "Selected ByT5 token length")
    save_hist(OUT_DIR / "coverage_histogram.png", selected_by_split, "bbox_union_coverage_ratio", "Selected bbox coverage")
    save_hist(OUT_DIR / "box_height_histogram.png", selected_by_split, "median_box_height_256", "Selected median text-box height")
    save_hist(OUT_DIR / "doc_like_histogram.png", selected_by_split, "doc_like_score", "Selected doc-like score")
    selected_all = [r for rows in selected_by_split.values() for r in rows]
    preview_grid(
        OUT_DIR / "source_preview_grid.png",
        selected_all,
        per_source={"TextScenesHQ": 40, "TextVisionBlend": 30, "StyledTextSynth": 20, "CleanTextSynth": 20, "LongWordsSubset-A": 20},
        title="realworld40 selected preview; red boxes are selection-only lightweight estimates",
    )
    preview_grid(OUT_DIR / "TextScenesHQ_preview_grid.png", selected_all, per_source={"TextScenesHQ": 60}, title="TextScenesHQ selected preview")
    preview_grid(OUT_DIR / "TextVisionBlend_preview_grid.png", selected_all, per_source={"TextVisionBlend": 60}, title="TextVisionBlend selected preview")

    source_counts = {
        split: dict(Counter(r.get("source") for r in rows))
        for split, rows in selected_by_split.items()
    }
    bucket_counts = {
        split: dict(Counter(r.get("coverage_bucket") for r in rows))
        for split, rows in selected_by_split.items()
    }
    source_bucket_counts = {
        split: {
            source: dict(Counter(r.get("coverage_bucket") for r in rows if r.get("source") == source))
            for source in SOURCES
        }
        for split, rows in selected_by_split.items()
    }
    duplicate_rows = [r for r in selected_by_split["train"] if r.get("duplicate_from")]
    duplicate_by_source = Counter(r.get("source") for r in duplicate_rows)
    duplicate_reason_counts = Counter(r.get("duplicate_reason") for r in duplicate_rows)

    stage_summary["A2_bbox_coverage_stats"] = {
        "status": "done",
        "input_paths": [str(p) for p in manifest_paths.values()],
        "output_paths": [str(OUT_DIR / "source_preview_grid.png")],
        "sample_count": sum(candidate_stats[split][source]["new_score_stats"].get("kept", 0) for split in candidate_stats for source in candidate_stats[split]),
        "missing_count": 0,
        "warning_count": 0,
        "silent_fallback": False,
        "fallback_reason": None,
        "affects_training": "No; lightweight boxes are selection-only and not final training bbox.",
        "bbox_usage": "selection_only_not_final_training_bbox",
        "bbox_source_for_selection": "lightweight_estimator",
        "deepseek_ocr_usage": "not_run_in_fast_v1; sample quality check deferred/not required for final bbox",
    }
    stage_summary["A3_A5_selection"] = {
        "status": "done",
        "input_paths": [str(OUT_DIR / "path_discovery_summary.json")],
        "output_paths": [str(OUT_DIR / outputs["train"][0]), str(OUT_DIR / outputs["val"][0]), str(OUT_DIR / outputs["holdout"][0])],
        "sample_count": sum(len(v) for v in selected_by_split.values()),
        "missing_count": 0,
        "warning_count": sum(len(v.get("shortages", {})) for split in source_select_stats.values() for v in split.values() if isinstance(v, dict)),
        "silent_fallback": False,
        "fallback_reason": None,
        "affects_training": "Yes after user confirmation; these manifests define candidate final data, but no bound preprocessing has run.",
    }
    stage_summary["A6_original_gigatok_easy_pilot"] = {
        "status": "skipped",
        "input_paths": [],
        "output_paths": [],
        "sample_count": 0,
        "missing_count": 0,
        "warning_count": 1,
        "silent_fallback": False,
        "fallback_reason": "fast_v1 explicitly avoids heavy original reconstruction; previous hardmix run already established original VQ path, but this run does not claim easy_sparse filtering.",
        "affects_training": "Potentially; user should inspect preview/summary before approving final preprocessing.",
    }

    summary = {
        "generated_at": now(),
        "output_dir": str(OUT_DIR),
        "preparation_only": True,
        "no_formal_training_started": True,
        "no_final_bound_preprocessing_started": True,
        "bbox_usage": "selection_only_not_final_training_bbox",
        "bbox_source_for_selection": "lightweight_estimator",
        "deepseek_ocr_usage": "not_run_in_fast_v1; final DeepSeek-OCR bbox/proxy deferred until selected manifests are confirmed",
        "source_manifest_paths": {split: str(path) for split, path in manifest_paths.items()},
        "tokenizer_path": discovery["tokenizer_path"],
        "transform_resize_pad_method_used": discovery["transform_resize_pad_method_used"],
        "source_subset_discovery_summary": {split: manifest_stats[split]["source_counts"] for split in manifest_stats},
        "stage_summaries": stage_summary,
        "manifest_stats": manifest_stats,
        "source_token_summary": source_token_summary,
        "source_processing": source_processing,
        "candidate_stats": candidate_stats,
        "source_selection_stats": source_select_stats,
        "shortage_backfill_records": shortage_backfill,
        "final_counts": {split: len(rows) for split, rows in selected_by_split.items()},
        "source_counts": source_counts,
        "bucket_counts": bucket_counts,
        "source_bucket_counts": source_bucket_counts,
        "TextScenesHQ_bucket_counts": {split: source_bucket_counts[split]["TextScenesHQ"] for split in source_bucket_counts},
        "TextVisionBlend_bucket_counts": {split: source_bucket_counts[split]["TextVisionBlend"] for split in source_bucket_counts},
        "overlap_checks_unique_image_path": overlaps,
        "token_len_distribution": {split: distribution(rows, "byt5_token_len") for split, rows in selected_by_split.items()},
        "coverage_distribution": {split: distribution(rows, "bbox_union_coverage_ratio") for split, rows in selected_by_split.items()},
        "box_height_distribution": {split: distribution(rows, "median_box_height_256") for split, rows in selected_by_split.items()},
        "doc_like_ratio": {split: doc_like_ratio(rows) for split, rows in selected_by_split.items()},
        "TextScenesHQ_bbox_quality_summary": {
            split: {
                "count": sum(1 for r in rows if r.get("source") == "TextScenesHQ"),
                "coverage": distribution([r for r in rows if r.get("source") == "TextScenesHQ"], "bbox_union_coverage_ratio"),
                "box_count": distribution([r for r in rows if r.get("source") == "TextScenesHQ"], "box_count"),
                "small_box_ratio": distribution([r for r in rows if r.get("source") == "TextScenesHQ"], "small_box_ratio"),
            }
            for split, rows in selected_by_split.items()
        },
        "TextVisionBlend_bbox_quality_summary": {
            split: {
                "count": sum(1 for r in rows if r.get("source") == "TextVisionBlend"),
                "coverage": distribution([r for r in rows if r.get("source") == "TextVisionBlend"], "bbox_union_coverage_ratio"),
                "box_count": distribution([r for r in rows if r.get("source") == "TextVisionBlend"], "box_count"),
                "small_box_ratio": distribution([r for r in rows if r.get("source") == "TextVisionBlend"], "small_box_ratio"),
            }
            for split, rows in selected_by_split.items()
        },
        "rejected_realworld_low_quality_count": len(rejected_realworld),
        "rejected_realworld_reasons": rejected_summary["reason_counts"],
        "selected_token_len": distribution(selected_all, "byt5_token_len"),
        "selected_coverage": distribution(selected_all, "bbox_union_coverage_ratio"),
        "duplicate_total": len(duplicate_rows),
        "duplicate_ratio": len(duplicate_rows) / max(1, len(selected_by_split["train"])),
        "duplicate_by_source": dict(duplicate_by_source),
        "duplicate_reason_counts": dict(duplicate_reason_counts),
        "original_gigatok_easy_pilot_status": "skipped_fast_v1_no_heavy_reconstruction",
        "warnings": [
            "Fast_v1 did not run original GigaTok reconstruction easy filtering.",
            "Fast_v1 did not run DeepSeek-OCR bbox/proxy; lightweight boxes are not final training boxes.",
        ],
        "output_files": {split: [str(OUT_DIR / f) for f in pair] for split, pair in outputs.items()},
        "preview_grids": {
            "source_preview_grid": str(OUT_DIR / "source_preview_grid.png"),
            "TextScenesHQ_preview_grid": str(OUT_DIR / "TextScenesHQ_preview_grid.png"),
            "TextVisionBlend_preview_grid": str(OUT_DIR / "TextVisionBlend_preview_grid.png"),
        },
    }
    jdump(OUT_DIR / "selection_summary.json", summary)
    jdump(OUT_DIR / "source_counts.json", source_counts)
    jdump(OUT_DIR / "bucket_counts.json", bucket_counts)
    jdump(OUT_DIR / "source_bucket_counts.json", source_bucket_counts)
    (state_dir / "done").write_text(now() + "\n", encoding="utf-8")
    try:
        (state_dir / "running").unlink()
    except FileNotFoundError:
        pass
    print(json.dumps({"done": True, "output_dir": str(OUT_DIR), "source_counts": source_counts, "bucket_counts": bucket_counts}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
