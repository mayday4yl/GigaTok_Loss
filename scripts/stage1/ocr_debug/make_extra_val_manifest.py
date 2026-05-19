#!/usr/bin/env python3
"""Build a held-out extra validation manifest for OCR/text reconstruction checks.

The script samples rows from an existing TextAtlas materialized manifest, excludes
the readable50 diagnostic set, and writes a small JSONL manifest that can be used
with ``evaluate_textatlas_reconstruction.py``. It does not download or modify
images.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Sequence

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    parser.add_argument("--subset", default="CleanTextSynth", help="Use empty string to allow all subsets.")
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--max-text-chars", type=int, default=0, help="0 means no upper bound.")
    parser.add_argument("--require-existing-images", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            row["_line_no"] = line_no
            yield row


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            clean = {key: value for key, value in row.items() if not key.startswith("_")}
            handle.write(json.dumps(clean, ensure_ascii=False))
            handle.write("\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def resolve_image_path(row: Mapping[str, Any], manifest_path: Path) -> Path:
    image_path = Path(str(row.get("image_path") or ""))
    if not image_path.is_absolute():
        image_path = manifest_path.resolve().parent / image_path
    return image_path.resolve()


def row_keys(row: Mapping[str, Any], manifest_path: Path) -> set[str]:
    keys = {
        str(resolve_image_path(row, manifest_path)),
        str(row.get("source_key") or ""),
        str(row.get("text") or ""),
    }
    return {key for key in keys if key}


def load_excluded_keys(paths: Sequence[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"exclude manifest not found: {path}")
        for row in iter_jsonl(path):
            excluded.update(row_keys(row, path))
    return excluded


def eligible_rows(args: argparse.Namespace, excluded: set[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    subset_filter = str(args.subset).strip()
    for row in iter_jsonl(args.source_manifest):
        if row_keys(row, args.source_manifest) & excluded:
            continue
        subset = str(row.get("subset") or "")
        if subset_filter and subset != subset_filter:
            continue
        text = str(row.get("text") or "").strip()
        if len(text) < args.min_text_chars:
            continue
        if args.max_text_chars > 0 and len(text) > args.max_text_chars:
            continue
        image_path = resolve_image_path(row, args.source_manifest)
        if args.require_existing_images and not image_path.exists():
            continue
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        rows.append(clean)
    return rows


def make_contact_sheet(path: Path, rows: Sequence[Mapping[str, Any]], source_manifest: Path) -> None:
    from PIL import Image, ImageDraw

    if not rows:
        return
    thumbs = []
    for idx, row in enumerate(rows, start=1):
        image_path = resolve_image_path(row, source_manifest)
        with Image.open(image_path) as img:
            thumb = img.convert("RGB").resize((160, 160), Image.Resampling.BICUBIC)
        thumbs.append((idx, thumb, str(row.get("source_key") or image_path.name)))

    cols = 5
    label_h = 22
    rows_n = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 160, rows_n * (160 + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for offset, (idx, thumb, label) in enumerate(thumbs):
        row = offset // cols
        col = offset % cols
        x = col * 160
        y = row * (160 + label_h)
        sheet.paste(thumb, (x, y + label_h))
        draw.rectangle((x, y, x + 159, y + label_h - 1), fill=(255, 255, 255))
        draw.text((x + 3, y + 4), f"{idx:02d} {label}"[:28], fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise SystemExit("--num-images must be positive")
    if args.min_text_chars < 0:
        raise SystemExit("--min-text-chars must be >= 0")
    if args.max_text_chars < 0:
        raise SystemExit("--max-text-chars must be >= 0")
    if not args.source_manifest.exists():
        raise SystemExit(f"source manifest not found: {args.source_manifest}")

    excluded = load_excluded_keys(args.exclude_manifest)
    candidates = eligible_rows(args, excluded)
    if len(candidates) < args.num_images:
        raise SystemExit(
            f"Only {len(candidates)} eligible rows found, need {args.num_images}. "
            "Lower --num-images or relax filters."
        )

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.num_images)
    selected.sort(key=lambda row: str(row.get("source_key") or row.get("image_path") or ""))

    write_jsonl(args.output_manifest, selected)
    summary_json = args.summary_json or args.output_manifest.with_name(f"{args.output_manifest.stem}_summary.json")
    text_lengths = [len(str(row.get("text") or "")) for row in selected]
    summary = {
        "source_manifest": str(args.source_manifest),
        "exclude_manifest": [str(path) for path in args.exclude_manifest],
        "output_manifest": str(args.output_manifest),
        "subset": args.subset,
        "num_images": len(selected),
        "seed": args.seed,
        "min_text_chars": args.min_text_chars,
        "max_text_chars": args.max_text_chars,
        "eligible_candidates": len(candidates),
        "excluded_keys": len(excluded),
        "text_chars_min": min(text_lengths),
        "text_chars_mean": mean(text_lengths),
        "text_chars_max": max(text_lengths),
        "examples": [
            {
                "image_path": row.get("image_path", ""),
                "source_key": row.get("source_key", ""),
                "text_preview": str(row.get("text") or "")[:160],
            }
            for row in selected[:5]
        ],
    }
    write_json(summary_json, summary)

    if args.contact_sheet is not None:
        make_contact_sheet(args.contact_sheet, selected, args.source_manifest)

    print(f"selected: {len(selected)} / candidates: {len(candidates)}")
    print(f"manifest: {args.output_manifest}")
    print(f"summary: {summary_json}")
    if args.contact_sheet is not None:
        print(f"contact_sheet: {args.contact_sheet}")


if __name__ == "__main__":
    main()
