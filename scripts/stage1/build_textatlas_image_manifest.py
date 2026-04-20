#!/usr/bin/env python3
"""Build image-only TextAtlas5M manifests for stage-1 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


DEFAULT_DATASET = "CSU-JPG/TextAtlas5M"
ALLOWED_SUBSETS = (
    "CleanTextSynth",
    "StyledTextSynth",
    "LongWordsSubset-M",
    "TextScenesHQ",
)
TEXT_SOURCE_BY_SUBSET = {
    "CleanTextSynth": "annotation",
    "StyledTextSynth": "annotation",
    "LongWordsSubset-M": "annotation",
    "TextScenesHQ": "raw_text",
}
_PIL_IMAGE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a small image-only TextAtlas5M pilot set and write "
            "MixedDatasetJson-compatible image path lists plus rich audit manifests."
        )
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=ALLOWED_SUBSETS, default=list(ALLOWED_SUBSETS))
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--train-per-subset", type=int, default=10_000)
    parser.add_argument("--val-per-subset", type=int, default=500)
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Number of HF rows to skip before collecting each subset. Keeps hf_row_idx exact.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Recorded for reproducibility; selection is sequential.")
    parser.add_argument("--save-format", choices=("jpeg", "png"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-source-rows-per-subset",
        type=int,
        default=0,
        help="Optional safety cap on source rows scanned per subset. 0 means no cap.",
    )
    return parser.parse_args()


def import_datasets():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it in the stage-1 environment "
            "before building the TextAtlas manifest."
        ) from exc
    return load_dataset


def import_pillow():
    global _PIL_IMAGE
    if _PIL_IMAGE is not None:
        return _PIL_IMAGE
    try:
        from PIL import Image as pil_image  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: Pillow. Install it in the stage-1 environment "
            "before building or checking the TextAtlas image manifest."
        ) from exc
    _PIL_IMAGE = pil_image
    return pil_image


def ensure_positive(args: argparse.Namespace) -> None:
    if args.train_per_subset < 0 or args.val_per_subset < 0:
        raise SystemExit("train/val counts must be non-negative.")
    if args.train_per_subset == 0 and args.val_per_subset == 0:
        raise SystemExit("At least one of train-per-subset or val-per-subset must be positive.")
    if args.start_offset < 0:
        raise SystemExit("start-offset must be non-negative.")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("jpeg-quality must be in [1, 100].")


def image_extension(save_format: str) -> str:
    return "jpg" if save_format == "jpeg" else "png"


def validate_saved_image(path: Path) -> Tuple[int, int]:
    pil_image = import_pillow()
    with pil_image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.load()
        return rgb.size


def row_to_rgb_image(value: Any) -> Any:
    pil_image = import_pillow()
    if isinstance(value, pil_image.Image):
        img = value
    elif isinstance(value, dict):
        if value.get("bytes") is not None:
            img = pil_image.open(BytesIO(value["bytes"]))
        elif value.get("path"):
            img = pil_image.open(value["path"])
        else:
            raise ValueError("Unsupported image dict; expected bytes or path.")
    else:
        raise TypeError(f"Unsupported image type: {type(value)!r}")

    rgb = img.convert("RGB")
    rgb.load()
    return rgb


def save_image(img: Any, path: Path, save_format: str, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "jpeg":
        img.save(path, format="JPEG", quality=jpeg_quality, subsampling=0)
    else:
        img.save(path, format="PNG", compress_level=3)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def append_jsonl(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def skip_rows(iterator: Iterator[Dict[str, Any]], count: int) -> int:
    skipped = 0
    while skipped < count:
        next(iterator)
        skipped += 1
    return skipped


def get_text_fields(subset: str, row: Dict[str, Any]) -> Tuple[str, str, str]:
    text_source = TEXT_SOURCE_BY_SUBSET[subset]
    text = row.get(text_source) or ""
    raw_annotation = row.get("annotation") or ""
    return str(text), text_source, str(raw_annotation)


def materialize_row(
    *,
    dataset_name: str,
    subset: str,
    hf_split: str,
    local_split: str,
    hf_row_idx: int,
    row: Dict[str, Any],
    image_root: Path,
    save_format: str,
    jpeg_quality: int,
    overwrite: bool,
) -> Dict[str, Any]:
    hf_image_path = str(row.get("image_path") or f"{hf_row_idx:010d}")
    ext = image_extension(save_format)
    image_path = (image_root / subset / local_split / f"{hf_row_idx:010d}.{ext}").resolve()

    if image_path.exists() and not overwrite:
        width, height = validate_saved_image(image_path)
    else:
        img = row_to_rgb_image(row.get("image"))
        save_image(img, image_path, save_format, jpeg_quality)
        width, height = validate_saved_image(image_path)

    text, text_source, raw_annotation = get_text_fields(subset, row)
    return {
        "dataset": dataset_name,
        "subset": subset,
        "split": local_split,
        "hf_split": hf_split,
        "hf_row_idx": hf_row_idx,
        "hf_image_path": hf_image_path,
        "image_path": str(image_path),
        "text": text,
        "text_source": text_source,
        "raw_annotation": raw_annotation,
        "width": width,
        "height": height,
        "file_size": image_path.stat().st_size,
    }


def collect_subset(
    *,
    load_dataset,
    args: argparse.Namespace,
    subset: str,
    bad_handle,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    dataset = load_dataset(args.dataset_name, subset, split=args.hf_split, streaming=True)
    iterator = iter(dataset)
    row_idx = skip_rows(iterator, args.start_offset)
    scanned = row_idx

    train_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    bad_count = 0
    source_limit = args.max_source_rows_per_subset

    while len(train_rows) < args.train_per_subset or len(val_rows) < args.val_per_subset:
        if source_limit and scanned - args.start_offset >= source_limit:
            raise RuntimeError(
                f"{subset}: reached max-source-rows-per-subset={source_limit} "
                f"before collecting requested train/val counts."
            )

        try:
            row = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(f"{subset}: source split ended before requested counts were collected.") from exc

        hf_row_idx = scanned
        scanned += 1
        local_split = "train" if len(train_rows) < args.train_per_subset else "val"

        try:
            record = materialize_row(
                dataset_name=args.dataset_name,
                subset=subset,
                hf_split=args.hf_split,
                local_split=local_split,
                hf_row_idx=hf_row_idx,
                row=row,
                image_root=args.image_root,
                save_format=args.save_format,
                jpeg_quality=args.jpeg_quality,
                overwrite=args.overwrite,
            )
        except Exception as exc:  # noqa: BLE001 - bad rows should be logged and skipped.
            bad_count += 1
            append_jsonl(
                bad_handle,
                {
                    "dataset": args.dataset_name,
                    "subset": subset,
                    "hf_split": args.hf_split,
                    "hf_row_idx": hf_row_idx,
                    "error": repr(exc),
                },
            )
            continue

        if local_split == "train":
            train_rows.append(record)
        else:
            val_rows.append(record)

    stats = {
        "subset": subset,
        "train": len(train_rows),
        "val": len(val_rows),
        "bad_rows": bad_count,
        "source_rows_scanned": scanned - args.start_offset,
        "start_offset": args.start_offset,
    }
    return train_rows, val_rows, stats


def summarize_counts(records_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"splits": {}, "total_images": 0, "total_bytes": 0}
    for split, records in records_by_split.items():
        by_subset = Counter(record["subset"] for record in records)
        bytes_by_subset: Dict[str, int] = defaultdict(int)
        for record in records:
            bytes_by_subset[record["subset"]] += int(record.get("file_size", 0))
        split_bytes = sum(bytes_by_subset.values())
        summary["splits"][split] = {
            "total": len(records),
            "by_subset": dict(sorted(by_subset.items())),
            "bytes_by_subset": dict(sorted(bytes_by_subset.items())),
            "total_bytes": split_bytes,
        }
        summary["total_images"] += len(records)
        summary["total_bytes"] += split_bytes
    return summary


def write_sha256_manifest(output_root: Path, paths: Iterable[Path], filename: str) -> Path:
    sha_path = output_root / filename
    with sha_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            digest = sha256_file(path)
            handle.write(f"{digest}  {path.name}\n")
    return sha_path


def main() -> None:
    args = parse_args()
    ensure_positive(args)
    args.output_root = args.output_root.resolve()
    args.image_root = args.image_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.image_root.mkdir(parents=True, exist_ok=True)

    load_dataset = import_datasets()
    started_at = time.strftime("%Y-%m-%d %H:%M:%S")

    records_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    subset_stats: List[Dict[str, Any]] = []
    bad_rows_path = args.output_root / "bad_rows.jsonl"

    with bad_rows_path.open("w", encoding="utf-8") as bad_handle:
        for subset in args.subsets:
            print(f"[build-textatlas] collecting subset={subset}", flush=True)
            train_rows, val_rows, stats = collect_subset(
                load_dataset=load_dataset,
                args=args,
                subset=subset,
                bad_handle=bad_handle,
            )
            records_by_split["train"].extend(train_rows)
            records_by_split["val"].extend(val_rows)
            subset_stats.append(stats)
            print(f"[build-textatlas] subset={subset} stats={stats}", flush=True)

    train_manifest = args.output_root / "train_manifest.jsonl"
    val_manifest = args.output_root / "val_manifest.jsonl"
    train_image_paths = args.output_root / "train_image_paths.json"
    val_image_paths = args.output_root / "val_image_paths.json"
    counts_path = args.output_root / "counts.json"
    build_config_path = args.output_root / "build_config.json"
    image_paths_sha_path = args.output_root / "image_paths.sha256"

    write_jsonl(train_manifest, records_by_split["train"])
    write_jsonl(val_manifest, records_by_split["val"])
    write_json(train_image_paths, [record["image_path"] for record in records_by_split["train"]])
    write_json(val_image_paths, [record["image_path"] for record in records_by_split["val"]])

    counts = summarize_counts(records_by_split)
    counts["subset_stats"] = subset_stats
    counts["started_at"] = started_at
    counts["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    counts["estimated_total_gb"] = round(counts["total_bytes"] / (1024 ** 3), 3)
    write_json(counts_path, counts)

    build_config = {
        "dataset_name": args.dataset_name,
        "hf_split": args.hf_split,
        "subsets": args.subsets,
        "train_per_subset": args.train_per_subset,
        "val_per_subset": args.val_per_subset,
        "start_offset": args.start_offset,
        "seed": args.seed,
        "selection_strategy": "sequential_streaming",
        "output_root": str(args.output_root),
        "image_root": str(args.image_root),
        "save_format": args.save_format,
        "jpeg_quality": args.jpeg_quality,
        "overwrite": args.overwrite,
    }
    write_json(build_config_path, build_config)

    image_paths_sha_path = write_sha256_manifest(
        args.output_root,
        [train_image_paths, val_image_paths],
        "image_paths.sha256",
    )
    sha_path = write_sha256_manifest(
        args.output_root,
        [
            train_manifest,
            val_manifest,
            train_image_paths,
            val_image_paths,
            counts_path,
            build_config_path,
            bad_rows_path,
            image_paths_sha_path,
        ],
        "manifest.sha256",
    )

    result = {
        "output_root": str(args.output_root),
        "image_root": str(args.image_root),
        "train_image_paths": str(train_image_paths),
        "val_image_paths": str(val_image_paths),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "bad_rows": str(bad_rows_path),
        "sha256": str(sha_path),
        "image_paths_sha256": str(image_paths_sha_path),
        "counts": counts,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
