#!/usr/bin/env python3
"""Materialize fixed local TextAtlas train/val images from source-only manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from textatlas_manifest_utils import (
    DEFAULT_STAGE1_SUBSETS,
    KNOWN_TEXTATLAS_SUBSETS,
    image_extension,
    iter_jsonl,
    load_streaming_dataset_with_retries,
    materialize_source_record,
    now_str,
    parse_pad_color,
    write_json,
    write_jsonl,
    write_sha256_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize fixed local TextAtlas split images.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "holdout"), default=["train", "val"])
    parser.add_argument("--save-format", choices=("jpeg", "png"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preprocess", choices=("none", "resize-pad"), default="resize-pad")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--load-timeout", type=int, default=60)
    parser.add_argument("--load-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=int, default=5)
    parser.add_argument("--max-bad-rows", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if len(set(args.splits)) != len(args.splits):
        raise SystemExit("--splits contains duplicates.")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive.")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be in [1, 100].")
    if args.max_bad_rows < 0:
        raise SystemExit("--max-bad-rows must be non-negative.")


def split_order_key(split: str) -> int:
    order = {"train": 0, "val": 1, "holdout": 2}
    return order[split]


def subset_order(subsets: Iterable[str]) -> List[str]:
    present = set(subsets)
    preferred = list(DEFAULT_STAGE1_SUBSETS) + [
        subset for subset in KNOWN_TEXTATLAS_SUBSETS if subset not in DEFAULT_STAGE1_SUBSETS
    ]
    return [subset for subset in preferred if subset in present]


def load_requested_rows(manifest_root: Path, splits: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in splits:
        manifest = manifest_root / f"{split}_manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest for split={split}: {manifest}")
        rows = list(iter_jsonl(manifest))
        if not rows:
            raise ValueError(f"manifest has no rows: {manifest}")
        rows_by_split[split] = rows
    return rows_by_split


def validate_dataset_identity(rows_by_split: Mapping[str, List[Dict[str, Any]]]) -> tuple[str, str]:
    dataset_names = {str(row["dataset"]) for rows in rows_by_split.values() for row in rows}
    hf_splits = {str(row["hf_split"]) for rows in rows_by_split.values() for row in rows}
    if len(dataset_names) != 1:
        raise ValueError(f"materialize requires one dataset name, got {sorted(dataset_names)}")
    if len(hf_splits) != 1:
        raise ValueError(f"materialize requires one HF split, got {sorted(hf_splits)}")
    return next(iter(dataset_names)), next(iter(hf_splits))


def main() -> None:
    args = parse_args()
    ensure_args(args)
    manifest_root = args.manifest_root.resolve()
    output_root = (args.output_root or manifest_root).resolve()
    image_root = args.image_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    ext = image_extension(args.save_format)
    pad_color = parse_pad_color(args.pad_color)
    started_at = now_str()

    rows_by_split = load_requested_rows(manifest_root, args.splits)
    dataset_name, hf_split = validate_dataset_identity(rows_by_split)

    requests_by_subset: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    source_counts_by_split = {split: len(rows) for split, rows in rows_by_split.items()}
    for split, rows in rows_by_split.items():
        for order, row in enumerate(rows):
            subset = str(row["subset"])
            if subset not in KNOWN_TEXTATLAS_SUBSETS:
                raise ValueError(f"Unsupported subset={subset!r}")
            row_idx = int(row["hf_row_idx"])
            output_path = image_root / split / subset / f"{row_idx:010d}.{ext}"
            requests_by_subset[subset][row_idx].append(
                {
                    "source_record": row,
                    "output_path": output_path,
                    "split": split,
                    "order": order,
                }
            )

    materialized_by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in args.splits}
    bad_rows: List[Dict[str, Any]] = []
    for subset in subset_order(requests_by_subset):
        total_subset_rows = sum(len(value) for value in requests_by_subset[subset].values())
        print(f"[materialize-local] subset={subset} rows={total_subset_rows}", flush=True)
        pending = {int(key): list(value) for key, value in requests_by_subset[subset].items()}
        dataset = load_streaming_dataset_with_retries(
            dataset_name=dataset_name,
            subset=subset,
            hf_split=hf_split,
            load_timeout=args.load_timeout,
            load_retries=args.load_retries,
            retry_sleep=args.retry_sleep,
        )
        for row_idx, row in enumerate(dataset):
            if row_idx not in pending:
                continue
            for request in pending.pop(row_idx):
                source_record = request["source_record"]
                split = str(request["split"])
                try:
                    record = materialize_source_record(
                        source_record=source_record,
                        row=row,
                        output_path=request["output_path"],
                        save_format=args.save_format,
                        jpeg_quality=args.jpeg_quality,
                        overwrite=args.overwrite,
                        preprocess=args.preprocess,
                        image_size=args.image_size,
                        pad_color=pad_color,
                    )
                    record["materialize_order"] = int(request["order"])
                    materialized_by_split[split].append(record)
                except Exception as exc:  # noqa: BLE001 - data failures should be auditable.
                    bad_rows.append(
                        {
                            "dataset": dataset_name,
                            "subset": subset,
                            "split": split,
                            "hf_split": hf_split,
                            "hf_row_idx": row_idx,
                            "error": repr(exc),
                        }
                    )
                    if len(bad_rows) > args.max_bad_rows:
                        raise RuntimeError(
                            f"bad row budget exceeded: {len(bad_rows)} > {args.max_bad_rows}. "
                            f"Latest subset={subset}, hf_row_idx={row_idx}"
                        ) from exc
            if not pending:
                break
        if pending:
            preview = sorted(pending)[:10]
            raise RuntimeError(f"{subset}: failed to materialize row indices {preview} (remaining={len(pending)}).")

    written_paths: List[Path] = []
    split_meta: Dict[str, Dict[str, Any]] = {}
    for split in sorted(args.splits, key=split_order_key):
        materialized = sorted(materialized_by_split[split], key=lambda row: int(row["materialize_order"]))
        image_paths_path = output_root / f"{split}_image_paths.json"
        materialized_manifest = output_root / f"{split}_materialized_manifest.jsonl"
        write_json(image_paths_path, [row["image_path"] for row in materialized])
        write_jsonl(materialized_manifest, materialized)
        written_paths.extend([image_paths_path, materialized_manifest])
        by_subset = Counter(str(row["subset"]) for row in materialized)
        total_bytes = sum(int(row.get("file_size", 0)) for row in materialized)
        split_meta[split] = {
            "source_rows": source_counts_by_split[split],
            "materialized_rows": len(materialized),
            "by_subset": dict(sorted(by_subset.items())),
            "total_bytes": total_bytes,
            "estimated_total_gb": round(total_bytes / (1024 ** 3), 3),
        }

    bad_rows_path = output_root / "bad_rows.jsonl"
    meta_path = output_root / "materialize_meta.json"
    write_jsonl(bad_rows_path, bad_rows)
    meta = {
        "manifest_root": str(manifest_root),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "splits": args.splits,
        "dataset_name": dataset_name,
        "hf_split": hf_split,
        "save_format": args.save_format,
        "jpeg_quality": args.jpeg_quality,
        "preprocess": args.preprocess,
        "image_size": args.image_size,
        "pad_color": list(pad_color),
        "load_timeout": args.load_timeout,
        "load_retries": args.load_retries,
        "retry_sleep": args.retry_sleep,
        "bad_rows": len(bad_rows),
        "max_bad_rows": args.max_bad_rows,
        "split_meta": split_meta,
        "started_at": started_at,
        "finished_at": now_str(),
    }
    write_json(meta_path, meta)
    written_paths.extend([bad_rows_path, meta_path])
    sha_path = write_sha256_manifest(output_root, written_paths, "materialized.sha256")

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "image_root": str(image_root),
                "sha256": str(sha_path),
                "meta": meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
