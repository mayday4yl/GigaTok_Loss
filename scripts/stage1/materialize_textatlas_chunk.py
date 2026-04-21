#!/usr/bin/env python3
"""Materialize one TextAtlas train chunk from train_manifest + chunk_order."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from textatlas_manifest_utils import (
    ALLOWED_SUBSETS,
    image_extension,
    read_json,
    read_jsonl,
    stream_materialize_records,
    write_json,
    write_jsonl,
    write_sha256_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a single stage-1 TextAtlas train chunk.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--chunk-order", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--save-format", choices=("jpeg", "png"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--load-timeout", type=int, default=60)
    parser.add_argument("--load-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_root = args.manifest_root.resolve()
    chunk_order_path = (args.chunk_order or manifest_root / "chunk_order.json").resolve()
    train_manifest_path = (args.train_manifest or manifest_root / "train_manifest.jsonl").resolve()
    output_root = args.output_root.resolve()
    image_root = output_root / "images"
    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    ext = image_extension(args.save_format)

    chunk_order = read_json(chunk_order_path)
    chunks = chunk_order.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError(f"{chunk_order_path} missing list field: chunks")
    chunk = next((item for item in chunks if int(item["chunk_id"]) == args.chunk_id), None)
    if chunk is None:
        raise ValueError(f"chunk_id={args.chunk_id} not found in {chunk_order_path}")
    manifest_indices = [int(index) for index in chunk["manifest_indices"]]
    if not manifest_indices:
        raise ValueError(f"chunk_id={args.chunk_id} has no manifest indices.")

    train_rows = read_jsonl(train_manifest_path)
    dataset_name = str(train_rows[0]["dataset"])
    hf_split = str(train_rows[0]["hf_split"])
    requests_by_subset: Dict[str, Dict[int, List[Dict[str, object]]]] = {
        subset: defaultdict(list) for subset in ALLOWED_SUBSETS
    }
    for order, manifest_idx in enumerate(manifest_indices):
        row = train_rows[manifest_idx]
        subset = str(row["subset"])
        row_idx = int(row["hf_row_idx"])
        filename = f"{order:08d}_{subset}_{row_idx:010d}.{ext}"
        output_path = image_root / filename
        requests_by_subset[subset][row_idx].append(
            {"source_record": row, "output_path": output_path, "order": order}
        )

    materialized = []
    for subset in ALLOWED_SUBSETS:
        if not requests_by_subset[subset]:
            continue
        print(f"[materialize-chunk] chunk={args.chunk_id} subset={subset} rows={sum(len(v) for v in requests_by_subset[subset].values())}", flush=True)
        materialized.extend(
            stream_materialize_records(
                dataset_name=dataset_name,
                hf_split=hf_split,
                subset=subset,
                requests_by_row_idx=requests_by_subset[subset],
                save_format=args.save_format,
                jpeg_quality=args.jpeg_quality,
                overwrite=args.overwrite,
                load_timeout=args.load_timeout,
                load_retries=args.load_retries,
                retry_sleep=args.retry_sleep,
            )
        )
    materialized.sort(key=lambda row: int(row["materialize_order"]))

    chunk_image_paths = output_root / "chunk_image_paths.json"
    chunk_manifest = output_root / "chunk_manifest.jsonl"
    chunk_meta = output_root / "chunk_meta.json"
    write_json(chunk_image_paths, [row["image_path"] for row in materialized])
    write_jsonl(chunk_manifest, materialized)
    by_subset = Counter(str(row["subset"]) for row in materialized)
    total_bytes = sum(int(row.get("file_size", 0)) for row in materialized)
    meta = {
        "chunk_id": args.chunk_id,
        "chunk_order": str(chunk_order_path),
        "train_manifest": str(train_manifest_path),
        "output_root": str(output_root),
        "image_root": str(image_root),
        "sample_count": len(materialized),
        "by_subset": dict(sorted(by_subset.items())),
        "total_bytes": total_bytes,
        "estimated_total_gb": round(total_bytes / (1024 ** 3), 3),
        "save_format": args.save_format,
        "jpeg_quality": args.jpeg_quality,
        "load_timeout": args.load_timeout,
        "load_retries": args.load_retries,
        "retry_sleep": args.retry_sleep,
    }
    write_json(chunk_meta, meta)
    sha_path = write_sha256_manifest(
        output_root,
        [chunk_image_paths, chunk_manifest, chunk_meta],
        "chunk_materialized.sha256",
    )
    print(
        json.dumps(
            {
                "chunk_image_paths": str(chunk_image_paths),
                "chunk_manifest": str(chunk_manifest),
                "chunk_meta": str(chunk_meta),
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
