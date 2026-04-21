#!/usr/bin/env python3
"""Materialize fixed TextAtlas validation images from a source-only val manifest."""

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
    iter_jsonl,
    now_str,
    stream_materialize_records,
    write_json,
    write_jsonl,
    write_sha256_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize fixed val images from TextAtlas val manifest.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
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
    val_manifest = (args.val_manifest or manifest_root / "val_manifest.jsonl").resolve()
    output_root = (args.output_root or manifest_root).resolve()
    image_root = args.image_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    ext = image_extension(args.save_format)

    rows = list(iter_jsonl(val_manifest))
    if not rows:
        raise SystemExit(f"No rows found in {val_manifest}")
    dataset_name = str(rows[0]["dataset"])
    hf_split = str(rows[0]["hf_split"])
    requests_by_subset: Dict[str, Dict[int, List[Dict[str, object]]]] = {
        subset: defaultdict(list) for subset in ALLOWED_SUBSETS
    }
    for order, row in enumerate(rows):
        subset = str(row["subset"])
        if subset not in ALLOWED_SUBSETS:
            raise ValueError(f"Unsupported subset={subset!r}")
        row_idx = int(row["hf_row_idx"])
        output_path = image_root / subset / f"{row_idx:010d}.{ext}"
        requests_by_subset[subset][row_idx].append(
            {"source_record": row, "output_path": output_path, "order": order}
        )

    materialized = []
    for subset in ALLOWED_SUBSETS:
        if not requests_by_subset[subset]:
            continue
        print(f"[materialize-val] subset={subset} rows={sum(len(v) for v in requests_by_subset[subset].values())}", flush=True)
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

    val_image_paths = output_root / "val_image_paths.json"
    materialized_manifest = output_root / "val_materialized_manifest.jsonl"
    meta_path = output_root / "val_materialize_meta.json"
    write_json(val_image_paths, [row["image_path"] for row in materialized])
    write_jsonl(materialized_manifest, materialized)
    by_subset = Counter(str(row["subset"]) for row in materialized)
    total_bytes = sum(int(row.get("file_size", 0)) for row in materialized)
    meta = {
        "source_val_manifest": str(val_manifest),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "save_format": args.save_format,
        "jpeg_quality": args.jpeg_quality,
        "load_timeout": args.load_timeout,
        "load_retries": args.load_retries,
        "retry_sleep": args.retry_sleep,
        "total": len(materialized),
        "by_subset": dict(sorted(by_subset.items())),
        "total_bytes": total_bytes,
        "estimated_total_gb": round(total_bytes / (1024 ** 3), 3),
        "finished_at": now_str(),
    }
    write_json(meta_path, meta)
    sha_path = write_sha256_manifest(
        output_root,
        [val_image_paths, materialized_manifest, meta_path],
        "val_materialized.sha256",
    )
    print(
        json.dumps(
            {
                "val_image_paths": str(val_image_paths),
                "val_materialized_manifest": str(materialized_manifest),
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
