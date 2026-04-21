#!/usr/bin/env python3
"""Build a frozen chunk order from a TextAtlas train source manifest."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from textatlas_manifest_utils import ALLOWED_SUBSETS, iter_jsonl, now_str, write_json, write_sha256_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build frozen chunk_order.json for stage-1 chunked train.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--strategy",
        choices=("subset-round-robin", "sequential", "shuffle"),
        default="subset-round-robin",
    )
    parser.add_argument("--first-chunk-size", type=int, default=20_000)
    parser.add_argument("--first-chunk-count", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if args.first_chunk_size <= 0 or args.chunk_size <= 0:
        raise SystemExit("chunk sizes must be positive.")
    if args.first_chunk_count < 0:
        raise SystemExit("--first-chunk-count must be non-negative.")


def weighted_round_robin(groups: Dict[str, List[int]]) -> List[int]:
    counts = {subset: len(groups.get(subset, [])) for subset in ALLOWED_SUBSETS}
    positions = {subset: 0 for subset in ALLOWED_SUBSETS}
    total = sum(counts.values())
    order: List[int] = []
    while len(order) < total:
        candidates = [subset for subset in ALLOWED_SUBSETS if positions[subset] < counts[subset]]
        subset = min(candidates, key=lambda key: positions[key] / counts[key] if counts[key] else 1.0)
        order.append(groups[subset][positions[subset]])
        positions[subset] += 1
    return order


def chunk_sizes(total: int, first_size: int, first_count: int, default_size: int) -> List[int]:
    sizes: List[int] = []
    remaining = total
    chunk_id = 0
    while remaining > 0:
        target = first_size if chunk_id < first_count else default_size
        size = min(target, remaining)
        sizes.append(size)
        remaining -= size
        chunk_id += 1
    return sizes


def main() -> None:
    args = parse_args()
    ensure_args(args)
    manifest_root = args.manifest_root.resolve()
    train_manifest = (args.train_manifest or manifest_root / "train_manifest.jsonl").resolve()
    output_root = (args.output_root or manifest_root).resolve()
    if output_root.exists() and (output_root / "chunk_order.json").exists() and not args.overwrite:
        raise SystemExit(f"chunk_order.json exists; pass --overwrite to replace: {output_root / 'chunk_order.json'}")
    output_root.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, List[int]] = defaultdict(list)
    rows_by_index: Dict[int, Dict[str, object]] = {}
    for line_idx, row in enumerate(iter_jsonl(train_manifest)):
        subset = str(row.get("subset"))
        if subset not in ALLOWED_SUBSETS:
            raise ValueError(f"{train_manifest}:{line_idx + 1}: unsupported subset={subset!r}")
        groups[subset].append(line_idx)
        rows_by_index[line_idx] = row

    total = len(rows_by_index)
    if args.strategy == "sequential":
        ordered_indices = list(range(total))
    elif args.strategy == "shuffle":
        ordered_indices = list(range(total))
        random.Random(args.seed).shuffle(ordered_indices)
    else:
        ordered_indices = weighted_round_robin(groups)

    sizes = chunk_sizes(total, args.first_chunk_size, args.first_chunk_count, args.chunk_size)
    chunks = []
    cursor = 0
    for chunk_id, size in enumerate(sizes):
        indices = ordered_indices[cursor : cursor + size]
        cursor += size
        by_subset = Counter(str(rows_by_index[index]["subset"]) for index in indices)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "sample_count": len(indices),
                "manifest_indices": indices,
                "by_subset": dict(sorted(by_subset.items())),
            }
        )

    chunk_order_path = output_root / "chunk_order.json"
    chunk_order = {
        "manifest_root": str(manifest_root),
        "train_manifest": str(train_manifest),
        "total_rows": total,
        "strategy": args.strategy,
        "seed": args.seed,
        "first_chunk_size": args.first_chunk_size,
        "first_chunk_count": args.first_chunk_count,
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "created_at": now_str(),
        "counts_by_subset": {subset: len(groups.get(subset, [])) for subset in ALLOWED_SUBSETS},
        "chunks": chunks,
    }
    write_json(chunk_order_path, chunk_order)
    sha_path = write_sha256_manifest(output_root, [chunk_order_path], "chunk_order.sha256")
    print(
        json.dumps(
            {
                "chunk_order": str(chunk_order_path),
                "chunk_order_sha256": str(sha_path),
                "chunk_count": len(chunks),
                "total_rows": total,
                "first_chunks": chunks[: min(3, len(chunks))],
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
