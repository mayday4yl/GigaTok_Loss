#!/usr/bin/env python3
"""Build source-only TextAtlas manifests for fixed-val and 3M chunked train."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set

from textatlas_manifest_utils import (
    ALLOWED_SUBSETS,
    DEFAULT_DATASET,
    allocate_proportional,
    build_source_record,
    get_exact_count,
    now_str,
    write_json,
    write_jsonl,
    write_sha256_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build exact-count TextAtlas source manifests for stage-1 full train.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=ALLOWED_SUBSETS, default=list(ALLOWED_SUBSETS))
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--train-total", type=int, default=3_000_000)
    parser.add_argument("--val-total", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--count-method",
        choices=("auto", "builder", "streaming"),
        default="auto",
        help="Use dataset metadata when available; streaming is exact but slower.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if args.train_total <= 0:
        raise SystemExit("--train-total must be positive.")
    if args.val_total < 0:
        raise SystemExit("--val-total must be non-negative.")
    if len(set(args.subsets)) != len(args.subsets):
        raise SystemExit("--subsets contains duplicates.")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"output root is not empty; pass --overwrite to replace: {args.output_root}")


def sample_available_indices(
    *,
    count: int,
    sample_count: int,
    forbidden: Set[int],
    rng: random.Random,
) -> List[int]:
    if sample_count == 0:
        return []
    available_count = count - len(forbidden)
    if sample_count > available_count:
        raise ValueError(f"Cannot sample {sample_count} rows from {available_count} available rows.")
    if not forbidden:
        return sorted(rng.sample(range(count), sample_count))

    sampled_positions = sorted(rng.sample(range(available_count), sample_count))
    selected: List[int] = []
    sampled_pos_idx = 0
    available_pos = 0
    for row_idx in range(count):
        if row_idx in forbidden:
            continue
        if available_pos == sampled_positions[sampled_pos_idx]:
            selected.append(row_idx)
            sampled_pos_idx += 1
            if sampled_pos_idx == len(sampled_positions):
                break
        available_pos += 1
    return selected


def source_rows(
    *,
    dataset_name: str,
    subset: str,
    split: str,
    hf_split: str,
    row_indices: Iterable[int],
    seed: int,
    selection_strategy: str,
) -> Iterable[Dict[str, object]]:
    for row_idx in row_indices:
        yield build_source_record(
            dataset_name=dataset_name,
            subset=subset,
            split=split,
            hf_split=hf_split,
            hf_row_idx=row_idx,
            selection_seed=seed,
            selection_strategy=selection_strategy,
        )


def main() -> None:
    args = parse_args()
    ensure_args(args)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started_at = now_str()

    exact_counts: Dict[str, int] = {}
    count_sources: Dict[str, str] = {}
    for subset in args.subsets:
        print(f"[build-full-manifest] counting subset={subset}", flush=True)
        count, source = get_exact_count(
            dataset_name=args.dataset_name,
            subset=subset,
            hf_split=args.hf_split,
            count_method=args.count_method,
        )
        exact_counts[subset] = count
        count_sources[subset] = source
        print(f"[build-full-manifest] subset={subset} exact_rows={count} source={source}", flush=True)

    val_counts = allocate_proportional(
        total=args.val_total,
        weights=exact_counts,
        capacities=exact_counts,
        keys=args.subsets,
    )
    train_capacities = {subset: exact_counts[subset] - val_counts[subset] for subset in args.subsets}
    train_counts = allocate_proportional(
        total=args.train_total,
        weights=train_capacities,
        capacities=train_capacities,
        keys=args.subsets,
    )

    rng = random.Random(args.seed)
    selected: Dict[str, Dict[str, List[int]]] = {}
    for subset in args.subsets:
        val_indices = sample_available_indices(
            count=exact_counts[subset],
            sample_count=val_counts[subset],
            forbidden=set(),
            rng=rng,
        )
        train_indices = sample_available_indices(
            count=exact_counts[subset],
            sample_count=train_counts[subset],
            forbidden=set(val_indices),
            rng=rng,
        )
        selected[subset] = {"train": train_indices, "val": val_indices}
        print(
            "[build-full-manifest] "
            f"subset={subset} train={len(train_indices)} val={len(val_indices)}",
            flush=True,
        )

    train_manifest = args.output_root / "train_manifest.jsonl"
    val_manifest = args.output_root / "val_manifest.jsonl"
    exact_count_path = args.output_root / "exact_count.json"
    build_config_path = args.output_root / "build_config.json"

    selection_strategy = "proportional_no_replacement_source_indices"
    train_iterables = [
        source_rows(
            dataset_name=args.dataset_name,
            subset=subset,
            split="train",
            hf_split=args.hf_split,
            row_indices=selected[subset]["train"],
            seed=args.seed,
            selection_strategy=selection_strategy,
        )
        for subset in args.subsets
    ]
    val_iterables = [
        source_rows(
            dataset_name=args.dataset_name,
            subset=subset,
            split="val",
            hf_split=args.hf_split,
            row_indices=selected[subset]["val"],
            seed=args.seed,
            selection_strategy=selection_strategy,
        )
        for subset in args.subsets
    ]

    def chain(parts: Sequence[Iterable[Dict[str, object]]]) -> Iterable[Dict[str, object]]:
        for part in parts:
            yield from part

    write_jsonl(train_manifest, chain(train_iterables))
    write_jsonl(val_manifest, chain(val_iterables))

    exact_count = {
        "dataset_name": args.dataset_name,
        "hf_split": args.hf_split,
        "subsets": args.subsets,
        "exact_rows_by_subset": exact_counts,
        "count_sources": count_sources,
        "val_total": args.val_total,
        "val_counts_by_subset": val_counts,
        "train_total": args.train_total,
        "train_counts_by_subset": train_counts,
        "train_capacity_after_val_by_subset": train_capacities,
        "selection_strategy": selection_strategy,
        "seed": args.seed,
        "started_at": started_at,
        "finished_at": now_str(),
    }
    write_json(exact_count_path, exact_count)

    build_config = {
        "dataset_name": args.dataset_name,
        "hf_split": args.hf_split,
        "subsets": args.subsets,
        "train_total": args.train_total,
        "val_total": args.val_total,
        "seed": args.seed,
        "count_method": args.count_method,
        "output_root": str(args.output_root),
        "source_manifest_only": True,
        "image_materialization": "fixed_val_once_train_chunked",
    }
    write_json(build_config_path, build_config)
    sha_path = write_sha256_manifest(
        args.output_root,
        [exact_count_path, train_manifest, val_manifest, build_config_path],
        "manifest.sha256",
    )

    result = {
        "output_root": str(args.output_root),
        "exact_count": str(exact_count_path),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "manifest_sha256": str(sha_path),
        "summary": exact_count,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
