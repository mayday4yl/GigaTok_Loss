#!/usr/bin/env python3
"""Build the fixed-count 5-subset TextAtlas manifest for the stage-1 validation run."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set

from textatlas_manifest_utils import (
    DEFAULT_DATASET,
    DEFAULT_STAGE1_SUBSETS,
    KNOWN_TEXTATLAS_SUBSETS,
    build_source_record,
    get_exact_count,
    now_str,
    write_json,
    write_jsonl,
    write_sha256_manifest,
)


TEXTSCENESHQ_SUBSET = "TextScenesHQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the fixed TextAtlas source manifest for stage-1.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subsets", nargs="+", choices=KNOWN_TEXTATLAS_SUBSETS, default=list(DEFAULT_STAGE1_SUBSETS))
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--textsceneshq-train-count", type=int, default=40_000)
    parser.add_argument("--same-train-count", type=int, default=50_000)
    parser.add_argument("--val-per-subset", type=int, default=2_000)
    parser.add_argument("--holdout-per-subset", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--count-method",
        choices=("auto", "viewer", "builder", "streaming"),
        default="auto",
    )
    parser.add_argument("--viewer-api-url", default="https://datasets-server.huggingface.co")
    parser.add_argument("--api-timeout", type=int, default=60)
    parser.add_argument(
        "--exact-counts-json",
        type=Path,
        default=None,
        help="Optional JSON mapping subset->row_count, or object with exact_rows_by_subset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if TEXTSCENESHQ_SUBSET not in args.subsets:
        raise SystemExit(f"--subsets must include {TEXTSCENESHQ_SUBSET}.")
    if args.textsceneshq_train_count < 0 or args.same_train_count < 0:
        raise SystemExit("train counts must be non-negative.")
    if args.val_per_subset < 0 or args.holdout_per_subset < 0:
        raise SystemExit("val/holdout counts must be non-negative.")
    if len(set(args.subsets)) != len(args.subsets):
        raise SystemExit("--subsets contains duplicates.")
    if args.output_root.exists() and any(args.output_root.iterdir()) and not args.overwrite:
        raise SystemExit(f"output root is not empty; pass --overwrite to replace: {args.output_root}")


def load_exact_counts(path: Path) -> Dict[str, int]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("exact_rows_by_subset"), dict):
        data = data["exact_rows_by_subset"]
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a subset->count mapping or exact_rows_by_subset object.")
    return {str(key): int(value) for key, value in data.items()}


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
    if sample_count == available_count:
        return [idx for idx in range(count) if idx not in forbidden]
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


def compute_train_counts(
    *,
    subsets: Sequence[str],
    exact_counts: Mapping[str, int],
    val_per_subset: int,
    holdout_per_subset: int,
    textsceneshq_train_count: int,
    same_train_count: int,
) -> Dict[str, Any]:
    fixed_eval_count = val_per_subset + holdout_per_subset
    capacities = {subset: int(exact_counts[subset]) - fixed_eval_count for subset in subsets}
    for subset, capacity in capacities.items():
        if capacity < 0:
            raise ValueError(
                f"subset={subset} has exact_count={exact_counts[subset]}, "
                f"smaller than val+holdout={fixed_eval_count}."
            )

    same_count_subsets = [subset for subset in subsets if subset != TEXTSCENESHQ_SUBSET]
    train_counts = {subset: same_train_count for subset in same_count_subsets}
    train_counts[TEXTSCENESHQ_SUBSET] = textsceneshq_train_count
    for subset, train_count in train_counts.items():
        if train_count > capacities[subset]:
            raise ValueError(
                f"subset={subset} requested train={train_count}, but capacity after "
                f"val/holdout is {capacities[subset]}."
            )
    train_counts = {subset: train_counts.get(subset, 0) for subset in subsets}
    return {
        "train_counts": train_counts,
        "same_count_subsets": same_count_subsets,
        "same_train_count": same_train_count,
        "textsceneshq_train_count": textsceneshq_train_count,
        "train_capacities_after_val_holdout": capacities,
    }


def chain(parts: Sequence[Iterable[Dict[str, object]]]) -> Iterable[Dict[str, object]]:
    for part in parts:
        yield from part


def main() -> None:
    args = parse_args()
    ensure_args(args)
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    started_at = now_str()
    exact_counts_override = load_exact_counts(args.exact_counts_json.resolve()) if args.exact_counts_json else None

    exact_counts: Dict[str, int] = {}
    count_sources: Dict[str, str] = {}
    for subset in args.subsets:
        print(f"[build-fixed-manifest] counting subset={subset}", flush=True)
        count, source = get_exact_count(
            dataset_name=args.dataset_name,
            subset=subset,
            hf_split=args.hf_split,
            count_method=args.count_method,
            viewer_api_url=args.viewer_api_url,
            api_timeout=args.api_timeout,
            exact_counts=exact_counts_override,
        )
        exact_counts[subset] = count
        count_sources[subset] = source
        print(f"[build-fixed-manifest] subset={subset} exact_rows={count} source={source}", flush=True)

    allocation = compute_train_counts(
        subsets=args.subsets,
        exact_counts=exact_counts,
        val_per_subset=args.val_per_subset,
        holdout_per_subset=args.holdout_per_subset,
        textsceneshq_train_count=args.textsceneshq_train_count,
        same_train_count=args.same_train_count,
    )
    train_counts: Dict[str, int] = allocation["train_counts"]
    val_counts = {subset: args.val_per_subset for subset in args.subsets}
    holdout_counts = {subset: args.holdout_per_subset for subset in args.subsets}

    rng = random.Random(args.seed)
    selected: Dict[str, Dict[str, List[int]]] = {}
    for subset in args.subsets:
        val_indices = sample_available_indices(
            count=exact_counts[subset],
            sample_count=val_counts[subset],
            forbidden=set(),
            rng=rng,
        )
        holdout_indices = sample_available_indices(
            count=exact_counts[subset],
            sample_count=holdout_counts[subset],
            forbidden=set(val_indices),
            rng=rng,
        )
        forbidden = set(val_indices) | set(holdout_indices)
        train_indices = sample_available_indices(
            count=exact_counts[subset],
            sample_count=train_counts[subset],
            forbidden=forbidden,
            rng=rng,
        )
        selected[subset] = {
            "train": train_indices,
            "val": val_indices,
            "holdout": holdout_indices,
        }
        print(
            "[build-fixed-manifest] "
            f"subset={subset} train={len(train_indices)} val={len(val_indices)} holdout={len(holdout_indices)}",
            flush=True,
        )

    selection_strategy = "fixed_counts_textsceneshq_40k_other_50k_source_indices"
    train_manifest = args.output_root / "train_manifest.jsonl"
    val_manifest = args.output_root / "val_manifest.jsonl"
    holdout_manifest = args.output_root / "holdout_manifest.jsonl"
    exact_count_path = args.output_root / "exact_count.json"
    build_config_path = args.output_root / "build_config.json"

    write_jsonl(
        train_manifest,
        chain(
            [
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
        ),
    )
    write_jsonl(
        val_manifest,
        chain(
            [
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
        ),
    )
    write_jsonl(
        holdout_manifest,
        chain(
            [
                source_rows(
                    dataset_name=args.dataset_name,
                    subset=subset,
                    split="holdout",
                    hf_split=args.hf_split,
                    row_indices=selected[subset]["holdout"],
                    seed=args.seed,
                    selection_strategy=selection_strategy,
                )
                for subset in args.subsets
            ]
        ),
    )

    exact_count = {
        "dataset_name": args.dataset_name,
        "hf_split": args.hf_split,
        "subsets": args.subsets,
        "exact_rows_by_subset": exact_counts,
        "count_sources": count_sources,
        "textsceneshq_train_count": args.textsceneshq_train_count,
        "same_train_count": args.same_train_count,
        "train_total": sum(train_counts.values()),
        "val_total": sum(val_counts.values()),
        "holdout_total": sum(holdout_counts.values()),
        "train_counts_by_subset": train_counts,
        "val_counts_by_subset": val_counts,
        "holdout_counts_by_subset": holdout_counts,
        "allocation": allocation,
        "selection_strategy": selection_strategy,
        "textatlas_eval_usage": "final_independent_benchmark_only",
        "seed": args.seed,
        "started_at": started_at,
        "finished_at": now_str(),
    }
    write_json(exact_count_path, exact_count)

    build_config = {
        "dataset_name": args.dataset_name,
        "hf_split": args.hf_split,
        "subsets": args.subsets,
        "textsceneshq_train_count": args.textsceneshq_train_count,
        "same_train_count": args.same_train_count,
        "val_per_subset": args.val_per_subset,
        "holdout_per_subset": args.holdout_per_subset,
        "seed": args.seed,
        "count_method": args.count_method,
        "viewer_api_url": args.viewer_api_url,
        "api_timeout": args.api_timeout,
        "exact_counts_json": str(args.exact_counts_json.resolve()) if args.exact_counts_json else None,
        "output_root": str(args.output_root),
        "source_manifest_only": True,
        "image_materialization": "fixed_local_train_val_from_manifest",
    }
    write_json(build_config_path, build_config)
    sha_path = write_sha256_manifest(
        args.output_root,
        [exact_count_path, train_manifest, val_manifest, holdout_manifest, build_config_path],
        "manifest.sha256",
    )

    result = {
        "output_root": str(args.output_root),
        "exact_count": str(exact_count_path),
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "holdout_manifest": str(holdout_manifest),
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
