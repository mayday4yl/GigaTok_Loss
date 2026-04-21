#!/usr/bin/env python3
"""Materialize the stage-1 TextAtlas set from sequential converted parquet files.

This is the fallback for ModelArts environments where Hugging Face datasets
streaming is too slow for random source-row sampling. It downloads contiguous
converted parquet shards, assigns the first rows of each subset to fixed
val/holdout/train splits, and writes the same image_paths JSON files used by
the existing training entrypoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import quote

from textatlas_manifest_utils import (
    DEFAULT_DATASET,
    DEFAULT_STAGE1_SUBSETS,
    KNOWN_TEXTATLAS_SUBSETS,
    build_source_record,
    image_extension,
    materialize_source_record,
    now_str,
    parse_pad_color,
    write_json,
    write_jsonl,
    write_sha256_manifest,
)


TEXTSCENESHQ_SUBSET = "TextScenesHQ"
CONVERT_PARQUET_REVISION = "refs/convert/parquet"
DEFAULT_EXACT_COUNTS = {
    "CleanTextSynth": 1_907_721,
    "StyledTextSynth": 425_826,
    "TextVisionBlend": 546_829,
    "TextScenesHQ": 48_935,
    "LongWordsSubset-A": 259_897,
}
DEFAULT_SHARD_COUNTS = {
    "CleanTextSynth": 67,
    "StyledTextSynth": 290,
    "TextVisionBlend": 22,
    "TextScenesHQ": 5,
    "LongWordsSubset-A": 11,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize sequential TextAtlas parquet rows.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--subsets", nargs="+", choices=KNOWN_TEXTATLAS_SUBSETS, default=list(DEFAULT_STAGE1_SUBSETS))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--parquet-cache-root", type=Path, required=True)
    parser.add_argument("--hf-endpoint", default=None, help="Defaults to HF_ENDPOINT or https://huggingface.co.")
    parser.add_argument("--textsceneshq-train-count", type=int, default=40_000)
    parser.add_argument("--same-train-count", type=int, default=50_000)
    parser.add_argument("--val-per-subset", type=int, default=2_000)
    parser.add_argument("--holdout-per-subset", type=int, default=500)
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "holdout"), default=["train", "val"])
    parser.add_argument("--save-format", choices=("jpeg", "png"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preprocess", choices=("none", "resize-pad"), default="resize-pad")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--curl-retries", type=int, default=8)
    parser.add_argument("--curl-connect-timeout", type=int, default=30)
    parser.add_argument("--curl-max-time", type=int, default=0, help="0 means no curl --max-time limit.")
    parser.add_argument("--curl-insecure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-bad-rows", type=int, default=0)
    parser.add_argument("--max-shards-per-subset", type=int, default=0, help="Debug limit. 0 means no limit.")
    parser.add_argument("--delete-parquet-after-materialize", action="store_true")
    parser.add_argument("--overwrite-manifest", action="store_true")
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if len(set(args.subsets)) != len(args.subsets):
        raise SystemExit("--subsets contains duplicates.")
    if len(set(args.splits)) != len(args.splits):
        raise SystemExit("--splits contains duplicates.")
    if TEXTSCENESHQ_SUBSET not in args.subsets:
        raise SystemExit(f"--subsets must include {TEXTSCENESHQ_SUBSET}.")
    if args.textsceneshq_train_count < 0 or args.same_train_count < 0:
        raise SystemExit("train counts must be non-negative.")
    if args.val_per_subset < 0 or args.holdout_per_subset < 0:
        raise SystemExit("val/holdout counts must be non-negative.")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive.")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be in [1, 100].")
    if args.curl_retries <= 0 or args.curl_connect_timeout <= 0 or args.curl_max_time < 0:
        raise SystemExit("curl retry/timeout values must be valid.")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.max_bad_rows < 0 or args.max_shards_per_subset < 0:
        raise SystemExit("max-bad-rows and max-shards-per-subset must be non-negative.")


def import_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyarrow.") from exc
    return pq


def split_order_key(split: str) -> int:
    order = {"train": 0, "val": 1, "holdout": 2}
    return order[split]


def train_count_for_subset(subset: str, textsceneshq_train_count: int, same_train_count: int) -> int:
    return textsceneshq_train_count if subset == TEXTSCENESHQ_SUBSET else same_train_count


def split_ranges_for_subset(
    *,
    subset: str,
    textsceneshq_train_count: int,
    same_train_count: int,
    val_per_subset: int,
    holdout_per_subset: int,
) -> Dict[str, range]:
    val_start = 0
    val_end = val_per_subset
    holdout_start = val_end
    holdout_end = holdout_start + holdout_per_subset
    train_start = holdout_end
    train_end = train_start + train_count_for_subset(subset, textsceneshq_train_count, same_train_count)
    return {
        "val": range(val_start, val_end),
        "holdout": range(holdout_start, holdout_end),
        "train": range(train_start, train_end),
    }


def build_source_manifests(
    *,
    dataset_name: str,
    hf_split: str,
    subsets: Sequence[str],
    textsceneshq_train_count: int,
    same_train_count: int,
    val_per_subset: int,
    holdout_per_subset: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "holdout": []}
    for subset in subsets:
        ranges = split_ranges_for_subset(
            subset=subset,
            textsceneshq_train_count=textsceneshq_train_count,
            same_train_count=same_train_count,
            val_per_subset=val_per_subset,
            holdout_per_subset=holdout_per_subset,
        )
        for split in ("train", "val", "holdout"):
            for row_idx in ranges[split]:
                rows_by_split[split].append(
                    build_source_record(
                        dataset_name=dataset_name,
                        subset=subset,
                        split=split,
                        hf_split=hf_split,
                        hf_row_idx=int(row_idx),
                        selection_seed=0,
                        selection_strategy="sequential_front_rows_from_converted_parquet",
                    )
                )
    return rows_by_split


def write_source_outputs(
    *,
    output_root: Path,
    rows_by_split: Mapping[str, List[Dict[str, Any]]],
    dataset_name: str,
    hf_split: str,
    subsets: Sequence[str],
    textsceneshq_train_count: int,
    same_train_count: int,
    val_per_subset: int,
    holdout_per_subset: int,
    started_at: str,
    overwrite_manifest: bool,
) -> Dict[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "train": output_root / "train_manifest.jsonl",
        "val": output_root / "val_manifest.jsonl",
        "holdout": output_root / "holdout_manifest.jsonl",
        "exact_count": output_root / "exact_count.json",
        "build_config": output_root / "build_config.json",
    }
    existing = [path for path in source_paths.values() if path.exists()]
    if existing and not overwrite_manifest:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"source manifest outputs already exist ({names}); pass --overwrite-manifest.")

    for split in ("train", "val", "holdout"):
        write_jsonl(source_paths[split], rows_by_split[split])

    train_counts = Counter(str(row["subset"]) for row in rows_by_split["train"])
    val_counts = Counter(str(row["subset"]) for row in rows_by_split["val"])
    holdout_counts = Counter(str(row["subset"]) for row in rows_by_split["holdout"])
    exact_count = {
        "dataset_name": dataset_name,
        "hf_split": hf_split,
        "subsets": list(subsets),
        "exact_rows_by_subset": {subset: DEFAULT_EXACT_COUNTS[subset] for subset in subsets},
        "shard_counts_by_subset": {subset: DEFAULT_SHARD_COUNTS[subset] for subset in subsets},
        "textsceneshq_train_count": textsceneshq_train_count,
        "same_train_count": same_train_count,
        "train_total": len(rows_by_split["train"]),
        "val_total": len(rows_by_split["val"]),
        "holdout_total": len(rows_by_split["holdout"]),
        "train_counts_by_subset": dict(sorted(train_counts.items())),
        "val_counts_by_subset": dict(sorted(val_counts.items())),
        "holdout_counts_by_subset": dict(sorted(holdout_counts.items())),
        "selection_strategy": "sequential_front_rows_from_converted_parquet",
        "textatlas_eval_usage": "final_independent_benchmark_only",
        "seed": 0,
        "started_at": started_at,
        "finished_at": now_str(),
    }
    build_config = {
        "dataset_name": dataset_name,
        "hf_split": hf_split,
        "subsets": list(subsets),
        "textsceneshq_train_count": textsceneshq_train_count,
        "same_train_count": same_train_count,
        "val_per_subset": val_per_subset,
        "holdout_per_subset": holdout_per_subset,
        "source_manifest_only": False,
        "image_materialization": "sequential_converted_parquet",
        "parquet_revision": CONVERT_PARQUET_REVISION,
        "output_root": str(output_root),
    }
    write_json(source_paths["exact_count"], exact_count)
    write_json(source_paths["build_config"], build_config)
    manifest_sha = write_sha256_manifest(
        output_root,
        [
            source_paths["exact_count"],
            source_paths["train"],
            source_paths["val"],
            source_paths["holdout"],
            source_paths["build_config"],
        ],
        "manifest.sha256",
    )
    source_paths["manifest_sha256"] = manifest_sha
    return source_paths


def hf_parquet_url(*, endpoint: str, dataset_name: str, subset: str, hf_split: str, shard_idx: int) -> str:
    dataset_path = "/".join(quote(part, safe="") for part in dataset_name.split("/"))
    subset_path = quote(subset, safe="")
    return (
        f"{endpoint.rstrip('/')}/datasets/{dataset_path}/resolve/"
        f"refs%2Fconvert%2Fparquet/{subset_path}/{quote(hf_split, safe='')}/{shard_idx:04d}.parquet"
    )


def parquet_local_path(
    *,
    cache_root: Path,
    dataset_name: str,
    subset: str,
    hf_split: str,
    shard_idx: int,
) -> Path:
    safe_dataset = dataset_name.replace("/", "__")
    return cache_root / safe_dataset / "convert_parquet" / subset / hf_split / f"{shard_idx:04d}.parquet"


def download_parquet_with_curl(
    *,
    url: str,
    output_path: Path,
    retries: int,
    connect_timeout: int,
    max_time: int,
    insecure: bool,
) -> Path:
    marker_path = output_path.with_suffix(output_path.suffix + ".complete")
    if output_path.exists() and marker_path.exists():
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        str(retries),
        "--connect-timeout",
        str(connect_timeout),
        "-C",
        "-",
        "-o",
        str(output_path),
        url,
    ]
    if insecure:
        cmd.insert(1, "-k")
    if max_time:
        cmd.extend(["--max-time", str(max_time)])
    print(f"[sequential-parquet] download {url}", flush=True)
    subprocess.run(cmd, check=True)
    marker_path.write_text(json.dumps({"url": url, "size": output_path.stat().st_size}, ensure_ascii=False) + "\n")
    return output_path


def needed_splits_for_row(row_idx: int, rows_by_split_by_subset: Mapping[str, Mapping[int, Dict[str, Any]]]) -> List[str]:
    return [split for split, rows in rows_by_split_by_subset.items() if row_idx in rows]


def materialize_parquet_rows(
    *,
    parquet_path: Path,
    shard_start_idx: int,
    rows_by_split_by_subset: Mapping[str, Mapping[int, Dict[str, Any]]],
    split_orders: Mapping[str, Mapping[int, int]],
    image_root: Path,
    ext: str,
    save_format: str,
    jpeg_quality: int,
    overwrite_images: bool,
    preprocess: str,
    image_size: int,
    pad_color: tuple[int, int, int],
    batch_size: int,
) -> Dict[str, List[Dict[str, Any]]]:
    pq = import_pyarrow()
    materialized: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    target_global_indices = {
        int(row_idx)
        for rows_by_idx in rows_by_split_by_subset.values()
        for row_idx in rows_by_idx
    }
    if not target_global_indices:
        return materialized

    parquet_file = pq.ParquetFile(parquet_path)
    local_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_len = batch.num_rows
        batch_start = local_offset
        batch_end = local_offset + batch_len
        needed_local = sorted(
            idx - shard_start_idx
            for idx in target_global_indices
            if batch_start <= idx - shard_start_idx < batch_end
        )
        if needed_local:
            rows = batch.to_pylist()
            for local_idx in needed_local:
                global_idx = shard_start_idx + local_idx
                row = rows[local_idx - batch_start]
                for split in needed_splits_for_row(global_idx, rows_by_split_by_subset):
                    source_record = rows_by_split_by_subset[split][global_idx]
                    output_path = image_root / split / str(source_record["subset"]) / f"{global_idx:010d}.{ext}"
                    record = materialize_source_record(
                        source_record=source_record,
                        row=row,
                        output_path=output_path,
                        save_format=save_format,
                        jpeg_quality=jpeg_quality,
                        overwrite=overwrite_images,
                        preprocess=preprocess,
                        image_size=image_size,
                        pad_color=pad_color,
                    )
                    record["materialize_order"] = int(split_orders[split][global_idx])
                    record["parquet_path"] = str(parquet_path)
                    record["parquet_global_start_idx"] = int(shard_start_idx)
                    record["parquet_local_row_idx"] = int(local_idx)
                    materialized[split].append(record)
        local_offset = batch_end
        if shard_start_idx + local_offset > max(target_global_indices):
            break
    return materialized


def rows_for_subset(rows_by_split: Mapping[str, List[Dict[str, Any]]], subset: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    output: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for split, rows in rows_by_split.items():
        subset_rows = {
            int(row["hf_row_idx"]): row
            for row in rows
            if str(row["subset"]) == subset
        }
        if subset_rows:
            output[split] = subset_rows
    return output


def split_orders_for_subset(rows_by_split: Mapping[str, List[Dict[str, Any]]], subset: str) -> Dict[str, Dict[int, int]]:
    output: Dict[str, Dict[int, int]] = {}
    for split, rows in rows_by_split.items():
        output[split] = {
            int(row["hf_row_idx"]): order
            for order, row in enumerate(rows)
            if str(row["subset"]) == subset
        }
    return output


def main() -> None:
    args = parse_args()
    ensure_args(args)
    started_at = now_str()
    output_root = args.output_root.resolve()
    image_root = args.image_root.resolve()
    parquet_cache_root = args.parquet_cache_root.resolve()
    endpoint = (args.hf_endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    pad_color = parse_pad_color(args.pad_color)
    ext = image_extension(args.save_format)

    rows_by_split = build_source_manifests(
        dataset_name=args.dataset_name,
        hf_split=args.hf_split,
        subsets=args.subsets,
        textsceneshq_train_count=args.textsceneshq_train_count,
        same_train_count=args.same_train_count,
        val_per_subset=args.val_per_subset,
        holdout_per_subset=args.holdout_per_subset,
    )
    source_paths = write_source_outputs(
        output_root=output_root,
        rows_by_split=rows_by_split,
        dataset_name=args.dataset_name,
        hf_split=args.hf_split,
        subsets=args.subsets,
        textsceneshq_train_count=args.textsceneshq_train_count,
        same_train_count=args.same_train_count,
        val_per_subset=args.val_per_subset,
        holdout_per_subset=args.holdout_per_subset,
        started_at=started_at,
        overwrite_manifest=args.overwrite_manifest,
    )

    image_root.mkdir(parents=True, exist_ok=True)
    parquet_cache_root.mkdir(parents=True, exist_ok=True)
    requested_splits = set(args.splits)
    materialized_by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in args.splits}
    bad_rows: List[Dict[str, Any]] = []
    shard_meta: Dict[str, List[Dict[str, Any]]] = {}

    for subset in args.subsets:
        subset_rows_all = rows_for_subset(rows_by_split, subset)
        subset_rows = {split: rows for split, rows in subset_rows_all.items() if split in requested_splits}
        if not subset_rows:
            continue
        split_orders = split_orders_for_subset(rows_by_split, subset)
        max_needed_idx = max(idx for rows in subset_rows.values() for idx in rows)
        shard_count = DEFAULT_SHARD_COUNTS[subset]
        if args.max_shards_per_subset:
            shard_count = min(shard_count, args.max_shards_per_subset)
        shard_start = 0
        processed: List[Dict[str, Any]] = []
        print(
            f"[sequential-parquet] subset={subset} max_needed_row={max_needed_idx} shards={shard_count}",
            flush=True,
        )
        for shard_idx in range(shard_count):
            url = hf_parquet_url(
                endpoint=endpoint,
                dataset_name=args.dataset_name,
                subset=subset,
                hf_split=args.hf_split,
                shard_idx=shard_idx,
            )
            parquet_path = parquet_local_path(
                cache_root=parquet_cache_root,
                dataset_name=args.dataset_name,
                subset=subset,
                hf_split=args.hf_split,
                shard_idx=shard_idx,
            )
            download_parquet_with_curl(
                url=url,
                output_path=parquet_path,
                retries=args.curl_retries,
                connect_timeout=args.curl_connect_timeout,
                max_time=args.curl_max_time,
                insecure=args.curl_insecure,
            )
            pq = import_pyarrow()
            parquet_file = pq.ParquetFile(parquet_path)
            row_count = int(parquet_file.metadata.num_rows)
            shard_end = shard_start + row_count
            needed_count = sum(
                1
                for rows in subset_rows.values()
                for row_idx in rows
                if shard_start <= row_idx < shard_end
            )
            print(
                f"[sequential-parquet] subset={subset} shard={shard_idx:04d} "
                f"rows={row_count} range=[{shard_start},{shard_end}) needed={needed_count}",
                flush=True,
            )
            if needed_count:
                try:
                    by_split = materialize_parquet_rows(
                        parquet_path=parquet_path,
                        shard_start_idx=shard_start,
                        rows_by_split_by_subset=subset_rows,
                        split_orders=split_orders,
                        image_root=image_root,
                        ext=ext,
                        save_format=args.save_format,
                        jpeg_quality=args.jpeg_quality,
                        overwrite_images=args.overwrite_images,
                        preprocess=args.preprocess,
                        image_size=args.image_size,
                        pad_color=pad_color,
                        batch_size=args.batch_size,
                    )
                    for split, records in by_split.items():
                        materialized_by_split[split].extend(records)
                except Exception as exc:  # noqa: BLE001 - data failures should be explicit.
                    bad_rows.append(
                        {
                            "subset": subset,
                            "shard_idx": shard_idx,
                            "row_start": shard_start,
                            "row_end": shard_end,
                            "error": repr(exc),
                        }
                    )
                    if len(bad_rows) > args.max_bad_rows:
                        raise RuntimeError(
                            f"bad row budget exceeded: {len(bad_rows)} > {args.max_bad_rows}; "
                            f"subset={subset}, shard={shard_idx:04d}"
                        ) from exc
            processed.append(
                {
                    "shard_idx": shard_idx,
                    "parquet_path": str(parquet_path),
                    "row_start": shard_start,
                    "row_end": shard_end,
                    "rows": row_count,
                    "needed_rows": needed_count,
                    "file_size": parquet_path.stat().st_size,
                }
            )
            if args.delete_parquet_after_materialize:
                parquet_path.unlink(missing_ok=True)
                parquet_path.with_suffix(parquet_path.suffix + ".complete").unlink(missing_ok=True)
            shard_start = shard_end
            if shard_start > max_needed_idx:
                break
        if shard_start <= max_needed_idx:
            raise RuntimeError(
                f"subset={subset} only reached row {shard_start}, but needed row {max_needed_idx}. "
                "Increase --max-shards-per-subset or check shard counts."
            )
        shard_meta[subset] = processed

    written_paths: List[Path] = []
    split_meta: Dict[str, Dict[str, Any]] = {}
    for split in sorted(args.splits, key=split_order_key):
        materialized = sorted(materialized_by_split[split], key=lambda row: int(row["materialize_order"]))
        expected = len(rows_by_split[split])
        if len(materialized) != expected:
            raise RuntimeError(f"split={split} materialized {len(materialized)} rows, expected {expected}.")
        image_paths_path = output_root / f"{split}_image_paths.json"
        materialized_manifest = output_root / f"{split}_materialized_manifest.jsonl"
        write_json(image_paths_path, [row["image_path"] for row in materialized])
        write_jsonl(materialized_manifest, materialized)
        written_paths.extend([image_paths_path, materialized_manifest])
        by_subset = Counter(str(row["subset"]) for row in materialized)
        total_bytes = sum(int(row.get("file_size", 0)) for row in materialized)
        split_meta[split] = {
            "source_rows": expected,
            "materialized_rows": len(materialized),
            "by_subset": dict(sorted(by_subset.items())),
            "total_bytes": total_bytes,
            "estimated_total_gb": round(total_bytes / (1024**3), 3),
        }

    bad_rows_path = output_root / "bad_rows.jsonl"
    meta_path = output_root / "materialize_meta.json"
    write_jsonl(bad_rows_path, bad_rows)
    meta = {
        "manifest_root": str(output_root),
        "image_root": str(image_root),
        "parquet_cache_root": str(parquet_cache_root),
        "source_manifest_paths": {key: str(path) for key, path in source_paths.items()},
        "splits": args.splits,
        "dataset_name": args.dataset_name,
        "hf_split": args.hf_split,
        "hf_endpoint": endpoint,
        "parquet_revision": CONVERT_PARQUET_REVISION,
        "selection_strategy": "sequential_front_rows_from_converted_parquet",
        "save_format": args.save_format,
        "jpeg_quality": args.jpeg_quality,
        "preprocess": args.preprocess,
        "image_size": args.image_size,
        "pad_color": list(pad_color),
        "curl_retries": args.curl_retries,
        "curl_connect_timeout": args.curl_connect_timeout,
        "curl_max_time": args.curl_max_time,
        "bad_rows": len(bad_rows),
        "max_bad_rows": args.max_bad_rows,
        "delete_parquet_after_materialize": args.delete_parquet_after_materialize,
        "split_meta": split_meta,
        "shard_meta": shard_meta,
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
                "parquet_cache_root": str(parquet_cache_root),
                "manifest_sha256": str(source_paths["manifest_sha256"]),
                "materialized_sha256": str(sha_path),
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
