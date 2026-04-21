#!/usr/bin/env python3
"""Materialize TextAtlas images via local parquet shard cache.

This avoids Hugging Face datasets streaming for unstable network environments:
download one parquet shard at a time, read selected rows locally, and keep
already downloaded shards/images for resume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from textatlas_manifest_utils import (
    DEFAULT_DATASET,
    DEFAULT_STAGE1_SUBSETS,
    KNOWN_TEXTATLAS_SUBSETS,
    image_extension,
    iter_jsonl,
    materialize_source_record,
    now_str,
    parse_pad_color,
    write_json,
    write_jsonl,
    write_sha256_manifest,
)


SHARD_RE = re.compile(r"^(?P<split>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.parquet$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize TextAtlas images from cached parquet shards.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--parquet-cache-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--hf-endpoint", default=None, help="Defaults to HF_ENDPOINT or https://huggingface.co.")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "holdout"), default=["train", "val"])
    parser.add_argument("--subsets", nargs="+", choices=KNOWN_TEXTATLAS_SUBSETS, default=list(DEFAULT_STAGE1_SUBSETS))
    parser.add_argument("--save-format", choices=("jpeg", "png"), default="png")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--preprocess", choices=("none", "resize-pad"), default="resize-pad")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--download-timeout", type=int, default=120)
    parser.add_argument("--download-retries", type=int, default=8)
    parser.add_argument("--retry-sleep", type=int, default=10)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-bad-rows", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0, help="Process at most N shards per subset. 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true", help="List required shards and exit without downloading.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_args(args: argparse.Namespace) -> None:
    if len(set(args.splits)) != len(args.splits):
        raise SystemExit("--splits contains duplicates.")
    if len(set(args.subsets)) != len(args.subsets):
        raise SystemExit("--subsets contains duplicates.")
    if args.image_size <= 0:
        raise SystemExit("--image-size must be positive.")
    if not (1 <= args.jpeg_quality <= 100):
        raise SystemExit("--jpeg-quality must be in [1, 100].")
    if args.download_timeout <= 0 or args.download_retries <= 0 or args.retry_sleep < 0:
        raise SystemExit("download timeout/retries/sleep must be positive.")
    if args.chunk_size_mb <= 0 or args.batch_size <= 0:
        raise SystemExit("chunk-size-mb and batch-size must be positive.")
    if args.max_bad_rows < 0 or args.max_shards < 0:
        raise SystemExit("max-bad-rows and max-shards must be non-negative.")


def split_order_key(split: str) -> int:
    order = {"train": 0, "val": 1, "holdout": 2}
    return order[split]


def subset_order(subsets: Iterable[str]) -> List[str]:
    present = set(subsets)
    preferred = list(DEFAULT_STAGE1_SUBSETS) + [
        subset for subset in KNOWN_TEXTATLAS_SUBSETS if subset not in DEFAULT_STAGE1_SUBSETS
    ]
    return [subset for subset in preferred if subset in present]


def load_requested_rows(manifest_root: Path, splits: Sequence[str], subsets: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    subset_set = set(subsets)
    rows_by_split: Dict[str, List[Dict[str, Any]]] = {}
    for split in splits:
        manifest = manifest_root / f"{split}_manifest.jsonl"
        if not manifest.exists():
            raise FileNotFoundError(f"missing manifest for split={split}: {manifest}")
        rows = [row for row in iter_jsonl(manifest) if str(row["subset"]) in subset_set]
        if not rows:
            raise ValueError(f"manifest has no requested rows for split={split}: {manifest}")
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


def build_requests(
    *,
    rows_by_split: Mapping[str, List[Dict[str, Any]]],
    image_root: Path,
    ext: str,
) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    requests_by_subset: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for split, rows in rows_by_split.items():
        for order, row in enumerate(rows):
            subset = str(row["subset"])
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
    return requests_by_subset


def import_hf_hub():
    try:
        from huggingface_hub import HfApi, hf_hub_url  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: huggingface_hub.") from exc
    return HfApi, hf_hub_url


def import_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: pyarrow.") from exc
    return pq


def list_parquet_shards(
    *,
    dataset_name: str,
    subset: str,
    hf_split: str,
    revision: str,
    endpoint: str,
    retries: int,
    retry_sleep: int,
) -> List[str]:
    HfApi, _ = import_hf_hub()
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            api = HfApi(endpoint=endpoint)
            files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset", revision=revision)
            prefix = f"{subset}/"
            shards = [
                path for path in files
                if path.startswith(prefix)
                and path.endswith(".parquet")
                and SHARD_RE.match(Path(path).name)
                and SHARD_RE.match(Path(path).name).group("split") == hf_split  # type: ignore[union-attr]
            ]
            shards.sort(key=shard_sort_key)
            if not shards:
                raise RuntimeError(f"No parquet shards found for subset={subset}, split={hf_split}.")
            return shards
        except Exception as exc:  # noqa: BLE001 - Hub failures are network dependent.
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = retry_sleep * attempt
            print(
                f"[parquet-shards] list files subset={subset} attempt={attempt}/{retries} failed: {exc!r}; "
                f"retrying in {sleep_s}s",
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to list parquet shards for subset={subset}.") from last_exc


def shard_sort_key(path: str) -> tuple[int, str]:
    match = SHARD_RE.match(Path(path).name)
    if not match:
        return (10**9, path)
    return (int(match.group("idx")), path)


def shard_local_path(cache_root: Path, dataset_name: str, revision: str, remote_path: str) -> Path:
    safe_dataset = dataset_name.replace("/", "__")
    safe_revision = revision.replace("/", "__")
    return cache_root / safe_dataset / safe_revision / remote_path


def download_shard(
    *,
    dataset_name: str,
    remote_path: str,
    revision: str,
    endpoint: str,
    output_path: Path,
    timeout: int,
    retries: int,
    retry_sleep: int,
    chunk_size_mb: int,
) -> Path:
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: requests.") from exc

    _, hf_hub_url = import_hf_hub()
    url = hf_hub_url(
        repo_id=dataset_name,
        filename=remote_path,
        repo_type="dataset",
        revision=revision,
        endpoint=endpoint,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")
    chunk_size = chunk_size_mb * 1024 * 1024
    last_exc: Optional[BaseException] = None

    for attempt in range(1, retries + 1):
        resume_at = part_path.stat().st_size if part_path.exists() else 0
        headers = {}
        mode = "wb"
        if resume_at > 0:
            headers["Range"] = f"bytes={resume_at}-"
            mode = "ab"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if resume_at > 0 and response.status_code != 206:
                    resume_at = 0
                    mode = "wb"
                response.raise_for_status()
                with part_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            handle.write(chunk)
            part_path.replace(output_path)
            return output_path
        except Exception as exc:  # noqa: BLE001 - retry any network or HTTP failure.
            last_exc = exc
            if attempt >= retries:
                break
            sleep_s = retry_sleep * attempt
            print(
                f"[parquet-shards] download {remote_path} attempt={attempt}/{retries} failed: {exc!r}; "
                f"retrying in {sleep_s}s",
                flush=True,
            )
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download parquet shard: {remote_path}") from last_exc


def materialize_from_parquet(
    *,
    parquet_path: Path,
    shard_start_idx: int,
    pending_local: Mapping[int, List[Dict[str, Any]]],
    save_format: str,
    jpeg_quality: int,
    overwrite: bool,
    preprocess: str,
    image_size: int,
    pad_color: tuple[int, int, int],
    batch_size: int,
) -> List[Dict[str, Any]]:
    pq = import_pyarrow()
    selected = set(int(key) for key in pending_local)
    if not selected:
        return []
    output: List[Dict[str, Any]] = []
    parquet_file = pq.ParquetFile(parquet_path)
    local_offset = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_len = batch.num_rows
        batch_start = local_offset
        batch_end = local_offset + batch_len
        needed = sorted(idx for idx in selected if batch_start <= idx < batch_end)
        if needed:
            rows = batch.to_pylist()
            for local_idx in needed:
                row = rows[local_idx - batch_start]
                for request in pending_local[local_idx]:
                    record = materialize_source_record(
                        source_record=request["source_record"],
                        row=row,
                        output_path=request["output_path"],
                        save_format=save_format,
                        jpeg_quality=jpeg_quality,
                        overwrite=overwrite,
                        preprocess=preprocess,
                        image_size=image_size,
                        pad_color=pad_color,
                    )
                    record["materialize_order"] = int(request["order"])
                    record["parquet_path"] = str(parquet_path)
                    record["parquet_local_row_idx"] = int(local_idx)
                    record["parquet_global_start_idx"] = int(shard_start_idx)
                    output.append(record)
        local_offset = batch_end
        if local_offset > max(selected):
            break
    return output


def main() -> None:
    args = parse_args()
    ensure_args(args)
    manifest_root = args.manifest_root.resolve()
    output_root = (args.output_root or manifest_root).resolve()
    image_root = args.image_root.resolve()
    parquet_cache_root = args.parquet_cache_root.resolve()
    endpoint = (args.hf_endpoint or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    pad_color = parse_pad_color(args.pad_color)
    ext = image_extension(args.save_format)
    started_at = now_str()

    output_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)
    parquet_cache_root.mkdir(parents=True, exist_ok=True)

    rows_by_split = load_requested_rows(manifest_root, args.splits, args.subsets)
    dataset_name, hf_split = validate_dataset_identity(rows_by_split)
    if dataset_name != args.dataset_name:
        raise ValueError(f"manifest dataset={dataset_name}, but --dataset-name={args.dataset_name}")
    requests_by_subset = build_requests(rows_by_split=rows_by_split, image_root=image_root, ext=ext)

    materialized_by_split: Dict[str, List[Dict[str, Any]]] = {split: [] for split in args.splits}
    bad_rows: List[Dict[str, Any]] = []
    shard_meta: Dict[str, Any] = {}

    for subset in subset_order(requests_by_subset):
        requested_global = sorted(requests_by_subset[subset])
        total_requests = sum(len(value) for value in requests_by_subset[subset].values())
        print(f"[parquet-shards] subset={subset} requested_rows={len(requested_global)} requests={total_requests}", flush=True)
        remote_shards = list_parquet_shards(
            dataset_name=dataset_name,
            subset=subset,
            hf_split=hf_split,
            revision=args.revision,
            endpoint=endpoint,
            retries=args.download_retries,
            retry_sleep=args.retry_sleep,
        )
        if args.max_shards:
            remote_shards = remote_shards[: args.max_shards]
        print(f"[parquet-shards] subset={subset} parquet_shards={len(remote_shards)} endpoint={endpoint}", flush=True)
        if args.dry_run:
            shard_meta[subset] = {"remote_shards": remote_shards, "requested_rows": len(requested_global)}
            continue

        pending = {int(key): list(value) for key, value in requests_by_subset[subset].items()}
        shard_start = 0
        processed_shards: List[Dict[str, Any]] = []
        for shard_idx, remote_path in enumerate(remote_shards, start=1):
            local_path = shard_local_path(parquet_cache_root, dataset_name, args.revision, remote_path)
            print(
                f"[parquet-shards] subset={subset} shard={shard_idx}/{len(remote_shards)} "
                f"remote={remote_path}",
                flush=True,
            )
            download_shard(
                dataset_name=dataset_name,
                remote_path=remote_path,
                revision=args.revision,
                endpoint=endpoint,
                output_path=local_path,
                timeout=args.download_timeout,
                retries=args.download_retries,
                retry_sleep=args.retry_sleep,
                chunk_size_mb=args.chunk_size_mb,
            )
            pq = import_pyarrow()
            row_count = int(pq.ParquetFile(local_path).metadata.num_rows)
            shard_end = shard_start + row_count
            needed_global = [idx for idx in pending if shard_start <= idx < shard_end]
            processed_shards.append(
                {
                    "remote_path": remote_path,
                    "local_path": str(local_path),
                    "row_start": shard_start,
                    "row_end": shard_end,
                    "rows": row_count,
                    "needed_rows": len(needed_global),
                }
            )
            if needed_global:
                pending_local = {
                    global_idx - shard_start: pending[global_idx]
                    for global_idx in needed_global
                }
                try:
                    materialized = materialize_from_parquet(
                        parquet_path=local_path,
                        shard_start_idx=shard_start,
                        pending_local=pending_local,
                        save_format=args.save_format,
                        jpeg_quality=args.jpeg_quality,
                        overwrite=args.overwrite,
                        preprocess=args.preprocess,
                        image_size=args.image_size,
                        pad_color=pad_color,
                        batch_size=args.batch_size,
                    )
                    for record in materialized:
                        materialized_by_split[str(record["split"])].append(record)
                    for global_idx in needed_global:
                        pending.pop(global_idx, None)
                except Exception as exc:  # noqa: BLE001 - data failures should be auditable.
                    for global_idx in needed_global:
                        for request in pending[global_idx]:
                            bad_rows.append(
                                {
                                    "dataset": dataset_name,
                                    "subset": subset,
                                    "split": request["split"],
                                    "hf_split": hf_split,
                                    "hf_row_idx": int(global_idx),
                                    "error": repr(exc),
                                    "parquet_path": str(local_path),
                                }
                            )
                    if len(bad_rows) > args.max_bad_rows:
                        raise RuntimeError(
                            f"bad row budget exceeded: {len(bad_rows)} > {args.max_bad_rows}. "
                            f"Latest subset={subset}, shard={remote_path}"
                        ) from exc
            shard_start = shard_end
            if not pending:
                break
        if pending:
            preview = sorted(pending)[:10]
            raise RuntimeError(f"{subset}: failed to materialize row indices {preview} (remaining={len(pending)}).")
        shard_meta[subset] = {
            "processed_shards": processed_shards,
            "requested_rows": len(requested_global),
            "requests": total_requests,
        }

    if args.dry_run:
        print(json.dumps({"dry_run": True, "shard_meta": shard_meta}, ensure_ascii=False, indent=2))
        return

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
            "source_rows": len(rows_by_split[split]),
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
        "parquet_cache_root": str(parquet_cache_root),
        "splits": args.splits,
        "subsets": args.subsets,
        "dataset_name": dataset_name,
        "hf_split": hf_split,
        "revision": args.revision,
        "hf_endpoint": endpoint,
        "save_format": args.save_format,
        "jpeg_quality": args.jpeg_quality,
        "preprocess": args.preprocess,
        "image_size": args.image_size,
        "pad_color": list(pad_color),
        "download_timeout": args.download_timeout,
        "download_retries": args.download_retries,
        "retry_sleep": args.retry_sleep,
        "bad_rows": len(bad_rows),
        "max_bad_rows": args.max_bad_rows,
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
