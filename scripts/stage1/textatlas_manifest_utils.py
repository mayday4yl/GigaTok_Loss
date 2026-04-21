#!/usr/bin/env python3
"""Shared helpers for TextAtlas stage-1 manifest and materialization scripts."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


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


def import_datasets():
    try:
        from datasets import load_dataset, load_dataset_builder  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: datasets.") from exc
    return load_dataset, load_dataset_builder


def import_pillow():
    global _PIL_IMAGE
    if _PIL_IMAGE is not None:
        return _PIL_IMAGE
    try:
        from PIL import Image as pil_image  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: Pillow.") from exc
    _PIL_IMAGE = pil_image
    return pil_image


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object.")
            rows.append(value)
    return rows


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object.")
            yield value


def write_sha256_manifest(output_root: Path, paths: Iterable[Path], filename: str) -> Path:
    sha_path = output_root / filename
    with sha_path.open("w", encoding="utf-8") as handle:
        for path in paths:
            handle.write(f"{sha256_file(path)}  {path.name}\n")
    return sha_path


def image_extension(save_format: str) -> str:
    return "jpg" if save_format == "jpeg" else "png"


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


def validate_saved_image(path: Path) -> Tuple[int, int]:
    pil_image = import_pillow()
    with pil_image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.load()
        return rgb.size


def get_text_fields(subset: str, row: Mapping[str, Any]) -> Tuple[str, str, str]:
    text_source = TEXT_SOURCE_BY_SUBSET[subset]
    text = row.get(text_source) or ""
    raw_annotation = row.get("annotation") or ""
    return str(text), text_source, str(raw_annotation)


def build_source_record(
    *,
    dataset_name: str,
    subset: str,
    split: str,
    hf_split: str,
    hf_row_idx: int,
    selection_seed: int,
    selection_strategy: str,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "subset": subset,
        "split": split,
        "hf_split": hf_split,
        "hf_row_idx": int(hf_row_idx),
        "source_key": f"{subset}:{hf_split}:{int(hf_row_idx)}",
        "image_path": None,
        "materialized": False,
        "selection_seed": selection_seed,
        "selection_strategy": selection_strategy,
    }


def materialize_source_record(
    *,
    source_record: Mapping[str, Any],
    row: Mapping[str, Any],
    output_path: Path,
    save_format: str,
    jpeg_quality: int,
    overwrite: bool,
) -> Dict[str, Any]:
    output_path = output_path.resolve()
    if output_path.exists() and not overwrite:
        width, height = validate_saved_image(output_path)
    else:
        img = row_to_rgb_image(row.get("image"))
        save_image(img, output_path, save_format, jpeg_quality)
        width, height = validate_saved_image(output_path)

    subset = str(source_record["subset"])
    text, text_source, raw_annotation = get_text_fields(subset, row)
    record = dict(source_record)
    record.update(
        {
            "hf_image_path": str(row.get("image_path") or f"{int(source_record['hf_row_idx']):010d}"),
            "image_path": str(output_path),
            "materialized": True,
            "text": text,
            "text_source": text_source,
            "raw_annotation": raw_annotation,
            "width": width,
            "height": height,
            "file_size": output_path.stat().st_size,
        }
    )
    return record


def allocate_proportional(
    *,
    total: int,
    weights: Mapping[str, int],
    capacities: Optional[Mapping[str, int]] = None,
    keys: Sequence[str] = ALLOWED_SUBSETS,
) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative.")
    if capacities is None:
        capacities = {key: total for key in keys}
    capacity_sum = sum(int(capacities.get(key, 0)) for key in keys)
    if total > capacity_sum:
        raise ValueError(f"Cannot allocate total={total}; only capacity={capacity_sum}.")
    weight_sum = sum(max(0, int(weights.get(key, 0))) for key in keys)
    if total and weight_sum <= 0:
        raise ValueError("Cannot allocate non-zero total with zero weights.")

    allocation = {key: 0 for key in keys}
    remainders: List[Tuple[int, int, str]] = []
    for order, key in enumerate(keys):
        weight = max(0, int(weights.get(key, 0)))
        capacity = max(0, int(capacities.get(key, 0)))
        raw = total * weight
        base = raw // weight_sum if weight_sum else 0
        allocation[key] = min(base, capacity)
        remainders.append((raw % weight_sum if weight_sum else 0, -order, key))

    remaining = total - sum(allocation.values())
    remainders.sort(reverse=True)
    while remaining > 0:
        progressed = False
        for _, _, key in remainders:
            if allocation[key] >= int(capacities.get(key, 0)):
                continue
            allocation[key] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("Allocation stalled; capacities are too small.")
    return allocation


def summarize_records(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    by_subset = Counter(str(row.get("subset")) for row in rows)
    return {
        "total": len(rows),
        "by_subset": dict(sorted(by_subset.items())),
    }


def get_exact_count(
    *,
    dataset_name: str,
    subset: str,
    hf_split: str,
    count_method: str,
) -> Tuple[int, str]:
    load_dataset, load_dataset_builder = import_datasets()
    if count_method in {"auto", "builder"}:
        try:
            builder = load_dataset_builder(dataset_name, subset)
            split_info = builder.info.splits.get(hf_split) if builder.info.splits is not None else None
            if split_info is not None and split_info.num_examples is not None:
                return int(split_info.num_examples), "builder.info.splits"
        except Exception:
            if count_method == "builder":
                raise
    if count_method not in {"auto", "streaming"}:
        raise ValueError(f"Unknown count method: {count_method}")

    count = 0
    dataset = load_dataset(dataset_name, subset, split=hf_split, streaming=True)
    for _ in dataset:
        count += 1
    return count, "streaming_iteration"


def stream_materialize_records(
    *,
    dataset_name: str,
    hf_split: str,
    subset: str,
    requests_by_row_idx: Mapping[int, List[Dict[str, Any]]],
    save_format: str,
    jpeg_quality: int,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    if not requests_by_row_idx:
        return []
    load_dataset, _ = import_datasets()
    pending = {int(key): list(value) for key, value in requests_by_row_idx.items()}
    materialized: List[Dict[str, Any]] = []
    dataset = load_dataset(dataset_name, subset, split=hf_split, streaming=True)

    for row_idx, row in enumerate(dataset):
        if row_idx not in pending:
            continue
        for request in pending.pop(row_idx):
            record = materialize_source_record(
                source_record=request["source_record"],
                row=row,
                output_path=request["output_path"],
                save_format=save_format,
                jpeg_quality=jpeg_quality,
                overwrite=overwrite,
            )
            record["materialize_order"] = request["order"]
            materialized.append(record)
        if not pending:
            break

    if pending:
        preview = sorted(pending)[:10]
        raise RuntimeError(f"{subset}: failed to materialize row indices {preview} (remaining={len(pending)}).")
    return materialized
