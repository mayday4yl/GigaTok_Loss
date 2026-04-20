#!/usr/bin/env python3
"""Validate image-only TextAtlas5M pilot manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


ALLOWED_SUBSETS = {
    "CleanTextSynth",
    "StyledTextSynth",
    "LongWordsSubset-M",
    "TextScenesHQ",
}
EXPECTED_TEXT_SOURCE = {
    "CleanTextSynth": "annotation",
    "StyledTextSynth": "annotation",
    "LongWordsSubset-M": "annotation",
    "TextScenesHQ": "raw_text",
}
REQUIRED_FIELDS = {
    "dataset",
    "subset",
    "split",
    "hf_split",
    "hf_row_idx",
    "hf_image_path",
    "image_path",
    "text",
    "text_source",
    "raw_annotation",
}
_PIL_IMAGE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TextAtlas image-only manifests.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--expected-train-per-subset", type=int, default=10_000)
    parser.add_argument("--expected-val-per-subset", type=int, default=500)
    parser.add_argument(
        "--max-image-checks",
        type=int,
        default=0,
        help="Validate at most this many images per split. 0 means validate all images.",
    )
    return parser.parse_args()


def import_pillow():
    global _PIL_IMAGE
    if _PIL_IMAGE is not None:
        return _PIL_IMAGE
    try:
        from PIL import Image as pil_image  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: Pillow. Install it in the stage-1 environment "
            "before checking the TextAtlas image manifest."
        ) from exc
    _PIL_IMAGE = pil_image
    return pil_image


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: manifest row must be a JSON object.")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> Tuple[int, int]:
    pil_image = import_pillow()
    with pil_image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.load()
        return rgb.size


def require(condition: bool, errors: List[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate_image_list(name: str, data: Any, errors: List[str]) -> List[str]:
    if not isinstance(data, list):
        errors.append(f"{name} must be a JSON array.")
        return []
    image_paths: List[str] = []
    for idx, item in enumerate(data):
        if not isinstance(item, str):
            errors.append(f"{name}[{idx}] must be a string path, got {type(item).__name__}.")
            continue
        image_paths.append(item)
    return image_paths


def validate_records(
    *,
    split: str,
    records: List[Dict[str, Any]],
    expected_per_subset: int,
    errors: List[str],
) -> Counter:
    counts = Counter()
    for idx, record in enumerate(records):
        missing = REQUIRED_FIELDS - set(record)
        require(not missing, errors, f"{split} manifest row {idx} missing fields: {sorted(missing)}")

        subset = record.get("subset")
        require(subset in ALLOWED_SUBSETS, errors, f"{split} row {idx} has disallowed subset: {subset!r}")
        require(record.get("split") == split, errors, f"{split} row {idx} has split={record.get('split')!r}")
        require(record.get("hf_split") == "train", errors, f"{split} row {idx} has hf_split={record.get('hf_split')!r}")
        require(isinstance(record.get("hf_row_idx"), int), errors, f"{split} row {idx} hf_row_idx must be int")
        require(isinstance(record.get("image_path"), str), errors, f"{split} row {idx} image_path must be string")

        if subset in EXPECTED_TEXT_SOURCE:
            expected_source = EXPECTED_TEXT_SOURCE[subset]
            require(
                record.get("text_source") == expected_source,
                errors,
                f"{split} row {idx} subset={subset} text_source={record.get('text_source')!r}, expected {expected_source!r}",
            )
        counts[subset] += 1

    for subset in sorted(ALLOWED_SUBSETS):
        require(
            counts[subset] == expected_per_subset,
            errors,
            f"{split} subset={subset} count={counts[subset]}, expected {expected_per_subset}",
        )
    return counts


def validate_train_val_disjoint(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    errors: List[str],
) -> None:
    train_by_subset: Dict[str, Set[int]] = {subset: set() for subset in ALLOWED_SUBSETS}
    val_by_subset: Dict[str, Set[int]] = {subset: set() for subset in ALLOWED_SUBSETS}

    for record in train_records:
        subset = record.get("subset")
        row_idx = record.get("hf_row_idx")
        if subset in train_by_subset and isinstance(row_idx, int):
            train_by_subset[subset].add(row_idx)

    for record in val_records:
        subset = record.get("subset")
        row_idx = record.get("hf_row_idx")
        if subset in val_by_subset and isinstance(row_idx, int):
            val_by_subset[subset].add(row_idx)

    for subset in sorted(ALLOWED_SUBSETS):
        overlap = train_by_subset[subset] & val_by_subset[subset]
        if overlap:
            preview = sorted(overlap)[:20]
            errors.append(
                f"subset={subset} has hf_row_idx overlap between train and val: "
                f"{preview} (total {len(overlap)})"
            )


def validate_paths_match(split: str, records: List[Dict[str, Any]], image_paths: List[str], errors: List[str]) -> None:
    manifest_paths = [record.get("image_path") for record in records]
    require(
        manifest_paths == image_paths,
        errors,
        f"{split}_image_paths.json must exactly match image_path order from {split}_manifest.jsonl.",
    )


def validate_images(split: str, image_paths: Iterable[str], max_checks: int, errors: List[str]) -> Dict[str, Any]:
    checked = 0
    total_bytes = 0
    for path_str in image_paths:
        if max_checks and checked >= max_checks:
            break
        path = Path(path_str)
        if not path.exists():
            errors.append(f"{split}: image does not exist: {path}")
            continue
        try:
            validate_image(path)
        except Exception as exc:  # noqa: BLE001 - report all validation failures.
            errors.append(f"{split}: image cannot convert/load as RGB: {path}: {exc!r}")
            continue
        total_bytes += path.stat().st_size
        checked += 1
    return {"checked": checked, "checked_bytes": total_bytes}


def read_sha_manifest(path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                digest, filename = line.split(None, 1)
            except ValueError:
                raise ValueError(f"invalid sha256 manifest line in {path}: {line!r}")
            values[filename.strip()] = digest
    return values


def validate_sha256_file(manifest_root: Path, sha_filename: str, errors: List[str]) -> Dict[str, str]:
    sha_path = manifest_root / sha_filename
    if not sha_path.exists():
        errors.append(f"required sha256 file missing: {sha_path}")
        return {}

    sha_manifest = read_sha_manifest(sha_path)
    checked: Dict[str, str] = {}
    for filename, expected_digest in sha_manifest.items():
        path = manifest_root / filename
        if not path.exists():
            errors.append(f"sha256 listed file is missing: {filename}")
            continue
        actual_digest = sha256_file(path)
        checked[filename] = actual_digest
        if actual_digest != expected_digest:
            errors.append(f"sha256 mismatch for {filename}: {actual_digest} != {expected_digest}")
    return checked


def validate_bad_rows(path: Path, errors: List[str]) -> Dict[str, Any]:
    if not path.exists():
        errors.append(f"required file missing: {path}")
        return {"rows": 0}

    rows = 0
    required = {"dataset", "subset", "hf_split", "hf_row_idx", "error"}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path}:{line_no}: invalid bad_rows JSONL row: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path}:{line_no}: bad_rows row must be a JSON object.")
                continue
            missing = required - set(row)
            if missing:
                errors.append(f"{path}:{line_no}: bad_rows row missing fields: {sorted(missing)}")
            if row.get("subset") not in ALLOWED_SUBSETS:
                errors.append(f"{path}:{line_no}: bad_rows row has disallowed subset: {row.get('subset')!r}")
    return {"rows": rows}


def validate_counts_json(
    path: Path,
    train_counts: Counter,
    val_counts: Counter,
    train_total: int,
    val_total: int,
    errors: List[str],
) -> Dict[str, Any]:
    if not path.exists():
        errors.append(f"required file missing: {path}")
        return {}

    counts = read_json(path)
    if not isinstance(counts, dict):
        errors.append(f"{path} must contain a JSON object.")
        return {}

    splits = counts.get("splits")
    if not isinstance(splits, dict):
        errors.append(f"{path} missing object field: splits")
        return counts

    expected = {
        "train": (train_counts, train_total),
        "val": (val_counts, val_total),
    }
    for split, (counter, total) in expected.items():
        split_counts = splits.get(split)
        if not isinstance(split_counts, dict):
            errors.append(f"{path} missing counts for split={split}")
            continue
        if split_counts.get("total") != total:
            errors.append(f"{path} split={split} total={split_counts.get('total')}, expected {total}")
        by_subset = split_counts.get("by_subset")
        if by_subset != dict(sorted(counter.items())):
            errors.append(f"{path} split={split} by_subset={by_subset}, expected {dict(sorted(counter.items()))}")

    return counts


def main() -> None:
    args = parse_args()
    manifest_root = args.manifest_root.resolve()
    errors: List[str] = []

    train_manifest_path = manifest_root / "train_manifest.jsonl"
    val_manifest_path = manifest_root / "val_manifest.jsonl"
    train_image_paths_path = manifest_root / "train_image_paths.json"
    val_image_paths_path = manifest_root / "val_image_paths.json"
    counts_path = manifest_root / "counts.json"
    bad_rows_path = manifest_root / "bad_rows.jsonl"
    manifest_sha_path = manifest_root / "manifest.sha256"
    image_paths_sha_path = manifest_root / "image_paths.sha256"

    for path in [
        train_manifest_path,
        val_manifest_path,
        train_image_paths_path,
        val_image_paths_path,
        counts_path,
        bad_rows_path,
        manifest_sha_path,
        image_paths_sha_path,
    ]:
        require(path.exists(), errors, f"required file missing: {path}")
    if errors:
        raise SystemExit("\n".join(errors))

    train_records = read_jsonl(train_manifest_path)
    val_records = read_jsonl(val_manifest_path)
    train_image_paths = validate_image_list("train_image_paths.json", read_json(train_image_paths_path), errors)
    val_image_paths = validate_image_list("val_image_paths.json", read_json(val_image_paths_path), errors)

    train_counts = validate_records(
        split="train",
        records=train_records,
        expected_per_subset=args.expected_train_per_subset,
        errors=errors,
    )
    val_counts = validate_records(
        split="val",
        records=val_records,
        expected_per_subset=args.expected_val_per_subset,
        errors=errors,
    )
    validate_paths_match("train", train_records, train_image_paths, errors)
    validate_paths_match("val", val_records, val_image_paths, errors)
    validate_train_val_disjoint(train_records, val_records, errors)

    train_image_stats = validate_images("train", train_image_paths, args.max_image_checks, errors)
    val_image_stats = validate_images("val", val_image_paths, args.max_image_checks, errors)
    bad_rows_stats = validate_bad_rows(bad_rows_path, errors)
    counts_checked = validate_counts_json(
        counts_path,
        train_counts,
        val_counts,
        len(train_records),
        len(val_records),
        errors,
    )
    manifest_sha_checked = validate_sha256_file(manifest_root, "manifest.sha256", errors)
    image_paths_sha_checked = validate_sha256_file(manifest_root, "image_paths.sha256", errors)

    summary = {
        "manifest_root": str(manifest_root),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_counts_by_subset": dict(sorted(train_counts.items())),
        "val_counts_by_subset": dict(sorted(val_counts.items())),
        "train_image_validation": train_image_stats,
        "val_image_validation": val_image_stats,
        "bad_rows": bad_rows_stats,
        "counts_checked": counts_checked,
        "manifest_sha256_checked": manifest_sha_checked,
        "image_paths_sha256_checked": image_paths_sha_checked,
        "errors": errors,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print("\n".join(errors[:50]), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
