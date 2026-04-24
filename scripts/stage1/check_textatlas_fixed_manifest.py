#!/usr/bin/env python3
"""Validate the fixed-count 5-subset TextAtlas source and materialized manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from textatlas_manifest_utils import (
    DEFAULT_STAGE1_SUBSETS,
    KNOWN_TEXTATLAS_SUBSETS,
    SOURCE_ONLY_REQUIRED_FIELDS,
    TEXT_SOURCE_BY_SUBSET,
)


MATERIALIZED_REQUIRED_FIELDS = set(SOURCE_ONLY_REQUIRED_FIELDS) | {
    "hf_image_path",
    "image_path",
    "materialized",
    "text",
    "text_source",
    "text_extraction_status",
    "raw_annotation",
    "width",
    "height",
    "file_size",
}
PROMPT_TEXT_PATTERNS = (
    r"displaying\s+the\s+text\s*[:：]",
    r"written\s+with\s+the\s+text\s*[:：]",
    r"showing\s+the\s+text\s*[:：]",
    r"containing\s+the\s+text\s*[:：]",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TextAtlas stage-1 fixed-count manifests.")
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, default=None)
    parser.add_argument("--subsets", nargs="+", choices=KNOWN_TEXTATLAS_SUBSETS, default=list(DEFAULT_STAGE1_SUBSETS))
    parser.add_argument("--expected-textsceneshq-train-count", type=int, default=40_000)
    parser.add_argument("--expected-same-train-count", type=int, default=50_000)
    parser.add_argument("--expected-val-per-subset", type=int, default=2_000)
    parser.add_argument("--expected-holdout-per-subset", type=int, default=500)
    parser.add_argument("--max-image-checks", type=int, default=100)
    parser.add_argument("--allow-missing-holdout", action="store_true")
    return parser.parse_args()


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
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object.")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: Pillow.") from exc
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.load()
        return rgb.size


def require(condition: bool, errors: List[str], message: str) -> None:
    if not condition:
        errors.append(message)


def looks_like_raw_prompt(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROMPT_TEXT_PATTERNS)


def validate_sha_manifest(root: Path, errors: List[str]) -> Dict[str, str]:
    sha_path = root / "manifest.sha256"
    if not sha_path.exists():
        errors.append(f"missing manifest.sha256: {sha_path}")
        return {}
    checked: Dict[str, str] = {}
    with sha_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            digest, filename = line.split(None, 1)
            path = root / filename.strip()
            if not path.exists():
                errors.append(f"sha256-listed file missing: {path}")
                continue
            actual = sha256_file(path)
            checked[path.name] = actual
            if actual != digest:
                errors.append(f"sha256 mismatch for {path.name}: {actual} != {digest}")
    return checked


def validate_source_rows(
    *,
    split: str,
    rows: List[Dict[str, Any]],
    subsets: Set[str],
    errors: List[str],
) -> Counter:
    counts = Counter()
    seen: Set[str] = set()
    for idx, row in enumerate(rows):
        missing = set(SOURCE_ONLY_REQUIRED_FIELDS) - set(row)
        require(not missing, errors, f"{split} row {idx} missing fields: {sorted(missing)}")
        subset = str(row.get("subset"))
        require(subset in subsets, errors, f"{split} row {idx} has disallowed subset={subset!r}")
        require(row.get("split") == split, errors, f"{split} row {idx} split={row.get('split')!r}")
        require(row.get("hf_split") == "train", errors, f"{split} row {idx} hf_split={row.get('hf_split')!r}")
        require(isinstance(row.get("hf_row_idx"), int), errors, f"{split} row {idx} hf_row_idx must be int")
        source_key = str(row.get("source_key"))
        require(source_key not in seen, errors, f"{split} duplicate source_key={source_key!r}")
        seen.add(source_key)
        counts[subset] += 1
    return counts


def validate_disjoint(split_rows: Dict[str, List[Dict[str, Any]]], errors: List[str]) -> None:
    seen: Dict[str, str] = {}
    for split, rows in split_rows.items():
        for row in rows:
            source_key = str(row.get("source_key"))
            previous = seen.get(source_key)
            if previous is not None:
                errors.append(f"source_key={source_key!r} appears in both {previous} and {split}")
            seen[source_key] = split


def validate_policy(
    *,
    exact_count: Dict[str, Any],
    train_counts: Counter,
    val_counts: Counter,
    holdout_counts: Counter,
    subsets: Sequence[str],
    expected_textsceneshq_train_count: int,
    expected_same_train_count: int,
    expected_val_per_subset: int,
    expected_holdout_per_subset: int,
    errors: List[str],
) -> None:
    for subset in subsets:
        require(
            val_counts[subset] == expected_val_per_subset,
            errors,
            f"val subset={subset} count={val_counts[subset]}, expected {expected_val_per_subset}",
        )
        require(
            holdout_counts[subset] == expected_holdout_per_subset,
            errors,
            f"holdout subset={subset} count={holdout_counts[subset]}, expected {expected_holdout_per_subset}",
        )

    same_count_subsets = [subset for subset in subsets if subset != "TextScenesHQ"]
    require(
        train_counts["TextScenesHQ"] == expected_textsceneshq_train_count,
        errors,
        f"TextScenesHQ train={train_counts['TextScenesHQ']}, expected {expected_textsceneshq_train_count}",
    )
    for subset in same_count_subsets:
        require(
            train_counts[subset] == expected_same_train_count,
            errors,
            f"train subset={subset} count={train_counts[subset]}, expected {expected_same_train_count}",
        )

    expected_train_counts = exact_count.get("train_counts_by_subset")
    if isinstance(expected_train_counts, dict):
        actual = {subset: train_counts[subset] for subset in subsets}
        require(actual == expected_train_counts, errors, f"train counts mismatch exact_count: {actual} != {expected_train_counts}")


def validate_materialized_split(
    *,
    materialized_root: Path,
    split: str,
    source_count: int,
    max_image_checks: int,
    errors: List[str],
) -> Dict[str, Any]:
    paths_path = materialized_root / f"{split}_image_paths.json"
    manifest_path = materialized_root / f"{split}_materialized_manifest.jsonl"
    if not paths_path.exists() or not manifest_path.exists():
        errors.append(f"missing materialized files for split={split} under {materialized_root}")
        return {}
    image_paths = read_json(paths_path)
    rows = read_jsonl(manifest_path)
    require(isinstance(image_paths, list), errors, f"{paths_path} must contain a JSON array")
    require(len(image_paths) == len(rows), errors, f"{split}: image path count != manifest count")
    require(len(rows) == source_count, errors, f"{split}: materialized rows={len(rows)}, source rows={source_count}")
    if isinstance(image_paths, list):
        manifest_paths = [row.get("image_path") for row in rows]
        require(
            manifest_paths == image_paths,
            errors,
            f"{split}: image paths JSON must exactly match materialized manifest order",
        )

    counts = Counter()
    checked = 0
    checked_bytes = 0
    for idx, row in enumerate(rows):
        missing = MATERIALIZED_REQUIRED_FIELDS - set(row)
        require(not missing, errors, f"{split} materialized row {idx} missing fields: {sorted(missing)}")
        subset = str(row.get("subset"))
        counts[subset] += 1
        extraction_status = str(row.get("text_extraction_status") or "")
        if subset in TEXT_SOURCE_BY_SUBSET:
            require(
                row.get("text_source") == TEXT_SOURCE_BY_SUBSET[subset] or extraction_status.startswith("fallback_"),
                errors,
                f"{split} row {idx} subset={subset} text_source={row.get('text_source')!r}",
            )
        text = str(row.get("text") or "").strip()
        raw_annotation = str(row.get("raw_annotation") or "").strip()
        require(bool(text), errors, f"{split} row {idx} subset={subset} has empty text")
        require(
            not (extraction_status.startswith("failed") or "_failed" in extraction_status),
            errors,
            f"{split} row {idx} subset={subset} failed text extraction: {extraction_status}",
        )
        require(
            not looks_like_raw_prompt(text),
            errors,
            f"{split} row {idx} subset={subset} text still looks like raw prompt: {text[:120]!r}",
        )
        if raw_annotation and row.get("text_source") != "raw_text" and extraction_status != "plain_annotation":
            require(
                text != raw_annotation,
                errors,
                f"{split} row {idx} subset={subset} text equals raw_annotation",
            )
        if max_image_checks and checked >= max_image_checks:
            continue
        image_path = Path(str(row.get("image_path")))
        if not image_path.exists():
            errors.append(f"{split}: missing image {image_path}")
            continue
        try:
            validate_image(image_path)
        except Exception as exc:  # noqa: BLE001 - report all validation failures.
            errors.append(f"{split}: cannot load image {image_path}: {exc!r}")
            continue
        checked += 1
        checked_bytes += image_path.stat().st_size
    return {"rows": len(rows), "by_subset": dict(sorted(counts.items())), "checked": checked, "checked_bytes": checked_bytes}


def main() -> None:
    args = parse_args()
    manifest_root = args.manifest_root.resolve()
    subsets = set(args.subsets)
    errors: List[str] = []

    required = ["exact_count.json", "train_manifest.jsonl", "val_manifest.jsonl", "manifest.sha256"]
    if not args.allow_missing_holdout:
        required.append("holdout_manifest.jsonl")
    for filename in required:
        require((manifest_root / filename).exists(), errors, f"required file missing: {manifest_root / filename}")
    if errors:
        raise SystemExit("\n".join(errors))

    exact_count = read_json(manifest_root / "exact_count.json")
    train_rows = read_jsonl(manifest_root / "train_manifest.jsonl")
    val_rows = read_jsonl(manifest_root / "val_manifest.jsonl")
    holdout_rows = [] if args.allow_missing_holdout and not (manifest_root / "holdout_manifest.jsonl").exists() else read_jsonl(manifest_root / "holdout_manifest.jsonl")

    train_counts = validate_source_rows(split="train", rows=train_rows, subsets=subsets, errors=errors)
    val_counts = validate_source_rows(split="val", rows=val_rows, subsets=subsets, errors=errors)
    holdout_counts = validate_source_rows(split="holdout", rows=holdout_rows, subsets=subsets, errors=errors)
    validate_disjoint({"train": train_rows, "val": val_rows, "holdout": holdout_rows}, errors)
    if holdout_rows:
        validate_policy(
            exact_count=exact_count,
            train_counts=train_counts,
            val_counts=val_counts,
            holdout_counts=holdout_counts,
            subsets=args.subsets,
            expected_textsceneshq_train_count=args.expected_textsceneshq_train_count,
            expected_same_train_count=args.expected_same_train_count,
            expected_val_per_subset=args.expected_val_per_subset,
            expected_holdout_per_subset=args.expected_holdout_per_subset,
            errors=errors,
        )
    manifest_sha_checked = validate_sha_manifest(manifest_root, errors)

    materialized_summary: Dict[str, Any] = {}
    if args.materialized_root is not None:
        materialized_root = args.materialized_root.resolve()
        for split, source_count in {
            "train": len(train_rows),
            "val": len(val_rows),
            "holdout": len(holdout_rows),
        }.items():
            if split == "holdout" and not (materialized_root / "holdout_image_paths.json").exists():
                continue
            materialized_summary[split] = validate_materialized_split(
                materialized_root=materialized_root,
                split=split,
                source_count=source_count,
                max_image_checks=args.max_image_checks,
                errors=errors,
            )

    summary = {
        "manifest_root": str(manifest_root),
        "materialized_root": str(args.materialized_root.resolve()) if args.materialized_root else None,
        "subsets": args.subsets,
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "holdout_count": len(holdout_rows),
        "train_counts_by_subset": dict(sorted(train_counts.items())),
        "val_counts_by_subset": dict(sorted(val_counts.items())),
        "holdout_counts_by_subset": dict(sorted(holdout_counts.items())),
        "manifest_sha256_checked": manifest_sha_checked,
        "materialized_summary": materialized_summary,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        print("\n".join(errors[:50]), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
