#!/usr/bin/env python3
"""Refresh TextAtlas materialized manifests with rendered text only.

This script keeps image paths and row order unchanged. It only rewrites the
text-related fields needed by the v2 text-conditioned decoder path.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from textatlas_manifest_utils import get_text_fields


PROMPT_TEXT_PATTERNS = (
    r"displaying\s+the\s+text\s*[:：]",
    r"written\s+with\s+the\s+text\s*[:：]",
    r"showing\s+the\s+text\s*[:：]",
    r"containing\s+the\s+text\s*[:：]",
    r"A\s+512x512\s+text\s+image\s+with\s+the\s+following\s+settings",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh rendered text fields in TextAtlas materialized manifests.")
    parser.add_argument("--manifest-root", type=Path, default=None, help="Directory containing split materialized manifests.")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=("train", "val", "holdout"))
    parser.add_argument("--output-root", type=Path, default=None, help="Output directory. Defaults to --manifest-root.")
    parser.add_argument("--suffix", default="_v2text", help="Suffix inserted before .jsonl for root/split mode.")
    parser.add_argument("--input-manifest", action="append", type=Path, default=[], help="Explicit input manifest path.")
    parser.add_argument("--output-manifest", action="append", type=Path, default=[], help="Explicit output manifest path.")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug limit per manifest. 0 means all rows.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def looks_like_raw_prompt(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROMPT_TEXT_PATTERNS)


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: row must be a JSON object.")
            yield line_no, value


def split_output_path(input_path: Path, output_root: Path, suffix: str) -> Path:
    name = input_path.name
    if not name.endswith(".jsonl"):
        return output_root / f"{name}{suffix}.jsonl"
    return output_root / f"{name[:-6]}{suffix}.jsonl"


def resolve_jobs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    jobs: List[Tuple[Path, Path]] = []
    if args.input_manifest:
        if len(args.output_manifest) != len(args.input_manifest):
            raise SystemExit("--output-manifest must be provided once for each --input-manifest.")
        return [(inp.resolve(), out.resolve()) for inp, out in zip(args.input_manifest, args.output_manifest)]

    if args.manifest_root is None:
        raise SystemExit("Use --manifest-root or explicit --input-manifest/--output-manifest pairs.")
    manifest_root = args.manifest_root.resolve()
    output_root = (args.output_root or manifest_root).resolve()
    for split in args.splits:
        input_path = manifest_root / f"{split}_materialized_manifest.jsonl"
        if not input_path.exists():
            if split == "holdout":
                continue
            raise FileNotFoundError(input_path)
        jobs.append((input_path, split_output_path(input_path, output_root, args.suffix)))
    return jobs


def row_for_text_extraction(row: Dict[str, Any]) -> Dict[str, Any]:
    prepared = dict(row)
    if not prepared.get("annotation") and prepared.get("raw_annotation"):
        prepared["annotation"] = prepared["raw_annotation"]
    if not prepared.get("raw_text") and prepared.get("text_source") == "raw_text" and prepared.get("text"):
        prepared["raw_text"] = prepared["text"]
    if not prepared.get("raw_annotation") and prepared.get("text_source") == "annotation" and prepared.get("text"):
        prepared["raw_annotation"] = prepared["text"]
        prepared["annotation"] = prepared["text"]
    return prepared


def refresh_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    subset = str(row.get("subset") or "")
    errors: List[str] = []
    if not subset:
        errors.append("missing subset")
        return dict(row), errors

    prepared = row_for_text_extraction(row)
    try:
        text, text_source, raw_annotation, status = get_text_fields(subset, prepared)
    except Exception as exc:  # noqa: BLE001 - report exact row failure.
        errors.append(f"text extraction raised {exc!r}")
        return dict(row), errors

    text = str(text or "").strip()
    updated = dict(row)
    updated["text"] = text
    updated["text_source"] = text_source
    updated["raw_annotation"] = raw_annotation
    updated["text_extraction_status"] = status

    if not text:
        errors.append("empty extracted text")
    if status.startswith("failed") or "_failed" in status:
        errors.append(f"failed extraction status={status}")
    if looks_like_raw_prompt(text):
        errors.append("extracted text still looks like raw prompt")
    if raw_annotation and text_source != "raw_text" and text == raw_annotation and status != "plain_annotation":
        errors.append("extracted text equals raw_annotation")
    return updated, errors


def refresh_manifest(
    *,
    input_path: Path,
    output_path: Path,
    max_rows: int,
    overwrite: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists() and not dry_run:
        tmp_path.unlink()

    by_subset = Counter()
    statuses = defaultdict(Counter)
    changed_by_subset = Counter()
    error_rows: List[str] = []
    first_source_key: Optional[str] = None
    last_source_key: Optional[str] = None
    rows = 0

    handle = None if dry_run else tmp_path.open("w", encoding="utf-8")
    try:
        for line_no, row in iter_jsonl(input_path):
            if max_rows and rows >= max_rows:
                break
            expected_order = row.get("materialize_order")
            if expected_order is not None and int(expected_order) != rows:
                error_rows.append(f"line {line_no}: materialize_order={expected_order}, expected {rows}")

            old_text = row.get("text")
            updated, errors = refresh_row(row)
            subset = str(updated.get("subset") or "")
            status = str(updated.get("text_extraction_status") or "")
            by_subset[subset] += 1
            statuses[subset][status] += 1
            if updated.get("text") != old_text:
                changed_by_subset[subset] += 1

            source_key = str(updated.get("source_key") or "")
            first_source_key = first_source_key or source_key
            last_source_key = source_key
            if errors:
                error_rows.append(f"line {line_no} source_key={source_key}: {'; '.join(errors)}")

            if handle is not None:
                handle.write(json.dumps(updated, ensure_ascii=False))
                handle.write("\n")
            rows += 1
    finally:
        if handle is not None:
            handle.close()

    if error_rows:
        if tmp_path.exists() and not dry_run:
            tmp_path.unlink()
    elif not dry_run:
        tmp_path.replace(output_path)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "dry_run": dry_run,
        "rows": rows,
        "by_subset": dict(sorted(by_subset.items())),
        "changed_by_subset": dict(sorted(changed_by_subset.items())),
        "status_by_subset": {
            subset: dict(counter.most_common())
            for subset, counter in sorted(statuses.items())
        },
        "first_source_key": first_source_key,
        "last_source_key": last_source_key,
        "errors": error_rows[:50],
        "error_count": len(error_rows),
    }


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args)
    summaries = [
        refresh_manifest(
            input_path=input_path,
            output_path=output_path,
            max_rows=args.max_rows,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        for input_path, output_path in jobs
    ]
    print(json.dumps({"jobs": summaries}, ensure_ascii=False, indent=2))
    if any(summary["error_count"] for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
