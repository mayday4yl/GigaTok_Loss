#!/usr/bin/env python3
"""构建多图机制验证用的 TextAtlas manifest。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_SUBSETS = (
    "CleanTextSynth",
)

ALLOWED_EMPTY_TEXT_STATUSES = {
    "styled_no_explicit_text",
    "textvision_no_explicit_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-source-manifest", type=Path, required=True)
    parser.add_argument("--val-source-manifest", type=Path, default=None)
    parser.add_argument("--holdout-source-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="cleantextsynth_dense_100")
    parser.add_argument("--per-subset", type=int, default=100, help="Training rows per subset.")
    parser.add_argument("--val-per-subset", type=int, default=0, help="Validation rows per subset. 0 disables val output.")
    parser.add_argument("--holdout-per-subset", type=int, default=0, help="Holdout rows per subset. 0 disables holdout output.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subset", action="append", default=None, help="Subset to include. Defaults to CleanTextSynth.")
    parser.add_argument(
        "--same-eval-as-train",
        action="store_true",
        help="Write val and holdout manifests as exact copies of the selected train rows.",
    )
    parser.add_argument(
        "--selection",
        choices=("random", "longest_text"),
        default="longest_text",
        help="Row selection strategy within each subset.",
    )
    parser.add_argument("--min-text-chars", type=int, default=1)
    parser.add_argument("--allow-empty-text", action="store_true")
    parser.add_argument("--require-existing-images", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_manifest_line_no"] = line_no
            yield row


def resolve_image_path(row: Mapping[str, Any], manifest_path: Path) -> Path:
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = manifest_path.resolve().parent / image_path
    return image_path


def row_is_eligible(
        row: Mapping[str, Any],
        manifest_path: Path,
        min_text_chars: int,
        allow_empty_text: bool,
        require_existing_images: bool,
) -> bool:
    image_path = row.get("image_path")
    if not image_path:
        return False
    if require_existing_images and not resolve_image_path(row, manifest_path).exists():
        return False
    text = str(row.get("text") or "").strip()
    status = str(row.get("text_extraction_status") or "")
    if not text:
        return allow_empty_text and status in ALLOWED_EMPTY_TEXT_STATUSES
    return len(text) >= min_text_chars


def group_rows_by_subset(
        manifest_path: Path,
        subsets: Sequence[str],
        min_text_chars: int,
        allow_empty_text: bool,
        require_existing_images: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    wanted = set(subsets)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in iter_jsonl(manifest_path):
        subset = str(row.get("subset") or "")
        if subset not in wanted:
            continue
        if not row_is_eligible(
            row,
            manifest_path=manifest_path,
            min_text_chars=min_text_chars,
            allow_empty_text=allow_empty_text,
            require_existing_images=require_existing_images,
        ):
            continue
        grouped[subset].append(row)
    return grouped


def select_rows(
        rows: Sequence[Dict[str, Any]],
        count: int,
        selection: str,
        rng: random.Random,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    if len(rows) < count:
        raise ValueError(f"Need {count} rows but only found {len(rows)} eligible rows.")
    rows = list(rows)
    if selection == "random":
        rng.shuffle(rows)
        selected = rows[:count]
    elif selection == "longest_text":
        selected = sorted(
            rows,
            key=lambda row: (
                -len(str(row.get("text") or "")),
                str(row.get("source_key") or ""),
                int(row.get("_manifest_line_no") or 0),
            ),
        )[:count]
    else:
        raise ValueError(f"Unsupported selection: {selection}")
    return sorted(selected, key=lambda row: (str(row.get("subset") or ""), int(row.get("_manifest_line_no") or 0)))


def strip_internal_fields(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(strip_internal_fields(row), ensure_ascii=False))
            handle.write("\n")


def subset_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    lengths = [len(str(row.get("text") or "")) for row in rows]
    if not lengths:
        return {"count": 0}
    examples = []
    for row in sorted(rows, key=lambda item: -len(str(item.get("text") or "")))[:3]:
        examples.append({
            "source_key": row.get("source_key"),
            "text_chars": len(str(row.get("text") or "")),
            "text_preview": str(row.get("text") or "")[:160],
        })
    return {
        "count": len(rows),
        "text_chars_min": min(lengths),
        "text_chars_mean": mean(lengths),
        "text_chars_max": max(lengths),
        "examples": examples,
    }


def build_split(
        manifest_path: Path,
        subsets: Sequence[str],
        count_per_subset: int,
        args: argparse.Namespace,
        rng: random.Random,
) -> List[Dict[str, Any]]:
    grouped = group_rows_by_subset(
        manifest_path,
        subsets=subsets,
        min_text_chars=args.min_text_chars,
        allow_empty_text=args.allow_empty_text,
        require_existing_images=args.require_existing_images,
    )
    missing = [subset for subset in subsets if subset not in grouped]
    if missing:
        raise SystemExit(f"Missing subset(s) in {manifest_path}: {missing}")

    selected: List[Dict[str, Any]] = []
    for subset in subsets:
        try:
            selected.extend(select_rows(grouped[subset], count_per_subset, args.selection, rng))
        except ValueError as exc:
            raise SystemExit(f"{manifest_path} subset={subset}: {exc}") from exc
    return selected


def write_summary(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    subsets = tuple(args.subset) if args.subset else DEFAULT_SUBSETS
    if args.per_subset <= 0:
        raise SystemExit("--per-subset must be positive.")
    if args.val_per_subset < 0:
        raise SystemExit("--val-per-subset must be >= 0.")
    if args.holdout_per_subset < 0:
        raise SystemExit("--holdout-per-subset must be >= 0.")

    rng = random.Random(args.seed)
    train_rows = build_split(
        args.train_source_manifest,
        subsets=subsets,
        count_per_subset=args.per_subset,
        args=args,
        rng=rng,
    )

    output_dir = args.output_dir
    train_out = output_dir / f"{args.name}_train.jsonl"
    write_jsonl(train_out, train_rows)

    val_out: Optional[Path] = None
    val_rows: List[Dict[str, Any]] = []
    holdout_out: Optional[Path] = None
    holdout_rows: List[Dict[str, Any]] = []
    if args.same_eval_as_train:
        val_rows = list(train_rows)
        holdout_rows = list(train_rows)
        val_out = output_dir / f"{args.name}_val.jsonl"
        holdout_out = output_dir / f"{args.name}_holdout.jsonl"
        write_jsonl(val_out, val_rows)
        write_jsonl(holdout_out, holdout_rows)
    elif args.val_source_manifest is not None and args.val_per_subset > 0:
        val_rows = build_split(
            args.val_source_manifest,
            subsets=subsets,
            count_per_subset=args.val_per_subset,
            args=args,
            rng=rng,
        )
        val_out = output_dir / f"{args.name}_val.jsonl"
        write_jsonl(val_out, val_rows)
        if args.holdout_source_manifest is not None and args.holdout_per_subset > 0:
            holdout_rows = build_split(
                args.holdout_source_manifest,
                subsets=subsets,
                count_per_subset=args.holdout_per_subset,
                args=args,
                rng=rng,
            )
            holdout_out = output_dir / f"{args.name}_holdout.jsonl"
            write_jsonl(holdout_out, holdout_rows)

    summary = {
        "name": args.name,
        "selection": args.selection,
        "seed": args.seed,
        "same_eval_as_train": bool(args.same_eval_as_train),
        "subsets": list(subsets),
        "train_source_manifest": str(args.train_source_manifest),
        "train_manifest": str(train_out),
        "train_total": len(train_rows),
        "train_per_subset": {
            subset: subset_summary([row for row in train_rows if row.get("subset") == subset])
            for subset in subsets
        },
        "val_source_manifest": str(args.val_source_manifest) if args.val_source_manifest is not None else None,
        "val_manifest": str(val_out) if val_out is not None else None,
        "val_total": len(val_rows),
        "val_per_subset": {
            subset: subset_summary([row for row in val_rows if row.get("subset") == subset])
            for subset in subsets
        } if val_rows else {},
        "holdout_source_manifest": str(args.holdout_source_manifest) if args.holdout_source_manifest is not None else None,
        "holdout_manifest": str(holdout_out) if holdout_out is not None else None,
        "holdout_total": len(holdout_rows),
        "holdout_per_subset": {
            subset: subset_summary([row for row in holdout_rows if row.get("subset") == subset])
            for subset in subsets
        } if holdout_rows else {},
    }
    summary_out = output_dir / f"{args.name}_summary.json"
    write_summary(summary_out, summary)

    print(f"train manifest: {train_out} ({len(train_rows)} rows)")
    if val_out is not None:
        print(f"val manifest: {val_out} ({len(val_rows)} rows)")
    if holdout_out is not None:
        print(f"holdout manifest: {holdout_out} ({len(holdout_rows)} rows)")
    print(f"summary: {summary_out}")


if __name__ == "__main__":
    main()
