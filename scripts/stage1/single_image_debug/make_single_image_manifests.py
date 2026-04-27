#!/usr/bin/env python3
"""从 TextAtlas train manifest 中抽取单图调试 manifest。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


DEFAULT_TARGETS = {
    "dense": "CleanTextSynth:train:2731",
    "medium": "TextScenesHQ:train:2803",
    "sparse": "LongWordsSubset-A:train:3149",
}


def parse_target(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--target must be NAME=SOURCE_KEY")
    name, source_key = value.split("=", 1)
    name = name.strip()
    source_key = source_key.strip()
    if not name or not source_key:
        raise argparse.ArgumentTypeError("--target must be NAME=SOURCE_KEY")
    return name, source_key


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target",
        action="append",
        type=parse_target,
        default=None,
        help="NAME=SOURCE_KEY. Can be passed multiple times. Defaults to dense/medium/sparse.",
    )
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


def main() -> None:
    args = parse_args()
    targets = dict(args.target) if args.target else dict(DEFAULT_TARGETS)
    wanted = set(targets.values())
    found: Dict[str, Dict[str, Any]] = {}

    for row in iter_jsonl(args.source_manifest):
        source_key = str(row.get("source_key") or "")
        if source_key in wanted:
            found[source_key] = row
            if len(found) == len(wanted):
                break

    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(f"Missing source_key(s) in {args.source_manifest}: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, source_key in targets.items():
        row = dict(found[source_key])
        line_no = row.pop("_manifest_line_no")
        out_path = args.output_dir / f"{name}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
        summary[name] = {
            "manifest": str(out_path),
            "source_key": source_key,
            "source_manifest_line_no": line_no,
            "subset": row.get("subset"),
            "image_path": row.get("image_path"),
            "text": row.get("text"),
            "text_chars": len(str(row.get("text") or "")),
            "text_extraction_status": row.get("text_extraction_status"),
        }
        print(f"{name}: {out_path}")
        print(f"  source_key: {source_key}")
        print(f"  image_path: {row.get('image_path')}")
        print(f"  text: {row.get('text')}")

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

