#!/usr/bin/env python3
"""Rotate stage-1 checkpoint files without touching protected aliases."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


PROTECTED_NAMES = {"latest.pt", "best.pt", "final.pt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-resume-dir", type=Path)
    parser.add_argument("--model-only-dir", type=Path)
    parser.add_argument("--keep-full", type=int, default=2)
    parser.add_argument("--keep-model", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Default is dry-run.")
    parser.add_argument("--json", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def step_value(path: Path) -> int:
    matches = re.findall(r"(\d+)", path.stem)
    if not matches:
        return -1
    return int(matches[-1])


def is_protected(path: Path) -> bool:
    if path.name in PROTECTED_NAMES:
        return True
    if path.name.endswith(".protected.pt"):
        return True
    if path.with_suffix(path.suffix + ".protected").exists():
        return True
    return False


def checkpoint_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted((p for p in directory.glob("*.pt") if p.is_file() or p.is_symlink()), key=step_value)


def rotate_directory(directory: Path | None, keep_count: int, execute: bool) -> dict:
    if directory is None:
        return {"directory": None, "kept": [], "deleted": [], "would_delete": [], "missing": False}

    result = {
        "directory": str(directory),
        "kept": [],
        "deleted": [],
        "would_delete": [],
        "missing": not directory.exists(),
    }
    if not directory.exists():
        return result

    files = checkpoint_files(directory)
    protected = [p for p in files if is_protected(p)]
    candidates = [p for p in files if not is_protected(p)]
    keep = set(protected + candidates[-keep_count:])

    for path in files:
        if path in keep:
            result["kept"].append(str(path))
            continue
        if execute:
            path.unlink(missing_ok=True)
            result["deleted"].append(str(path))
        else:
            result["would_delete"].append(str(path))

    return result


def print_human(results: Iterable[dict], execute: bool) -> None:
    mode = "EXECUTE" if execute else "DRY-RUN"
    print(f"== rotate_stage1_checkpoints ({mode}) ==")
    for item in results:
        directory = item["directory"]
        if directory is None:
            continue
        print(f"-- {directory}")
        if item["missing"]:
            print("  missing")
            continue
        for path in item["kept"]:
            print(f"  keep        {path}")
        key = "deleted" if execute else "would_delete"
        for path in item[key]:
            action = "delete" if execute else "would delete"
            print(f"  {action:<11} {path}")


def main() -> None:
    args = parse_args()
    results = [
        rotate_directory(args.full_resume_dir, args.keep_full, args.execute),
        rotate_directory(args.model_only_dir, args.keep_model, args.execute),
    ]
    if args.json:
        print(json.dumps({"execute": args.execute, "results": results}, indent=2, ensure_ascii=False))
    else:
        print_human(results, args.execute)


if __name__ == "__main__":
    main()
