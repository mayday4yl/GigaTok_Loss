#!/usr/bin/env python3
"""Prune bulky stage-1 eval outputs while keeping metrics and sample grids."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


KEEP_FILENAMES = {
    "metrics.json",
    "metrics.txt",
    "eval_config.json",
    "sample_grid.png",
    "sample_grid.jpg",
    "sample_grid.jpeg",
    "grid.png",
    "grid.jpg",
    "grid.jpeg",
}

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LARGE_SUFFIXES = IMAGE_SUFFIXES | {".npz", ".npy"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--max-dir-size-gb", type=float, default=1.0)
    parser.add_argument("--execute", action="store_true", help="Actually delete files. Default is dry-run.")
    parser.add_argument("--remove-empty-dirs", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def is_grid_like(path: Path) -> bool:
    name = path.name.lower()
    return "grid" in name and path.suffix.lower() in IMAGE_SUFFIXES


def is_core_keep(path: Path) -> bool:
    return path.name in KEEP_FILENAMES or is_grid_like(path)


def should_delete(path: Path, eval_root: Path, sample_keep: set[Path]) -> bool:
    rel_parts = path.relative_to(eval_root).parts
    if not rel_parts:
        return False
    if rel_parts[0] == "tmp":
        return True
    if is_core_keep(path):
        return False
    if path in sample_keep:
        return False
    if path.suffix.lower() in LARGE_SUFFIXES:
        return True
    return False


def sample_candidates(eval_root: Path) -> list[Path]:
    candidates = []
    for path in eval_root.rglob("*"):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(eval_root).parts
        if rel_parts and rel_parts[0] == "tmp":
            continue
        if is_core_keep(path):
            continue
        if path.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(path)
    return sorted(candidates)


def path_size(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return 0


def total_size(root: Path, deleted: set[Path]) -> int:
    total = 0
    for path in root.rglob("*"):
        if path in deleted:
            continue
        if path.is_file() or path.is_symlink():
            total += path.stat().st_size
    return total


def remove_empty_dirs(root: Path, execute: bool, actions: list[dict]) -> None:
    for directory in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        try:
            next(directory.iterdir())
        except StopIteration:
            actions.append({"action": "rmdir" if execute else "would_rmdir", "path": str(directory)})
            if execute:
                directory.rmdir()


def main() -> None:
    args = parse_args()
    eval_root = args.eval_root.resolve()
    actions: list[dict] = []

    if not eval_root.exists():
        raise SystemExit(f"eval root does not exist: {eval_root}")
    if not eval_root.is_dir():
        raise SystemExit(f"eval root is not a directory: {eval_root}")

    sample_keep = set(sample_candidates(eval_root)[: max(args.max_samples, 0)])
    deleted: set[Path] = set()

    for path in sorted((p for p in eval_root.rglob("*") if p.is_file() or p.is_symlink())):
        if not should_delete(path, eval_root, sample_keep):
            continue
        actions.append({"action": "delete" if args.execute else "would_delete", "path": str(path)})
        deleted.add(path)
        if args.execute:
            path.unlink(missing_ok=True)

    max_bytes = int(args.max_dir_size_gb * 1024 * 1024 * 1024)
    if max_bytes >= 0:
        kept_samples = [p for p in sorted(sample_keep) if p.exists() and p not in deleted]
        while total_size(eval_root, deleted) > max_bytes and kept_samples:
            victim = kept_samples.pop()
            actions.append({
                "action": "delete_size_cap" if args.execute else "would_delete_size_cap",
                "path": str(victim),
            })
            deleted.add(victim)
            if args.execute:
                victim.unlink(missing_ok=True)
        remaining_size = total_size(eval_root, deleted)
        if remaining_size > max_bytes:
            actions.append({
                "action": "size_cap_still_exceeded",
                "path": str(eval_root),
                "bytes": remaining_size,
                "max_bytes": max_bytes,
            })

    for directory in sorted((p for p in eval_root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        rel_parts = directory.relative_to(eval_root).parts
        if rel_parts and rel_parts[0] == "tmp":
            if args.execute:
                shutil.rmtree(directory, ignore_errors=True)
                actions.append({"action": "rmtree", "path": str(directory)})
            else:
                actions.append({"action": "would_rmtree", "path": str(directory)})

    if args.remove_empty_dirs:
        remove_empty_dirs(eval_root, args.execute, actions)

    if args.json:
        print(json.dumps({"execute": args.execute, "actions": actions}, indent=2, ensure_ascii=False))
    else:
        mode = "EXECUTE" if args.execute else "DRY-RUN"
        print(f"== prune_stage1_eval_outputs ({mode}) ==")
        for item in actions:
            print(f"{item['action']:<14} {item['path']}")
        if not actions:
            print("nothing to prune")


if __name__ == "__main__":
    main()
