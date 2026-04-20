#!/usr/bin/env python3
"""Rewrite absolute image paths in TextAtlas stage-1 manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


IMAGE_PATH_FILES = ("train_image_paths.json", "val_image_paths.json")
RICH_MANIFEST_FILES = ("train_manifest.jsonl", "val_manifest.jsonl")
OTHER_MANIFEST_FILES = ("counts.json", "build_config.json", "bad_rows.jsonl")
SHA_IMAGE_PATHS = "image_paths.sha256"
SHA_ALL = "manifest.sha256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite old absolute image prefixes in TextAtlas manifests.")
    parser.add_argument("--manifest-root", type=Path, required=True, help="Existing manifest directory.")
    parser.add_argument("--old-prefix", required=True, help="Old absolute path prefix to replace.")
    parser.add_argument("--new-prefix", required=True, help="New absolute path prefix.")
    parser.add_argument("--output-root", type=Path, default=None, help="Write rewritten manifest files here.")
    parser.add_argument("--in-place", action="store_true", help="Rewrite manifest-root in place.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output-root.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned replacements without writing files.")
    return parser.parse_args()


def normalize_prefix(value: str) -> str:
    return value.rstrip("/")


def replace_prefix(value: str, old_prefix: str, new_prefix: str) -> str:
    if value == old_prefix:
        return new_prefix
    if value.startswith(old_prefix + "/"):
        return new_prefix + value[len(old_prefix) :]
    return value


def rewrite_json_strings(value: Any, old_prefix: str, new_prefix: str) -> Any:
    if isinstance(value, str):
        return replace_prefix(value, old_prefix, new_prefix)
    if isinstance(value, list):
        return [rewrite_json_strings(item, old_prefix, new_prefix) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_json_strings(item, old_prefix, new_prefix) for key, item in value.items()}
    return value


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: JSONL row must be an object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_manifest(root: Path, filenames: Iterable[str], output_name: str) -> None:
    with (root / output_name).open("w", encoding="utf-8") as handle:
        for filename in filenames:
            path = root / filename
            handle.write(f"{sha256_file(path)}  {filename}\n")


def prepare_output_root(args: argparse.Namespace) -> Path:
    source = args.manifest_root.resolve()
    if args.in_place and args.output_root is not None:
        raise SystemExit("--in-place and --output-root are mutually exclusive.")
    if not args.in_place and args.output_root is None:
        raise SystemExit("Pass --in-place or --output-root.")
    if args.in_place:
        return source

    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output-root already exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    shutil.copytree(source, output_root)
    return output_root


def count_replacements_in_text(path: Path, old_prefix: str) -> int:
    return path.read_text(encoding="utf-8").count(old_prefix)


def main() -> None:
    args = parse_args()
    source_root = args.manifest_root.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"manifest-root does not exist: {source_root}")

    old_prefix = normalize_prefix(args.old_prefix)
    new_prefix = normalize_prefix(args.new_prefix)
    required = list(IMAGE_PATH_FILES) + list(RICH_MANIFEST_FILES) + list(OTHER_MANIFEST_FILES)
    missing = [name for name in required if not (source_root / name).exists()]
    if missing:
        raise SystemExit(f"manifest-root missing required files: {missing}")

    planned = {
        name: count_replacements_in_text(source_root / name, old_prefix)
        for name in required
    }
    print(json.dumps({"source_root": str(source_root), "planned_replacements": planned}, indent=2))
    if args.dry_run:
        return

    output_root = prepare_output_root(args)

    for filename in IMAGE_PATH_FILES:
        data = read_json(output_root / filename)
        if not isinstance(data, list):
            raise ValueError(f"{filename} must contain a JSON array.")
        write_json(output_root / filename, rewrite_json_strings(data, old_prefix, new_prefix))

    for filename in RICH_MANIFEST_FILES:
        rows = read_jsonl(output_root / filename)
        for row in rows:
            if "image_path" in row:
                row["image_path"] = replace_prefix(str(row["image_path"]), old_prefix, new_prefix)
        write_jsonl(output_root / filename, rows)

    build_config_path = output_root / "build_config.json"
    build_config = read_json(build_config_path)
    build_config = rewrite_json_strings(build_config, old_prefix, new_prefix)
    if isinstance(build_config, dict):
        build_config["output_root"] = str(output_root)
    write_json(build_config_path, build_config)

    write_sha256_manifest(output_root, IMAGE_PATH_FILES, SHA_IMAGE_PATHS)
    write_sha256_manifest(
        output_root,
        [
            "train_manifest.jsonl",
            "val_manifest.jsonl",
            "train_image_paths.json",
            "val_image_paths.json",
            "counts.json",
            "build_config.json",
            "bad_rows.jsonl",
            SHA_IMAGE_PATHS,
        ],
        SHA_ALL,
    )

    print(json.dumps({"rewritten_manifest_root": str(output_root)}, indent=2))


if __name__ == "__main__":
    main()
