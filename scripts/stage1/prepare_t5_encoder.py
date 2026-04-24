#!/usr/bin/env python3
"""Download or verify the frozen T5 encoder used by v2 text conditioning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare T5 tokenizer and encoder files for text-conditioned GigaTok.")
    parser.add_argument("--config-yaml", type=Path, default=None, help="Read text_conditioning fields from a v2 YAML config.")
    parser.add_argument("--encoder-name", default=None, help="Hugging Face model id or local directory.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def load_text_conditioning_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: PyYAML is required for --config-yaml.") from exc
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return dict((config or {}).get("text_conditioning") or {})


def main() -> None:
    args = parse_args()
    cfg = load_text_conditioning_config(args.config_yaml)
    encoder_name = args.encoder_name or cfg.get("encoder_name") or "google/t5-v1_1-xl"
    cache_dir = args.cache_dir if args.cache_dir is not None else cfg.get("cache_dir")
    local_files_only = bool(args.local_files_only or cfg.get("local_files_only", False))

    try:
        from transformers import AutoTokenizer, T5EncoderModel  # type: ignore
    except ImportError as exc:
        raise SystemExit("Missing dependency: transformers.") from exc

    kwargs = {
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, **kwargs)
    encoder = T5EncoderModel.from_pretrained(encoder_name, **kwargs)
    summary = {
        "encoder_name": encoder_name,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "tokenizer_class": tokenizer.__class__.__name__,
        "encoder_class": encoder.__class__.__name__,
        "d_model": getattr(encoder.config, "d_model", None),
        "num_layers": getattr(encoder.config, "num_layers", None),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
