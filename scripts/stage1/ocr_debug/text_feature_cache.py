#!/usr/bin/env python3
"""Loader for frozen Glyph-ByT5 text feature caches."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


REQUIRED_METADATA = {
    "backend": "glyph_byt5",
    "feature_mode": "glyph_mapped",
    "feature_dim": 2048,
    "num_layers": 1,
}


class TextFeatureCache:
    """CPU-resident lookup table for frozen Glyph-ByT5 mapped features."""

    def __init__(self, cache_path: str | Path):
        self.cache_path = Path(cache_path).expanduser().resolve()
        if not self.cache_path.exists():
            raise FileNotFoundError(f"text feature cache not found: {self.cache_path}")

        payload = torch.load(self.cache_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError(f"Expected mapping payload in {self.cache_path}, got {type(payload)}")

        features_by_text = payload.get("features_by_text")
        if not isinstance(features_by_text, Mapping):
            raise KeyError(f"{self.cache_path} must contain features_by_text mapping")

        metadata_path = self.cache_path.with_name("metadata.json")
        metadata = payload.get("metadata", {})
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise TypeError("Text feature cache metadata must be a mapping")

        self.features_by_text = dict(features_by_text)
        self.metadata = dict(metadata)
        self.metadata_path = metadata_path
        self._validate_payload()

    def _validate_payload(self) -> None:
        for key, expected in REQUIRED_METADATA.items():
            actual = self.metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"text feature cache metadata mismatch for {key}: expected {expected!r}, got {actual!r}"
                )
        if not self.features_by_text:
            raise ValueError(f"text feature cache is empty: {self.cache_path}")

    @property
    def num_unique_texts(self) -> int:
        return len(self.features_by_text)

    @property
    def feature_dim(self) -> int:
        return int(self.metadata["feature_dim"])

    @property
    def max_length(self) -> int:
        return int(self.metadata["max_length"])

    def validate_config(self, text_conditioning_cfg: Mapping[str, Any]) -> None:
        checks = {
            "backend": str(text_conditioning_cfg.get("encoder_backend", "t5")).lower(),
            "feature_dim": int(text_conditioning_cfg.get("glyph_byt5", {}).get("mapper_output_dim", 2048)),
            "max_length": int(text_conditioning_cfg.get("max_length", 128)),
        }
        expected_mode = "glyph_mapped"
        if self.metadata.get("feature_mode") != expected_mode:
            raise ValueError(
                f"Only feature_mode={expected_mode!r} is supported, got {self.metadata.get('feature_mode')!r}"
            )
        for key, actual in checks.items():
            cached = self.metadata.get(key)
            if cached != actual:
                raise ValueError(
                    f"text feature cache/config mismatch for {key}: cache={cached!r}, config={actual!r}"
                )
        if int(self.metadata.get("num_layers", -1)) != 1:
            raise ValueError("cached Glyph-ByT5 mapped output must expose exactly one layer")

    def lookup_batch(
            self,
            texts: Sequence[str],
            device: torch.device | str,
            dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = []
        masks = []
        missing = []
        for text in texts:
            key = str(text)
            item = self.features_by_text.get(key)
            if item is None:
                missing.append(key[:120])
                continue
            feature = item.get("feature") if isinstance(item, Mapping) else None
            attention_mask = item.get("attention_mask") if isinstance(item, Mapping) else None
            if not torch.is_tensor(feature) or not torch.is_tensor(attention_mask):
                raise TypeError(f"Invalid cache entry for text={key[:120]!r}")
            features.append(feature)
            masks.append(attention_mask)
        if missing:
            raise KeyError(
                "Texts missing from text feature cache: "
                + "; ".join(repr(text) for text in missing[:5])
            )
        if not features:
            raise ValueError("lookup_batch received an empty text batch")

        feature_batch = torch.stack(features, dim=0)
        mask_batch = torch.stack(masks, dim=0)
        if dtype is not None:
            feature_batch = feature_batch.to(dtype=dtype)
        return (
            feature_batch.to(device=device, non_blocking=True).detach(),
            mask_batch.to(device=device, non_blocking=True).detach(),
        )
