#!/usr/bin/env python3
"""Cache frozen Glyph-ByT5 mapped text features for readable50 diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.ocr_debug.text_feature_cache import TextFeatureCache  # noqa: E402
from tokenizer.tokenizer_image.vq.glyph_byt5 import (  # noqa: E402
    get_text_layer_states,
    load_text_encoder_from_config,
    tokenize_texts_with_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--feature-mode", choices=("glyph_mapped",), default="glyph_mapped")
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def unique_texts(paths: Sequence[Path]) -> List[str]:
    seen = set()
    out: List[str] = []
    for path in paths:
        for row in iter_jsonl(path):
            text = str(row.get("text") or "").strip()
            if text not in seen:
                seen.add(text)
                out.append(text)
    return out


def prepare_device(device_backend: str, device_id: int) -> torch.device:
    if device_backend == "cpu":
        return torch.device("cpu")
    if device_backend == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    if device_backend == "npu":
        import torch_npu  # noqa: F401

        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is not available")
        torch.npu.set_device(device_id)
        return torch.device(f"npu:{device_id}")
    raise ValueError(f"unsupported device backend: {device_backend}")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def autocast_context(device_backend: str, dtype: torch.dtype):
    if dtype == torch.float32 or device_backend == "cpu":
        return nullcontext()
    if device_backend == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if device_backend == "npu":
        npu_amp = getattr(getattr(torch, "npu", None), "amp", None)
        if npu_amp is not None and hasattr(npu_amp, "autocast"):
            return npu_amp.autocast(dtype=dtype)
        return torch.autocast(device_type="npu", dtype=dtype)
    raise ValueError(f"unsupported device backend: {device_backend}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_batch(
        *,
        texts: Sequence[str],
        tokenizer: Any,
        encoder: torch.nn.Module,
        backend: str,
        max_length: int,
        device: torch.device,
        device_backend: str,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    text_inputs, _ = tokenize_texts_with_stats(
        tokenizer,
        texts,
        max_length=max_length,
        device=device,
        backend=backend,
    )
    attention_mask = text_inputs["attention_mask"]
    with torch.no_grad():
        with autocast_context(device_backend, dtype):
            outputs = encoder(
                input_ids=text_inputs["input_ids"],
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
    layer_states = get_text_layer_states(outputs, backend)
    if len(layer_states) != 1:
        raise RuntimeError(f"glyph_mapped cache expects one mapped hidden state, got {len(layer_states)}")
    return layer_states[0].detach(), attention_mask.detach()


def build_cache(args: argparse.Namespace) -> tuple[Path, Path]:
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    text_cfg = config.get("text_conditioning", {})
    if not text_cfg.get("enabled", False):
        raise ValueError("text_conditioning.enabled must be true to build text feature cache")
    backend = str(text_cfg.get("encoder_backend", "t5")).lower()
    if backend != "glyph_byt5":
        raise ValueError(f"Only glyph_byt5 cache is supported, got backend={backend}")

    max_length = int(text_cfg.get("max_length", 128))
    device = prepare_device(args.device_backend, args.device_id)
    dtype = dtype_from_name(args.dtype)
    encoder_result = load_text_encoder_from_config(text_cfg, device)
    if encoder_result is None:
        raise RuntimeError("load_text_encoder_from_config returned None")
    if encoder_result.backend != "glyph_byt5" or int(encoder_result.feature_dim) != 2048 or int(encoder_result.num_layers) != 1:
        raise ValueError(
            "cache supports only frozen Glyph-ByT5 mapped output "
            f"(backend={encoder_result.backend}, feature_dim={encoder_result.feature_dim}, "
            f"num_layers={encoder_result.num_layers})"
        )

    texts = unique_texts([path.expanduser().resolve() for path in args.manifest_jsonl])
    if not texts:
        raise ValueError("No text found in manifest_jsonl inputs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features_by_text: Dict[str, Dict[str, torch.Tensor]] = {}
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start:start + args.batch_size]
        features, attention_mask = encode_batch(
            texts=batch_texts,
            tokenizer=encoder_result.tokenizer,
            encoder=encoder_result.encoder,
            backend=encoder_result.backend,
            max_length=max_length,
            device=device,
            device_backend=args.device_backend,
            dtype=dtype,
        )
        features = features.to("cpu", dtype=dtype)
        attention_mask = attention_mask.to("cpu", dtype=torch.bool)
        for idx, text in enumerate(batch_texts):
            features_by_text[text] = {
                "feature": features[idx].contiguous(),
                "attention_mask": attention_mask[idx].contiguous(),
            }
        print(f"cached {min(start + len(batch_texts), len(texts))}/{len(texts)} texts", flush=True)

    metadata = {
        "backend": "glyph_byt5",
        "feature_mode": args.feature_mode,
        "feature_dim": 2048,
        "num_layers": 1,
        "max_length": max_length,
        "dtype": args.dtype,
        "num_unique_texts": len(texts),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "manifest_jsonl": [str(path.expanduser().resolve()) for path in args.manifest_jsonl],
    }
    cache_path = args.output_dir / "text_features.pt"
    metadata_path = args.output_dir / "metadata.json"
    torch.save({"features_by_text": features_by_text, "metadata": metadata}, cache_path)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return cache_path, metadata_path


def verify_cache(args: argparse.Namespace, cache_path: Path) -> None:
    config_path = args.config.expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    text_cfg = config["text_conditioning"]
    max_length = int(text_cfg.get("max_length", 128))
    device = prepare_device(args.device_backend, args.device_id)
    dtype = dtype_from_name(args.dtype)
    encoder_result = load_text_encoder_from_config(text_cfg, device)
    cache = TextFeatureCache(cache_path)
    cache.validate_config(text_cfg)
    texts = unique_texts([path.expanduser().resolve() for path in args.manifest_jsonl])
    sample_texts = texts[: min(len(texts), args.batch_size)]
    online_features, online_mask = encode_batch(
        texts=sample_texts,
        tokenizer=encoder_result.tokenizer,
        encoder=encoder_result.encoder,
        backend=encoder_result.backend,
        max_length=max_length,
        device=device,
        device_backend=args.device_backend,
        dtype=dtype,
    )
    cached_features, cached_mask = cache.lookup_batch(sample_texts, device=device, dtype=dtype)
    diff = (online_features.to(dtype=dtype).float() - cached_features.float()).abs()
    print("verify:")
    print(f"  texts: {len(sample_texts)}")
    print(f"  max_abs_diff: {float(diff.max().item()):.8g}")
    print(f"  mean_abs_diff: {float(diff.mean().item()):.8g}")
    print(f"  attention_mask_equal: {bool(torch.equal(online_mask.bool(), cached_mask.bool()))}")


def main() -> None:
    args = parse_args()
    cache_path, metadata_path = build_cache(args)
    print(f"wrote cache: {cache_path}")
    print(f"wrote metadata: {metadata_path}")
    if args.verify:
        verify_cache(args, cache_path)


if __name__ == "__main__":
    main()
