#!/usr/bin/env python3
"""Evaluate baseline vs HR tokenizer reconstruction on local TextAtlas images."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import tempfile
import unicodedata
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_init import custom_load, load_model_from_config  # noqa: E402
from tokenizer.tokenizer_image.vq.glyph_byt5 import (  # noqa: E402
    get_text_layer_states,
    load_text_encoder_from_config,
    tokenize_texts_with_stats,
)


try:
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:  # pragma: no cover - optional dependency on some servers
    ssim_metric = None


RunSpec = Tuple[str, Path, Path]


TEXT_CONDITIONING_MISSING_PREFIXES = ("text_projection.",)
TEXT_CONDITIONING_MISSING_NAMES = {
    "text_type_embedding",
    "visual_type_embedding",
    "text_gate_logit",
    "visual_mask_token",
    "residual_head_gate",
    "residual_gate",
}
TEXT_CONDITIONING_MISSING_PREFIXES = TEXT_CONDITIONING_MISSING_PREFIXES + (
    "residual_head_mlp.",
    "residual_text_mlp.",
    "adaln_mlps.",
    "s1to2decoder.residual_cross_attn_layers.",
    "s1to2decoder.residual_cross_attn_projs.",
    "s1to2decoder.residual_cross_attn_scales.",
)


def parse_run(value: str) -> RunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run must be NAME:CONFIG:CKPT")
    name, config, ckpt = parts
    if not name:
        raise argparse.ArgumentTypeError("run NAME must not be empty")
    return name, Path(config), Path(ckpt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME:CONFIG:CKPT. Pass twice for baseline and HR.")
    parser.add_argument("--image-paths-json", type=Path, default=None, help="Optional JSON list of image paths. If omitted, paths are read from --manifest-jsonl.")
    parser.add_argument("--manifest-jsonl", type=Path, default=None, help="Optional materialized val manifest for subset labels and rendered text.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--max-images", type=int, default=10_000, help="0 means all images.")
    parser.add_argument("--grid-samples", type=int, default=24)
    parser.add_argument("--grid-seed", type=int, default=0)
    parser.add_argument("--save-per-sample", type=int, default=0, help="Save first N comparison rows as individual PNGs.")
    parser.add_argument("--ocr-backend", choices=("none", "pytesseract", "paddleocr", "easyocr", "deepseek_ocr"), default="none")
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.0)
    parser.add_argument("--ocr-jsonl", type=Path, default=None, help="Optional per-sample OCR prediction output JSONL.")
    parser.add_argument("--deepseek-ocr-model", default=None, help="Local path or HF id for --ocr-backend=deepseek_ocr. Defaults to DEEPSEEK_OCR_MODEL or deepseek-ai/DeepSeek-OCR.")
    parser.add_argument("--deepseek-ocr-prompt", default="<image>\n<|grounding|>OCR this image.", help="Prompt passed to DeepSeek-OCR infer().")
    parser.add_argument("--deepseek-ocr-base-size", type=int, default=1024)
    parser.add_argument("--deepseek-ocr-image-size", type=int, default=640)
    parser.add_argument("--deepseek-ocr-crop-mode", action="store_true", help="Enable DeepSeek-OCR crop_mode.")
    parser.add_argument("--deepseek-ocr-save-results", action="store_true", help="Let DeepSeek-OCR write its own visualized outputs.")
    parser.add_argument("--deepseek-ocr-test-compress", action="store_true", help="Pass test_compress=True to DeepSeek-OCR infer() when supported.")
    parser.add_argument("--deepseek-ocr-attn-implementation", default="eager", help="Transformers attention implementation, e.g. eager, sdpa, flash_attention_2, or none.")
    parser.add_argument("--deepseek-ocr-dtype", choices=("auto", "fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--deepseek-ocr-output-dir", type=Path, default=None, help="Temporary/output dir used by DeepSeek-OCR infer().")
    parser.add_argument("--deepseek-ocr-max-new-tokens", type=int, default=1024, help="Cap DeepSeek-OCR generation length by wrapping generate().")
    parser.add_argument("--text-layer-pair-index", type=int, default=0, help="Deterministic [T5 layer, decoder layer] pair index for text-conditioned reconstruction.")
    parser.add_argument("--checkpoint-weight-key", choices=("auto", "model", "ema", "state_dict"), default="auto", help="Checkpoint weights to evaluate. auto preserves the historical preference: ema, then model, then state_dict.")
    parser.add_argument("--text-input-mode", choices=("correct", "empty", "shuffled", "wrong", "length_matched_wrong"), default="correct", help="Text used by text-conditioned runs; use empty/shuffled/length_matched_wrong for sensitivity checks.")
    parser.add_argument("--wrong-text-seed", type=int, default=0, help="Seed for --text-input-mode=shuffled.")
    parser.add_argument("--fixed-wrong-text", default="THIS IS A FIXED WRONG TEXT 0123456789", help="Text used for --text-input-mode=wrong.")
    parser.add_argument("--strip-punctuation", action="store_true", help="Remove punctuation before CER/NED/exact-match metrics.")
    parser.add_argument("--remove-spaces", action="store_true", help="Remove spaces before CER/NED/exact-match metrics.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_color(value: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3:
        raise ValueError("--pad-color must be one integer or R,G,B")
    color = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("--pad-color channels must be in [0, 255]")
    return color  # type: ignore[return-value]


def resize_pad_image(img: Image.Image, image_size: int, pad_color: Tuple[int, int, int]) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    if width == image_size and height == image_size:
        return img
    scale = min(image_size / width, image_size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = img.resize(resized_size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), pad_color)
    canvas.paste(resized, ((image_size - resized_size[0]) // 2, (image_size - resized_size[1]) // 2))
    return canvas


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(127.5 * x + 128.0, 0, 255)
    return x.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


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


def autocast_context(device_backend: str, mixed_precision: str, dtype: Optional[torch.dtype] = None):
    if mixed_precision == "none" or device_backend == "cpu":
        return nullcontext()
    if device_backend == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if device_backend == "npu":
        npu_amp = getattr(getattr(torch, "npu", None), "amp", None)
        if npu_amp is not None and hasattr(npu_amp, "autocast"):
            return npu_amp.autocast(dtype=dtype)
        return torch.autocast(device_type="npu", dtype=dtype)
    raise ValueError(f"unsupported device backend: {device_backend}")


def dtype_from_mixed_precision(mixed_precision: str) -> Optional[torch.dtype]:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def load_manifest_metadata(manifest_jsonl: Path | None) -> Tuple[List[Path], Dict[str, Dict[str, str]]]:
    if manifest_jsonl is None or not manifest_jsonl.exists():
        return [], {}
    manifest_dir = manifest_jsonl.resolve().parent
    image_paths: List[Path] = []
    metadata_by_path: Dict[str, Dict[str, str]] = {}
    for row in iter_jsonl(manifest_jsonl):
        image_path = row.get("image_path")
        if not image_path:
            continue
        path = Path(str(image_path))
        if not path.is_absolute():
            path = manifest_dir / path
        image_paths.append(path)
        metadata_by_path[str(path.resolve())] = {
            "subset": str(row.get("subset") or "unknown"),
            "text": str(row.get("text") or ""),
            "source_key": str(row.get("source_key") or ""),
            "text_extraction_status": str(row.get("text_extraction_status") or ""),
        }
    return image_paths, metadata_by_path


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def text_conditioning_missing_keys(model: torch.nn.Module) -> List[str]:
    missing_keys = []
    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if name in TEXT_CONDITIONING_MISSING_NAMES or name.startswith(TEXT_CONDITIONING_MISSING_PREFIXES):
            missing_keys.append(name)
    return missing_keys


def load_text_encoder(config: Mapping[str, Any], device: torch.device):
    text_cfg = config.get("text_conditioning", {})
    result = load_text_encoder_from_config(text_cfg, device)
    if result is None:
        return None, None, None, None
    return result.tokenizer, result.encoder, int(result.feature_dim), result.num_layers


def build_layer_pairs_from_cfg(
        cfg: Mapping[str, Any],
        decoder_num_layers: int,
        text_num_layers: Optional[int],
        config_name: str,
) -> List[Tuple[int, int]]:
    raw_pairs = cfg.get("layer_pairs", None)
    if not raw_pairs:
        shared_layers = min(decoder_num_layers, text_num_layers) if text_num_layers is not None else decoder_num_layers
        start = shared_layers // 3
        end = min(shared_layers, start + max(1, shared_layers // 3))
        raw_pairs = [[layer_idx, layer_idx] for layer_idx in range(start, end)]

    layer_pairs = []
    for pair in raw_pairs:
        if len(pair) != 2:
            raise ValueError(f"{config_name}.layer_pairs entries must be [t5_layer, decoder_layer], got {pair}")
        text_layer, decoder_layer = int(pair[0]), int(pair[1])
        if decoder_layer < 0 or decoder_layer >= decoder_num_layers:
            raise ValueError(
                f"decoder_layer={decoder_layer} is out of range for decoder_num_layers={decoder_num_layers}"
            )
        if text_num_layers is not None and (text_layer < 0 or text_layer >= text_num_layers):
            raise ValueError(f"text_layer={text_layer} is out of range for text_num_layers={text_num_layers}")
        layer_pairs.append((text_layer, decoder_layer))
    if not layer_pairs:
        raise ValueError("No text layer pairs selected.")
    return layer_pairs


def build_text_layer_pairs(config: Mapping[str, Any], decoder_num_layers: int, text_num_layers: Optional[int]) -> List[Tuple[int, int]]:
    return build_layer_pairs_from_cfg(config.get("text_hr", {}), decoder_num_layers, text_num_layers, "text_hr")


def build_text_recon_layer_pairs(config: Mapping[str, Any], decoder_num_layers: int, text_num_layers: Optional[int]) -> List[Tuple[int, int]]:
    text_recon_cfg = config.get("text_recon_conditioning", {})
    return build_layer_pairs_from_cfg(
        text_recon_cfg, decoder_num_layers, text_num_layers, "text_recon_conditioning")


def validate_text_recon_config(config: Mapping[str, Any]) -> None:
    text_recon_cfg = config.get("text_recon_conditioning", {})
    if not bool(text_recon_cfg.get("enabled", False)):
        return
    mode = str(text_recon_cfg.get("mode", "concat_memory"))
    visual_memory_mask_cfg = text_recon_cfg.get("visual_memory_mask", {})
    visual_memory_mask_enabled = bool(visual_memory_mask_cfg.get("enabled", False))
    if mode not in {
        "concat_memory",
        "concat_memory_visual_mask",
        "residual_head",
        "residual_pooled_layer",
        "adaln",
        "residual_cross_attn_visual_mask",
    }:
        raise NotImplementedError(
            "Only text_recon_conditioning.mode=concat_memory, concat_memory_visual_mask, "
            "residual_head, residual_pooled_layer, adaln, or residual_cross_attn_visual_mask "
            "is implemented in the current commit."
        )
    visual_mask_modes = {"concat_memory_visual_mask", "residual_cross_attn_visual_mask"}
    if mode not in visual_mask_modes and visual_memory_mask_enabled:
        raise ValueError(
            "text_recon_conditioning.visual_memory_mask.enabled must be false unless "
            "mode=concat_memory_visual_mask or residual_cross_attn_visual_mask."
        )
    if mode in visual_mask_modes and not visual_memory_mask_enabled:
        raise ValueError(f"text_recon_conditioning.visual_memory_mask.enabled must be true for mode={mode}.")
    if mode in {
        "residual_head",
        "residual_pooled_layer",
        "adaln",
    } and bool(config.get("text_hr", {}).get("enabled", False)):
        raise ValueError(f"text_recon_conditioning.mode={mode} requires text_hr.enabled=false.")
    if mode == "residual_cross_attn_visual_mask" and bool(config.get("text_hr", {}).get("enabled", False)):
        configured_image_token_len = int(
            config.get("text_hr", {}).get(
                "image_token_len",
                config.get("model", {}).get("init_args", {}).get("num_latent_tokens", 256),
            )
        )
        if configured_image_token_len != 0:
            raise ValueError(
                "residual_cross_attn_visual_mask text_hr uses text-only branch attention, "
                "so text_hr.image_token_len must be 0."
            )
    if visual_memory_mask_enabled:
        if str(visual_memory_mask_cfg.get("mode", "learned_mask_token")) != "learned_mask_token":
            raise NotImplementedError("Only visual_memory_mask.mode=learned_mask_token is implemented.")
        ratio = float(visual_memory_mask_cfg.get("ratio", 0.0))
        if ratio < 0.0 or ratio >= 1.0:
            raise ValueError(f"text_recon_conditioning.visual_memory_mask.ratio must be in [0, 1), got {ratio}")
        strategy = str(visual_memory_mask_cfg.get("strategy", "token_random"))
        if strategy not in {"token_random", "block_random"}:
            raise NotImplementedError("Only visual_memory_mask.strategy=token_random or block_random is implemented.")
        if int(visual_memory_mask_cfg.get("block_size", 1)) <= 0:
            raise ValueError("text_recon_conditioning.visual_memory_mask.block_size must be positive.")
    if mode != "residual_head" and not text_recon_cfg.get("layers", None) and not text_recon_cfg.get("layer_pairs", None):
        raise ValueError("text_recon_conditioning.enabled=True requires non-empty layers or layer_pairs.")
    for block_name in ("residual",):
        block_cfg = text_recon_cfg.get(block_name, {})
        if isinstance(block_cfg, Mapping) and bool(block_cfg.get("enabled", False)):
            raise NotImplementedError(f"text_recon_conditioning.{block_name} is planned but not implemented.")
    adaln_cfg = text_recon_cfg.get("adaln", {})
    if mode != "adaln" and isinstance(adaln_cfg, Mapping) and bool(adaln_cfg.get("enabled", False)):
        raise NotImplementedError("text_recon_conditioning.adaln is only implemented for mode=adaln.")


class TextContext:
    def __init__(
            self,
            tokenizer: Any,
            encoder: torch.nn.Module,
            layer_pairs: Sequence[Tuple[int, int]],
            text_recon_layer_pairs: Optional[Sequence[Tuple[int, int]]],
            text_injection_layers: Optional[Sequence[int]],
            text_recon_mode: Optional[str],
            head_text_layer: Optional[int],
            visual_memory_mask_enabled: bool,
            visual_memory_mask_ratio: float,
            visual_memory_mask_strategy: str,
            visual_memory_mask_block_size: int,
            visual_memory_mask_fixed_pattern: bool,
            visual_memory_mask_seed: int,
            visual_memory_mask_apply_in_eval: bool,
            max_length: int,
            pair_index: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.layer_pairs = list(layer_pairs)
        self.text_recon_layer_pairs = list(text_recon_layer_pairs or [])
        self.text_injection_layers = list(text_injection_layers or [])
        self.text_recon_mode = text_recon_mode
        self.head_text_layer = head_text_layer
        self.visual_memory_mask_enabled = visual_memory_mask_enabled
        self.visual_memory_mask_ratio = visual_memory_mask_ratio
        self.visual_memory_mask_strategy = visual_memory_mask_strategy
        self.visual_memory_mask_block_size = visual_memory_mask_block_size
        self.visual_memory_mask_fixed_pattern = visual_memory_mask_fixed_pattern
        self.visual_memory_mask_seed = visual_memory_mask_seed
        self.visual_memory_mask_apply_in_eval = visual_memory_mask_apply_in_eval
        self.max_length = max_length
        if pair_index < 0 or pair_index >= len(self.layer_pairs):
            raise ValueError(f"--text-layer-pair-index={pair_index} out of range for {len(self.layer_pairs)} pairs")
        self.pair_index = pair_index

    @property
    def selected_pair(self) -> Tuple[int, int]:
        return self.layer_pairs[self.pair_index]


def load_tokenizer_model(
        config_path: Path,
        ckpt_path: Path,
        device: torch.device,
        text_layer_pair_index: int,
        checkpoint_weight_key: str = "auto",
) -> Tuple[torch.nn.Module, Mapping[str, Any], Optional[TextContext]]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_text_recon_config(config)

    tokenizer, text_encoder, text_feature_dim, text_num_layers = load_text_encoder(config, device)
    model = load_model_from_config(config)
    if config.get("text_conditioning", {}).get("enabled", False):
        text_recon_cfg = config.get("text_recon_conditioning", {})
        text_recon_on = bool(text_recon_cfg.get("enabled", False))
        text_recon_mode = str(text_recon_cfg.get("mode", "concat_memory"))
        visual_memory_mask_cfg = text_recon_cfg.get("visual_memory_mask", {})
        text_gate_cfg = text_recon_cfg.get("text_gate", {}) if text_recon_on else {}
        residual_head_cfg = text_recon_cfg.get("residual_head", {}) if text_recon_on else {}
        residual_pooled_cfg = text_recon_cfg.get("residual_pooled_layer", {}) if text_recon_on else {}
        adaln_cfg = text_recon_cfg.get("adaln", {}) if text_recon_on else {}
        residual_cross_attn_cfg = text_recon_cfg.get("residual_cross_attn", {}) if text_recon_on else {}
        concat_memory_mode = text_recon_mode in {"concat_memory", "concat_memory_visual_mask"}
        adaln_layers = None
        residual_cross_attn_layers = None
        if text_recon_on and text_recon_mode == "adaln":
            raw_layers = text_recon_cfg.get("layers", None)
            if not raw_layers:
                raw_layers = [pair[1] for pair in text_recon_cfg.get("layer_pairs", [])]
            adaln_layers = [int(layer_idx) for layer_idx in raw_layers]
        if text_recon_on and text_recon_mode == "residual_cross_attn_visual_mask":
            raw_layers = text_recon_cfg.get("layers", None)
            if not raw_layers:
                raw_layers = [pair[1] for pair in text_recon_cfg.get("layer_pairs", [])]
            residual_cross_attn_layers = [int(layer_idx) for layer_idx in raw_layers]
        model.configure_text_conditioning(
            text_feature_dim=text_feature_dim,
            text_projection=config.get("text_conditioning", {}).get("text_projection", "linear_layernorm"),
            text_type_embedding=config.get("text_conditioning", {}).get("text_type_embedding", True),
            visual_type_embedding=text_recon_on and concat_memory_mode and bool(text_recon_cfg.get("visual_type_embedding", False)),
            text_gate_enabled=text_recon_on and concat_memory_mode and bool(text_gate_cfg.get("enabled", False)),
            text_gate_init=float(text_gate_cfg.get("init", 0.1)),
            visual_memory_mask_enabled=(
                text_recon_on and text_recon_mode in {"concat_memory_visual_mask", "residual_cross_attn_visual_mask"}
            ),
            text_recon_mode=text_recon_mode if text_recon_on else None,
            residual_head_gate_init=float(residual_head_cfg.get("gate_init", 1e-3)),
            residual_head_mlp_hidden_mult=float(residual_head_cfg.get("mlp_hidden_mult", 4.0)),
            residual_gate_init=float(residual_pooled_cfg.get("gate_init", 1e-3)),
            residual_mlp_hidden_mult=float(residual_pooled_cfg.get("mlp_hidden_mult", 4.0)),
            adaln_layers=adaln_layers,
            adaln_mlp_hidden_mult=float(adaln_cfg.get("mlp_hidden_mult", 4.0)),
            adaln_zero_init_last=bool(adaln_cfg.get("zero_init_last", True)),
            residual_cross_attn_layers=residual_cross_attn_layers,
            residual_cross_attn_scale_init=residual_cross_attn_cfg.get("scale_init", None),
            residual_cross_attn_scale_learnable=bool(residual_cross_attn_cfg.get("scale_learnable", True)),
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if checkpoint_weight_key != "auto":
        if checkpoint_weight_key not in checkpoint:
            raise KeyError(f"checkpoint weight key {checkpoint_weight_key!r} not found in {ckpt_path}")
        weights = checkpoint[checkpoint_weight_key]
    elif "ema" in checkpoint:
        weights = checkpoint["ema"]
    elif "model" in checkpoint:
        weights = checkpoint["model"]
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    else:
        raise KeyError(f"cannot find model weights in {ckpt_path}")
    ignore_missing = text_conditioning_missing_keys(model) if config.get("text_conditioning", {}).get("enabled", False) else None
    custom_load(model, strip_module_prefix(weights), ignore_missing_keys=ignore_missing)
    del checkpoint
    model.to(device)
    model.eval()
    text_context = None
    if tokenizer is not None and text_encoder is not None:
        decoder_num_layers = int(getattr(model.s1to2decoder, "num_layers"))
        layer_pairs = build_text_layer_pairs(config, decoder_num_layers, text_num_layers)
        text_recon_layer_pairs = None
        text_injection_layers = None
        text_recon_mode = None
        head_text_layer = None
        visual_memory_mask_cfg = {}
        if bool(config.get("text_recon_conditioning", {}).get("enabled", False)):
            text_recon_cfg = config.get("text_recon_conditioning", {})
            text_recon_mode = str(text_recon_cfg.get("mode", "concat_memory"))
            visual_memory_mask_cfg = text_recon_cfg.get("visual_memory_mask", {})
            if text_recon_mode == "residual_head":
                head_text_layer = int(text_recon_cfg.get("head_text_layer", 15))
                if text_num_layers is not None and (head_text_layer < 0 or head_text_layer >= text_num_layers):
                    raise ValueError(
                        f"text_recon_conditioning.head_text_layer={head_text_layer} is out of range "
                        f"for T5 num_layers={text_num_layers}"
                    )
            else:
                text_recon_layer_pairs = build_text_recon_layer_pairs(config, decoder_num_layers, text_num_layers)
                text_injection_layers = [decoder_layer for _, decoder_layer in text_recon_layer_pairs]
        text_context = TextContext(
            tokenizer=tokenizer,
            encoder=text_encoder,
            layer_pairs=layer_pairs,
            text_recon_layer_pairs=text_recon_layer_pairs,
            text_injection_layers=text_injection_layers,
            text_recon_mode=text_recon_mode,
            head_text_layer=head_text_layer,
            visual_memory_mask_enabled=bool(visual_memory_mask_cfg.get("enabled", False)),
            visual_memory_mask_ratio=float(visual_memory_mask_cfg.get("ratio", 0.0)),
            visual_memory_mask_strategy=str(visual_memory_mask_cfg.get("strategy", "token_random")),
            visual_memory_mask_block_size=int(visual_memory_mask_cfg.get("block_size", 1)),
            visual_memory_mask_fixed_pattern=bool(visual_memory_mask_cfg.get("fixed_pattern", False)),
            visual_memory_mask_seed=int(visual_memory_mask_cfg.get("seed", 0)),
            visual_memory_mask_apply_in_eval=bool(visual_memory_mask_cfg.get("apply_in_eval", False)),
            max_length=int(config.get("text_conditioning", {}).get("max_length", 128)),
            pair_index=text_layer_pair_index,
        )
    return model, config, text_context


def reconstruct_batch(
        model: torch.nn.Module,
        batch: torch.Tensor,
        texts: Optional[Sequence[str]],
        text_context: Optional[TextContext],
        device_backend: str,
        mixed_precision: str,
) -> torch.Tensor:
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    with torch.inference_mode():
        if text_context is None:
            latent, _, info = model.encode(batch)
            indices = info[2]
            return model.decode_code(indices, latent.shape)
        if texts is None:
            raise ValueError("Text-conditioned reconstruction requires rendered text from --manifest-jsonl.")

        selected_text_layer, selected_decoder_layer = text_context.selected_pair
        text_backend = getattr(text_context.encoder, "encoder_backend", "t5")
        text_inputs, _ = tokenize_texts_with_stats(
            text_context.tokenizer,
            [str(text) for text in texts],
            max_length=text_context.max_length,
            device=batch.device,
            backend=text_backend,
        )
        input_ids = text_inputs["input_ids"]
        text_attention_mask = text_inputs["attention_mask"]
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            text_outputs = text_context.encoder(
                input_ids=input_ids,
                attention_mask=text_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        t5_layer_states = get_text_layer_states(text_outputs, text_backend)
        decoder_text_key_padding_mask = ~text_attention_mask.bool()
        decoder_text_features = None
        decoder_text_features_by_layer = None
        decoder_head_text_features = None
        call_selected_decoder_layer = selected_decoder_layer
        if text_context.text_recon_mode == "residual_head":
            head_text_layer = int(text_context.head_text_layer)
            if head_text_layer >= len(t5_layer_states):
                raise ValueError(
                    f"head_text_layer={head_text_layer} is out of range for "
                    f"T5 layer outputs={len(t5_layer_states)}"
                )
            decoder_head_text_features = t5_layer_states[head_text_layer].detach()
            call_selected_decoder_layer = None
        elif text_context.text_recon_layer_pairs:
            decoder_text_features_by_layer = {}
            for text_layer, decoder_layer in text_context.text_recon_layer_pairs:
                if text_layer >= len(t5_layer_states):
                    raise ValueError(
                        f"text_layer={text_layer} is out of range for T5 layer outputs={len(t5_layer_states)}"
                    )
                decoder_text_features_by_layer[int(decoder_layer)] = t5_layer_states[text_layer].detach()
            call_selected_decoder_layer = None
        else:
            if selected_text_layer >= len(t5_layer_states):
                raise ValueError(
                    f"selected_text_layer={selected_text_layer} is out of range for "
                    f"T5 layer outputs={len(t5_layer_states)}"
                )
            decoder_text_features = t5_layer_states[selected_text_layer].detach()
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            outputs = model(
                batch,
                selected_decoder_layer=call_selected_decoder_layer,
                decoder_text_features=decoder_text_features,
                decoder_text_features_by_layer=decoder_text_features_by_layer,
                decoder_head_text_features=decoder_head_text_features,
                decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                text_injection_layers=text_context.text_injection_layers or None,
                visual_memory_mask_enabled=text_context.visual_memory_mask_enabled,
                visual_memory_mask_ratio=text_context.visual_memory_mask_ratio,
                visual_memory_mask_strategy=text_context.visual_memory_mask_strategy,
                visual_memory_mask_block_size=text_context.visual_memory_mask_block_size,
                visual_memory_mask_fixed_pattern=text_context.visual_memory_mask_fixed_pattern,
                visual_memory_mask_seed=text_context.visual_memory_mask_seed,
                visual_memory_mask_apply_in_eval=text_context.visual_memory_mask_apply_in_eval,
            )
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def metric_values(gt: np.ndarray, rec: np.ndarray) -> Dict[str, float]:
    gt_f = gt.astype(np.float32) / 255.0
    rec_f = rec.astype(np.float32) / 255.0
    diff = rec_f - gt_f
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 0 else float(-10.0 * math.log10(mse))
    if ssim_metric is None:
        ssim = float("nan")
    else:
        ssim = float(ssim_metric(gt_f, rec_f, channel_axis=-1, data_range=1.0))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def normalize_text_for_metric(
        text: str,
        *,
        lowercase: bool,
        strip_punctuation: bool,
        remove_spaces: bool,
) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"\s+", " ", text)
    if strip_punctuation:
        text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        text = re.sub(r"\s+", " ", text).strip()
    if remove_spaces:
        text = re.sub(r"\s+", "", text)
    if lowercase:
        text = text.lower()
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def text_metric_values(
        gt_text: str,
        pred_text: str,
        *,
        strip_punctuation: bool,
        remove_spaces: bool,
) -> Dict[str, float]:
    gt_exact = normalize_text_for_metric(
        gt_text,
        lowercase=False,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    pred_exact = normalize_text_for_metric(
        pred_text,
        lowercase=False,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    gt_ci = normalize_text_for_metric(
        gt_text,
        lowercase=True,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    pred_ci = normalize_text_for_metric(
        pred_text,
        lowercase=True,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    edit = levenshtein_distance(pred_ci, gt_ci)
    cer = edit / max(1, len(gt_ci))
    ned_similarity = 1.0 - edit / max(1, len(gt_ci), len(pred_ci))
    return {
        "ocr_exact_acc": float(pred_exact == gt_exact),
        "ocr_exact_acc_ci": float(pred_ci == gt_ci),
        "ocr_cer": float(cer),
        "ocr_ned_similarity": float(ned_similarity),
        "ocr_edit_distance": float(edit),
        "ocr_gt_chars": float(len(gt_ci)),
        "ocr_pred_chars": float(len(pred_ci)),
    }


def extract_paddle_texts(result: Any, min_confidence: float) -> List[str]:
    texts: List[str] = []
    if result is None:
        return texts
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[0], str):
        score = float(result[1]) if isinstance(result[1], (int, float)) else 1.0
        if score >= min_confidence:
            texts.append(result[0])
        return texts
    if (
        isinstance(result, (list, tuple))
        and len(result) >= 2
        and isinstance(result[1], tuple)
        and len(result[1]) >= 2
        and isinstance(result[1][0], str)
    ):
        score = float(result[1][1]) if isinstance(result[1][1], (int, float)) else 1.0
        if score >= min_confidence:
            texts.append(result[1][0])
        return texts
    if isinstance(result, (list, tuple)):
        for item in result:
            texts.extend(extract_paddle_texts(item, min_confidence))
    return texts


def clean_deepseek_ocr_text(text: str) -> str:
    text = re.sub(r"<\|ref\|>(.*?)<\|/ref\|>", r" \1 ", text, flags=re.DOTALL)
    text = re.sub(r"<\|det\|>.*?(?:<\|/det\|>|$)", " ", text, flags=re.DOTALL)
    text = re.sub(r"\[\[[0-9,\\s\\.-]*(?:\]\])?", " ", text)
    text = re.sub(r"</?s>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def stringify_deepseek_ocr_result(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return clean_deepseek_ocr_text(result)
    if isinstance(result, Mapping):
        for key in ("text", "result", "markdown", "output", "prediction", "pred_text"):
            if key in result:
                text = stringify_deepseek_ocr_result(result[key])
                if text:
                    return text
        return " ".join(str(value) for value in result.values() if value is not None)
    if isinstance(result, (list, tuple)):
        return " ".join(stringify_deepseek_ocr_result(item) for item in result if item is not None).strip()
    return clean_deepseek_ocr_text(str(result))


def read_deepseek_saved_text(output_dir: Path) -> str:
    candidates = [
        output_dir / "result.mmd",
        output_dir / "result_ori.mmd",
    ]
    candidates.extend(sorted(output_dir.glob("*.mmd"), key=lambda path: path.stat().st_mtime, reverse=True))
    for path in candidates:
        if path.exists() and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            text = re.sub(r"<\|det\|>.*?<\|/det\|>", " ", text, flags=re.DOTALL)
            text = clean_deepseek_ocr_text(text)
            if text:
                return text
    return ""


def enable_deepseek_ocr_npu_cuda_compat() -> None:
    if getattr(torch, "_deepseek_ocr_npu_cuda_compat", False):
        return
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - server optional dependency
        raise RuntimeError("DeepSeek-OCR NPU mode requires torch_npu because upstream infer() calls .cuda().") from exc

    original_autocast = torch.autocast

    def tensor_cuda_to_npu(self: torch.Tensor, device: Any = None, non_blocking: bool = False, **kwargs: Any) -> torch.Tensor:
        return self.npu(non_blocking=non_blocking)

    def module_cuda_to_npu(self: torch.nn.Module, device: Any = None) -> torch.nn.Module:
        return self.npu()

    def autocast_redirect(device_type: str, *args: Any, **kwargs: Any) -> Any:
        if device_type == "cuda":
            device_type = "npu"
        return original_autocast(device_type, *args, **kwargs)

    torch.Tensor.cuda = tensor_cuda_to_npu  # type: ignore[assignment]
    torch.nn.Module.cuda = module_cuda_to_npu  # type: ignore[assignment]
    torch.autocast = autocast_redirect  # type: ignore[assignment]
    torch._deepseek_ocr_npu_cuda_compat = True  # type: ignore[attr-defined]


def cap_model_generate(model: torch.nn.Module, max_new_tokens: int) -> None:
    if max_new_tokens <= 0 or getattr(model, "_deepseek_ocr_generate_capped", False):
        return
    original_generate = model.generate

    def generate_with_cap(*args: Any, **kwargs: Any) -> Any:
        current = kwargs.get("max_new_tokens")
        if current is None:
            kwargs["max_new_tokens"] = max_new_tokens
        else:
            kwargs["max_new_tokens"] = min(int(current), max_new_tokens)
        return original_generate(*args, **kwargs)

    model.generate = generate_with_cap  # type: ignore[assignment]
    model._deepseek_ocr_generate_capped = True  # type: ignore[attr-defined]


class OCRReader:
    def __init__(
        self,
        backend: str,
        lang: str,
        min_confidence: float,
        device_backend: str,
        device: torch.device,
        deepseek_ocr_model: Optional[str],
        deepseek_ocr_prompt: str,
        deepseek_ocr_base_size: int,
        deepseek_ocr_image_size: int,
        deepseek_ocr_crop_mode: bool,
        deepseek_ocr_save_results: bool,
        deepseek_ocr_test_compress: bool,
        deepseek_ocr_attn_implementation: str,
        deepseek_ocr_dtype: str,
        deepseek_ocr_output_dir: Optional[Path],
        deepseek_ocr_max_new_tokens: int,
    ) -> None:
        self.backend = backend
        self.lang = lang
        self.min_confidence = min_confidence
        self.reader = None
        self.tokenizer = None
        self.device = device
        self.deepseek_ocr_prompt = deepseek_ocr_prompt
        self.deepseek_ocr_base_size = deepseek_ocr_base_size
        self.deepseek_ocr_image_size = deepseek_ocr_image_size
        self.deepseek_ocr_crop_mode = deepseek_ocr_crop_mode
        self.deepseek_ocr_save_results = deepseek_ocr_save_results
        self.deepseek_ocr_test_compress = deepseek_ocr_test_compress
        self.deepseek_ocr_max_new_tokens = deepseek_ocr_max_new_tokens
        self.deepseek_ocr_counter = 0
        self.deepseek_ocr_tmp = tempfile.TemporaryDirectory(prefix="deepseek_ocr_eval_") if backend == "deepseek_ocr" and deepseek_ocr_output_dir is None else None
        self.deepseek_ocr_output_dir = deepseek_ocr_output_dir or (Path(self.deepseek_ocr_tmp.name) if self.deepseek_ocr_tmp is not None else None)
        if backend == "none":
            return
        if backend == "pytesseract":
            import pytesseract  # type: ignore

            self.reader = pytesseract
        elif backend == "paddleocr":
            from paddleocr import PaddleOCR  # type: ignore

            self.reader = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        elif backend == "easyocr":
            import easyocr  # type: ignore

            self.reader = easyocr.Reader([lang], gpu=device_backend != "cpu")
        elif backend == "deepseek_ocr":
            if device.type == "npu":
                enable_deepseek_ocr_npu_cuda_compat()
            try:
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as exc:  # pragma: no cover - optional server dependency
                raise ImportError("Install transformers before using --ocr-backend=deepseek_ocr.") from exc

            model_name_or_path = deepseek_ocr_model or os.environ.get("DEEPSEEK_OCR_MODEL") or "deepseek-ai/DeepSeek-OCR"
            load_kwargs: Dict[str, Any] = {
                "trust_remote_code": True,
                "use_safetensors": True,
            }
            if deepseek_ocr_attn_implementation and deepseek_ocr_attn_implementation != "none":
                load_kwargs["_attn_implementation"] = deepseek_ocr_attn_implementation
            dtype_map = {
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }
            if deepseek_ocr_dtype in dtype_map:
                load_kwargs["torch_dtype"] = dtype_map[deepseek_ocr_dtype]

            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.reader = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
            self.reader = self.reader.eval().to(device)  # type: ignore[union-attr]
            if deepseek_ocr_dtype in dtype_map:
                self.reader = self.reader.to(dtype_map[deepseek_ocr_dtype])  # type: ignore[union-attr]
            cap_model_generate(self.reader, deepseek_ocr_max_new_tokens)  # type: ignore[arg-type]
            if self.deepseek_ocr_output_dir is not None:
                self.deepseek_ocr_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def recognize(self, image: np.ndarray) -> str:
        if self.backend == "none":
            return ""
        if self.backend == "pytesseract":
            text = self.reader.image_to_string(Image.fromarray(image), lang=self.lang)  # type: ignore[union-attr]
            return " ".join(str(text).split())
        if self.backend == "paddleocr":
            result = self.reader.ocr(image, cls=True)  # type: ignore[union-attr]
            return " ".join(extract_paddle_texts(result, self.min_confidence))
        if self.backend == "easyocr":
            result = self.reader.readtext(image, detail=1, paragraph=False)  # type: ignore[union-attr]
            texts = []
            for item in result:
                if len(item) >= 3 and float(item[2]) >= self.min_confidence:
                    texts.append(str(item[1]))
            return " ".join(texts)
        if self.backend == "deepseek_ocr":
            if self.reader is None or self.tokenizer is None or self.deepseek_ocr_output_dir is None:
                return ""
            self.deepseek_ocr_counter += 1
            image_path = self.deepseek_ocr_output_dir / f"recon_{self.deepseek_ocr_counter:06d}.png"
            Image.fromarray(image).save(image_path)
            infer_kwargs: Dict[str, Any] = {
                "prompt": self.deepseek_ocr_prompt,
                "image_file": str(image_path),
                "output_path": str(self.deepseek_ocr_output_dir),
                "base_size": self.deepseek_ocr_base_size,
                "image_size": self.deepseek_ocr_image_size,
                "crop_mode": self.deepseek_ocr_crop_mode,
                "save_results": self.deepseek_ocr_save_results,
                "test_compress": self.deepseek_ocr_test_compress,
                "eval_mode": True,
            }
            try:
                result = self.reader.infer(self.tokenizer, **infer_kwargs)  # type: ignore[union-attr]
            except TypeError:
                infer_kwargs.pop("save_results", None)
                infer_kwargs.pop("test_compress", None)
                result = self.reader.infer(self.tokenizer, **infer_kwargs)  # type: ignore[union-attr]
            text = stringify_deepseek_ocr_result(result)
            if not text:
                text = read_deepseek_saved_text(self.deepseek_ocr_output_dir)
            return text
        raise ValueError(f"Unsupported OCR backend: {self.backend}")


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums: Dict[str, float] = defaultdict(float)

    def update(self, values: Mapping[str, float]) -> None:
        self.count += 1
        for key, value in values.items():
            if not math.isnan(value):
                self.sums[key] += float(value)

    def mean(self) -> Dict[str, float]:
        if self.count == 0:
            return {"count": 0}
        result = {"count": self.count}
        for key, value in sorted(self.sums.items()):
            result[key] = value / self.count
        return result


def select_grid_indices(num_images: int, grid_samples: int, seed: int) -> List[int]:
    grid_samples = min(max(grid_samples, 0), num_images)
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_images), grid_samples))


def make_text_inputs_for_mode(
        correct_texts: Sequence[str],
        mode: str,
        wrong_text_seed: int,
        fixed_wrong_text: str,
) -> List[str]:
    if mode == "correct":
        return list(correct_texts)
    if mode == "empty":
        return [""] * len(correct_texts)
    if mode == "wrong":
        return [fixed_wrong_text] * len(correct_texts)
    if mode == "shuffled":
        shuffled = list(correct_texts)
        random.Random(wrong_text_seed).shuffle(shuffled)
        if len(shuffled) > 1 and shuffled == list(correct_texts):
            shuffled = shuffled[1:] + shuffled[:1]
        return shuffled
    if mode == "length_matched_wrong":
        texts = [str(text or "") for text in correct_texts]
        non_empty = [(idx, text) for idx, text in enumerate(texts) if text.strip()]
        rng = random.Random(wrong_text_seed)
        wrong_texts: List[str] = []
        for idx, text in enumerate(texts):
            candidates = [(other_idx, other_text) for other_idx, other_text in non_empty if other_idx != idx]
            if not text.strip() or not candidates:
                wrong_texts.append(fixed_wrong_text)
                continue
            target_len = len(text)
            candidates.sort(key=lambda item: abs(len(item[1]) - target_len))
            nearest = candidates[: min(8, len(candidates))]
            wrong_texts.append(rng.choice(nearest)[1])
        return wrong_texts
    raise ValueError(f"unsupported text_input_mode: {mode}")


def evaluate_run(
    *,
    run_name: str,
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    subset_by_path: Mapping[str, str],
    device: torch.device,
    batch_size: int,
    image_size: int,
    pad_color: Tuple[int, int, int],
    grid_indices: Sequence[int],
    metadata_by_path: Mapping[str, Mapping[str, str]],
    text_context: Optional[TextContext],
    ocr_reader: OCRReader,
    ocr_jsonl: Optional[Path],
    device_backend: str,
    mixed_precision: str,
    text_input_mode: str,
    wrong_text_seed: int,
    fixed_wrong_text: str,
    strip_punctuation: bool,
    remove_spaces: bool,
) -> Tuple[Dict[str, Any], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, str]]:
    overall = MetricAccumulator()
    by_subset: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    grid_recons: Dict[int, np.ndarray] = {}
    grid_gts: Dict[int, np.ndarray] = {}
    grid_subsets: Dict[int, str] = {}
    grid_set = set(grid_indices)
    correct_texts = [
        metadata_by_path.get(str(path.resolve()), {}).get("text", "")
        for path in image_paths
    ]
    text_inputs_for_model = make_text_inputs_for_mode(
        correct_texts,
        text_input_mode,
        wrong_text_seed,
        fixed_wrong_text,
    )

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        pil_images = [resize_pad_image(Image.open(path), image_size, pad_color) for path in batch_paths]
        batch = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        batch_texts = text_inputs_for_model[start : start + len(batch_paths)]
        rec_batch = tensor_to_uint8(
            reconstruct_batch(
                model,
                batch,
                batch_texts if text_context is not None else None,
                text_context,
                device_backend=device_backend,
                mixed_precision=mixed_precision,
            )
        )
        gt_batch = [np.asarray(img, dtype=np.uint8) for img in pil_images]
        ocr_rows = []

        for offset, (path, gt, rec) in enumerate(zip(batch_paths, gt_batch, rec_batch)):
            index = start + offset
            metadata = metadata_by_path.get(str(path.resolve()), {})
            subset = metadata.get("subset", subset_by_path.get(str(path.resolve()), "unknown"))
            values = metric_values(gt, rec)
            gt_text = metadata.get("text", "")
            pred_text = ""
            if ocr_reader.enabled:
                pred_text = ocr_reader.recognize(rec)
                values.update(
                    text_metric_values(
                        gt_text,
                        pred_text,
                        strip_punctuation=strip_punctuation,
                        remove_spaces=remove_spaces,
                    )
                )
            overall.update(values)
            by_subset[subset].update(values)
            if ocr_jsonl is not None and ocr_reader.enabled:
                ocr_rows.append({
                    "run": run_name,
                    "index": index,
                    "image_path": str(path),
                    "subset": subset,
                    "source_key": metadata.get("source_key", ""),
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "metrics": {key: values[key] for key in values if key.startswith("ocr_")},
                })
            if index in grid_set:
                grid_gts[index] = gt
                grid_recons[index] = rec
                grid_subsets[index] = subset

        if ocr_rows and ocr_jsonl is not None:
            append_jsonl(ocr_jsonl, ocr_rows)

        if (start // batch_size + 1) % 50 == 0:
            print(f"[{run_name}] evaluated {min(start + batch_size, len(image_paths))}/{len(image_paths)} images", flush=True)

    metrics = {
        "overall": overall.mean(),
        "by_subset": {subset: acc.mean() for subset, acc in sorted(by_subset.items())},
    }
    return metrics, grid_recons, grid_gts, grid_subsets


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    x, y = xy
    draw.rectangle((x, y, x + 255, y + 21), fill=(255, 255, 255))
    draw.text((x + 4, y + 4), text, fill=(0, 0, 0))


def build_grid(
    *,
    output_path: Path,
    run_names: Sequence[str],
    grid_indices: Sequence[int],
    grid_gts: Mapping[int, np.ndarray],
    grid_recons_by_run: Mapping[str, Mapping[int, np.ndarray]],
    grid_subsets: Mapping[int, str],
    save_per_sample: int,
    output_dir: Path,
) -> None:
    if not grid_indices:
        return
    cell = 256
    label_h = 22
    cols = 1 + len(run_names)
    rows = len(grid_indices)
    grid = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    for row, index in enumerate(grid_indices):
        y0 = row * (cell + label_h)
        subset = grid_subsets.get(index, "unknown")
        images = [("gt", grid_gts[index])] + [(name, grid_recons_by_run[name][index]) for name in run_names]
        row_canvas = Image.new("RGB", (cols * cell, cell + label_h), (240, 240, 240))
        row_draw = ImageDraw.Draw(row_canvas)
        for col, (label, image_arr) in enumerate(images):
            x0 = col * cell
            row_canvas.paste(Image.fromarray(image_arr), (x0, label_h))
            label_text = f"{label} | #{index} | {subset}" if col == 0 else label
            draw_label(row_draw, (x0, 0), label_text[:42])
        grid.paste(row_canvas, (0, y0))
        if row < save_per_sample:
            row_canvas.save(output_dir / f"sample_{row:03d}_idx_{index:05d}.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"saved grid: {output_path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if len(args.run) < 1:
        raise SystemExit("at least one --run is required")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pad_color = parse_color(args.pad_color)
    device = prepare_device(args.device_backend, args.device_id)

    manifest_image_paths, metadata_by_path = load_manifest_metadata(args.manifest_jsonl)
    if args.image_paths_json is not None:
        image_paths = [Path(path) for path in read_json(args.image_paths_json)]
    else:
        image_paths = manifest_image_paths
    if args.ocr_backend != "none" and not metadata_by_path:
        raise SystemExit("--ocr-backend requires --manifest-jsonl with rendered text.")
    if args.image_paths_json is None and args.manifest_jsonl is None:
        raise SystemExit("Provide --image-paths-json or --manifest-jsonl.")
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit("no images to evaluate")

    missing = [str(path) for path in image_paths[:1000] if not path.exists()]
    if missing:
        raise SystemExit(f"missing image files, first missing: {missing[0]}")

    subset_by_path = {path: row.get("subset", "unknown") for path, row in metadata_by_path.items()}
    grid_indices = select_grid_indices(len(image_paths), args.grid_samples, args.grid_seed)
    ocr_reader = OCRReader(
        backend=args.ocr_backend,
        lang=args.ocr_lang,
        min_confidence=args.ocr_min_confidence,
        device_backend=args.device_backend,
        device=device,
        deepseek_ocr_model=args.deepseek_ocr_model,
        deepseek_ocr_prompt=args.deepseek_ocr_prompt,
        deepseek_ocr_base_size=args.deepseek_ocr_base_size,
        deepseek_ocr_image_size=args.deepseek_ocr_image_size,
        deepseek_ocr_crop_mode=args.deepseek_ocr_crop_mode,
        deepseek_ocr_save_results=args.deepseek_ocr_save_results,
        deepseek_ocr_test_compress=args.deepseek_ocr_test_compress,
        deepseek_ocr_attn_implementation=args.deepseek_ocr_attn_implementation,
        deepseek_ocr_dtype=args.deepseek_ocr_dtype,
        deepseek_ocr_output_dir=args.deepseek_ocr_output_dir,
        deepseek_ocr_max_new_tokens=args.deepseek_ocr_max_new_tokens,
    )
    ocr_jsonl = args.ocr_jsonl or (output_dir / "ocr_predictions.jsonl" if ocr_reader.enabled else None)
    if ocr_jsonl is not None and ocr_jsonl.exists():
        ocr_jsonl.unlink()

    metrics_by_run: Dict[str, Any] = {}
    grid_recons_by_run: Dict[str, Dict[int, np.ndarray]] = {}
    grid_gts: Dict[int, np.ndarray] = {}
    grid_subsets: Dict[int, str] = {}
    run_names: List[str] = []

    for run_name, config_path, ckpt_path in args.run:
        print(f"== evaluating {run_name} ==")
        run_names.append(run_name)
        model, config, text_context = load_tokenizer_model(
            config_path,
            ckpt_path,
            device,
            text_layer_pair_index=args.text_layer_pair_index,
            checkpoint_weight_key=args.checkpoint_weight_key,
        )
        if text_context is not None:
            print(f"[{run_name}] text-conditioned reconstruction uses pair {text_context.selected_pair}")
        metrics, recons, gts, subsets = evaluate_run(
            run_name=run_name,
            model=model,
            image_paths=image_paths,
            subset_by_path=subset_by_path,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            pad_color=pad_color,
            grid_indices=grid_indices,
            metadata_by_path=metadata_by_path,
            text_context=text_context,
            ocr_reader=ocr_reader,
            ocr_jsonl=ocr_jsonl,
            device_backend=args.device_backend,
            mixed_precision=args.mixed_precision,
            text_input_mode=args.text_input_mode,
            wrong_text_seed=args.wrong_text_seed,
            fixed_wrong_text=args.fixed_wrong_text,
            strip_punctuation=args.strip_punctuation,
            remove_spaces=args.remove_spaces,
        )
        metrics_by_run[run_name] = metrics
        grid_recons_by_run[run_name] = recons
        grid_gts.update(gts)
        grid_subsets.update(subsets)
        del model
        if args.device_backend == "cuda":
            torch.cuda.empty_cache()
        elif args.device_backend == "npu":
            torch.npu.empty_cache()

    summary = {
        "num_images": len(image_paths),
        "image_paths_json": str(args.image_paths_json),
        "manifest_jsonl": str(args.manifest_jsonl) if args.manifest_jsonl else None,
        "device_backend": args.device_backend,
        "device_id": args.device_id,
        "mixed_precision": args.mixed_precision,
        "batch_size": args.batch_size,
        "ocr_backend": args.ocr_backend,
        "ocr_lang": args.ocr_lang,
        "ocr_min_confidence": args.ocr_min_confidence,
        "ocr_jsonl": str(ocr_jsonl) if ocr_jsonl is not None else None,
        "deepseek_ocr_model": args.deepseek_ocr_model,
        "deepseek_ocr_prompt": args.deepseek_ocr_prompt,
        "deepseek_ocr_base_size": args.deepseek_ocr_base_size,
        "deepseek_ocr_image_size": args.deepseek_ocr_image_size,
        "deepseek_ocr_crop_mode": args.deepseek_ocr_crop_mode,
        "deepseek_ocr_save_results": args.deepseek_ocr_save_results,
        "deepseek_ocr_test_compress": args.deepseek_ocr_test_compress,
        "deepseek_ocr_attn_implementation": args.deepseek_ocr_attn_implementation,
        "deepseek_ocr_dtype": args.deepseek_ocr_dtype,
        "deepseek_ocr_output_dir": str(args.deepseek_ocr_output_dir) if args.deepseek_ocr_output_dir else None,
        "deepseek_ocr_max_new_tokens": args.deepseek_ocr_max_new_tokens,
        "text_layer_pair_index": args.text_layer_pair_index,
        "text_input_mode": args.text_input_mode,
        "wrong_text_seed": args.wrong_text_seed,
        "fixed_wrong_text": args.fixed_wrong_text,
        "strip_punctuation": args.strip_punctuation,
        "remove_spaces": args.remove_spaces,
        "runs": metrics_by_run,
    }
    write_json(output_dir / "metrics.json", summary)

    build_grid(
        output_path=output_dir / "comparison_grid.png",
        run_names=run_names,
        grid_indices=grid_indices,
        grid_gts=grid_gts,
        grid_recons_by_run=grid_recons_by_run,
        grid_subsets=grid_subsets,
        save_per_sample=args.save_per_sample,
        output_dir=output_dir,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
