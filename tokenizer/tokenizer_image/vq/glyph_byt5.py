"""Local Glyph-ByT5 text encoder backend.

This module intentionally keeps only the pure text-encoder pieces needed by
stage-1 tokenizer experiments. It does not import the Glyph-SDXL pipeline,
UNet, timestep conditioning, or diffusers-specific model base classes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    from transformers.models.t5.modeling_t5 import T5LayerFF, T5LayerNorm, T5LayerSelfAttention
except ImportError:  # pragma: no cover - handled by caller with a clear error.
    AutoTokenizer = None
    T5ForConditionalGeneration = None
    T5LayerFF = None
    T5LayerNorm = None
    T5LayerSelfAttention = None


@dataclass
class TextEncoderLoadResult:
    tokenizer: Any
    encoder: nn.Module
    feature_dim: int
    num_layers: int
    backend: str


class T5EncoderBlock(nn.Module):
    """Minimal copy of Glyph-SDXL's mapper block without diffusers dependency."""

    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            layer_head_mask=None,
            output_attentions=False,
    ):
        self_attn_past_key_value = None
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=False,
            output_attentions=output_attentions,
        )
        hidden_states = self_attention_outputs[0]
        attention_outputs = self_attention_outputs[2:]

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        hidden_states = self.layer[-1](hidden_states)

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states,) + attention_outputs


class T5EncoderBlockByT5Mapper(nn.Module):
    """Glyph-ByT5 mapper: [B, T, 1472] -> [B, T, 2048]."""

    def __init__(self, byt5_config, num_layers, sdxl_channels=None):
        super().__init__()
        if num_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    T5EncoderBlock(byt5_config, has_relative_attention_bias=bool(i == 0))
                    for i in range(num_layers)
                ]
            )
        else:
            self.blocks = None
        self.layer_norm = T5LayerNorm(byt5_config.d_model, eps=byt5_config.layer_norm_epsilon)
        if sdxl_channels is not None:
            self.channel_mapper = nn.Linear(byt5_config.d_model, sdxl_channels)
            self.final_layer_norm = T5LayerNorm(sdxl_channels, eps=byt5_config.layer_norm_epsilon)
        else:
            self.channel_mapper = None
            self.final_layer_norm = None

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def get_extended_attention_mask(
            self,
            attention_mask: Tensor,
            input_shape: Tuple[int, ...],
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
    ) -> Tensor:
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = attention_mask.device
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        extended_attention_mask = extended_attention_mask.to(device=device, dtype=dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(self, inputs_embeds, attention_mask):
        input_shape = inputs_embeds.size()[:-1]
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask,
            input_shape,
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )

        hidden_states = inputs_embeds
        position_bias = None

        if self.blocks is not None:
            for layer_module in self.blocks:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                )
                hidden_states = layer_outputs[0]
                if len(layer_outputs) > 1:
                    position_bias = layer_outputs[1]
        hidden_states = self.layer_norm(hidden_states)
        if self.channel_mapper is not None:
            hidden_states = self.channel_mapper(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class GlyphByT5Encoder(nn.Module):
    """Frozen Glyph-ByT5 encoder plus mapper with a T5-like forward signature."""

    def __init__(self, encoder: nn.Module, mapper: nn.Module, feature_dim: int, num_layers: int):
        super().__init__()
        self.encoder = encoder
        self.mapper = mapper
        self.config = SimpleNamespace(d_model=feature_dim, num_layers=num_layers)
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        self.encoder_backend = "glyph_byt5"

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, return_dict=True, **kwargs):
        del output_hidden_states, kwargs
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        mapped = self.mapper(encoder_outputs.last_hidden_state, attention_mask)
        if return_dict:
            return SimpleNamespace(last_hidden_state=mapped, hidden_states=(mapped,))
        return (mapped,)


def _json_values(path: Path) -> List[Tuple[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, Mapping):
        return [(str(key), value) for key, value in data.items()]
    if isinstance(data, Sequence):
        return [(str(idx), value) for idx, value in enumerate(data)]
    raise TypeError(f"Unsupported JSON annotation format: {path}")


def _glyph_special_tokens(
        glyph_model_dir: Path,
        *,
        color_special_token: bool,
        font_special_token: bool,
        multilingual: bool,
        color_ann_path: str,
        font_ann_path: str,
) -> List[str]:
    tokens: List[str] = []
    if color_special_token:
        for _, color_idx in _json_values(glyph_model_dir / color_ann_path):
            tokens.append(f"<color-{color_idx}>")
    if font_special_token:
        for font_code, font_idx in _json_values(glyph_model_dir / font_ann_path):
            if multilingual:
                tokens.append(f"<{font_code[:2]}-font-{font_idx}>")
            else:
                tokens.append(f"<font-{font_idx}>")
    return tokens


def add_glyph_byt5_special_tokens(
        tokenizer,
        text_model,
        glyph_model_dir: Path,
        glyph_cfg: Mapping[str, Any],
) -> int:
    tokens = _glyph_special_tokens(
        glyph_model_dir,
        color_special_token=bool(glyph_cfg.get("color_special_token", True)),
        font_special_token=bool(glyph_cfg.get("font_special_token", True)),
        multilingual=bool(glyph_cfg.get("multilingual", True)),
        color_ann_path=str(glyph_cfg.get("color_ann_path", "assets/color_idx.json")),
        font_ann_path=str(glyph_cfg.get("font_ann_path", "assets/multilingual_10-lang_idx.json")),
    )
    if tokens:
        tokenizer.add_tokens(tokens, special_tokens=True)
        text_model.resize_token_embeddings(len(tokenizer))
    return len(tokens)


def _load_state_dict(path: Path, map_location="cpu") -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location=map_location)
    if isinstance(state, Mapping) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected state dict at {path}, got {type(state)}")
    return dict(state)


def load_glyph_byt5_text_encoder(text_cfg: Mapping[str, Any], device: torch.device) -> TextEncoderLoadResult:
    if AutoTokenizer is None or T5ForConditionalGeneration is None:
        raise ImportError("Glyph-ByT5 backend requires transformers with T5ForConditionalGeneration.")

    glyph_model_dir = Path(text_cfg.get("encoder_name", text_cfg.get("glyph_model_dir", "")))
    if not glyph_model_dir:
        raise ValueError("text_conditioning.encoder_name or glyph_model_dir must point to Glyph-SDXL-v2.")
    glyph_model_dir = glyph_model_dir.expanduser()
    byt5_base_name = text_cfg.get("byt5_base_name", text_cfg.get("byt5_name", "google/byt5-small"))
    local_files_only = bool(text_cfg.get("local_files_only", False))
    glyph_cfg = text_cfg.get("glyph_byt5", {}) or {}

    tokenizer = AutoTokenizer.from_pretrained(byt5_base_name, local_files_only=local_files_only)
    text_model = T5ForConditionalGeneration.from_pretrained(byt5_base_name, local_files_only=local_files_only)
    add_glyph_byt5_special_tokens(tokenizer, text_model, glyph_model_dir, glyph_cfg)
    encoder = text_model.get_encoder()

    encoder_ckpt = glyph_model_dir / str(glyph_cfg.get("byt5_model_path", "checkpoints/byt5_model.pt"))
    mapper_ckpt = glyph_model_dir / str(glyph_cfg.get("byt5_mapper_path", "checkpoints/byt5_mapper.pt"))
    encoder_state = _load_state_dict(encoder_ckpt)
    embed_weight = encoder_state.get("embed_tokens.weight")
    if embed_weight is None:
        raise KeyError(f"{encoder_ckpt} does not contain embed_tokens.weight")
    if int(embed_weight.shape[0]) != len(tokenizer):
        raise ValueError(
            "Glyph-ByT5 tokenizer/checkpoint vocab mismatch: "
            f"checkpoint={int(embed_weight.shape[0])}, tokenizer={len(tokenizer)}. "
            "Check Glyph special-token config before loading."
        )
    encoder.load_state_dict(encoder_state, strict=True)

    mapper_num_layers = int(glyph_cfg.get("mapper_num_layers", 4))
    mapper_output_dim = int(glyph_cfg.get("mapper_output_dim", 2048))
    mapper = T5EncoderBlockByT5Mapper(
        encoder.config,
        num_layers=mapper_num_layers,
        sdxl_channels=mapper_output_dim,
    )
    mapper_state = _load_state_dict(mapper_ckpt)
    mapper.load_state_dict(mapper_state, strict=True)

    wrapped = GlyphByT5Encoder(encoder=encoder, mapper=mapper, feature_dim=mapper_output_dim, num_layers=1)
    if text_cfg.get("freeze", True):
        wrapped.requires_grad_(False)
    wrapped.eval().to(device)
    return TextEncoderLoadResult(
        tokenizer=tokenizer,
        encoder=wrapped,
        feature_dim=mapper_output_dim,
        num_layers=1,
        backend="glyph_byt5",
    )


def load_text_encoder_from_config(text_cfg: Mapping[str, Any], device: torch.device) -> Optional[TextEncoderLoadResult]:
    if not text_cfg.get("enabled", False):
        return None
    backend = str(text_cfg.get("encoder_backend", "t5")).lower()
    if backend == "glyph_byt5":
        return load_glyph_byt5_text_encoder(text_cfg, device)
    if backend != "t5":
        raise ValueError(f"Unsupported text_conditioning.encoder_backend={backend}")
    if AutoTokenizer is None:
        raise ImportError("T5 text backend requires transformers.")
    from transformers import T5EncoderModel

    encoder_name = text_cfg.get("encoder_name", "google/t5-v1_1-xl")
    pretrained_kwargs = {
        "cache_dir": text_cfg.get("cache_dir", None),
        "local_files_only": bool(text_cfg.get("local_files_only", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, **pretrained_kwargs)
    encoder = T5EncoderModel.from_pretrained(encoder_name, **pretrained_kwargs)
    if text_cfg.get("freeze", True):
        encoder.requires_grad_(False)
    encoder.eval().to(device)
    feature_dim = getattr(encoder.config, "d_model", None)
    num_layers = getattr(encoder.config, "num_layers", None)
    if feature_dim is None:
        raise ValueError(f"Cannot read d_model from T5 encoder config: {encoder_name}")
    return TextEncoderLoadResult(
        tokenizer=tokenizer,
        encoder=encoder,
        feature_dim=int(feature_dim),
        num_layers=num_layers,
        backend="t5",
    )


def get_text_layer_states(text_outputs, backend: str, exclude_embedding_output: bool = True):
    hidden_states = text_outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Text encoder did not return hidden_states.")
    if backend == "glyph_byt5":
        return tuple(hidden_states)
    if exclude_embedding_output:
        return tuple(hidden_states[1:])
    return tuple(hidden_states)


def tokenize_texts_with_stats(
        tokenizer,
        texts: Sequence[str],
        *,
        max_length: int,
        device: Optional[torch.device] = None,
        non_blocking: bool = True,
        backend: str = "t5",
):
    text_inputs = tokenizer(
        [str(text) for text in texts],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    stats: Dict[str, torch.Tensor] = {}
    if backend == "glyph_byt5":
        raw_inputs = tokenizer(
            [str(text) for text in texts],
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        lengths = [len(input_ids) for input_ids in raw_inputs["input_ids"]]
        length_tensor = torch.tensor(lengths, dtype=torch.float32)
        truncated = length_tensor > float(max_length)
        stats = {
            "glyph_input_tokens_mean": length_tensor.mean(),
            "glyph_input_tokens_max": length_tensor.max(),
            "glyph_truncated_count": truncated.float().sum(),
            "glyph_truncated_ratio": truncated.float().mean(),
            "glyph_max_length": torch.tensor(float(max_length), dtype=torch.float32),
        }
    if device is not None:
        for key, value in list(text_inputs.items()):
            if torch.is_tensor(value):
                text_inputs[key] = value.to(device, non_blocking=non_blocking)
        stats = {
            key: value.to(device, non_blocking=non_blocking) if torch.is_tensor(value) else value
            for key, value in stats.items()
        }
    return text_inputs, stats
