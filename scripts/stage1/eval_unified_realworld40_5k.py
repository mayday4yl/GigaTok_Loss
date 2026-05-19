#!/usr/bin/env python3
"""Unified holdout evaluation for realworld40 5k tokenizer runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover
    structural_similarity = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.ocr_box_gate import (  # noqa: E402
    build_ocr_box_gate,
    load_ocr_box_index,
    ocr_box_gate_config_from_dict,
)
from scripts.stage1.ocr_debug.text_feature_cache import TextFeatureCache  # noqa: E402
from utils.model_init import custom_load, load_model_from_config  # noqa: E402


TEXT_CONDITIONING_MISSING_PREFIXES = (
    "text_projection.",
    "residual_head_mlp.",
    "residual_text_mlp.",
    "adaln_mlps.",
    "s1to2decoder.residual_cross_attn_layers.",
    "s1to2decoder.residual_cross_attn_projs.",
    "s1to2decoder.residual_cross_attn_scales.",
)
TEXT_CONDITIONING_MISSING_NAMES = {
    "text_type_embedding",
    "visual_type_embedding",
    "text_gate_logit",
    "visual_mask_token",
    "residual_head_gate",
    "residual_gate",
}


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_model_hash(path: Path) -> Tuple[str, Optional[int]]:
    obj = torch.load(path, map_location="cpu")
    step = None
    if isinstance(obj, Mapping):
        for key in ("step", "train_steps", "global_step"):
            if key in obj:
                try:
                    step = int(obj[key])
                except Exception:
                    step = None
                break
        weights = obj.get("model", obj.get("state_dict", obj.get("ema", None)))
    else:
        weights = None
    if not isinstance(weights, Mapping):
        raise KeyError(f"cannot find model/state_dict/ema weights in {path}")
    h = hashlib.sha256()
    for key in sorted(weights.keys()):
        value = weights[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    del obj
    return h.hexdigest(), step


def resize_pad_image(image: Image.Image, image_size: int = 256, pad_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), pad_color)
    canvas.paste(resized, ((image_size - resized_size[0]) // 2, (image_size - resized_size[1]) // 2))
    return canvas


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1) * 2.0 - 1.0


def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    x = torch.clamp((tensor.float() + 1.0) * 0.5, 0.0, 1.0)
    return (x.permute(0, 2, 3, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)


def metric_values(gt_u8: np.ndarray, rec_u8: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    gt = gt_u8.astype(np.float32) / 255.0
    rec = rec_u8.astype(np.float32) / 255.0
    if mask is not None:
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            return {"mse": float("nan"), "mae": float("nan"), "psnr": float("nan"), "ssim": float("nan")}
        diff = rec[mask_bool] - gt[mask_bool]
    else:
        diff = rec - gt
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 0 else float(-10.0 * math.log10(mse))
    if structural_similarity is None:
        ssim = float("nan")
    elif mask is not None:
        ys, xs = np.where(mask.astype(bool))
        if len(xs) == 0:
            ssim = float("nan")
        else:
            x0, x1 = max(0, int(xs.min()) - 4), min(gt.shape[1], int(xs.max()) + 5)
            y0, y1 = max(0, int(ys.min()) - 4), min(gt.shape[0], int(ys.max()) + 5)
            if x1 - x0 < 7 or y1 - y0 < 7:
                ssim = float("nan")
            else:
                ssim = float(structural_similarity(gt[y0:y1, x0:x1], rec[y0:y1, x0:x1], channel_axis=-1, data_range=1.0))
    else:
        ssim = float(structural_similarity(gt, rec, channel_axis=-1, data_range=1.0))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


def prefixed(prefix: str, values: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def autocast_context(device: torch.device, mixed_precision: str):
    if device.type == "cpu" or mixed_precision == "none":
        return nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    if device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if device.type == "npu":
        npu_amp = getattr(getattr(torch, "npu", None), "amp", None)
        if npu_amp is not None and hasattr(npu_amp, "autocast"):
            return npu_amp.autocast(dtype=dtype)
        return torch.autocast(device_type="npu", dtype=dtype)
    return torch.autocast(device_type=device.type, dtype=dtype)


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
            raise RuntimeError("NPU is not available")
        torch.npu.set_device(device_id)
        return torch.device(f"npu:{device_id}")
    raise ValueError(f"unsupported device backend={device_backend}")


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "npu":
        torch.npu.empty_cache()


def text_conditioning_missing_keys(model: torch.nn.Module) -> List[str]:
    missing = []
    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if name in TEXT_CONDITIONING_MISSING_NAMES or name.startswith(TEXT_CONDITIONING_MISSING_PREFIXES):
            missing.append(name)
    return missing


def build_layer_pairs(raw_pairs: Sequence[Sequence[int]], decoder_num_layers: int) -> List[Tuple[int, int]]:
    pairs = []
    for pair in raw_pairs:
        if len(pair) != 2:
            raise ValueError(f"layer pair must be [text_layer, decoder_layer], got {pair}")
        text_layer, decoder_layer = int(pair[0]), int(pair[1])
        if decoder_layer < 0 or decoder_layer >= decoder_num_layers:
            raise ValueError(f"decoder_layer={decoder_layer} out of range for {decoder_num_layers}")
        pairs.append((text_layer, decoder_layer))
    if not pairs:
        raise ValueError("empty text layer pairs")
    return pairs


def configure_text_model(model: torch.nn.Module, config: Mapping[str, Any], text_feature_cache: Optional[TextFeatureCache]) -> Dict[str, Any]:
    text_cfg = config.get("text_conditioning", {})
    text_recon_cfg = config.get("text_recon_conditioning", {})
    if not bool(text_cfg.get("enabled", False)):
        return {"enabled": False}
    if text_feature_cache is None:
        raise ValueError("text_conditioning.enabled requires --text-feature-cache for unified eval")
    text_feature_cache.validate_config(text_cfg)
    text_feature_dim = int(text_feature_cache.metadata["feature_dim"])
    text_recon_on = bool(text_recon_cfg.get("enabled", False))
    text_recon_mode = str(text_recon_cfg.get("mode", "concat_memory"))
    visual_memory_mask_cfg = text_recon_cfg.get("visual_memory_mask", {})
    text_gate_cfg = text_recon_cfg.get("text_gate", {}) if text_recon_on else {}
    residual_head_cfg = text_recon_cfg.get("residual_head", {}) if text_recon_on else {}
    residual_pooled_cfg = text_recon_cfg.get("residual_pooled_layer", {}) if text_recon_on else {}
    adaln_cfg = text_recon_cfg.get("adaln", {}) if text_recon_on else {}
    residual_cross_attn_cfg = text_recon_cfg.get("residual_cross_attn", {}) if text_recon_on else {}
    concat_memory_mode = text_recon_mode in {"concat_memory", "concat_memory_visual_mask"}
    residual_cross_attn_layers = None
    adaln_layers = None
    if text_recon_on and text_recon_mode == "residual_cross_attn_visual_mask":
        raw_layers = text_recon_cfg.get("layers") or [pair[1] for pair in text_recon_cfg.get("layer_pairs", [])]
        residual_cross_attn_layers = [int(layer) for layer in raw_layers]
    if text_recon_on and text_recon_mode == "adaln":
        raw_layers = text_recon_cfg.get("layers") or [pair[1] for pair in text_recon_cfg.get("layer_pairs", [])]
        adaln_layers = [int(layer) for layer in raw_layers]
    model.configure_text_conditioning(
        text_feature_dim=text_feature_dim,
        text_projection=text_cfg.get("text_projection", "linear_layernorm"),
        text_type_embedding=text_cfg.get("text_type_embedding", True),
        visual_type_embedding=text_recon_on and concat_memory_mode and bool(text_recon_cfg.get("visual_type_embedding", False)),
        text_gate_enabled=text_recon_on and concat_memory_mode and bool(text_gate_cfg.get("enabled", False)),
        text_gate_init=float(text_gate_cfg.get("init", 0.1)),
        visual_memory_mask_enabled=text_recon_on and text_recon_mode in {"concat_memory_visual_mask", "residual_cross_attn_visual_mask"},
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
    decoder_num_layers = int(getattr(model.s1to2decoder, "num_layers"))
    return {
        "enabled": True,
        "text_recon_mode": text_recon_mode,
        "layer_pairs": build_layer_pairs(text_recon_cfg.get("layer_pairs", []), decoder_num_layers),
        "text_injection_layers": [int(layer) for layer in (text_recon_cfg.get("layers") or [])],
        "visual_memory_mask_enabled": bool(visual_memory_mask_cfg.get("enabled", False)),
        "visual_memory_mask_ratio": float(visual_memory_mask_cfg.get("ratio", 0.0)),
        "visual_memory_mask_strategy": str(visual_memory_mask_cfg.get("strategy", "token_random")),
        "visual_memory_mask_block_size": int(visual_memory_mask_cfg.get("block_size", 1)),
        "visual_memory_mask_fixed_pattern": bool(visual_memory_mask_cfg.get("fixed_pattern", False)),
        "visual_memory_mask_seed": int(visual_memory_mask_cfg.get("seed", 0)),
        "visual_memory_mask_apply_in_eval": bool(visual_memory_mask_cfg.get("apply_in_eval", False)),
    }


def load_model(config_path: Path, ckpt_path: Path, device: torch.device, text_feature_cache: Optional[TextFeatureCache]) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = load_model_from_config(config)
    text_state = configure_text_model(model, config, text_feature_cache)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    weights = checkpoint.get("model", checkpoint.get("state_dict", checkpoint.get("ema", None)))
    if weights is None:
        raise KeyError(f"cannot find model weights in {ckpt_path}")
    ignore_missing = text_conditioning_missing_keys(model) if config.get("text_conditioning", {}).get("enabled", False) else None
    custom_load(model, {key.removeprefix("module."): value for key, value in weights.items()}, ignore_missing_keys=ignore_missing)
    del checkpoint
    model.to(device).eval()
    return model, config, text_state


def build_text_mask(boxes: Sequence[Mapping[str, Any]], image_size: int = 256) -> np.ndarray:
    mask = np.zeros((image_size, image_size), dtype=bool)
    for box in boxes:
        raw = box.get("bbox") or box.get("bbox_256")
        if raw is None or len(raw) != 4:
            continue
        x0, y0, x1, y1 = [float(x) for x in raw]
        x0i = max(0, min(image_size, int(math.floor(x0))))
        y0i = max(0, min(image_size, int(math.floor(y0))))
        x1i = max(0, min(image_size, int(math.ceil(x1))))
        y1i = max(0, min(image_size, int(math.ceil(y1))))
        if x1i > x0i and y1i > y0i:
            mask[y0i:y1i, x0i:x1i] = True
    return mask


def load_rows(manifest: Path, bbox_jsonl: Path, max_images: int, seed: int, smoke: bool) -> List[Dict[str, Any]]:
    bbox_index = load_ocr_box_index(str(bbox_jsonl))
    rows = list(iter_jsonl(manifest))
    if smoke:
        rng = random.Random(seed)
        rows = rng.sample(rows, min(max_images, len(rows)))
    elif max_images > 0:
        rows = rows[:max_images]
    out = []
    missing = []
    for idx, row in enumerate(rows):
        image_path = str(row["image_path"])
        if not Path(image_path).exists():
            missing.append(image_path)
            continue
        bbox_entry = bbox_index.get(image_path)
        if bbox_entry is None:
            raise KeyError(f"bbox missing for {image_path}")
        out.append({"eval_index": idx, "row": row, "bbox_entry": bbox_entry})
    if missing:
        raise FileNotFoundError(f"{len(missing)} missing image paths, first={missing[:3]}")
    return out


def make_batches(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def reconstruct(
        model: torch.nn.Module,
        images: torch.Tensor,
        texts: Sequence[str],
        gates: Optional[torch.Tensor],
        text_state: Mapping[str, Any],
        text_feature_cache: Optional[TextFeatureCache],
        config: Mapping[str, Any],
        device: torch.device,
        mixed_precision: str,
) -> torch.Tensor:
    causal_type = config["model"]["causal_settings"]["causal_type"]
    with torch.inference_mode(), autocast_context(device, mixed_precision):
        if not text_state.get("enabled", False):
            outputs = model(images, causal_type=causal_type)
        else:
            assert text_feature_cache is not None
            cached_features, attention_mask = text_feature_cache.lookup_batch(
                texts,
                device=device,
                dtype=torch.bfloat16 if mixed_precision == "bf16" else None,
            )
            t5_states = (cached_features.detach(),)
            decoder_text_key_padding_mask = ~attention_mask.bool()
            decoder_features_by_layer = {}
            for text_layer, decoder_layer in text_state["layer_pairs"]:
                decoder_features_by_layer[int(decoder_layer)] = t5_states[int(text_layer)].detach()
            outputs = model(
                images,
                causal_type=causal_type,
                selected_decoder_layer=None,
                decoder_text_features=None,
                decoder_text_features_by_layer=decoder_features_by_layer,
                decoder_head_text_features=None,
                decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                text_injection_layers=text_state.get("text_injection_layers") or None,
                visual_memory_mask_enabled=bool(text_state.get("visual_memory_mask_enabled", False)),
                visual_memory_mask_ratio=float(text_state.get("visual_memory_mask_ratio", 0.0)),
                visual_memory_mask_strategy=str(text_state.get("visual_memory_mask_strategy", "token_random")),
                visual_memory_mask_block_size=int(text_state.get("visual_memory_mask_block_size", 1)),
                visual_memory_mask_fixed_pattern=bool(text_state.get("visual_memory_mask_fixed_pattern", False)),
                visual_memory_mask_seed=int(text_state.get("visual_memory_mask_seed", 0)),
                visual_memory_mask_apply_in_eval=bool(text_state.get("visual_memory_mask_apply_in_eval", False)),
                ocr_box_gate=gates,
                ocr_box_gate_layers=config.get("ocr_box_gate", {}).get("apply_layers", None),
            )
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def aggregate(rows: Sequence[Mapping[str, Any]], group_keys: Sequence[str]) -> List[Dict[str, Any]]:
    metric_keys = [key for key in rows[0].keys() if key.endswith(("mse", "mae", "psnr", "ssim"))] if rows else []
    groups: Dict[Tuple[str, ...], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "unknown")) for key in group_keys)].append(row)
    out = []
    for group, items in sorted(groups.items()):
        record = {key: value for key, value in zip(group_keys, group)}
        record["count"] = len(items)
        for key in metric_keys:
            vals = [float(item[key]) for item in items if item.get(key) is not None and math.isfinite(float(item[key]))]
            record[key] = float(np.mean(vals)) if vals else float("nan")
        out.append(record)
    return out


def save_comparison_grid(samples: Sequence[Dict[str, Any]], run_names: Sequence[str], output_path: Path) -> None:
    if not samples:
        return
    cell_w, cell_h = 256, 300
    cols = 1 + len(run_names)
    rows = len(samples)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for r, sample in enumerate(samples):
        y = r * cell_h
        gt = Image.open(sample["gt_path"]).convert("RGB").resize((256, 256))
        canvas.paste(gt, (0, y + 28))
        draw.text((4, y + 4), f"GT {sample['source']} {sample['bucket']}", fill=(0, 0, 0))
        for c, run_name in enumerate(run_names, start=1):
            rec_path = sample["recon_paths"].get(run_name)
            if rec_path and Path(rec_path).exists():
                rec = Image.open(rec_path).convert("RGB").resize((256, 256))
                canvas.paste(rec, (c * cell_w, y + 28))
            label = sample["labels"].get(run_name, run_name)
            draw.text((c * cell_w + 4, y + 4), label[:42], fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def parse_run(value: str) -> Tuple[str, Path, Path, Optional[Path]]:
    parts = value.split(":", 3)
    if len(parts) not in {3, 4}:
        raise argparse.ArgumentTypeError("--run NAME:CONFIG:CKPT[:TEXT_CACHE]")
    name, config, ckpt = parts[:3]
    text_cache = Path(parts[3]) if len(parts) == 4 and parts[3] else None
    return name, Path(config), Path(ckpt), text_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bbox-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--mode", choices=("smoke", "holdout"), default="smoke")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="cuda")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--grid-seed", type=int, default=0)
    args = parser.parse_args()

    device = prepare_device(args.device_backend, args.device_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    items = load_rows(args.manifest, args.bbox_jsonl, args.max_images, seed=args.grid_seed, smoke=args.mode == "smoke")
    if args.shard_count > 1:
        items = [item for idx, item in enumerate(items) if idx % args.shard_count == args.shard_index]

    run_metrics: List[Dict[str, Any]] = []
    per_sample_path = args.output_dir / f"per_sample_metrics.shard{args.shard_index:02d}.jsonl"
    per_sample_path.parent.mkdir(parents=True, exist_ok=True)
    per_sample_handle = per_sample_path.open("w", encoding="utf-8")
    grid_candidates: Dict[str, Dict[str, Any]] = {}

    for run_name, config_path, ckpt_path, text_cache_path in args.run:
        text_cache = TextFeatureCache(text_cache_path) if text_cache_path is not None else None
        model, config, text_state = load_model(config_path, ckpt_path, device, text_cache)
        config_checks = {
            "text_conditioning": bool(config.get("text_conditioning", {}).get("enabled", False)),
            "text_hr": bool(config.get("text_hr", {}).get("enabled", False)),
            "local_similarity_loss": bool(config.get("local_similarity_loss", {}).get("enabled", False)),
            "ocr_visual_alignment_loss": bool(config.get("ocr_visual_alignment_loss", {}).get("enabled", False)),
            "ocr_box_gate": bool(config.get("ocr_box_gate", {}).get("enabled", False)),
            "bbox_jsonl": config.get("ocr_box_gate", {}).get("bbox_jsonl"),
        }
        ckpt_file_hash = sha256_file(ckpt_path)
        ckpt_model_hash, ckpt_step = checkpoint_model_hash(ckpt_path)
        recon_dir = args.output_dir / "reconstructions" / run_name
        recon_dir.mkdir(parents=True, exist_ok=True)

        rows_for_run = []
        for batch_items in make_batches(items, args.batch_size):
            pil_images = [resize_pad_image(Image.open(item["row"]["image_path"])) for item in batch_items]
            image_tensors = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device)
            texts = [str(item["row"].get("text") or "") for item in batch_items]
            gates = None
            if config_checks["ocr_box_gate"]:
                gate_cfg = ocr_box_gate_config_from_dict(config.get("ocr_box_gate", {}), image_size=256)
                gate_tensors = [build_ocr_box_gate(item["bbox_entry"]["boxes"], gate_cfg)[0] for item in batch_items]
                gates = torch.stack(gate_tensors).to(device=device, dtype=torch.float32)
            recons = reconstruct(
                model=model,
                images=image_tensors,
                texts=texts,
                gates=gates,
                text_state=text_state,
                text_feature_cache=text_cache,
                config=config,
                device=device,
                mixed_precision=args.mixed_precision,
            )
            rec_u8_batch = tensor_to_uint8(recons)
            gt_u8_batch = np.stack([np.asarray(img, dtype=np.uint8) for img in pil_images], axis=0)
            for local_idx, item in enumerate(batch_items):
                row = item["row"]
                boxes = item["bbox_entry"].get("boxes", [])
                text_mask = build_text_mask(boxes, image_size=256)
                nontext_mask = ~text_mask
                gt_u8 = gt_u8_batch[local_idx]
                rec_u8 = rec_u8_batch[local_idx]
                image_id = f"{item['eval_index']:06d}_{Path(row['image_path']).stem}"
                rec_path = recon_dir / f"{image_id}.png"
                Image.fromarray(rec_u8).save(rec_path)
                gt_path = args.output_dir / "gt_256" / f"{image_id}.png"
                gt_path.parent.mkdir(parents=True, exist_ok=True)
                if not gt_path.exists():
                    Image.fromarray(gt_u8).save(gt_path)
                whole = metric_values(gt_u8, rec_u8)
                text_vals = metric_values(gt_u8, rec_u8, text_mask)
                nontext_vals = metric_values(gt_u8, rec_u8, nontext_mask)
                sample_metric = {
                    "run": run_name,
                    "image_path": row["image_path"],
                    "source": row.get("source") or row.get("subset") or "unknown",
                    "bucket": row.get("coverage_bucket") or "unknown",
                    "text_mask_coverage": float(text_mask.mean()),
                    "nontext_mask_coverage": float(nontext_mask.mean()),
                    "box_count": int(len(boxes)),
                    "empty_box": bool(len(boxes) == 0),
                    "warning": bool(len(boxes) == 0 or float(text_mask.mean()) < 0.005),
                    "reconstruction_path": str(rec_path),
                    **prefixed("whole", whole),
                    **prefixed("text", text_vals),
                    **prefixed("nontext", nontext_vals),
                }
                per_sample_handle.write(json.dumps(sample_metric, ensure_ascii=False) + "\n")
                rows_for_run.append(sample_metric)
                key = row["image_path"]
                entry = grid_candidates.setdefault(key, {
                    "gt_path": str(gt_path),
                    "source": sample_metric["source"],
                    "bucket": sample_metric["bucket"],
                    "recon_paths": {},
                    "labels": {},
                })
                entry["recon_paths"][run_name] = str(rec_path)
                entry["labels"][run_name] = f"{run_name} PSNR={whole['psnr']:.2f} text={text_vals['psnr']:.2f}"
        del model
        if text_cache is not None:
            del text_cache
        empty_device_cache(device)
        aggregate_all = aggregate(rows_for_run, ["run"])
        aggregate_source_bucket = aggregate(rows_for_run, ["run", "source", "bucket"])
        run_metrics.append({
            "run": run_name,
            "config_path": str(config_path),
            "checkpoint_path": str(ckpt_path),
            "checkpoint_step": ckpt_step,
            "file_sha256": ckpt_file_hash,
            "model_state_hash": ckpt_model_hash,
            "config_checks": config_checks,
            "aggregate": aggregate_all[0] if aggregate_all else {},
            "source_bucket": aggregate_source_bucket,
        })

    per_sample_handle.close()
    write_json(args.output_dir / f"eval_summary.shard{args.shard_index:02d}.json", {
        "mode": args.mode,
        "manifest": str(args.manifest),
        "bbox_jsonl": str(args.bbox_jsonl),
        "sample_count": len(items),
        "ssim_implementation": "skimage.metrics.structural_similarity" if structural_similarity is not None else None,
        "image_range": "[0,1]",
        "data_range": 1.0,
        "runs": run_metrics,
    })
    if args.shard_index == 0:
        rng = random.Random(args.grid_seed)
        samples = list(grid_candidates.values())
        grid_samples = rng.sample(samples, min(20, len(samples))) if samples else []
        save_comparison_grid(grid_samples, [run[0] for run in args.run], args.output_dir / "holdout_qual20_grid.png")


if __name__ == "__main__":
    main()
