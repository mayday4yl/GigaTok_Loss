#!/usr/bin/env python3
"""Check setting-eval image metrics against training validation metrics.

This script recomputes reconstruction MSE/PSNR in the same tensor space used by
``vq_train.py`` validation:

    diff = (reconstruction - gt) * 0.5
    mse = mean(diff ** 2)
    psnr = -10 * log10(mse)

It is intended for debugging metric drift only. It does not retrain and does not
run OCR.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    autocast_context,
    dtype_from_mixed_precision,
    load_manifest_metadata,
    load_tokenizer_model,
    make_text_inputs_for_mode,
    parse_color,
    pil_to_tensor,
    prepare_device,
    reconstruct_batch,
    resize_pad_image,
)
from tokenizer.tokenizer_image.vq.glyph_byt5 import (  # noqa: E402
    get_text_layer_states,
    tokenize_texts_with_stats,
)
from tokenizer.tokenizer_image.vq import vq_train as vq_train_module  # noqa: E402
from tokenizer.tokenizer_image.vq.vq_train import resize_pad_arr  # noqa: E402
from scripts.stage1.ocr_debug.text_feature_cache import TextFeatureCache  # noqa: E402
from scripts.stage1.ocr_debug.evaluate_setting_metrics import (  # noqa: E402
    PAPER_FIELDS,
    csv_value,
    fmt,
    load_runs,
    parse_float,
    paper_rows,
    write_csv,
    write_markdown_table,
)


RUN_ORDER = ["orig", "noocr", "w001", "w002", "w003", "w005"]
SANITY_FIELDS = [
    "run",
    "checkpoint_weight_key",
    "num_images",
    "train_metric_source",
    "train_metric_step",
    "train_val_images",
    "train_final_val_mse",
    "train_final_val_psnr",
    "train_final_val_ssim",
    "setting_eval_tensor_mse",
    "setting_eval_psnr_from_mse",
    "vq_train_compute_mse",
    "vq_train_compute_psnr",
    "ratio",
    "psnr_gap",
    "vq_train_ratio",
    "vq_train_psnr_gap",
    "setting_vs_vq_train_mse_ratio",
    "gt_tensor_min",
    "gt_tensor_max",
    "gt_tensor_mean",
    "recon_tensor_min",
    "recon_tensor_max",
    "recon_tensor_mean",
    "gt_metric_image_min",
    "gt_metric_image_max",
    "gt_metric_image_mean",
    "recon_metric_image_min",
    "recon_metric_image_max",
    "recon_metric_image_mean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-json", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--setting-metrics-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--max-images", type=int, default=0, help="0 means all rows in manifest.")
    parser.add_argument("--text-input-mode", choices=("correct", "empty", "shuffled", "wrong", "length_matched_wrong"), default="correct")
    parser.add_argument("--wrong-text-seed", type=int, default=0)
    parser.add_argument("--fixed-wrong-text", default="THIS IS A FIXED WRONG TEXT 0123456789")
    parser.add_argument("--checkpoint-weight-key", choices=("model", "ema", "auto", "state_dict"), default="model")
    parser.add_argument("--text-layer-pair-index", type=int, default=0)
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--text-feature-cache", type=Path, default=None, help="Optional cache used to mirror cached text validation.")
    return parser.parse_args()


class ManifestTensorDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        metadata_by_path: Mapping[str, Mapping[str, str]],
        *,
        image_size: int,
        pad_color: str,
    ) -> None:
        self.image_paths = list(image_paths)
        self.metadata_by_path = metadata_by_path
        self.image_size = int(image_size)
        color = parse_color(pad_color)
        if color[0] != color[1] or color[0] != color[2]:
            raise ValueError("vq_train resize_pad_arr supports grayscale fill only; use equal RGB pad_color.")
        self.fill = int(color[0])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        with Image.open(path) as img:
            img = resize_pad_arr(img, self.image_size, fill=self.fill)
        tensor = pil_to_tensor(img)
        text = self.metadata_by_path.get(str(path.resolve()), {}).get("text", "")
        return tensor, str(text)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_sanity_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SANITY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in SANITY_FIELDS})


def val_metric_candidate(path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    if not rows:
        return {
            "path": str(path),
            "exists": path.exists(),
            "rows": 0,
            "step": None,
            "val_mse": None,
            "val_psnr": None,
            "val_ssim": None,
            "val_images": None,
        }
    final = rows[-1]
    return {
        "path": str(path),
        "exists": True,
        "rows": len(rows),
        "step": parse_float(final.get("step")),
        "val_mse": parse_float(final.get("val_mse")),
        "val_psnr": parse_float(final.get("val_psnr")),
        "val_ssim": parse_float(final.get("val_ssim")),
        "val_images": parse_float(final.get("val_images")),
    }


def val_metric_candidates(run_dir: Path) -> list[dict[str, Any]]:
    return [
        val_metric_candidate(run_dir / "val.csv"),
        val_metric_candidate(run_dir / "metrics" / "val_metrics.csv"),
    ]


def final_val_metrics(run_dir: Path) -> dict[str, float | None]:
    candidates = val_metric_candidates(run_dir)
    for candidate in candidates:
        if candidate["rows"]:
            return {
                "source": candidate["path"],
                "step": candidate["step"],
                "val_mse": candidate["val_mse"],
                "val_psnr": candidate["val_psnr"],
                "val_ssim": candidate["val_ssim"],
                "val_images": candidate["val_images"],
            }
    return {"source": None, "step": None, "val_mse": None, "val_psnr": None, "val_ssim": None, "val_images": None}


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    y = x.detach().float()
    return {
        "min": float(y.min().item()),
        "max": float(y.max().item()),
        "mean": float(y.mean().item()),
    }


def tensor_hash(x: torch.Tensor) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def causal_type_from_config(config: Mapping[str, Any]) -> str | None:
    return (config.get("model", {}).get("causal_settings", {}) or {}).get("causal_type", None)


def legacy_setting_reconstruct_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    texts: Sequence[str] | None,
    text_context: Any,
    *,
    device_backend: str,
    mixed_precision: str,
) -> torch.Tensor:
    """Reproduce the old setting_eval path before causal_type alignment."""
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    with torch.inference_mode():
        if text_context is None:
            latent, _, info = model.encode(batch)
            indices = info[2]
            return model.decode_code(indices, latent.shape)
        if texts is None:
            raise ValueError("Text-conditioned reconstruction requires text.")

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
            decoder_head_text_features = t5_layer_states[int(text_context.head_text_layer)].detach()
            call_selected_decoder_layer = None
        elif text_context.text_recon_layer_pairs:
            decoder_text_features_by_layer = {
                int(decoder_layer): t5_layer_states[text_layer].detach()
                for text_layer, decoder_layer in text_context.text_recon_layer_pairs
            }
            call_selected_decoder_layer = None
        else:
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


def train_val_like_reconstruct_batch(
    model: torch.nn.Module,
    batch: torch.Tensor,
    texts: Sequence[str] | None,
    text_context: Any,
    *,
    device_backend: str,
    mixed_precision: str,
    causal_type: str | None,
) -> torch.Tensor:
    """Mirror the forward path inside vq_train.compute_reconstruction_metrics for one batch."""
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    selected_decoder_layer = None
    decoder_text_features = None
    decoder_text_features_by_layer = None
    decoder_head_text_features = None
    decoder_text_key_padding_mask = None

    if text_context is not None:
        if texts is None:
            raise ValueError("Text-conditioned validation requires text strings.")
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
        if text_context.text_recon_mode == "residual_head":
            decoder_head_text_features = t5_layer_states[int(text_context.head_text_layer)].detach()
        elif text_context.text_recon_layer_pairs:
            decoder_text_features_by_layer = {
                int(decoder_layer): t5_layer_states[text_layer].detach()
                for text_layer, decoder_layer in text_context.text_recon_layer_pairs
            }
        else:
            selected_text_layer, selected_decoder_layer = text_context.selected_pair
            decoder_text_features = t5_layer_states[selected_text_layer].detach()

    with torch.inference_mode():
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            outputs = model(
                batch,
                causal_type=causal_type,
                selected_decoder_layer=selected_decoder_layer,
                decoder_text_features=decoder_text_features,
                decoder_text_features_by_layer=decoder_text_features_by_layer,
                decoder_head_text_features=decoder_head_text_features,
                decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                text_injection_layers=(text_context.text_injection_layers if text_context is not None else None) or None,
                visual_memory_mask_enabled=bool(getattr(text_context, "visual_memory_mask_enabled", False)),
                visual_memory_mask_ratio=float(getattr(text_context, "visual_memory_mask_ratio", 0.0)),
                visual_memory_mask_strategy=str(getattr(text_context, "visual_memory_mask_strategy", "token_random")),
                visual_memory_mask_block_size=int(getattr(text_context, "visual_memory_mask_block_size", 1)),
                visual_memory_mask_fixed_pattern=bool(getattr(text_context, "visual_memory_mask_fixed_pattern", False)),
                visual_memory_mask_seed=int(getattr(text_context, "visual_memory_mask_seed", 0)),
                visual_memory_mask_apply_in_eval=bool(getattr(text_context, "visual_memory_mask_apply_in_eval", False)),
            )
    return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


def compute_vq_train_metrics_local(
    *,
    model: torch.nn.Module,
    text_context: Any,
    text_feature_cache: TextFeatureCache | None,
    image_paths: Sequence[Path],
    metadata_by_path: Mapping[str, Mapping[str, str]],
    device: torch.device,
    args: argparse.Namespace,
    causal_type: str | None,
) -> dict[str, float]:
    dataset = ManifestTensorDataset(
        image_paths,
        metadata_by_path,
        image_size=args.image_size,
        pad_color=args.pad_color,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    local_args = SimpleNamespace(
        device_backend=args.device_backend,
        mixed_precision=args.mixed_precision,
        val_compute_ssim=True,
    )

    original_all_reduce = vq_train_module.dist.all_reduce
    vq_train_module.dist.all_reduce = lambda tensor, op=None: tensor
    try:
        return vq_train_module.compute_reconstruction_metrics(
            vq_model=model,
            val_loader=loader,
            device=device,
            args=local_args,
            causal_type=causal_type,
            text_tokenizer=(text_context.tokenizer if text_context is not None and text_feature_cache is None else None),
            text_encoder=(text_context.encoder if text_context is not None and text_feature_cache is None else None),
            text_feature_cache=(text_feature_cache if text_context is not None else None),
            text_layer_pairs=(text_context.layer_pairs if text_context is not None else None),
            text_recon_layer_pairs=(text_context.text_recon_layer_pairs if text_context is not None else None),
            text_injection_layers=(text_context.text_injection_layers if text_context is not None else None),
            text_recon_mode=(text_context.text_recon_mode if text_context is not None else None),
            text_recon_head_text_layer=(text_context.head_text_layer if text_context is not None else None),
            visual_memory_mask_enabled=bool(getattr(text_context, "visual_memory_mask_enabled", False)),
            visual_memory_mask_ratio=float(getattr(text_context, "visual_memory_mask_ratio", 0.0)),
            visual_memory_mask_strategy=str(getattr(text_context, "visual_memory_mask_strategy", "token_random")),
            visual_memory_mask_block_size=int(getattr(text_context, "visual_memory_mask_block_size", 1)),
            visual_memory_mask_fixed_pattern=bool(getattr(text_context, "visual_memory_mask_fixed_pattern", False)),
            visual_memory_mask_seed=int(getattr(text_context, "visual_memory_mask_seed", 0)),
            visual_memory_mask_apply_in_eval=bool(getattr(text_context, "visual_memory_mask_apply_in_eval", False)),
            text_max_length=(text_context.max_length if text_context is not None else None),
            mixed_precision_dtype=dtype_from_mixed_precision(args.mixed_precision),
        )
    finally:
        vq_train_module.dist.all_reduce = original_all_reduce


def evaluate_tensor_metrics(
    *,
    run_name: str,
    spec: Mapping[str, Any],
    image_paths: Sequence[Path],
    metadata_by_path: Mapping[str, Mapping[str, str]],
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    model, config, text_context = load_tokenizer_model(
        Path(str(spec["config"])),
        Path(str(spec["ckpt"])),
        device,
        text_layer_pair_index=args.text_layer_pair_index,
        checkpoint_weight_key=args.checkpoint_weight_key,
    )
    causal_type = causal_type_from_config(config)
    text_feature_cache = None
    if text_context is not None and args.text_feature_cache is not None:
        text_feature_cache = TextFeatureCache(args.text_feature_cache)
        text_feature_cache.validate_config(config.get("text_conditioning", {}))
    pad_color = parse_color(args.pad_color)
    correct_texts = [
        metadata_by_path.get(str(path.resolve()), {}).get("text", "")
        for path in image_paths
    ]
    text_inputs_for_model = make_text_inputs_for_mode(
        correct_texts,
        args.text_input_mode,
        args.wrong_text_seed,
        args.fixed_wrong_text,
    )

    mse_sum = 0.0
    elem_count = 0
    range_stats: dict[str, float] | None = None

    for start in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[start:start + args.batch_size]
        pil_images = [resize_pad_image(Image.open(path), args.image_size, pad_color) for path in batch_paths]
        batch = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        batch_texts = text_inputs_for_model[start:start + len(batch_paths)]
        rec = reconstruct_batch(
            model,
            batch,
            batch_texts if text_context is not None else None,
            text_context,
            device_backend=args.device_backend,
            mixed_precision=args.mixed_precision,
            causal_type=causal_type,
        )
        legacy_rec = None
        train_like_rec = None
        if start == 0 and args.debug_dir is not None:
            train_like_rec = train_val_like_reconstruct_batch(
                model,
                batch,
                batch_texts if text_context is not None else None,
                text_context,
                device_backend=args.device_backend,
                mixed_precision=args.mixed_precision,
                causal_type=causal_type,
            )
            legacy_rec = legacy_setting_reconstruct_batch(
                model,
                batch,
                batch_texts if text_context is not None else None,
                text_context,
                device_backend=args.device_backend,
                mixed_precision=args.mixed_precision,
            )
        diff = (rec.float() - batch.float()) * 0.5
        mse_sum += float(diff.pow(2).sum().item())
        elem_count += int(diff.numel())

        if range_stats is None:
            gt_tensor = tensor_stats(batch)
            rec_tensor = tensor_stats(rec)
            gt_metric = tensor_stats(torch.clamp((batch.float() + 1.0) * 0.5, 0.0, 1.0))
            rec_metric = tensor_stats(torch.clamp((rec.float() + 1.0) * 0.5, 0.0, 1.0))
            range_stats = {
                "gt_tensor_min": gt_tensor["min"],
                "gt_tensor_max": gt_tensor["max"],
                "gt_tensor_mean": gt_tensor["mean"],
                "recon_tensor_min": rec_tensor["min"],
                "recon_tensor_max": rec_tensor["max"],
                "recon_tensor_mean": rec_tensor["mean"],
                "gt_metric_image_min": gt_metric["min"],
                "gt_metric_image_max": gt_metric["max"],
                "gt_metric_image_mean": gt_metric["mean"],
                "recon_metric_image_min": rec_metric["min"],
                "recon_metric_image_max": rec_metric["max"],
                "recon_metric_image_mean": rec_metric["mean"],
            }
            if legacy_rec is not None and train_like_rec is not None and args.debug_dir is not None:
                fixed_mse = float(diff.pow(2).mean().item())
                train_like_diff = (train_like_rec.float() - batch.float()) * 0.5
                train_like_mse = float(train_like_diff.pow(2).mean().item())
                legacy_diff = (legacy_rec.float() - batch.float()) * 0.5
                legacy_mse = float(legacy_diff.pow(2).mean().item())
                compare_lines = [
                    f"run: {run_name}",
                    f"config: {spec['config']}",
                    f"ckpt: {spec['ckpt']}",
                    f"causal_type: {causal_type}",
                    f"text_context: {text_context is not None}",
                    f"text_recon_mode: {getattr(text_context, 'text_recon_mode', None)}",
                    f"text_injection_layers: {getattr(text_context, 'text_injection_layers', None)}",
                    f"text_recon_layer_pairs: {getattr(text_context, 'text_recon_layer_pairs', None)}",
                    f"visual_memory_mask_enabled: {getattr(text_context, 'visual_memory_mask_enabled', None)}",
                    f"visual_memory_mask_ratio: {getattr(text_context, 'visual_memory_mask_ratio', None)}",
                    f"visual_memory_mask_apply_in_eval: {getattr(text_context, 'visual_memory_mask_apply_in_eval', None)}",
                    f"uses_text_feature_cache_for_vq_train_compute: {text_feature_cache is not None}",
                    f"gt_tensor_min/max/mean: {gt_tensor['min']:.8g} {gt_tensor['max']:.8g} {gt_tensor['mean']:.8g}",
                    f"setting_recon_tensor_min/max/mean: {rec_tensor['min']:.8g} {rec_tensor['max']:.8g} {rec_tensor['mean']:.8g}",
                    f"train_like_recon_tensor_min/max/mean: {tensor_stats(train_like_rec)['min']:.8g} {tensor_stats(train_like_rec)['max']:.8g} {tensor_stats(train_like_rec)['mean']:.8g}",
                    f"legacy_recon_tensor_min/max/mean: {tensor_stats(legacy_rec)['min']:.8g} {tensor_stats(legacy_rec)['max']:.8g} {tensor_stats(legacy_rec)['mean']:.8g}",
                    f"gt_tensor_max_abs_diff_A_vs_B: 0.0",
                    f"recon_tensor_max_abs_diff_vq_train_like_vs_setting_eval: {float((train_like_rec.float() - rec.float()).abs().max().item()):.8g}",
                    f"recon_tensor_max_abs_diff_setting_eval_vs_legacy: {float((rec.float() - legacy_rec.float()).abs().max().item()):.8g}",
                    f"recon_tensor_max_abs_diff_vq_train_like_vs_legacy: {float((train_like_rec.float() - legacy_rec.float()).abs().max().item()):.8g}",
                    f"mse_A_vq_train_like_batch: {train_like_mse:.10g}",
                    f"mse_B_setting_eval_batch: {fixed_mse:.10g}",
                    f"mse_B_legacy_setting: {legacy_mse:.10g}",
                    f"psnr_A_vq_train_like_batch: {-10.0 * math.log10(max(train_like_mse, 1e-12)):.8g}",
                    f"psnr_B_setting_eval_batch: {-10.0 * math.log10(max(fixed_mse, 1e-12)):.8g}",
                    f"psnr_C_legacy_setting: {-10.0 * math.log10(max(legacy_mse, 1e-12)):.8g}",
                ]
                args.debug_dir.mkdir(parents=True, exist_ok=True)
                (args.debug_dir / f"path_compare_{run_name}.txt").write_text(
                    "\n".join(compare_lines) + "\n",
                    encoding="utf-8",
                )

    mse = mse_sum / max(1, elem_count)
    psnr = -10.0 * math.log10(max(mse, 1e-12))
    vq_train_metrics = compute_vq_train_metrics_local(
        model=model,
        text_context=text_context,
        text_feature_cache=text_feature_cache,
        image_paths=image_paths,
        metadata_by_path=metadata_by_path,
        device=device,
        args=args,
        causal_type=causal_type,
    )
    out = {
        "run": run_name,
        "checkpoint_weight_key": args.checkpoint_weight_key,
        "num_images": len(image_paths),
        "setting_eval_tensor_mse": mse,
        "setting_eval_psnr_from_mse": psnr,
        "vq_train_compute_mse": vq_train_metrics.get("val_mse"),
        "vq_train_compute_psnr": vq_train_metrics.get("val_psnr"),
    }
    if range_stats:
        out.update(range_stats)
    del model
    if args.device_backend == "cuda":
        torch.cuda.empty_cache()
    elif args.device_backend == "npu":
        torch.npu.empty_cache()
    return out


def read_setting_metrics(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    return {row.get("run", ""): row for row in read_csv_rows(path)}


def write_sanity_md(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Setting Eval Metric Sanity Check",
        "",
        "| run | train val_mse | setting tensor mse | setting ratio | vq_train mse | vq_train ratio | train PSNR | setting PSNR | setting gap | vq_train PSNR | vq_train gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {run} | {train_mse} | {tensor_mse} | {ratio} | {vq_mse} | {vq_ratio} | {train_psnr} | {tensor_psnr} | {gap} | {vq_psnr} | {vq_gap} |".format(
                run=row.get("run", ""),
                train_mse=fmt(row.get("train_final_val_mse"), 6),
                tensor_mse=fmt(row.get("setting_eval_tensor_mse"), 6),
                ratio=fmt(row.get("ratio"), 3),
                vq_mse=fmt(row.get("vq_train_compute_mse"), 6),
                vq_ratio=fmt(row.get("vq_train_ratio"), 3),
                train_psnr=fmt(row.get("train_final_val_psnr"), 4),
                tensor_psnr=fmt(row.get("setting_eval_psnr_from_mse"), 4),
                gap=fmt(row.get("psnr_gap"), 4),
                vq_psnr=fmt(row.get("vq_train_compute_psnr"), 4),
                vq_gap=fmt(row.get("vq_train_psnr_gap"), 4),
            )
        )
    lines.extend([
        "",
        "The tensor metric uses `diff = (reconstruction - gt) * 0.5`, matching `vq_train.py` validation.",
        "`vq_train mse/PSNR` is computed by calling `vq_train.compute_reconstruction_metrics()` in this script with distributed all-reduce patched to a no-op.",
        "If the ratio is still far from 1, inspect checkpoint weight key, preprocessing, and text-conditioning path.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_valpsnr_tables(
    *,
    output_dir: Path,
    runs: Mapping[str, Mapping[str, Any]],
    sanity_rows: Sequence[Mapping[str, Any]],
    setting_metrics: Mapping[str, Mapping[str, str]],
) -> None:
    by_run = {str(row.get("run")): row for row in sanity_rows}
    rows = []
    for run, spec in runs.items():
        train = by_run.get(run, {})
        setting = setting_metrics.get(run, {})
        rows.append({
            "Setting": run,
            "Text Injection": "yes" if spec.get("text_injection") else "no",
            "OCR Loss": "yes" if spec.get("ocr_loss") else "no",
            "OCR Weight": "-" if spec.get("ocr_weight") is None else spec.get("ocr_weight"),
            "PSNR↑": fmt(train.get("train_final_val_psnr"), 3),
            "SSIM↑": fmt(train.get("train_final_val_ssim"), 4),
            "LPIPS↓": "",
            "Third-party OCR CER↓": fmt(setting.get("thirdparty_ocr_cer"), 4),
            "Third-party OCR NED↑": fmt(setting.get("thirdparty_ocr_ned"), 4),
            "DeepSeek OCR CER↓": fmt(setting.get("deepseek_ocr_cer"), 4),
            "Best Setting Note": "image metrics from training val.csv; OCR metrics from setting_eval",
        })
    write_csv(output_dir / "paper_table_valpsnr.csv", PAPER_FIELDS, rows)
    write_markdown_table(
        output_dir / "paper_table_valpsnr.md",
        PAPER_FIELDS,
        rows,
        [
            "Image quality columns use each run's training `val.csv` final `val_psnr` / `val_ssim`.",
            "OCR columns are copied from `setting_eval/setting_metrics.csv`.",
            "Use this table until setting-eval image metric alignment is confirmed.",
        ],
    )


def extract_logged_val_manifest(run_dir: Path) -> str:
    """Best-effort extraction of the validation manifest used during training."""
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Validation dataset contains [^\n]*\(([^)]+)\)", text)
    if match:
        return match.group(1)
    match = re.search(r"val_json_path=['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)
    return ""


def write_manifest_check(
    path: Path,
    manifest_jsonl: Path,
    image_paths: Sequence[Path],
    metadata_by_path: Mapping[str, Mapping[str, str]],
    runs: Mapping[str, Mapping[str, Any]],
) -> None:
    lines = [
        f"setting_eval manifest path: {manifest_jsonl}",
        f"num images: {len(image_paths)}",
        "",
        "training val manifest paths from run logs:",
    ]
    for run, spec in runs.items():
        run_dir = Path(str(spec["run_dir"]))
        logged = extract_logged_val_manifest(run_dir)
        lines.append(f"{run}: {logged or '<not found in log.txt>'}")
    lines.extend([
        "",
        "first 5 image_path/text:",
    ])
    for idx, image_path in enumerate(image_paths[:5]):
        meta = metadata_by_path.get(str(image_path.resolve()), {})
        lines.append(f"{idx}: image_path={image_path}")
        lines.append(f"{idx}: text={meta.get('text', '')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_preprocess_check(path: Path, image_paths: Sequence[Path], image_size: int, pad_color: str) -> None:
    color = parse_color(pad_color)
    with Image.open(image_paths[0]) as img:
        setting_pil = resize_pad_image(img, image_size, color)
    with Image.open(image_paths[0]) as img:
        train_pil = resize_pad_arr(img, image_size, fill=color[0])
    setting_tensor = pil_to_tensor(setting_pil)
    train_tensor = pil_to_tensor(train_pil)
    lines = [
        f"image_path: {image_paths[0]}",
        f"image_size: {image_size}",
        f"pad_color: {color}",
        f"setting tensor shape: {tuple(setting_tensor.shape)}",
        f"train tensor shape: {tuple(train_tensor.shape)}",
        f"max_abs_diff: {float((setting_tensor - train_tensor).abs().max().item()):.8g}",
        f"setting min/max/mean: {float(setting_tensor.min()):.8g} {float(setting_tensor.max()):.8g} {float(setting_tensor.mean()):.8g}",
        f"train min/max/mean: {float(train_tensor.min()):.8g} {float(train_tensor.max()):.8g} {float(train_tensor.mean()):.8g}",
        f"setting hash: {tensor_hash(setting_tensor)}",
        f"train hash: {tensor_hash(train_tensor)}",
        f"setting first pixels: {setting_tensor.flatten()[:12].tolist()}",
        f"train first pixels: {train_tensor.flatten()[:12].tolist()}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config_ckpt_check(path: Path, runs: Mapping[str, Mapping[str, Any]]) -> None:
    import yaml

    lines = []
    for run, spec in runs.items():
        cfg_path = Path(str(spec["config"]))
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        text_cfg = cfg.get("text_conditioning", {})
        text_recon = cfg.get("text_recon_conditioning", {})
        ocr_tf = cfg.get("ocr_teacher_forcing_loss", {})
        text_hr = cfg.get("text_hr", {})
        trainer = cfg.get("trainer", {})
        ckpt_path = Path(str(spec["ckpt"]))
        ckpt_info: dict[str, Any] = {}
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                if isinstance(ckpt, Mapping):
                    ckpt_info = {
                        "keys": sorted(str(key) for key in ckpt.keys()),
                        "steps": ckpt.get("steps"),
                        "has_model": "model" in ckpt,
                        "has_ema": "ema" in ckpt,
                        "has_state_dict": "state_dict" in ckpt,
                    }
                del ckpt
            except Exception as exc:  # pragma: no cover - diagnostic only
                ckpt_info = {"error": repr(exc)}
        lines.extend([
            f"[{run}]",
            f"config path: {cfg_path}",
            f"ckpt path: {ckpt_path}",
            f"ckpt exists: {ckpt_path.exists()}",
            f"ckpt info: {ckpt_info}",
            f"run_dir: {spec.get('run_dir')}",
            f"logged val manifest: {extract_logged_val_manifest(Path(str(spec['run_dir'])))}",
            f"text_conditioning.enabled: {text_cfg.get('enabled')}",
            f"text_recon_conditioning.enabled: {text_recon.get('enabled')}",
            f"text_recon_conditioning.mode: {text_recon.get('mode')}",
            f"text_recon_conditioning.layers: {text_recon.get('layers')}",
            f"text_recon_conditioning.layer_pairs: {text_recon.get('layer_pairs')}",
            f"text_recon_conditioning.visual_memory_mask: {text_recon.get('visual_memory_mask')}",
            f"text_recon_conditioning.residual_cross_attn: {text_recon.get('residual_cross_attn')}",
            f"ocr_teacher_forcing_loss.enabled: {ocr_tf.get('enabled')}",
            f"ocr_teacher_forcing_loss.weight: {ocr_tf.get('weight')}",
            f"text_hr.enabled: {text_hr.get('enabled')}",
            f"trainer.hr_on: {trainer.get('hr_on')}",
            f"trainer.hr_loss_weight: {trainer.get('hr_loss_weight')}",
            f"model.causal_settings.causal_type: {causal_type_from_config(cfg)}",
            "",
        ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_train_metric_source_check(path: Path, runs: Mapping[str, Mapping[str, Any]]) -> None:
    lines = []
    for run, spec in runs.items():
        run_dir = Path(str(spec["run_dir"]))
        lines.append(f"[{run}]")
        lines.append(f"run_dir: {run_dir}")
        for candidate in val_metric_candidates(run_dir):
            lines.append(
                "candidate: path={path} exists={exists} rows={rows} step={step} "
                "val_mse={val_mse} val_psnr={val_psnr} val_ssim={val_ssim} val_images={val_images}".format(
                    **candidate
                )
            )
        log_path = run_dir / "log.txt"
        if log_path.exists():
            val_lines = [
                line.strip()
                for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if "Val MSE:" in line
            ]
            lines.append("last log Val MSE lines:")
            lines.extend(val_lines[-3:] or ["<none>"])
        else:
            lines.append("log.txt: <missing>")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.runs_json)
    image_paths, metadata_by_path = load_manifest_metadata(args.manifest_jsonl)
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise SystemExit("no images to evaluate")
    if args.debug_dir is not None:
        args.debug_dir.mkdir(parents=True, exist_ok=True)
        write_manifest_check(args.debug_dir / "manifest_check.txt", args.manifest_jsonl, image_paths, metadata_by_path, runs)
        write_preprocess_check(args.debug_dir / "preprocess_check.txt", image_paths, args.image_size, args.pad_color)
        write_config_ckpt_check(args.debug_dir / "config_ckpt_check.txt", runs)
        write_train_metric_source_check(args.debug_dir / "train_metric_source_check.txt", runs)

    device = prepare_device(args.device_backend, args.device_id)
    rows = []
    for run, spec in runs.items():
        print(f"== sanity {run} ({args.checkpoint_weight_key}) ==", flush=True)
        train = final_val_metrics(Path(str(spec["run_dir"])))
        row = evaluate_tensor_metrics(
            run_name=run,
            spec=spec,
            image_paths=image_paths,
            metadata_by_path=metadata_by_path,
            device=device,
            args=args,
        )
        row["train_final_val_mse"] = train["val_mse"]
        row["train_final_val_psnr"] = train["val_psnr"]
        row["train_final_val_ssim"] = train["val_ssim"]
        row["train_metric_source"] = train["source"]
        row["train_metric_step"] = train["step"]
        row["train_val_images"] = train["val_images"]
        if train["val_mse"] is not None:
            row["ratio"] = row["setting_eval_tensor_mse"] / train["val_mse"]
            if row.get("vq_train_compute_mse") is not None:
                row["vq_train_ratio"] = row["vq_train_compute_mse"] / train["val_mse"]
            if row.get("vq_train_compute_mse") is not None and row["vq_train_compute_mse"] != 0:
                row["setting_vs_vq_train_mse_ratio"] = row["setting_eval_tensor_mse"] / row["vq_train_compute_mse"]
        if train["val_psnr"] is not None:
            row["psnr_gap"] = row["setting_eval_psnr_from_mse"] - train["val_psnr"]
            if row.get("vq_train_compute_psnr") is not None:
                row["vq_train_psnr_gap"] = row["vq_train_compute_psnr"] - train["val_psnr"]
        rows.append(row)

    write_sanity_csv(args.output_dir / "metric_sanity.csv", rows)
    write_sanity_md(args.output_dir / "metric_sanity.md", rows)
    write_sanity_md(args.output_dir / "metric_sanity_fixed.md", rows)
    if args.debug_dir is not None:
        write_sanity_md(args.debug_dir / "metric_sanity_fixed.md", rows)
    setting_metrics = read_setting_metrics(args.setting_metrics_csv)
    write_valpsnr_tables(
        output_dir=args.output_dir,
        runs=runs,
        sanity_rows=rows,
        setting_metrics=setting_metrics,
    )
    print(f"sanity csv: {args.output_dir / 'metric_sanity.csv'}")
    print(f"sanity md: {args.output_dir / 'metric_sanity.md'}")
    print(f"valpsnr table: {args.output_dir / 'paper_table_valpsnr.md'}")


if __name__ == "__main__":
    main()
