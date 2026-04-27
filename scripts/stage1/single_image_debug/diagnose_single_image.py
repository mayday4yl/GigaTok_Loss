#!/usr/bin/env python3
"""单图逐层 text-HR 诊断：重建、cross-attention SVD、特征统计。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    autocast_context,
    dtype_from_mixed_precision,
    load_tokenizer_model,
    metric_values,
    parse_color,
    pil_to_tensor,
    prepare_device,
    resize_pad_image,
    tensor_to_uint8,
    write_json,
)
from tokenizer.tokenizer_image.vq.vq_loss import (  # noqa: E402
    _frobenius_normalize_attention_matrix,
    _gram_scaled_identity_loss,
    _log_participation_ratio_loss,
    _sigma_mean_mse_loss,
    _singular_energy_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--layer-mode", choices=("config_pairs", "all_decoder"), default="all_decoder")
    parser.add_argument("--text-layer-mode", choices=("same", "fixed"), default="same")
    parser.add_argument("--fixed-text-layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0, help="Seed used before model construction for deterministic missing-key init.")
    parser.add_argument("--tau", type=float, default=None, help="Override text_hr.tau from config for reporting current code loss.")
    parser.add_argument("--save-npy", action="store_true", default=True)
    parser.add_argument("--no-save-npy", action="store_false", dest="save_npy")
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def first_manifest_row(path: Path) -> Dict[str, Any]:
    for row in iter_jsonl(path):
        return row
    raise ValueError(f"No rows found in {path}")


def resolve_image_path(row: Mapping[str, Any], manifest_jsonl: Path) -> Path:
    image_path = Path(str(row["image_path"]))
    if not image_path.is_absolute():
        image_path = manifest_jsonl.resolve().parent / image_path
    return image_path


def tensor_stats(prefix: str, tensor: torch.Tensor) -> Dict[str, float]:
    x = tensor.detach().float()
    return {
        f"{prefix}_mean": float(x.mean().item()),
        f"{prefix}_std": float(x.std(unbiased=False).item()),
        f"{prefix}_rms": float(torch.sqrt(torch.mean(x * x)).item()),
        f"{prefix}_min": float(x.min().item()),
        f"{prefix}_max": float(x.max().item()),
        f"{prefix}_max_abs": float(x.abs().max().item()),
    }


def scalar_mean(value: Any) -> float:
    """把 tensor/list/dict 形式的辅助 loss 汇总成一个诊断标量。"""
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Mapping):
        values = [scalar_mean(item) for item in value.values()]
    elif isinstance(value, (list, tuple)):
        values = [scalar_mean(item) for item in value]
    else:
        return float("nan")
    values = [item for item in values if math.isfinite(item)]
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "npu"):
        try:
            torch.npu.manual_seed_all(seed)
        except Exception:
            pass


def normalized_spectrum_loss(sigma: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = sigma / (sigma.sum() + eps)
    r = sigma.numel()
    loss = (p - (1.0 / max(1, r))).pow(2).mean()
    entropy = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum()
    effective_rank = entropy.exp()
    return loss, p, effective_rank


def rank_at_energy(p: torch.Tensor, threshold: float) -> int:
    if p.numel() == 0:
        return 0
    cdf = torch.cumsum(p, dim=0)
    return int((cdf < threshold).sum().item() + 1)


def attention_svd_metrics(
        attn: torch.Tensor,
        text_attention_mask: torch.Tensor,
        image_token_len: int,
        tau: float,
        out_npy: Optional[Path],
) -> Dict[str, float]:
    if attn.dim() != 4 or attn.shape[0] != 1:
        raise ValueError(f"Expected attention shape [1,H,Q,K], got {tuple(attn.shape)}")
    _, num_heads, num_queries, num_keys = attn.shape
    valid_mask = text_attention_mask[0].bool().to(attn.device)
    valid_text_tokens = int(valid_mask.sum().item())
    text_token_len = int(text_attention_mask.shape[1])
    if num_keys < image_token_len + text_token_len:
        raise ValueError(
            f"Attention key length {num_keys} < image_token_len + text_token_len "
            f"({image_token_len} + {text_token_len})"
        )
    if valid_text_tokens < 1:
        raise ValueError("No valid text tokens for SVD diagnosis.")

    text_attn = attn[:, :, :, image_token_len:image_token_len + text_token_len]
    valid_text_attn = text_attn[0, :, :, valid_mask]
    text_mass = valid_text_attn.sum(dim=-1)
    matrix = valid_text_attn.reshape(num_heads * num_queries, valid_text_tokens).float()
    legacy_matrix = matrix / math.sqrt(num_heads)
    matrix_normed, fro_norm = _frobenius_normalize_attention_matrix(matrix, eps=1e-8)
    sigma = torch.linalg.svdvals(matrix)
    legacy_sigma = torch.linalg.svdvals(legacy_matrix)
    sigma_fro = torch.linalg.svdvals(matrix_normed)
    sigma_sorted = sigma.detach().cpu()
    sigma_fro_sorted = sigma_fro.detach().cpu()
    if out_npy is not None:
        out_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_npy, sigma_sorted.numpy())
        np.save(out_npy.with_name(f"{out_npy.stem}_frobenius{out_npy.suffix}"), sigma_fro_sorted.numpy())

    norm_loss, p, effective_rank = normalized_spectrum_loss(sigma)
    raw_energy_stats = _singular_energy_stats(sigma, eps=1e-8)
    fro_energy_stats = _singular_energy_stats(sigma_fro, eps=1e-8)
    raw_energy_p = raw_energy_stats["energy"]
    fro_p = fro_energy_stats["energy"]
    raw_energy_loss = (raw_energy_p - sigma.new_tensor(1.0 / max(1, raw_energy_p.numel()))).pow(2).mean()
    fro_loss = (fro_p - sigma_fro.new_tensor(1.0 / max(1, fro_p.numel()))).pow(2).mean()
    current_loss = torch.abs(legacy_sigma - legacy_sigma.new_tensor(tau)).mean()
    top1_ratio = p.max() if p.numel() else sigma.new_tensor(0.0)
    top5_ratio = torch.topk(p, min(5, p.numel())).values.sum() if p.numel() else sigma.new_tensor(0.0)
    sigma_mean_mse_loss = _sigma_mean_mse_loss(sigma_fro)
    gram_scaled_identity_loss = _gram_scaled_identity_loss(matrix_normed)
    log_participation_ratio_loss = _log_participation_ratio_loss(sigma_fro, eps=1e-8)

    return {
        "num_heads": float(num_heads),
        "num_queries": float(num_queries),
        "num_keys": float(num_keys),
        "valid_text_tokens": float(valid_text_tokens),
        "text_attention_mass_mean": float(text_mass.float().mean().item()),
        "text_attention_mass_std": float(text_mass.float().std(unbiased=False).item()),
        "sigma_count": float(sigma.numel()),
        "sigma_mean": float(sigma.mean().item()),
        "sigma_std": float(sigma.std(unbiased=False).item()),
        "sigma_min": float(sigma.min().item()),
        "sigma_max": float(sigma.max().item()),
        "sigma_top1": float(sigma[0].item()),
        "sigma_top1_ratio": float(top1_ratio.item()),
        "sigma_top5_ratio": float(top5_ratio.item()),
        "sum_normalized_effective_rank": float(effective_rank.item()),
        "rank90": float(rank_at_energy(p, 0.90)),
        "rank95": float(rank_at_energy(p, 0.95)),
        "raw_energy_top1_ratio": float(raw_energy_stats["energy_top1"].item()),
        "raw_energy_top5_ratio": float(raw_energy_stats["energy_top5"].item()),
        "raw_energy_effective_rank": float(raw_energy_stats["effective_rank"].item()),
        "raw_energy_rank90": float(rank_at_energy(raw_energy_p, 0.90)),
        "raw_energy_rank95": float(rank_at_energy(raw_energy_p, 0.95)),
        "raw_energy_uniform_loss": float(raw_energy_loss.item()),
        "frobenius_norm": float(fro_norm.item()),
        "normed_sigma_mean": float(sigma_fro.mean().item()),
        "normed_sigma_std": float(sigma_fro.std(unbiased=False).item()),
        "normed_sigma_min": float(sigma_fro.min().item()),
        "normed_sigma_max": float(sigma_fro.max().item()),
        "frobenius_sigma_mean": float(sigma_fro.mean().item()),
        "frobenius_sigma_std": float(sigma_fro.std(unbiased=False).item()),
        "frobenius_sigma_min": float(sigma_fro.min().item()),
        "frobenius_sigma_max": float(sigma_fro.max().item()),
        "frobenius_sigma_top1": float(sigma_fro[0].item()),
        "frobenius_energy_top1_ratio": float(fro_energy_stats["energy_top1"].item()),
        "frobenius_energy_top5_ratio": float(fro_energy_stats["energy_top5"].item()),
        "frobenius_effective_rank": float(fro_energy_stats["effective_rank"].item()),
        "frobenius_rank90": float(rank_at_energy(fro_p, 0.90)),
        "frobenius_rank95": float(rank_at_energy(fro_p, 0.95)),
        "frobenius_uniform_loss": float(fro_loss.item()),
        "participation_rank": float(fro_energy_stats["participation_rank"].item()),
        "participation_rank_ratio": float(fro_energy_stats["participation_rank_ratio"].item()),
        "effective_rank": float(fro_energy_stats["effective_rank"].item()),
        "energy_top1_ratio": float(fro_energy_stats["energy_top1"].item()),
        "sigma_mean_mse_loss": float(sigma_mean_mse_loss.item()),
        "gram_scaled_identity_loss": float(gram_scaled_identity_loss.item()),
        "log_participation_ratio_loss": float(log_participation_ratio_loss.item()),
        "current_code_tau_loss": float(current_loss.item()),
        "normalized_spectrum_loss": float(norm_loss.item()),
    }


def save_tensor_image(path: Path, tensor: torch.Tensor) -> np.ndarray:
    arr = tensor_to_uint8(tensor.detach().float())[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)
    return arr


def save_absdiff(path: Path, target_path: Path, recon_path: Path) -> None:
    target = Image.open(target_path).convert("RGB")
    recon = Image.open(recon_path).convert("RGB")
    diff = ImageChops.difference(target, recon)
    path.parent.mkdir(parents=True, exist_ok=True)
    diff.save(path)


def build_pairs(args: argparse.Namespace, model: torch.nn.Module, text_context: Any) -> List[Tuple[int, int]]:
    if text_context is None:
        raise ValueError("This diagnostic requires text_conditioning.enabled=True.")
    if args.layer_mode == "config_pairs":
        return list(text_context.layer_pairs)

    decoder_num_layers = int(getattr(model.s1to2decoder, "num_layers"))
    text_num_layers = int(getattr(text_context.encoder.config, "num_layers"))
    pairs = []
    for decoder_layer in range(decoder_num_layers):
        if args.text_layer_mode == "fixed":
            text_layer = args.fixed_text_layer
        else:
            text_layer = min(decoder_layer, text_num_layers - 1)
        if text_layer < 0 or text_layer >= text_num_layers:
            raise ValueError(f"text_layer={text_layer} is out of range for T5 num_layers={text_num_layers}")
        pairs.append((text_layer, decoder_layer))
    return pairs


def attach_layer_hooks(model: torch.nn.Module, feature_stats: Dict[int, Dict[str, float]]):
    handles = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                feature_stats[layer_idx] = tensor_stats("decoder_layer_feature", tensor)
        return hook

    for layer_idx, block in enumerate(model.s1to2decoder.transformer):
        handles.append(block.register_forward_hook(make_hook(layer_idx)))
    return handles


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    row = first_manifest_row(args.manifest_jsonl)
    image_path = resolve_image_path(row, args.manifest_jsonl)
    text = str(row.get("text") or "")
    if not text:
        raise ValueError("Single-image diagnosis requires non-empty text.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = prepare_device(args.device_backend, args.device_id)
    model, config, text_context = load_tokenizer_model(
        args.config,
        args.ckpt,
        device,
        text_layer_pair_index=0,
    )
    pairs = build_pairs(args, model, text_context)
    text_hr_cfg = config.get("text_hr", {})
    image_token_len = int(text_hr_cfg.get("image_token_len", 256))
    tau = float(args.tau if args.tau is not None else text_hr_cfg.get("tau", 1.0))
    ptdtype = dtype_from_mixed_precision(args.mixed_precision)

    pad_color = parse_color(args.pad_color)
    pil = resize_pad_image(Image.open(image_path).convert("RGB"), args.image_size, pad_color)
    target_path = args.output_dir / "target_256.png"
    pil.save(target_path)
    image = pil_to_tensor(pil).unsqueeze(0).to(device)

    tokenized = text_context.tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=text_context.max_length,
        return_tensors="pt",
    )
    input_ids = tokenized["input_ids"].to(device)
    text_attention_mask = tokenized["attention_mask"].to(device)
    text_key_padding_mask = ~text_attention_mask.bool()

    with torch.inference_mode():
        with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
            text_outputs = text_context.encoder(
                input_ids=input_ids,
                attention_mask=text_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            quant, diff, encoder_spatial = model.encode(image, return_code=False)

    hidden_states = text_outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("T5 encoder did not return hidden states.")

    rows = []
    saved_recons = {}
    with torch.inference_mode():
        for text_layer, decoder_layer in pairs:
            hidden_state_index = text_layer + 1
            if hidden_state_index >= len(hidden_states):
                raise ValueError(
                    f"text_layer={text_layer} maps to hidden_states[{hidden_state_index}], "
                    f"but T5 returned {len(hidden_states)} hidden states."
                )
            decoder_text_features = hidden_states[hidden_state_index].detach()
            with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
                text_memory = model.project_text_memory(decoder_text_features)

            feature_stats: Dict[int, Dict[str, float]] = {}
            hooks = attach_layer_hooks(model, feature_stats)
            try:
                with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
                    recon, rec_spatial, attn = model.decode(
                        quant,
                        selected_decoder_layer=decoder_layer,
                        decoder_text_features=decoder_text_features,
                        decoder_text_key_padding_mask=text_key_padding_mask,
                    )
            finally:
                for handle in hooks:
                    handle.remove()

            if attn is None:
                raise RuntimeError(f"No cross-attention returned for decoder_layer={decoder_layer}")

            layer_tag = f"layer{decoder_layer:02d}_text{text_layer:02d}"
            recon_path = args.output_dir / f"recon_{layer_tag}.png"
            diff_path = args.output_dir / f"diff_{layer_tag}.png"
            recon_arr = save_tensor_image(recon_path, recon)
            save_absdiff(diff_path, target_path, recon_path)
            target_arr = np.asarray(pil, dtype=np.uint8)
            saved_recons[layer_tag] = str(recon_path)

            npy_path = args.output_dir / f"singular_values_{layer_tag}.npy" if args.save_npy else None
            metrics = {
                "text_layer": text_layer,
                "decoder_layer": decoder_layer,
                "source_key": row.get("source_key", ""),
                "subset": row.get("subset", ""),
                "text": text,
                "image_path": str(image_path),
                "ckpt": str(args.ckpt),
                "config": str(args.config),
                "tau": tau,
                "vq_diff": scalar_mean(diff),
                "feature_rec_mse": float(F.mse_loss(encoder_spatial.float(), rec_spatial.float()).item()),
                "recon_path": str(recon_path),
                "diff_path": str(diff_path),
            }
            metrics.update(metric_values(target_arr, recon_arr))
            metrics.update(tensor_stats("t5_hidden", decoder_text_features))
            metrics.update(tensor_stats("projected_text_memory", text_memory))
            metrics.update(tensor_stats("encoder_spatial", encoder_spatial))
            metrics.update(tensor_stats("rec_spatial", rec_spatial))
            metrics.update(feature_stats.get(decoder_layer, {}))
            metrics.update(attention_svd_metrics(
                attn,
                text_attention_mask=text_attention_mask,
                image_token_len=image_token_len,
                tau=tau,
                out_npy=npy_path,
            ))
            rows.append(metrics)
            print(
                f"decoder_layer={decoder_layer:02d} text_layer={text_layer:02d} "
                f"mse={metrics['mse']:.6f} psnr={metrics['psnr']:.3f} "
                f"text_mass={metrics['text_attention_mass_mean']:.4f} "
                f"sigma_top1_ratio={metrics['sigma_top1_ratio']:.4f} "
                f"tau_loss={metrics['current_code_tau_loss']:.4f} "
                f"frob_loss={metrics['frobenius_uniform_loss']:.6f} "
                f"log_pr_loss={metrics['log_participation_ratio_loss']:.4f} "
                f"part_ratio={metrics['participation_rank_ratio']:.4f}"
            )

    write_csv(args.output_dir / "per_layer_metrics.csv", rows)
    summary = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "image_path": str(image_path),
        "source_key": row.get("source_key", ""),
        "subset": row.get("subset", ""),
        "text": text,
        "text_chars": len(text),
        "valid_t5_tokens": int(text_attention_mask.sum().item()),
        "config": str(args.config),
        "ckpt": str(args.ckpt),
        "seed": args.seed,
        "layer_mode": args.layer_mode,
        "text_layer_mode": args.text_layer_mode,
        "pairs": [{"text_layer": t, "decoder_layer": d} for t, d in pairs],
        "target_256": str(target_path),
        "recon_paths": saved_recons,
        "metrics_csv": str(args.output_dir / "per_layer_metrics.csv"),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"Wrote: {args.output_dir / 'per_layer_metrics.csv'}")
    print(f"Wrote: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
