#!/usr/bin/env python3
"""Probe whether OCR teacher-forcing loss reaches tokenizer parameters.

This is a mechanism check only. It runs a differentiable reconstruction for a
small batch, backpropagates either OCR CE or image MSE, and reports gradient
norms grouped by tokenizer module.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    TextContext,
    autocast_context,
    dtype_from_mixed_precision,
    get_text_layer_states,
    load_tokenizer_model,
    parse_color,
    parse_run,
    pil_to_tensor,
    prepare_device,
    resize_pad_image,
    tokenize_texts_with_stats,
    write_json,
)
from scripts.stage1.ocr_debug.probe_deepseek_ocr_teacher_forcing_loss import (  # noqa: E402
    load_deepseek_ocr,
    load_manifest_rows,
    make_texts,
    ocr_teacher_forcing_loss,
)
from tokenizer.tokenizer_image.vq.vq_loss import VQLoss  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=parse_run, required=True, help="NAME:CONFIG:CKPT")
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deepseek-ocr-model", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--ocr-dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--ocr-image-size", type=int, default=512)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=1)
    text_modes = ("correct", "empty", "shuffled", "wrong", "length_matched_wrong")
    parser.add_argument("--text-input-mode", choices=text_modes, default="correct")
    parser.add_argument("--target-text-mode", choices=text_modes, default="correct")
    parser.add_argument("--wrong-text-seed", type=int, default=0)
    parser.add_argument("--fixed-wrong-text", default="THIS IS A FIXED WRONG TEXT 0123456789")
    parser.add_argument("--text-layer-pair-index", type=int, default=0)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--prompt", default="<image>\n<|grounding|>OCR this image.")
    parser.add_argument("--target-max-tokens", type=int, default=256)
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--ocr-loss-weight", type=float, default=0.005)
    parser.add_argument(
        "--probe-losses",
        default="weighted_ocr,mse,full_no_ocr,full_with_ocr",
        help=(
            "Comma-separated losses to probe. Supported: weighted_ocr,ocr,mse,"
            "full_no_ocr,full_with_ocr."
        ),
    )
    parser.add_argument("--global-step", type=int, default=200, help="Global step used for full VQLoss diagnostics.")
    parser.add_argument(
        "--small-cosine-groups",
        default="text_projection,residual_cross_attn_projs,residual_cross_attn_scales,residual_cross_attn_layers",
        help="Comma-separated module groups whose gradients are copied to CPU for cosine diagnostics.",
    )
    return parser.parse_args()


def apply_training_freeze(model: torch.nn.Module, config: Mapping[str, Any]) -> None:
    trainer_cfg = config.get("trainer", {})
    if hasattr(model, "apply_stage1_finetune_freeze"):
        model.apply_stage1_finetune_freeze(
            freeze_encoder=bool(trainer_cfg.get("freeze_encoder", False)),
            freeze_quantizer=bool(trainer_cfg.get("freeze_quantizer", False)),
            freeze_codebook=bool(trainer_cfg.get("freeze_codebook", False)),
            freeze_post_quant_conv=bool(trainer_cfg.get("freeze_post_quant_conv", False)),
        )


def ensure_single_process_dist() -> tempfile.TemporaryDirectory:
    """Initialize a rank-0 process group because VQLoss calls dist.get_rank()."""
    tmpdir = tempfile.TemporaryDirectory(prefix="ocr_grad_probe_dist_")
    if dist.is_available() and not dist.is_initialized():
        init_path = Path(tmpdir.name) / "init"
        dist.init_process_group(
            backend="gloo",
            init_method=f"file://{init_path}",
            rank=0,
            world_size=1,
        )
    return tmpdir


def build_vq_loss(config: Mapping[str, Any], device: torch.device) -> VQLoss:
    params = config["loss"]["params"]
    vq_loss = VQLoss(
        disc_start=params["disc_start"],
        gen_start=params.get("gen_start", 0),
        disc_weight=params["disc_weight"],
        disc_dim=params["disc_dim"],
        disc_num_layers=params.get("disc_num_layers", 3),
        disc_type=params["disc_type"],
        disc_loss=params["disc_loss"],
        gen_adv_loss=params["gen_adv_loss"],
        image_size=params["image_size"],
        perceptual_weight=params["perceptual_weight"],
        reconstruction_weight=params["reconstruction_weight"],
        reconstruction_loss=params["reconstruction_loss"],
        codebook_weight=params["codebook_weight"],
        aux_loss_end=params["aux_loss_end"],
        entropy_loss_end=params.get("entropy_loss_end", None),
        norm=params["norm"],
        use_direct_rec_loss=params.get("use_direct_rec_loss", False),
        kw=params["kw"],
        blur_ds=params.get("blur_ds", False),
        lecam=params.get("lecam", False),
        proj_weight=params.get("proj_weight", 0.5),
        disc_semantic_type=params.get("disc_semantic_type", "local"),
        use_semantic_input=params.get("use_semantic_input", False),
        perceptual_model=params.get("perceptual_model", "vgg"),
        gamma=params.get("gamma", 15),
        discriminator_device=device,
    ).to(device)
    vq_loss.eval()
    for param in vq_loss.parameters():
        param.requires_grad_(False)
    return vq_loss


def parse_model_outputs(outputs: Any, *, selected_decoder_layer: Optional[int], text_recon_on: bool) -> Dict[str, Any]:
    if not isinstance(outputs, (list, tuple)):
        return {
            "recons_imgs": [outputs],
            "inter_loss_set": None,
            "hr_attn_weights": None,
            "text_recon_stats": {},
        }
    if selected_decoder_layer is not None:
        if text_recon_on:
            recons_imgs, inter_loss_set, hr_attn_weights, text_recon_stats = outputs
        else:
            recons_imgs, inter_loss_set, hr_attn_weights = outputs
            text_recon_stats = {}
    else:
        if text_recon_on:
            recons_imgs, inter_loss_set, text_recon_stats = outputs
        else:
            recons_imgs, inter_loss_set = outputs
            text_recon_stats = {}
        hr_attn_weights = None
    return {
        "recons_imgs": recons_imgs,
        "inter_loss_set": inter_loss_set,
        "hr_attn_weights": hr_attn_weights,
        "text_recon_stats": text_recon_stats or {},
    }


def differentiable_forward(
        model: torch.nn.Module,
        batch: torch.Tensor,
        texts: Optional[Sequence[str]],
        text_context: Optional[TextContext],
        config: Mapping[str, Any],
        device_backend: str,
        mixed_precision: str,
) -> Dict[str, Any]:
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    if text_context is None:
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            outputs = model(batch)
        parsed = parse_model_outputs(outputs, selected_decoder_layer=None, text_recon_on=False)
        parsed["rec"] = primary_reconstruction(parsed["recons_imgs"])
        parsed["text_attention_mask"] = None
        parsed["selected_text_layer"] = None
        parsed["selected_decoder_layer"] = None
        return parsed

    if texts is None:
        raise ValueError("Text-conditioned reconstruction requires text input.")

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
    with torch.no_grad():
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
    text_recon_on = bool(config.get("text_recon_conditioning", {}).get("enabled", False))
    text_hr_on = bool(config.get("text_hr", {}).get("enabled", False))
    call_selected_decoder_layer = selected_decoder_layer if (text_hr_on or not text_recon_on) else None

    if text_context.text_recon_mode == "residual_head":
        head_text_layer = int(text_context.head_text_layer)
        decoder_head_text_features = t5_layer_states[head_text_layer].detach()
        call_selected_decoder_layer = selected_decoder_layer if text_hr_on else None
    elif text_context.text_recon_layer_pairs:
        decoder_text_features_by_layer = {}
        for text_layer, decoder_layer in text_context.text_recon_layer_pairs:
            decoder_text_features_by_layer[int(decoder_layer)] = t5_layer_states[text_layer].detach()
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
            return_text_recon_stats=text_recon_on,
        )
    parsed = parse_model_outputs(
        outputs,
        selected_decoder_layer=call_selected_decoder_layer,
        text_recon_on=text_recon_on,
    )
    parsed["rec"] = primary_reconstruction(parsed["recons_imgs"])
    parsed["text_attention_mask"] = text_attention_mask
    parsed["selected_text_layer"] = selected_text_layer
    parsed["selected_decoder_layer"] = selected_decoder_layer
    return parsed


def differentiable_reconstruct(
        model: torch.nn.Module,
        batch: torch.Tensor,
        texts: Optional[Sequence[str]],
        text_context: Optional[TextContext],
        config: Mapping[str, Any],
        device_backend: str,
        mixed_precision: str,
) -> torch.Tensor:
    return differentiable_forward(
        model,
        batch,
        texts,
        text_context,
        config,
        device_backend,
        mixed_precision,
    )["rec"]


def primary_reconstruction(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        first = outputs[0]
        if isinstance(first, (list, tuple)):
            return first[0]
        return first
    return outputs


def group_name(name: str) -> str:
    if name.startswith("s1to2decoder.residual_cross_attn_scales."):
        return "residual_cross_attn_scales"
    if name.startswith("s1to2decoder.residual_cross_attn_projs."):
        return "residual_cross_attn_projs"
    if name.startswith("s1to2decoder.residual_cross_attn_layers."):
        return "residual_cross_attn_layers"
    if name.startswith("text_projection."):
        return "text_projection"
    if name.startswith("s1to2decoder."):
        return "s1to2decoder"
    if name.startswith("decoder."):
        return "decoder"
    if name.startswith("post_quant_conv."):
        return "post_quant_conv"
    if name.startswith(("encoder.", "s2to1encoder.", "quant_conv.")):
        return "frozen_encoder"
    if name.startswith("quantize."):
        return "frozen_quantizer"
    return "other"


def grad_summary(model: torch.nn.Module) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    top_params = []
    for name, param in model.named_parameters():
        group = group_name(name)
        item = groups.setdefault(
            group,
            {
                "param_count": 0,
                "trainable_param_count": 0,
                "grad_param_count": 0,
                "grad_norm_sq": 0.0,
                "max_param_grad_norm": 0.0,
            },
        )
        item["param_count"] += int(param.numel())
        if param.requires_grad:
            item["trainable_param_count"] += int(param.numel())
        if param.grad is None:
            continue
        grad_norm = float(param.grad.detach().float().norm().cpu())
        item["grad_param_count"] += 1
        item["grad_norm_sq"] += grad_norm * grad_norm
        item["max_param_grad_norm"] = max(float(item["max_param_grad_norm"]), grad_norm)
        top_params.append((grad_norm, name))

    for item in groups.values():
        item["grad_norm"] = math.sqrt(float(item.pop("grad_norm_sq")))
    top_params.sort(reverse=True)
    return {
        "groups": groups,
        "top_params": [{"name": name, "grad_norm": norm} for norm, name in top_params[:20]],
    }


def zero_model_grads(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.grad = None


def collect_group_grad_vectors(model: torch.nn.Module, groups: Sequence[str]) -> Dict[str, torch.Tensor]:
    wanted = set(groups)
    chunks: Dict[str, List[torch.Tensor]] = {group: [] for group in wanted}
    for name, param in model.named_parameters():
        group = group_name(name)
        if group not in wanted or not param.requires_grad:
            continue
        if param.grad is None:
            chunks[group].append(torch.zeros(param.numel(), dtype=torch.float32))
        else:
            chunks[group].append(param.grad.detach().float().cpu().reshape(-1))
    return {
        group: torch.cat(values) if values else torch.empty(0, dtype=torch.float32)
        for group, values in chunks.items()
    }


def clear_device_cache(device_backend: str) -> None:
    if device_backend == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device_backend == "cuda":
        torch.cuda.empty_cache()


def run_probe_loss(
        *,
        loss_name: str,
        model: torch.nn.Module,
        config: Mapping[str, Any],
        vq_loss: Optional[VQLoss],
        text_context: Optional[TextContext],
        gt: torch.Tensor,
        recon_texts: Sequence[str],
        target_texts: Sequence[str],
        tokenizer: Any,
        ocr_model: torch.nn.Module,
        ocr_model_dtype: torch.dtype,
        args: argparse.Namespace,
        small_cosine_groups: Sequence[str],
) -> Dict[str, Any]:
    zero_model_grads(model)
    model.train()
    forward_result = differentiable_forward(
        model,
        gt,
        recon_texts if text_context is not None else None,
        text_context,
        config,
        args.device_backend,
        args.mixed_precision,
    )
    rec = forward_result["rec"]
    rec.retain_grad()

    if loss_name in {"ocr", "weighted_ocr"}:
        ptdtype = dtype_from_mixed_precision(args.mixed_precision)
        with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
            raw_loss, stats = ocr_teacher_forcing_loss(
                tokenizer,
                ocr_model,
                rec,
                target_texts,
                prompt=args.prompt,
                image_size=args.ocr_image_size,
                model_dtype=ocr_model_dtype,
                target_max_tokens=args.target_max_tokens,
                include_eos=args.include_eos,
            )
        weighted_loss = raw_loss * float(args.ocr_loss_weight)
        loss = weighted_loss
        extra = {
            "loss_name_alias": "weighted_ocr",
            "raw_ocr_tf_loss": float(raw_loss.detach().float().cpu()),
            "ocr_loss_weight": float(args.ocr_loss_weight),
            "weighted_ocr_tf_loss": float(weighted_loss.detach().float().cpu()),
        }
        extra.update({key: float(value.detach().float().mean().cpu()) for key, value in stats.items()})
    elif loss_name == "mse":
        raw_loss = F.mse_loss(rec.float(), gt.float())
        loss = raw_loss
        extra = {"mse_loss": float(raw_loss.detach().float().cpu())}
    elif loss_name in {"full_no_ocr", "full_with_ocr"}:
        if vq_loss is None:
            raise ValueError("full loss probes require a VQLoss instance.")
        text_hr_cfg = config.get("text_hr", {})
        text_hr_on = bool(text_hr_cfg.get("enabled", False))
        ptdtype = dtype_from_mixed_precision(args.mixed_precision)
        with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
            raw_full_loss = vq_loss(
                forward_result["inter_loss_set"],
                gt,
                forward_result["recons_imgs"],
                exp_dir=str(args.output_dir),
                optimizer_idx=0,
                global_step=args.global_step,
                last_layer=None,
                logger=None,
                log_every=10 ** 9,
                ckpt_every=10 ** 9,
                text_hr_attn_weights=forward_result["hr_attn_weights"] if text_hr_on else None,
                text_attention_mask=forward_result["text_attention_mask"] if text_hr_on else None,
                text_hr_loss_weight=float(text_hr_cfg.get("hr_loss_weight", 0.0)) if text_hr_on else 0.0,
                selected_layer=forward_result["selected_decoder_layer"],
                selected_text_layer=forward_result["selected_text_layer"],
                text_hr_tau=float(text_hr_cfg.get("tau", 1.0)),
                text_hr_svd_mode=str(text_hr_cfg.get("svd_mode", "frobenius_uniform")),
                text_hr_eps=float(text_hr_cfg.get("eps", 1e-8)),
                text_hr_skip_if_valid_tokens_lt=int(text_hr_cfg.get("skip_if_valid_tokens_lt", 2)),
                text_hr_image_token_len=int(text_hr_cfg.get("image_token_len", 256)),
                text_recon_stats=forward_result["text_recon_stats"],
            )
        loss = raw_full_loss
        extra = {"full_loss_without_ocr": float(raw_full_loss.detach().float().cpu())}
        if loss_name == "full_with_ocr":
            ptdtype = dtype_from_mixed_precision(args.mixed_precision)
            with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
                raw_ocr_loss, stats = ocr_teacher_forcing_loss(
                    tokenizer,
                    ocr_model,
                    rec,
                    target_texts,
                    prompt=args.prompt,
                    image_size=args.ocr_image_size,
                    model_dtype=ocr_model_dtype,
                    target_max_tokens=args.target_max_tokens,
                    include_eos=args.include_eos,
                )
            weighted_ocr_loss = raw_ocr_loss * float(args.ocr_loss_weight)
            loss = raw_full_loss + weighted_ocr_loss
            extra.update({
                "raw_ocr_tf_loss": float(raw_ocr_loss.detach().float().cpu()),
                "ocr_loss_weight": float(args.ocr_loss_weight),
                "weighted_ocr_tf_loss": float(weighted_ocr_loss.detach().float().cpu()),
                "full_loss_with_ocr": float(loss.detach().float().cpu()),
            })
            extra.update({key: float(value.detach().float().mean().cpu()) for key, value in stats.items()})
    else:
        raise ValueError(f"unsupported probe loss: {loss_name}")

    loss.backward()
    rec_grad = rec.grad
    result = {
        "loss_name": loss_name,
        "reconstruction_requires_grad": bool(rec.requires_grad),
        "reconstruction_grad_norm": (
            float(rec_grad.detach().float().flatten(1).norm(dim=1).mean().cpu())
            if rec_grad is not None else 0.0
        ),
    }
    result.update(extra)
    result.update(grad_summary(model))
    result["_small_grad_vectors"] = collect_group_grad_vectors(model, small_cosine_groups)
    zero_model_grads(model)
    return result


def safe_ratio(numer: float, denom: float) -> Optional[float]:
    return None if denom == 0.0 else numer / denom


def safe_cosine(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    if a.numel() == 0 or b.numel() == 0 or a.numel() != b.numel():
        return None
    a_norm = a.norm()
    b_norm = b.norm()
    if float(a_norm) == 0.0 or float(b_norm) == 0.0:
        return None
    return float(torch.dot(a, b) / (a_norm * b_norm))


def add_gradient_comparisons(results: Dict[str, Any], vectors_by_loss: Mapping[str, Dict[str, torch.Tensor]]) -> None:
    probe_losses = results["probe_losses"]
    group_names = sorted({
        group
        for payload in probe_losses.values()
        for group in payload.get("groups", {}).keys()
    })
    comparisons: Dict[str, Any] = {"ratios": {}, "cosines": {}}
    for loss_name, payload in probe_losses.items():
        if loss_name == "full_with_ocr":
            continue
        comparisons["ratios"][loss_name] = {}
        for group in group_names:
            group_norm = float(payload.get("groups", {}).get(group, {}).get("grad_norm", 0.0))
            full_norm = float(probe_losses.get("full_with_ocr", {}).get("groups", {}).get(group, {}).get("grad_norm", 0.0))
            mse_norm = float(probe_losses.get("mse", {}).get("groups", {}).get(group, {}).get("grad_norm", 0.0))
            comparisons["ratios"][loss_name][group] = {
                "grad_over_full_with_ocr": safe_ratio(group_norm, full_norm),
                "grad_over_mse": safe_ratio(group_norm, mse_norm),
            }

    cosine_refs = ["mse", "full_no_ocr", "full_with_ocr"]
    weighted_key = "weighted_ocr" if "weighted_ocr" in vectors_by_loss else "ocr"
    for group, ocr_vec in vectors_by_loss.get(weighted_key, {}).items():
        comparisons["cosines"][group] = {}
        for ref_name in cosine_refs:
            ref_vec = vectors_by_loss.get(ref_name, {}).get(group)
            if ref_vec is None:
                comparisons["cosines"][group][f"{weighted_key}_vs_{ref_name}"] = None
            else:
                comparisons["cosines"][group][f"{weighted_key}_vs_{ref_name}"] = safe_cosine(ocr_vec, ref_vec)
    results["gradient_comparisons"] = comparisons


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = prepare_device(args.device_backend, args.device_id)
    pad_color = parse_color(args.pad_color)

    run_name, config_path, ckpt_path = args.run
    dist_tmpdir = ensure_single_process_dist()
    model, config, text_context = load_tokenizer_model(
        config_path,
        ckpt_path,
        device,
        text_layer_pair_index=args.text_layer_pair_index,
    )
    apply_training_freeze(model, config)
    vq_loss = build_vq_loss(config, device)

    tokenizer, ocr_model, ocr_model_dtype = load_deepseek_ocr(
        args.deepseek_ocr_model,
        device,
        dtype_name=args.ocr_dtype,
        attn_implementation=args.attn_implementation,
    )

    rows = load_manifest_rows(args.manifest_jsonl, args.max_images)
    batch_rows = rows[: args.batch_size]
    pil_images = [resize_pad_image(Image.open(str(row["image_path"])), args.image_size, pad_color) for row in batch_rows]
    gt = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
    recon_texts = make_texts(
        batch_rows,
        args.text_input_mode,
        args.wrong_text_seed,
        args.fixed_wrong_text,
        corpus_rows=rows,
    )
    target_texts = make_texts(
        batch_rows,
        args.target_text_mode,
        args.wrong_text_seed,
        args.fixed_wrong_text,
        corpus_rows=rows,
    )

    results: Dict[str, Any] = {
        "run": run_name,
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "manifest_jsonl": str(args.manifest_jsonl),
        "image_paths": [str(row["image_path"]) for row in batch_rows],
        "text_input_mode": args.text_input_mode,
        "target_text_mode": args.target_text_mode,
        "probe_losses": {},
    }

    probe_losses = [item.strip() for item in args.probe_losses.split(",") if item.strip()]
    small_cosine_groups = [item.strip() for item in args.small_cosine_groups.split(",") if item.strip()]
    vectors_by_loss: Dict[str, Dict[str, torch.Tensor]] = {}
    for loss_name in probe_losses:
        result = run_probe_loss(
            loss_name=loss_name,
            model=model,
            config=config,
            vq_loss=vq_loss,
            text_context=text_context,
            gt=gt,
            recon_texts=recon_texts,
            target_texts=target_texts,
            tokenizer=tokenizer,
            ocr_model=ocr_model,
            ocr_model_dtype=ocr_model_dtype,
            args=args,
            small_cosine_groups=small_cosine_groups,
        )
        vectors_by_loss[loss_name] = result.pop("_small_grad_vectors")
        results["probe_losses"][loss_name] = result
        clear_device_cache(args.device_backend)

    add_gradient_comparisons(results, vectors_by_loss)
    write_json(output_dir / "param_grad_metrics.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    dist_tmpdir.cleanup()


if __name__ == "__main__":
    main()
