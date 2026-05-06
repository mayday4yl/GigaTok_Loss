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
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
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
    parser.add_argument("--text-input-mode", choices=("correct", "empty", "shuffled", "wrong"), default="correct")
    parser.add_argument("--target-text-mode", choices=("correct", "empty", "shuffled", "wrong"), default="correct")
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
        default="ocr,mse",
        help="Comma-separated losses to probe. Supported: ocr,mse.",
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


def differentiable_reconstruct(
        model: torch.nn.Module,
        batch: torch.Tensor,
        texts: Optional[Sequence[str]],
        text_context: Optional[TextContext],
        device_backend: str,
        mixed_precision: str,
) -> torch.Tensor:
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    if text_context is None:
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            outputs = model(batch)
        return primary_reconstruction(outputs)

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
    call_selected_decoder_layer = selected_decoder_layer

    if text_context.text_recon_mode == "residual_head":
        head_text_layer = int(text_context.head_text_layer)
        decoder_head_text_features = t5_layer_states[head_text_layer].detach()
        call_selected_decoder_layer = None
    elif text_context.text_recon_layer_pairs:
        decoder_text_features_by_layer = {}
        for text_layer, decoder_layer in text_context.text_recon_layer_pairs:
            decoder_text_features_by_layer[int(decoder_layer)] = t5_layer_states[text_layer].detach()
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
    return primary_reconstruction(outputs)


def primary_reconstruction(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, (list, tuple)):
        first = outputs[0]
        if isinstance(first, (list, tuple)):
            return first[0]
        return first
    return outputs


def group_name(name: str) -> str:
    if name.startswith("s1to2decoder.residual_cross_attn_scales."):
        return "text_cross_attn_scale"
    if name.startswith("s1to2decoder.residual_cross_attn_projs."):
        return "text_cross_attn_proj"
    if name.startswith("s1to2decoder.residual_cross_attn_layers."):
        return "text_cross_attn_mha"
    if name.startswith("text_projection."):
        return "text_projection"
    if name.startswith("s1to2decoder."):
        return "transformer_decoder"
    if name.startswith("decoder."):
        return "cnn_decoder"
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


def clear_device_cache(device_backend: str) -> None:
    if device_backend == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()
    elif device_backend == "cuda":
        torch.cuda.empty_cache()


def run_probe_loss(
        *,
        loss_name: str,
        model: torch.nn.Module,
        text_context: Optional[TextContext],
        gt: torch.Tensor,
        recon_texts: Sequence[str],
        target_texts: Sequence[str],
        tokenizer: Any,
        ocr_model: torch.nn.Module,
        ocr_model_dtype: torch.dtype,
        args: argparse.Namespace,
) -> Dict[str, Any]:
    zero_model_grads(model)
    model.train()
    rec = differentiable_reconstruct(
        model,
        gt,
        recon_texts if text_context is not None else None,
        text_context,
        args.device_backend,
        args.mixed_precision,
    )
    rec.retain_grad()

    if loss_name == "ocr":
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
            "raw_ocr_tf_loss": float(raw_loss.detach().float().cpu()),
            "ocr_loss_weight": float(args.ocr_loss_weight),
            "weighted_ocr_tf_loss": float(weighted_loss.detach().float().cpu()),
        }
        extra.update({key: float(value.detach().float().mean().cpu()) for key, value in stats.items()})
    elif loss_name == "mse":
        raw_loss = F.mse_loss(rec.float(), gt.float())
        loss = raw_loss
        extra = {"mse_loss": float(raw_loss.detach().float().cpu())}
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
    zero_model_grads(model)
    return result


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = prepare_device(args.device_backend, args.device_id)
    pad_color = parse_color(args.pad_color)

    run_name, config_path, ckpt_path = args.run
    model, config, text_context = load_tokenizer_model(
        config_path,
        ckpt_path,
        device,
        text_layer_pair_index=args.text_layer_pair_index,
    )
    apply_training_freeze(model, config)

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
    recon_texts = make_texts(batch_rows, args.text_input_mode, args.wrong_text_seed, args.fixed_wrong_text)
    target_texts = make_texts(batch_rows, args.target_text_mode, args.wrong_text_seed, args.fixed_wrong_text)

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
    for loss_name in probe_losses:
        results["probe_losses"][loss_name] = run_probe_loss(
            loss_name=loss_name,
            model=model,
            text_context=text_context,
            gt=gt,
            recon_texts=recon_texts,
            target_texts=target_texts,
            tokenizer=tokenizer,
            ocr_model=ocr_model,
            ocr_model_dtype=ocr_model_dtype,
            args=args,
        )
        clear_device_cache(args.device_backend)

    write_json(output_dir / "param_grad_metrics.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
