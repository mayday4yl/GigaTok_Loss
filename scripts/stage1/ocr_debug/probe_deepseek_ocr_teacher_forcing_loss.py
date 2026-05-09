#!/usr/bin/env python3
"""Probe differentiable DeepSeek-OCR teacher-forcing CE on reconstructions.

This script is a mechanism check, not a training script. It reconstructs images
with one or more tokenizer checkpoints, feeds each reconstruction into frozen
DeepSeek-OCR, and computes causal-LM CE against a target text string. The OCR
image tokens are inserted manually through inputs_embeds so gradients can flow
from CE loss back to the reconstructed image tensor.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    MetricAccumulator,
    TextContext,
    autocast_context,
    dtype_from_mixed_precision,
    iter_jsonl,
    load_tokenizer_model,
    metric_values,
    parse_color,
    parse_run,
    pil_to_tensor,
    prepare_device,
    reconstruct_batch,
    resize_pad_image,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME:CONFIG:CKPT. Can be passed multiple times.")
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
    parser.add_argument("--text-input-mode", choices=text_modes, default="correct", help="Text fed into GigaTok text-conditioned runs.")
    parser.add_argument("--target-text-mode", choices=text_modes, default="correct", help="Target label text for OCR CE.")
    parser.add_argument("--wrong-text-seed", type=int, default=0)
    parser.add_argument("--fixed-wrong-text", default="THIS IS A FIXED WRONG TEXT 0123456789")
    parser.add_argument("--text-layer-pair-index", type=int, default=0)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--prompt", default="<image>\n<|grounding|>OCR this image.")
    parser.add_argument("--target-max-tokens", type=int, default=256)
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--include-gt", action="store_true", help="Also compute OCR CE on GT images without reconstruction.")
    return parser.parse_args()


def load_manifest_rows(path: Path, max_images: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    manifest_dir = path.resolve().parent
    for row in iter_jsonl(path):
        image_path = row.get("image_path")
        if not image_path:
            continue
        path_obj = Path(str(image_path))
        if not path_obj.is_absolute():
            path_obj = manifest_dir / path_obj
        row = dict(row)
        row["image_path"] = str(path_obj)
        rows.append(row)
        if max_images > 0 and len(rows) >= max_images:
            break
    if not rows:
        raise SystemExit(f"no rows found in {path}")
    return rows


def make_texts(
        rows: Sequence[Mapping[str, Any]],
        mode: str,
        wrong_text_seed: int,
        fixed_wrong_text: str,
        *,
        corpus_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[str]:
    if mode == "empty":
        return [""] * len(rows)
    if mode == "wrong":
        return [fixed_wrong_text] * len(rows)
    texts = [str(row.get("text", "")) for row in rows]
    if mode == "shuffled":
        shuffled = list(texts)
        random.Random(wrong_text_seed).shuffle(shuffled)
        if len(shuffled) > 1 and shuffled == texts:
            shuffled = shuffled[1:] + shuffled[:1]
        return shuffled
    if mode == "length_matched_wrong":
        candidates = [
            str(row.get("text", ""))
            for row in (corpus_rows or rows)
            if str(row.get("text", "")).strip()
        ]
        rng = random.Random(wrong_text_seed)
        wrong_texts = []
        for text in texts:
            valid = [candidate for candidate in candidates if candidate and candidate != text]
            if not valid:
                wrong_texts.append(fixed_wrong_text)
                continue
            target_len = len(text)
            best_delta = min(abs(len(candidate) - target_len) for candidate in valid)
            nearest = [candidate for candidate in valid if abs(len(candidate) - target_len) == best_delta]
            wrong_texts.append(rng.choice(nearest))
        return wrong_texts
    return texts


def load_deepseek_ocr(model_path: Path, device: torch.device, dtype_name: str, attn_implementation: str):
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    dtype = dtype_map[dtype_name]
    kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "use_safetensors": True,
    }
    if attn_implementation and attn_implementation != "none":
        kwargs["_attn_implementation"] = attn_implementation
    if dtype_name != "fp32":
        kwargs["torch_dtype"] = dtype
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModel.from_pretrained(str(model_path), **kwargs).eval().to(device)
    if dtype_name != "fp32":
        model = model.to(dtype)
    for param in model.parameters():
        param.requires_grad_(False)
    return tokenizer, model, dtype


def preprocess_for_ocr(images: torch.Tensor, image_size: int, dtype: torch.dtype) -> torch.Tensor:
    images = images.clamp(-1, 1)
    if images.shape[-2:] != (image_size, image_size):
        images = F.interpolate(images, size=(image_size, image_size), mode="bicubic", align_corners=False)
    return images.clamp(-1, 1).to(dtype=dtype)


def encode_text(tokenizer, text: str, bos: bool = False, eos: bool = False) -> List[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if bos:
        token_ids = [0] + token_ids
    if eos:
        token_ids = token_ids + [1]
    return token_ids


def extract_ocr_image_token_features(ocr_model: torch.nn.Module, images_norm: torch.Tensor) -> torch.Tensor:
    core = ocr_model.get_model() if hasattr(ocr_model, "get_model") else ocr_model.model
    sam_features = core.sam_model(images_norm)
    vision_features = core.vision_model(images_norm, sam_features)
    features = torch.cat((vision_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)), dim=-1)
    features = core.projector(features)
    batch_size, hw, width = features.shape
    side = int(math.sqrt(hw))
    if side * side != hw:
        raise ValueError(f"OCR feature token count must be square before newline insertion, got {hw}")
    features = features.view(batch_size, side, side, width)
    newline = core.image_newline.to(device=features.device, dtype=features.dtype).view(1, 1, 1, width)
    newline = newline.expand(batch_size, side, 1, width)
    features = torch.cat([features, newline], dim=2).reshape(batch_size, side * (side + 1), width)
    view_separator = core.view_seperator.to(device=features.device, dtype=features.dtype).view(1, 1, width)
    view_separator = view_separator.expand(batch_size, 1, width)
    return torch.cat([features, view_separator], dim=1)


def build_teacher_forcing_inputs(
        tokenizer,
        ocr_model: torch.nn.Module,
        images: torch.Tensor,
        target_texts: Sequence[str],
        *,
        prompt: str,
        image_size: int,
        model_dtype: torch.dtype,
        target_max_tokens: int,
        include_eos: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    if "<image>" not in prompt:
        raise ValueError("prompt must contain <image> so OCR image tokens can be inserted.")
    image_features = extract_ocr_image_token_features(
        ocr_model,
        preprocess_for_ocr(images, image_size, model_dtype),
    )
    batch_size, image_token_count, _ = image_features.shape
    image_token_id = 128815
    bos_id = 0
    text_before, text_after = prompt.split("<image>", 1)
    before_ids = encode_text(tokenizer, text_before, bos=False, eos=False)
    after_ids = encode_text(tokenizer, text_after, bos=False, eos=False)

    input_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    target_token_counts = []
    truncated_count = 0
    prompt_len = 1 + len(before_ids) + image_token_count + len(after_ids)
    for target_text in target_texts:
        target_ids = encode_text(tokenizer, str(target_text), bos=False, eos=include_eos)
        original_len = len(target_ids)
        if target_max_tokens > 0 and len(target_ids) > target_max_tokens:
            target_ids = target_ids[:target_max_tokens]
            truncated_count += 1
        token_ids = [bos_id] + before_ids + ([image_token_id] * image_token_count) + after_ids + target_ids
        labels = [-100] * prompt_len + target_ids
        input_rows.append(token_ids)
        label_rows.append(labels)
        target_token_counts.append(len(target_ids))

    max_len = max(len(row) for row in input_rows)
    input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id or 0, dtype=torch.long, device=images.device)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=images.device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=images.device)
    image_positions = torch.zeros((batch_size, max_len), dtype=torch.bool, device=images.device)
    for idx, (tokens, row_labels) in enumerate(zip(input_rows, label_rows)):
        row_len = len(tokens)
        input_ids[idx, :row_len] = torch.tensor(tokens, dtype=torch.long, device=images.device)
        labels[idx, :row_len] = torch.tensor(row_labels, dtype=torch.long, device=images.device)
        attention_mask[idx, :row_len] = 1
        image_start = 1 + len(before_ids)
        image_positions[idx, image_start:image_start + image_token_count] = True

    inputs_embeds = ocr_model.get_input_embeddings()(input_ids).clone()
    for idx in range(batch_size):
        inputs_embeds[idx, image_positions[idx], :] = image_features[idx].to(dtype=inputs_embeds.dtype)

    stats = {
        "ocr_tf_prompt_len": torch.tensor(float(prompt_len), device=images.device),
        "ocr_tf_image_tokens": torch.tensor(float(image_token_count), device=images.device),
        "ocr_tf_target_tokens_mean": torch.tensor(
            float(sum(target_token_counts) / max(1, len(target_token_counts))),
            device=images.device,
        ),
        "ocr_tf_truncated_count": torch.tensor(float(truncated_count), device=images.device),
    }
    return input_ids, inputs_embeds, attention_mask, labels, stats


def ocr_teacher_forcing_loss(
        tokenizer,
        ocr_model: torch.nn.Module,
        images: torch.Tensor,
        target_texts: Sequence[str],
        *,
        prompt: str,
        image_size: int,
        model_dtype: torch.dtype,
        target_max_tokens: int,
        include_eos: bool,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    input_ids, inputs_embeds, attention_mask, labels, stats = build_teacher_forcing_inputs(
        tokenizer,
        ocr_model,
        images,
        target_texts,
        prompt=prompt,
        image_size=image_size,
        model_dtype=model_dtype,
        target_max_tokens=target_max_tokens,
        include_eos=include_eos,
    )
    core = ocr_model.get_model() if hasattr(ocr_model, "get_model") else ocr_model.model
    original_sam_model = core.sam_model
    try:
        # Skip DeepSeek-OCR's built-in image replacement path; we already placed
        # differentiable image token features into inputs_embeds.
        core.sam_model = None
        outputs = ocr_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
            return_dict=True,
        )
    finally:
        core.sam_model = original_sam_model
    stats["ocr_tf_ce_loss"] = outputs.loss.detach()
    return outputs.loss, stats


def evaluate_run(
        *,
        run_name: str,
        model: torch.nn.Module,
        text_context: Optional[TextContext],
        rows: Sequence[Mapping[str, Any]],
        tokenizer,
        ocr_model: torch.nn.Module,
        model_dtype: torch.dtype,
        device: torch.device,
        device_backend: str,
        mixed_precision: str,
        batch_size: int,
        image_size: int,
        ocr_image_size: int,
        pad_color: Tuple[int, int, int],
        text_input_mode: str,
        target_text_mode: str,
        wrong_text_seed: int,
        fixed_wrong_text: str,
        prompt: str,
        target_max_tokens: int,
        include_eos: bool,
) -> Dict[str, Any]:
    acc = MetricAccumulator()
    rows_out: List[Dict[str, Any]] = []
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    recon_texts_all = make_texts(rows, text_input_mode, wrong_text_seed, fixed_wrong_text)
    target_texts_all = make_texts(rows, target_text_mode, wrong_text_seed, fixed_wrong_text)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pil_images = [resize_pad_image(Image.open(str(row["image_path"])), image_size, pad_color) for row in batch_rows]
        gt = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        recon_texts = recon_texts_all[start : start + len(batch_rows)]
        target_texts = target_texts_all[start : start + len(batch_rows)]
        rec = reconstruct_batch(
            model,
            gt,
            recon_texts if text_context is not None else None,
            text_context,
            device_backend=device_backend,
            mixed_precision=mixed_precision,
        )
        rec_leaf = rec.detach().clone().requires_grad_(True)
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            ce_loss, ce_stats = ocr_teacher_forcing_loss(
                tokenizer,
                ocr_model,
                rec_leaf,
                target_texts,
                prompt=prompt,
                image_size=ocr_image_size,
                model_dtype=model_dtype,
                target_max_tokens=target_max_tokens,
                include_eos=include_eos,
            )
        ce_loss.backward()
        grad = rec_leaf.grad
        grad_norm = grad.flatten(1).float().norm(dim=1) if grad is not None else torch.zeros(len(batch_rows), device=device)

        gt_uint8 = ((gt.detach().clamp(-1, 1) + 1.0) * 127.5).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        rec_uint8 = ((rec.detach().clamp(-1, 1) + 1.0) * 127.5).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        batch_stats = {key: float(value.detach().float().mean().cpu()) for key, value in ce_stats.items()}
        grad_norm_cpu = grad_norm.detach().float().cpu()
        for i, row in enumerate(batch_rows):
            values = metric_values(gt_uint8[i], rec_uint8[i])
            values.update(batch_stats)
            values["ocr_tf_grad_norm"] = float(grad_norm_cpu[i])
            acc.update(values)
            rows_out.append(
                {
                    "run": run_name,
                    "index": start + i,
                    "image_path": str(row["image_path"]),
                    "recon_text": recon_texts[i],
                    "target_text": target_texts[i],
                    "metrics": values,
                }
            )
    return {"overall": acc.mean(), "samples": rows_out}


def evaluate_gt(
        *,
        rows: Sequence[Mapping[str, Any]],
        tokenizer,
        ocr_model: torch.nn.Module,
        model_dtype: torch.dtype,
        device: torch.device,
        device_backend: str,
        mixed_precision: str,
        batch_size: int,
        image_size: int,
        ocr_image_size: int,
        pad_color: Tuple[int, int, int],
        target_text_mode: str,
        wrong_text_seed: int,
        fixed_wrong_text: str,
        prompt: str,
        target_max_tokens: int,
        include_eos: bool,
) -> Dict[str, Any]:
    acc = MetricAccumulator()
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    target_texts_all = make_texts(rows, target_text_mode, wrong_text_seed, fixed_wrong_text)
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        pil_images = [resize_pad_image(Image.open(str(row["image_path"])), image_size, pad_color) for row in batch_rows]
        gt = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        gt_leaf = gt.detach().clone().requires_grad_(True)
        target_texts = target_texts_all[start : start + len(batch_rows)]
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            ce_loss, ce_stats = ocr_teacher_forcing_loss(
                tokenizer,
                ocr_model,
                gt_leaf,
                target_texts,
                prompt=prompt,
                image_size=ocr_image_size,
                model_dtype=model_dtype,
                target_max_tokens=target_max_tokens,
                include_eos=include_eos,
            )
        ce_loss.backward()
        grad = gt_leaf.grad
        values = {key: float(value.detach().float().mean().cpu()) for key, value in ce_stats.items()}
        values["ocr_tf_grad_norm"] = float(grad.flatten(1).float().norm(dim=1).mean().detach().cpu()) if grad is not None else 0.0
        acc.update(values)
    return {"overall": acc.mean()}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = prepare_device(args.device_backend, args.device_id)
    pad_color = parse_color(args.pad_color)
    rows = load_manifest_rows(args.manifest_jsonl, args.max_images)
    tokenizer, ocr_model, model_dtype = load_deepseek_ocr(
        args.deepseek_ocr_model,
        device,
        dtype_name=args.ocr_dtype,
        attn_implementation=args.attn_implementation,
    )

    results: Dict[str, Any] = {
        "manifest_jsonl": str(args.manifest_jsonl),
        "num_images": len(rows),
        "deepseek_ocr_model": str(args.deepseek_ocr_model),
        "device_backend": args.device_backend,
        "mixed_precision": args.mixed_precision,
        "ocr_dtype": args.ocr_dtype,
        "text_input_mode": args.text_input_mode,
        "target_text_mode": args.target_text_mode,
        "target_max_tokens": args.target_max_tokens,
        "include_eos": args.include_eos,
        "runs": {},
    }

    if args.include_gt:
        results["runs"]["gt"] = evaluate_gt(
            rows=rows,
            tokenizer=tokenizer,
            ocr_model=ocr_model,
            model_dtype=model_dtype,
            device=device,
            device_backend=args.device_backend,
            mixed_precision=args.mixed_precision,
            batch_size=args.batch_size,
            image_size=args.image_size,
            ocr_image_size=args.ocr_image_size,
            pad_color=pad_color,
            target_text_mode=args.target_text_mode,
            wrong_text_seed=args.wrong_text_seed,
            fixed_wrong_text=args.fixed_wrong_text,
            prompt=args.prompt,
            target_max_tokens=args.target_max_tokens,
            include_eos=args.include_eos,
        )["overall"]

    for run_name, config_path, ckpt_path in args.run:
        model, _, text_context = load_tokenizer_model(
            config_path,
            ckpt_path,
            device,
            text_layer_pair_index=args.text_layer_pair_index,
        )
        result = evaluate_run(
            run_name=run_name,
            model=model,
            text_context=text_context,
            rows=rows,
            tokenizer=tokenizer,
            ocr_model=ocr_model,
            model_dtype=model_dtype,
            device=device,
            device_backend=args.device_backend,
            mixed_precision=args.mixed_precision,
            batch_size=args.batch_size,
            image_size=args.image_size,
            ocr_image_size=args.ocr_image_size,
            pad_color=pad_color,
            text_input_mode=args.text_input_mode,
            target_text_mode=args.target_text_mode,
            wrong_text_seed=args.wrong_text_seed,
            fixed_wrong_text=args.fixed_wrong_text,
            prompt=args.prompt,
            target_max_tokens=args.target_max_tokens,
            include_eos=args.include_eos,
        )
        results["runs"][run_name] = result["overall"]
        write_json(output_dir / f"{run_name}_samples.json", result["samples"])
        del model
        if args.device_backend == "npu":
            torch.npu.empty_cache()
        elif args.device_backend == "cuda":
            torch.cuda.empty_cache()

    write_json(output_dir / "metrics.json", results)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
