#!/usr/bin/env python3
"""Train a tiny post-reconstruction refiner with frozen OCR loss.

This is a diagnostic experiment for the "separate network after reconstruction"
idea. The tokenizer is frozen and only produces detached reconstructions. A
small residual CNN then refines those images and is optimized by frozen
DeepSeek-OCR teacher-forcing loss plus an optional pixel MSE anchor.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    autocast_context,
    dtype_from_mixed_precision,
    load_tokenizer_model,
    parse_color,
    parse_run,
    pil_to_tensor,
    prepare_device,
    reconstruct_batch,
    resize_pad_image,
    write_json,
)
from scripts.stage1.ocr_debug.probe_deepseek_ocr_teacher_forcing_loss import (  # noqa: E402
    load_deepseek_ocr,
    load_manifest_rows,
    make_texts,
    ocr_teacher_forcing_loss,
)


class TinyResidualRefiner(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4, max_delta: float = 0.25, scale_init: float = 0.1) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        layers: List[nn.Module] = [
            nn.Conv2d(3, width, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        ]
        for _ in range(depth - 2):
            layers.extend([
                nn.Conv2d(width, width, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
            ])
        layers.append(nn.Conv2d(width, 3, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        self.max_delta = float(max_delta)
        self.scale = nn.Parameter(torch.tensor(float(scale_init)))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.net(image.float())).to(dtype=image.dtype)
        refined = image + self.scale.to(dtype=image.dtype) * self.max_delta * delta
        return refined.clamp(-1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=parse_run, required=True, help="Frozen tokenizer NAME:CONFIG:CKPT")
    parser.add_argument("--train-manifest-jsonl", type=Path, required=True)
    parser.add_argument("--val-manifest-jsonl", type=Path, default=None)
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
    parser.add_argument("--max-train-images", type=int, default=50)
    parser.add_argument("--max-val-images", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--ocr-loss-weight", type=float, default=0.005)
    parser.add_argument("--mse-loss-weight", type=float, default=1.0)
    parser.add_argument("--target-max-tokens", type=int, default=256)
    parser.add_argument("--prompt", default="<image>\n<|grounding|>OCR this image.")
    parser.add_argument("--include-eos", action="store_true")
    parser.add_argument("--text-input-mode", choices=("correct", "empty", "shuffled", "wrong"), default="correct")
    parser.add_argument("--target-text-mode", choices=("correct", "empty", "shuffled", "wrong"), default="correct")
    parser.add_argument("--wrong-text-seed", type=int, default=0)
    parser.add_argument("--fixed-wrong-text", default="THIS IS A FIXED WRONG TEXT 0123456789")
    parser.add_argument("--text-layer-pair-index", type=int, default=0)
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--refiner-width", type=int, default=64)
    parser.add_argument("--refiner-depth", type=int, default=4)
    parser.add_argument("--refiner-max-delta", type=float, default=0.25)
    parser.add_argument("--refiner-scale-init", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_batch(
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        image_size: int,
        pad_color: Tuple[int, int, int],
        device: torch.device,
) -> Tuple[torch.Tensor, List[Mapping[str, Any]]]:
    selected = [rows[index] for index in indices]
    pil_images = [resize_pad_image(Image.open(str(row["image_path"])), image_size, pad_color) for row in selected]
    images = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
    return images, selected


def reconstruct_detached(
        tokenizer_model: torch.nn.Module,
        gt: torch.Tensor,
        texts: Sequence[str],
        text_context,
        device_backend: str,
        mixed_precision: str,
) -> torch.Tensor:
    tokenizer_model.eval()
    rec = reconstruct_batch(
        tokenizer_model,
        gt,
        texts if text_context is not None else None,
        text_context,
        device_backend=device_backend,
        mixed_precision=mixed_precision,
    )
    return rec.detach()


def compute_losses(
        *,
        refiner: nn.Module,
        recon: torch.Tensor,
        gt: torch.Tensor,
        target_texts: Sequence[str],
        tokenizer,
        ocr_model: torch.nn.Module,
        ocr_model_dtype: torch.dtype,
        args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    ptdtype = dtype_from_mixed_precision(args.mixed_precision)
    with autocast_context(args.device_backend, args.mixed_precision, dtype=ptdtype):
        refined = refiner(recon)
        raw_ocr_loss, ocr_stats = ocr_teacher_forcing_loss(
            tokenizer,
            ocr_model,
            refined,
            target_texts,
            prompt=args.prompt,
            image_size=args.ocr_image_size,
            model_dtype=ocr_model_dtype,
            target_max_tokens=args.target_max_tokens,
            include_eos=args.include_eos,
        )
        mse_loss = F.mse_loss(refined.float(), gt.float())
        loss = args.ocr_loss_weight * raw_ocr_loss + args.mse_loss_weight * mse_loss

    stats = {
        "loss": float(loss.detach().float().cpu()),
        "raw_ocr_tf_loss": float(raw_ocr_loss.detach().float().cpu()),
        "weighted_ocr_tf_loss": float((args.ocr_loss_weight * raw_ocr_loss).detach().float().cpu()),
        "mse_loss": float(mse_loss.detach().float().cpu()),
        "weighted_mse_loss": float((args.mse_loss_weight * mse_loss).detach().float().cpu()),
        "refiner_scale": float(refiner.scale.detach().float().cpu()),
    }
    stats.update({key: float(value.detach().float().mean().cpu()) for key, value in ocr_stats.items()})
    return loss, stats


@torch.no_grad()
def evaluate(
        *,
        rows: Sequence[Mapping[str, Any]],
        tokenizer_model: torch.nn.Module,
        text_context,
        refiner: nn.Module,
        tokenizer,
        ocr_model: torch.nn.Module,
        ocr_model_dtype: torch.dtype,
        args: argparse.Namespace,
        device: torch.device,
        pad_color: Tuple[int, int, int],
) -> Dict[str, float]:
    refiner.eval()
    total: Dict[str, float] = {}
    count = 0
    for start in range(0, len(rows), args.batch_size):
        indices = list(range(start, min(start + args.batch_size, len(rows))))
        gt, batch_rows = load_batch(rows, indices, args.image_size, pad_color, device)
        recon_texts = make_texts(batch_rows, args.text_input_mode, args.wrong_text_seed, args.fixed_wrong_text)
        target_texts = make_texts(batch_rows, args.target_text_mode, args.wrong_text_seed, args.fixed_wrong_text)
        recon = reconstruct_detached(
            tokenizer_model,
            gt,
            recon_texts,
            text_context,
            args.device_backend,
            args.mixed_precision,
        )
        loss, stats = compute_losses(
            refiner=refiner,
            recon=recon,
            gt=gt,
            target_texts=target_texts,
            tokenizer=tokenizer,
            ocr_model=ocr_model,
            ocr_model_dtype=ocr_model_dtype,
            args=args,
        )
        del loss
        for key, value in stats.items():
            total[key] = total.get(key, 0.0) + value
        count += 1
    return {key: value / max(1, count) for key, value in total.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pad_color = parse_color(args.pad_color)
    device = prepare_device(args.device_backend, args.device_id)

    run_name, config_path, ckpt_path = args.run
    tokenizer_model, _, text_context = load_tokenizer_model(
        config_path,
        ckpt_path,
        device,
        text_layer_pair_index=args.text_layer_pair_index,
    )
    tokenizer_model.eval()
    for param in tokenizer_model.parameters():
        param.requires_grad_(False)

    tokenizer, ocr_model, ocr_model_dtype = load_deepseek_ocr(
        args.deepseek_ocr_model,
        device,
        dtype_name=args.ocr_dtype,
        attn_implementation=args.attn_implementation,
    )

    train_rows = load_manifest_rows(args.train_manifest_jsonl, args.max_train_images)
    val_rows = load_manifest_rows(args.val_manifest_jsonl, args.max_val_images) if args.val_manifest_jsonl else train_rows[: args.max_val_images]

    refiner = TinyResidualRefiner(
        width=args.refiner_width,
        depth=args.refiner_depth,
        max_delta=args.refiner_max_delta,
        scale_init=args.refiner_scale_init,
    ).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    order = list(range(len(train_rows)))
    for step in range(1, args.steps + 1):
        refiner.train()
        if (step - 1) % len(order) == 0:
            random.shuffle(order)
        batch_indices = [
            order[((step - 1) * args.batch_size + offset) % len(order)]
            for offset in range(args.batch_size)
        ]
        gt, batch_rows = load_batch(train_rows, batch_indices, args.image_size, pad_color, device)
        recon_texts = make_texts(batch_rows, args.text_input_mode, args.wrong_text_seed, args.fixed_wrong_text)
        target_texts = make_texts(batch_rows, args.target_text_mode, args.wrong_text_seed, args.fixed_wrong_text)
        recon = reconstruct_detached(
            tokenizer_model,
            gt,
            recon_texts,
            text_context,
            args.device_backend,
            args.mixed_precision,
        )

        optimizer.zero_grad(set_to_none=True)
        loss, stats = compute_losses(
            refiner=refiner,
            recon=recon,
            gt=gt,
            target_texts=target_texts,
            tokenizer=tokenizer,
            ocr_model=ocr_model,
            ocr_model_dtype=ocr_model_dtype,
            args=args,
        )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=10.0)
        optimizer.step()

        stats["step"] = float(step)
        stats["grad_norm"] = float(grad_norm.detach().float().cpu())
        history.append(stats)
        if step == 1 or step % args.log_every == 0:
            print(json.dumps({"train": stats}, ensure_ascii=False), flush=True)

        if step == 1 or (args.val_every > 0 and step % args.val_every == 0):
            val_stats = evaluate(
                rows=val_rows,
                tokenizer_model=tokenizer_model,
                text_context=text_context,
                refiner=refiner,
                tokenizer=tokenizer,
                ocr_model=ocr_model,
                ocr_model_dtype=ocr_model_dtype,
                args=args,
                device=device,
                pad_color=pad_color,
            )
            val_stats["step"] = float(step)
            print(json.dumps({"val": val_stats}, ensure_ascii=False), flush=True)
            write_json(output_dir / "latest_val.json", val_stats)

    torch.save(
        {
            "refiner": refiner.state_dict(),
            "args": vars(args),
            "history": history,
            "source_run": run_name,
            "source_config": str(config_path),
            "source_checkpoint": str(ckpt_path),
        },
        output_dir / "refiner_last.pt",
    )
    write_json(output_dir / "history.json", history)


if __name__ == "__main__":
    main()
