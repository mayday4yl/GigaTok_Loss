#!/usr/bin/env python3
"""Measure reconstruction sensitivity between correct and wrong text inputs.

This script keeps the checkpoint/config fixed and only changes the text fed to
the text branch. It reports both normal reconstruction metrics and pairwise
image differences against the correct-text reconstruction. The top-5% diff is
included because dense text usually occupies a small part of the full image.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.stage1.evaluate_textatlas_reconstruction import (  # noqa: E402
    MetricAccumulator,
    load_tokenizer_model,
    metric_values,
    parse_color,
    parse_run,
    pil_to_tensor,
    prepare_device,
    reconstruct_batch,
    resize_pad_image,
    tensor_to_uint8,
    write_json,
)
from scripts.stage1.ocr_debug.probe_deepseek_ocr_teacher_forcing_loss import (  # noqa: E402
    load_manifest_rows,
    make_texts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=parse_run, required=True, help="NAME:CONFIG:CKPT")
    parser.add_argument("--manifest-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=50)
    parser.add_argument("--text-layer-pair-index", type=int, default=0)
    parser.add_argument(
        "--compare-modes",
        default="empty,shuffled,length_matched_wrong",
        help="Comma-separated modes to compare against correct.",
    )
    parser.add_argument("--wrong-text-seed", type=int, default=0)
    parser.add_argument("--fixed-wrong-text", default="THIS IS A FIXED WRONG TEXT 0123456789")
    return parser.parse_args()


def mean_top_fraction(values: np.ndarray, fraction: float) -> float:
    flat = values.reshape(-1)
    if flat.size == 0:
        return float("nan")
    count = max(1, int(round(flat.size * fraction)))
    if count >= flat.size:
        return float(np.mean(flat))
    top = np.partition(flat, flat.size - count)[-count:]
    return float(np.mean(top))


def image_diff_values(correct: np.ndarray, other: np.ndarray) -> Dict[str, float]:
    correct_f = correct.astype(np.float32) / 255.0
    other_f = other.astype(np.float32) / 255.0
    per_pixel = np.mean(np.abs(correct_f - other_f), axis=-1)
    return {
        "full_image_mean_abs_diff": float(np.mean(per_pixel)),
        "top5_pixel_mean_abs_diff": mean_top_fraction(per_pixel, 0.05),
        "max_pixel_mean_abs_diff": float(np.max(per_pixel)),
    }


def load_batch(
        rows: Sequence[Mapping[str, Any]],
        image_size: int,
        pad_color: Tuple[int, int, int],
        device: torch.device,
) -> Tuple[torch.Tensor, List[np.ndarray]]:
    pil_images = [resize_pad_image(Image.open(str(row["image_path"])), image_size, pad_color) for row in rows]
    images = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
    gt_arrays = [np.asarray(img, dtype=np.uint8) for img in pil_images]
    return images, gt_arrays


def update_prefixed(acc: MetricAccumulator, prefix: str, values: Mapping[str, float]) -> None:
    acc.update({f"{prefix}_{key}": value for key, value in values.items()})


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    compare_modes = [mode.strip() for mode in args.compare_modes.split(",") if mode.strip()]
    allowed_modes = {"empty", "shuffled", "wrong", "length_matched_wrong"}
    unknown = sorted(set(compare_modes) - allowed_modes)
    if unknown:
        raise SystemExit(f"unsupported compare mode(s): {', '.join(unknown)}")

    rows = load_manifest_rows(args.manifest_jsonl, args.max_images)
    if not rows:
        raise SystemExit(f"no rows found in {args.manifest_jsonl}")

    device = prepare_device(args.device_backend, args.device_id)
    pad_color = parse_color(args.pad_color)
    run_name, config_path, ckpt_path = args.run
    model, _, text_context = load_tokenizer_model(
        config_path,
        ckpt_path,
        device,
        text_layer_pair_index=args.text_layer_pair_index,
    )
    if text_context is None:
        raise SystemExit("run is not text-conditioned; text-mode diff is not meaningful")

    mode_texts = {
        "correct": make_texts(rows, "correct", args.wrong_text_seed, args.fixed_wrong_text),
    }
    for mode in compare_modes:
        mode_texts[mode] = make_texts(rows, mode, args.wrong_text_seed, args.fixed_wrong_text)

    recon_metrics: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    diff_metrics: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)

    for start in range(0, len(rows), args.batch_size):
        end = min(start + args.batch_size, len(rows))
        batch_rows = rows[start:end]
        batch, gt_arrays = load_batch(batch_rows, args.image_size, pad_color, device)

        with torch.no_grad():
            correct_rec = tensor_to_uint8(
                reconstruct_batch(
                    model,
                    batch,
                    mode_texts["correct"][start:end],
                    text_context,
                    device_backend=args.device_backend,
                    mixed_precision=args.mixed_precision,
                )
            )
            mode_recons = {"correct": correct_rec}
            for mode in compare_modes:
                mode_recons[mode] = tensor_to_uint8(
                    reconstruct_batch(
                        model,
                        batch,
                        mode_texts[mode][start:end],
                        text_context,
                        device_backend=args.device_backend,
                        mixed_precision=args.mixed_precision,
                    )
                )

        for offset, gt in enumerate(gt_arrays):
            correct_arr = mode_recons["correct"][offset]
            recon_metrics["correct"].update(metric_values(gt, correct_arr))
            for mode in compare_modes:
                other_arr = mode_recons[mode][offset]
                recon_metrics[mode].update(metric_values(gt, other_arr))
                diff_metrics[mode].update(image_diff_values(correct_arr, other_arr))

        print(f"[{run_name}] processed {end}/{len(rows)} images", flush=True)

    summary = {
        "run": run_name,
        "config": str(config_path),
        "checkpoint": str(ckpt_path),
        "manifest_jsonl": str(args.manifest_jsonl),
        "num_images": len(rows),
        "text_layer_pair_index": args.text_layer_pair_index,
        "compare_modes": compare_modes,
        "reconstruction_metrics": {
            mode: acc.mean() for mode, acc in sorted(recon_metrics.items())
        },
        "diff_vs_correct": {
            mode: acc.mean() for mode, acc in sorted(diff_metrics.items())
        },
    }
    write_json(output_dir / "text_mode_image_diff.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
