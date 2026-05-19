#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.ocr_box_gate import (
    build_ocr_box_gate,
    load_ocr_box_index,
    ocr_box_gate_config_from_dict,
    preflight_ocr_box_coverage,
    render_ocr_box_gate_debug,
    validate_ocr_box_gate_config,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Render OCR bbox overlay and 16x16 gate debug images.")
    parser.add_argument("--manifest", required=True, help="TextAtlas JSONL manifest with image_path fields.")
    parser.add_argument("--bbox-jsonl", required=True, help="OCR bbox JSONL keyed by exact image_path.")
    parser.add_argument("--output-dir", required=True, help="Directory for debug PNG/JSON outputs.")
    parser.add_argument("--index", type=int, default=0, help="Manifest row index to render.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--num-latent-tokens", type=int, default=256)
    parser.add_argument("--inside-value", type=float, default=1.0)
    parser.add_argument("--edge-value", type=float, default=0.3)
    parser.add_argument("--outside-value", type=float, default=0.05)
    parser.add_argument("--dilate-tokens", type=int, default=1)
    parser.add_argument("--low-conf-enabled", action="store_true")
    parser.add_argument("--low-conf-alpha", type=float, default=1.0)
    parser.add_argument("--low-conf-max", type=float, default=1.2)
    return parser.parse_args()


def load_manifest_row(manifest_path: Path, index: int):
    manifest_dir = manifest_path.resolve().parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        for row_idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            if row_idx == index:
                row = json.loads(line)
                image_path = str(row["image_path"])
                if not os.path.isabs(image_path):
                    image_path = str((manifest_dir / image_path).resolve())
                return row, image_path
    raise IndexError(f"index={index} out of range for {manifest_path}")


def tensor_to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().float().mean().item())
    return float(value)


def main():
    args = parse_args()
    cfg = ocr_box_gate_config_from_dict(
        {
            "enabled": True,
            "bbox_jsonl": args.bbox_jsonl,
            "grid_size": args.grid_size,
            "image_size": args.image_size,
            "inside_value": args.inside_value,
            "edge_value": args.edge_value,
            "outside_value": args.outside_value,
            "dilate_tokens": args.dilate_tokens,
            "missing_policy": "error",
            "empty_boxes_policy": "error",
            "low_conf_enabled": args.low_conf_enabled,
            "low_conf_alpha": args.low_conf_alpha,
            "low_conf_max": args.low_conf_max,
        },
        image_size=args.image_size,
    )
    validate_ocr_box_gate_config(cfg, args.num_latent_tokens)
    bbox_index = load_ocr_box_index(args.bbox_jsonl)
    preflight = preflight_ocr_box_coverage([args.manifest], bbox_index, cfg)
    row, image_path = load_manifest_row(Path(args.manifest), args.index)
    entry = bbox_index.get(image_path)
    if entry is None:
        raise KeyError(f"OCR bbox missing for image_path={image_path}")
    gate, stats = build_ocr_box_gate(entry["boxes"], cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = render_ocr_box_gate_debug(
        image_path=image_path,
        boxes=entry["boxes"],
        gate=gate,
        output_dir=str(output_dir),
        config=cfg,
        prefix=f"idx{args.index:06d}",
    )
    summary = {
        "status": "ok",
        "manifest": args.manifest,
        "bbox_jsonl": args.bbox_jsonl,
        "index": args.index,
        "image_path": image_path,
        "text": row.get("text", ""),
        "box_count": len(entry["boxes"]),
        "preflight": preflight,
        "gate_shape": list(gate.shape),
        "gate_stats": {key: tensor_to_float(value) for key, value in stats.items()},
        "outputs": paths,
    }
    summary_path = output_dir / f"idx{args.index:06d}_ocr_box_gate_debug.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] image_path={image_path}")
    print(f"[OK] box_count={len(entry['boxes'])}")
    print(f"[OK] gate_shape={list(gate.shape)}")
    for key in sorted(summary["gate_stats"]):
        print(f"{key}={summary['gate_stats'][key]:.6g}")
    print(f"[OK] wrote {paths['bbox_debug_overlay']}")
    print(f"[OK] wrote {paths['gate_debug_16x16']}")
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
