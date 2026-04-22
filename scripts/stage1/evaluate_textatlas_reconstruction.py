#!/usr/bin/env python3
"""Evaluate baseline vs HR tokenizer reconstruction on local TextAtlas images."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.model_init import custom_load, load_model_from_config  # noqa: E402


try:
    from skimage.metrics import structural_similarity as ssim_metric
except ImportError:  # pragma: no cover - optional dependency on some servers
    ssim_metric = None


RunSpec = Tuple[str, Path, Path]


def parse_run(value: str) -> RunSpec:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--run must be NAME:CONFIG:CKPT")
    name, config, ckpt = parts
    if not name:
        raise argparse.ArgumentTypeError("run NAME must not be empty")
    return name, Path(config), Path(ckpt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="NAME:CONFIG:CKPT. Pass twice for baseline and HR.")
    parser.add_argument("--image-paths-json", type=Path, required=True)
    parser.add_argument("--manifest-jsonl", type=Path, default=None, help="Optional materialized val manifest for subset labels.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--max-images", type=int, default=10_000, help="0 means all images.")
    parser.add_argument("--grid-samples", type=int, default=24)
    parser.add_argument("--grid-seed", type=int, default=0)
    parser.add_argument("--save-per-sample", type=int, default=0, help="Save first N comparison rows as individual PNGs.")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def parse_color(value: str) -> Tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) == 1:
        parts *= 3
    if len(parts) != 3:
        raise ValueError("--pad-color must be one integer or R,G,B")
    color = tuple(int(part) for part in parts)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("--pad-color channels must be in [0, 255]")
    return color  # type: ignore[return-value]


def resize_pad_image(img: Image.Image, image_size: int, pad_color: Tuple[int, int, int]) -> Image.Image:
    img = img.convert("RGB")
    width, height = img.size
    if width == image_size and height == image_size:
        return img
    scale = min(image_size / width, image_size / height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = img.resize(resized_size, resample=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), pad_color)
    canvas.paste(resized, ((image_size - resized_size[0]) // 2, (image_size - resized_size[1]) // 2))
    return canvas


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor * 2.0 - 1.0


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(127.5 * x + 128.0, 0, 255)
    return x.permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()


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
            raise RuntimeError("Ascend NPU is not available")
        torch.npu.set_device(device_id)
        return torch.device(f"npu:{device_id}")
    raise ValueError(f"unsupported device backend: {device_backend}")


def load_subset_map(manifest_jsonl: Path | None) -> Dict[str, str]:
    if manifest_jsonl is None or not manifest_jsonl.exists():
        return {}
    subset_by_path: Dict[str, str] = {}
    for row in iter_jsonl(manifest_jsonl):
        image_path = row.get("image_path")
        subset = row.get("subset")
        if image_path and subset:
            subset_by_path[str(Path(str(image_path)).resolve())] = str(subset)
    return subset_by_path


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_tokenizer_model(config_path: Path, ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = load_model_from_config(config)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        weights = checkpoint["ema"]
    elif "model" in checkpoint:
        weights = checkpoint["model"]
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    else:
        raise KeyError(f"cannot find model weights in {ckpt_path}")
    custom_load(model, strip_module_prefix(weights))
    del checkpoint
    model.to(device)
    model.eval()
    return model


def reconstruct_batch(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        latent, _, info = model.encode(batch)
        indices = info[2]
        return model.decode_code(indices, latent.shape)


def metric_values(gt: np.ndarray, rec: np.ndarray) -> Dict[str, float]:
    gt_f = gt.astype(np.float32) / 255.0
    rec_f = rec.astype(np.float32) / 255.0
    diff = rec_f - gt_f
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    psnr = float("inf") if mse <= 0 else float(-10.0 * math.log10(mse))
    if ssim_metric is None:
        ssim = float("nan")
    else:
        ssim = float(ssim_metric(gt_f, rec_f, channel_axis=-1, data_range=1.0))
    return {"mse": mse, "mae": mae, "psnr": psnr, "ssim": ssim}


class MetricAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sums: Dict[str, float] = defaultdict(float)

    def update(self, values: Mapping[str, float]) -> None:
        self.count += 1
        for key, value in values.items():
            if not math.isnan(value):
                self.sums[key] += float(value)

    def mean(self) -> Dict[str, float]:
        if self.count == 0:
            return {"count": 0}
        result = {"count": self.count}
        for key, value in sorted(self.sums.items()):
            result[key] = value / self.count
        return result


def select_grid_indices(num_images: int, grid_samples: int, seed: int) -> List[int]:
    grid_samples = min(max(grid_samples, 0), num_images)
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_images), grid_samples))


def evaluate_run(
    *,
    run_name: str,
    model: torch.nn.Module,
    image_paths: Sequence[Path],
    subset_by_path: Mapping[str, str],
    device: torch.device,
    batch_size: int,
    image_size: int,
    pad_color: Tuple[int, int, int],
    grid_indices: Sequence[int],
) -> Tuple[Dict[str, Any], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, str]]:
    overall = MetricAccumulator()
    by_subset: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    grid_recons: Dict[int, np.ndarray] = {}
    grid_gts: Dict[int, np.ndarray] = {}
    grid_subsets: Dict[int, str] = {}
    grid_set = set(grid_indices)

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        pil_images = [resize_pad_image(Image.open(path), image_size, pad_color) for path in batch_paths]
        batch = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        rec_batch = tensor_to_uint8(reconstruct_batch(model, batch))
        gt_batch = [np.asarray(img, dtype=np.uint8) for img in pil_images]

        for offset, (path, gt, rec) in enumerate(zip(batch_paths, gt_batch, rec_batch)):
            index = start + offset
            subset = subset_by_path.get(str(path.resolve()), "unknown")
            values = metric_values(gt, rec)
            overall.update(values)
            by_subset[subset].update(values)
            if index in grid_set:
                grid_gts[index] = gt
                grid_recons[index] = rec
                grid_subsets[index] = subset

        if (start // batch_size + 1) % 50 == 0:
            print(f"[{run_name}] evaluated {min(start + batch_size, len(image_paths))}/{len(image_paths)} images", flush=True)

    metrics = {
        "overall": overall.mean(),
        "by_subset": {subset: acc.mean() for subset, acc in sorted(by_subset.items())},
    }
    return metrics, grid_recons, grid_gts, grid_subsets


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str) -> None:
    x, y = xy
    draw.rectangle((x, y, x + 255, y + 21), fill=(255, 255, 255))
    draw.text((x + 4, y + 4), text, fill=(0, 0, 0))


def build_grid(
    *,
    output_path: Path,
    run_names: Sequence[str],
    grid_indices: Sequence[int],
    grid_gts: Mapping[int, np.ndarray],
    grid_recons_by_run: Mapping[str, Mapping[int, np.ndarray]],
    grid_subsets: Mapping[int, str],
    save_per_sample: int,
    output_dir: Path,
) -> None:
    if not grid_indices:
        return
    cell = 256
    label_h = 22
    cols = 1 + len(run_names)
    rows = len(grid_indices)
    grid = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (240, 240, 240))
    draw = ImageDraw.Draw(grid)

    for row, index in enumerate(grid_indices):
        y0 = row * (cell + label_h)
        subset = grid_subsets.get(index, "unknown")
        images = [("gt", grid_gts[index])] + [(name, grid_recons_by_run[name][index]) for name in run_names]
        row_canvas = Image.new("RGB", (cols * cell, cell + label_h), (240, 240, 240))
        row_draw = ImageDraw.Draw(row_canvas)
        for col, (label, image_arr) in enumerate(images):
            x0 = col * cell
            row_canvas.paste(Image.fromarray(image_arr), (x0, label_h))
            label_text = f"{label} | #{index} | {subset}" if col == 0 else label
            draw_label(row_draw, (x0, 0), label_text[:42])
        grid.paste(row_canvas, (0, y0))
        if row < save_per_sample:
            row_canvas.save(output_dir / f"sample_{row:03d}_idx_{index:05d}.png")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)
    print(f"saved grid: {output_path}")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if len(args.run) < 1:
        raise SystemExit("at least one --run is required")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pad_color = parse_color(args.pad_color)
    device = prepare_device(args.device_backend, args.device_id)

    image_paths = [Path(path) for path in read_json(args.image_paths_json)]
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit("no images to evaluate")

    missing = [str(path) for path in image_paths[:1000] if not path.exists()]
    if missing:
        raise SystemExit(f"missing image files, first missing: {missing[0]}")

    subset_by_path = load_subset_map(args.manifest_jsonl)
    grid_indices = select_grid_indices(len(image_paths), args.grid_samples, args.grid_seed)

    metrics_by_run: Dict[str, Any] = {}
    grid_recons_by_run: Dict[str, Dict[int, np.ndarray]] = {}
    grid_gts: Dict[int, np.ndarray] = {}
    grid_subsets: Dict[int, str] = {}
    run_names: List[str] = []

    for run_name, config_path, ckpt_path in args.run:
        print(f"== evaluating {run_name} ==")
        run_names.append(run_name)
        model = load_tokenizer_model(config_path, ckpt_path, device)
        metrics, recons, gts, subsets = evaluate_run(
            run_name=run_name,
            model=model,
            image_paths=image_paths,
            subset_by_path=subset_by_path,
            device=device,
            batch_size=args.batch_size,
            image_size=args.image_size,
            pad_color=pad_color,
            grid_indices=grid_indices,
        )
        metrics_by_run[run_name] = metrics
        grid_recons_by_run[run_name] = recons
        grid_gts.update(gts)
        grid_subsets.update(subsets)
        del model
        if args.device_backend == "cuda":
            torch.cuda.empty_cache()
        elif args.device_backend == "npu":
            torch.npu.empty_cache()

    summary = {
        "num_images": len(image_paths),
        "image_paths_json": str(args.image_paths_json),
        "manifest_jsonl": str(args.manifest_jsonl) if args.manifest_jsonl else None,
        "device_backend": args.device_backend,
        "device_id": args.device_id,
        "batch_size": args.batch_size,
        "runs": metrics_by_run,
    }
    write_json(output_dir / "metrics.json", summary)

    build_grid(
        output_path=output_dir / "comparison_grid.png",
        run_names=run_names,
        grid_indices=grid_indices,
        grid_gts=grid_gts,
        grid_recons_by_run=grid_recons_by_run,
        grid_subsets=grid_subsets,
        save_per_sample=args.save_per_sample,
        output_dir=output_dir,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
