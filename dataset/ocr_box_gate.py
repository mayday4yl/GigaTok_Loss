import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageDraw


_BOX_KEYS = ("bbox", "box", "xyxy", "coords", "coordinates")
_CONF_KEYS = ("confidence", "conf", "score")


@dataclass(frozen=True)
class OCRBoxGateConfig:
    enabled: bool = False
    bbox_jsonl: Optional[str] = None
    grid_size: int = 16
    image_size: int = 256
    inside_value: float = 1.0
    edge_value: float = 0.3
    outside_value: float = 0.05
    dilate_tokens: int = 1
    missing_policy: str = "error"
    empty_boxes_policy: str = "error"
    low_conf_enabled: bool = False
    low_conf_alpha: float = 1.0
    low_conf_max: float = 1.2
    apply_layers: Optional[Tuple[int, ...]] = None
    log_debug: bool = True


def ocr_box_gate_config_from_dict(raw_cfg: Optional[Dict[str, Any]], image_size: int = 256) -> OCRBoxGateConfig:
    if isinstance(raw_cfg, OCRBoxGateConfig):
        return raw_cfg
    raw_cfg = raw_cfg or {}
    apply_layers = raw_cfg.get("apply_layers", None)
    if apply_layers is not None:
        apply_layers = tuple(int(layer_idx) for layer_idx in apply_layers)
    return OCRBoxGateConfig(
        enabled=bool(raw_cfg.get("enabled", False)),
        bbox_jsonl=raw_cfg.get("bbox_jsonl", None),
        grid_size=int(raw_cfg.get("grid_size", 16)),
        image_size=int(raw_cfg.get("image_size", image_size)),
        inside_value=float(raw_cfg.get("inside_value", 1.0)),
        edge_value=float(raw_cfg.get("edge_value", 0.3)),
        outside_value=float(raw_cfg.get("outside_value", 0.05)),
        dilate_tokens=int(raw_cfg.get("dilate_tokens", 1)),
        missing_policy=str(raw_cfg.get("missing_policy", "error")),
        empty_boxes_policy=str(raw_cfg.get("empty_boxes_policy", "error")),
        low_conf_enabled=bool(raw_cfg.get("low_conf_enabled", False)),
        low_conf_alpha=float(raw_cfg.get("low_conf_alpha", 1.0)),
        low_conf_max=float(raw_cfg.get("low_conf_max", 1.2)),
        apply_layers=apply_layers,
        log_debug=bool(raw_cfg.get("log_debug", True)),
    )


def _as_float_list(value: Any, expected_len: int = 4) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    if not isinstance(value, (list, tuple)):
        return None
    if len(value) != expected_len:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _parse_box(box: Any) -> Dict[str, Any]:
    if isinstance(box, dict):
        coords = None
        for key in _BOX_KEYS:
            coords = _as_float_list(box.get(key))
            if coords is not None:
                break
        if coords is None and all(key in box for key in ("x1", "y1", "x2", "y2")):
            coords = [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
        if coords is None and all(key in box for key in ("x", "y", "w", "h")):
            x, y, w, h = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
            coords = [x, y, x + w, y + h]
        if coords is None:
            raise ValueError(f"OCR box has no supported bbox fields: keys={sorted(box.keys())}")
        conf = None
        for key in _CONF_KEYS:
            if key in box and box[key] is not None:
                conf = float(box[key])
                break
        return {"bbox": coords, "confidence": conf, "text": box.get("text", None)}

    coords = _as_float_list(box)
    if coords is None:
        raise ValueError(f"Unsupported OCR box format: {type(box)!r}")
    return {"bbox": coords, "confidence": None, "text": None}


def _canonical_image_path(row: Dict[str, Any], manifest_dir: Optional[str] = None) -> str:
    image_path = row.get("image_path")
    if not image_path:
        raise ValueError("manifest/bbox row missing image_path")
    image_path = str(image_path)
    if manifest_dir is not None and not os.path.isabs(image_path):
        image_path = os.path.join(manifest_dir, image_path)
    return os.path.abspath(image_path) if os.path.isabs(image_path) else image_path


def load_ocr_box_index(bbox_jsonl: str) -> Dict[str, Dict[str, Any]]:
    if not bbox_jsonl:
        raise ValueError("ocr_box_gate.bbox_jsonl is required when ocr_box_gate.enabled=true")
    if not os.path.exists(bbox_jsonl):
        raise FileNotFoundError(bbox_jsonl)

    box_index: Dict[str, Dict[str, Any]] = {}
    with open(bbox_jsonl, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            image_path = _canonical_image_path(row)
            if image_path in box_index:
                raise ValueError(f"{bbox_jsonl}:{line_no}: duplicate image_path={image_path}")
            raw_boxes = row.get("boxes", None)
            if raw_boxes is None:
                raise ValueError(f"{bbox_jsonl}:{line_no}: missing boxes")
            if not isinstance(raw_boxes, list):
                raise ValueError(f"{bbox_jsonl}:{line_no}: boxes must be a list")
            box_index[image_path] = {
                "image_path": image_path,
                "boxes": [_parse_box(box) for box in raw_boxes],
            }
    if not box_index:
        raise ValueError(f"No OCR boxes found in {bbox_jsonl}")
    return box_index


def validate_ocr_box_gate_config(config: OCRBoxGateConfig, num_latent_tokens: int) -> None:
    if not config.enabled:
        return
    if config.missing_policy != "error":
        raise ValueError("ocr_box_gate.missing_policy must be 'error' in v1.")
    if config.empty_boxes_policy != "error":
        raise ValueError("ocr_box_gate.empty_boxes_policy must be 'error' in v1.")
    if config.grid_size <= 0:
        raise ValueError(f"ocr_box_gate.grid_size must be positive, got {config.grid_size}")
    if config.grid_size * config.grid_size != int(num_latent_tokens):
        raise ValueError(
            "ocr_box_gate.grid_size * grid_size must equal num_latent_tokens, "
            f"got {config.grid_size}*{config.grid_size}!={num_latent_tokens}"
        )
    if config.image_size <= 0:
        raise ValueError(f"ocr_box_gate.image_size must be positive, got {config.image_size}")
    if config.dilate_tokens < 0:
        raise ValueError(f"ocr_box_gate.dilate_tokens must be >= 0, got {config.dilate_tokens}")
    if config.outside_value < 0.0 or config.edge_value < 0.0 or config.inside_value < 0.0:
        raise ValueError("ocr_box_gate values must be non-negative")
    if config.low_conf_max < 1.0:
        raise ValueError("ocr_box_gate.low_conf_max must be >= 1.0")


def preflight_ocr_box_coverage(
        manifest_paths: Iterable[str],
        bbox_index: Dict[str, Dict[str, Any]],
        config: OCRBoxGateConfig,
        max_images_by_manifest: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    if not config.enabled:
        return {"enabled": False}

    max_images_by_manifest = max_images_by_manifest or {}
    total_rows = 0
    missing: List[str] = []
    empty: List[str] = []
    for manifest_path in manifest_paths:
        if not manifest_path:
            continue
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        row_limit = int(max_images_by_manifest.get(manifest_path, 0) or 0)
        seen_rows = 0
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if row_limit > 0 and seen_rows >= row_limit:
                    break
                row = json.loads(line)
                image_path = _canonical_image_path(row, manifest_dir=manifest_dir)
                seen_rows += 1
                total_rows += 1
                entry = bbox_index.get(image_path)
                if entry is None:
                    missing.append(image_path)
                    continue
                if not entry["boxes"]:
                    empty.append(image_path)

    if missing and config.missing_policy == "error":
        preview = ", ".join(missing[:5])
        raise ValueError(f"OCR bbox preflight missing {len(missing)} image_path entries. First missing: {preview}")
    if empty and config.empty_boxes_policy == "error":
        preview = ", ".join(empty[:5])
        raise ValueError(f"OCR bbox preflight found boxes=[] for {len(empty)} entries. First empty: {preview}")
    return {
        "enabled": True,
        "manifest_rows_checked": total_rows,
        "missing_count": len(missing),
        "empty_boxes_count": len(empty),
        "bbox_index_entries": len(bbox_index),
    }


def _bbox_to_token_bounds(bbox: Sequence[float], config: OCRBoxGateConfig) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    x1 = max(0.0, min(float(config.image_size), x1))
    x2 = max(0.0, min(float(config.image_size), x2))
    y1 = max(0.0, min(float(config.image_size), y1))
    y2 = max(0.0, min(float(config.image_size), y2))
    if x2 <= x1 or y2 <= y1:
        return None

    token_size = float(config.image_size) / float(config.grid_size)
    col0 = int(math.floor(x1 / token_size))
    col1 = int(math.ceil(x2 / token_size)) - 1
    row0 = int(math.floor(y1 / token_size))
    row1 = int(math.ceil(y2 / token_size)) - 1
    col0 = max(0, min(config.grid_size - 1, col0))
    col1 = max(0, min(config.grid_size - 1, col1))
    row0 = max(0, min(config.grid_size - 1, row0))
    row1 = max(0, min(config.grid_size - 1, row1))
    if col1 < col0 or row1 < row0:
        return None
    return row0, row1, col0, col1


def _low_conf_factor(confidence: Optional[float], config: OCRBoxGateConfig) -> float:
    if not config.low_conf_enabled:
        return 1.0
    if confidence is None:
        return 1.0
    conf = max(0.0, min(1.0, float(confidence)))
    return min(float(config.low_conf_max), max(1.0, 1.0 + float(config.low_conf_alpha) * (1.0 - conf)))


def build_ocr_box_gate(boxes: Sequence[Dict[str, Any]], config: OCRBoxGateConfig) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if not boxes and config.empty_boxes_policy == "error":
        raise ValueError("OCR bbox entry has boxes=[] and empty_boxes_policy='error'")

    gate_2d = torch.full(
        (config.grid_size, config.grid_size),
        float(config.outside_value),
        dtype=torch.float32,
    )
    confidences: List[float] = []
    low_conf_factors: List[float] = []

    for box in boxes:
        bbox = box.get("bbox")
        bounds = _bbox_to_token_bounds(bbox, config)
        if bounds is None:
            continue
        row0, row1, col0, col1 = bounds
        dilate = int(config.dilate_tokens)
        edge_row0 = max(0, row0 - dilate)
        edge_row1 = min(config.grid_size - 1, row1 + dilate)
        edge_col0 = max(0, col0 - dilate)
        edge_col1 = min(config.grid_size - 1, col1 + dilate)
        gate_2d[edge_row0:edge_row1 + 1, edge_col0:edge_col1 + 1] = torch.maximum(
            gate_2d[edge_row0:edge_row1 + 1, edge_col0:edge_col1 + 1],
            torch.tensor(float(config.edge_value), dtype=torch.float32),
        )
        confidence = box.get("confidence", None)
        if confidence is not None:
            confidences.append(float(confidence))
        low_conf_factor = _low_conf_factor(confidence, config)
        low_conf_factors.append(low_conf_factor)
        inside_value = float(config.inside_value) * low_conf_factor
        gate_2d[row0:row1 + 1, col0:col1 + 1] = torch.maximum(
            gate_2d[row0:row1 + 1, col0:col1 + 1],
            torch.tensor(inside_value, dtype=torch.float32),
        )

    gate = gate_2d.flatten()
    eps = 1e-6
    stats = {
        "ocr_box_gate_mean": gate.mean(),
        "ocr_box_gate_min": gate.min(),
        "ocr_box_gate_max": gate.max(),
        "ocr_box_gate_nonzero_ratio": (gate > eps).float().mean(),
        "ocr_box_gate_above_outside_ratio": (gate > float(config.outside_value) + eps).float().mean(),
        "ocr_box_gate_inside_ratio": (gate >= float(config.inside_value) - eps).float().mean(),
        "ocr_box_gate_edge_ratio": (
            (gate > float(config.outside_value) + eps) & (gate < float(config.inside_value) - eps)
        ).float().mean(),
        "ocr_box_count": torch.tensor(float(len(boxes)), dtype=torch.float32),
        "ocr_confidence_mean": torch.tensor(
            float(sum(confidences) / len(confidences)) if confidences else 0.0,
            dtype=torch.float32,
        ),
        "ocr_low_conf_enabled": torch.tensor(1.0 if config.low_conf_enabled else 0.0, dtype=torch.float32),
        "ocr_low_conf_factor_mean": torch.tensor(
            float(sum(low_conf_factors) / len(low_conf_factors)) if low_conf_factors else 1.0,
            dtype=torch.float32,
        ),
    }
    return gate, stats


def render_ocr_box_gate_debug(
        image_path: str,
        boxes: Sequence[Dict[str, Any]],
        gate: torch.Tensor,
        output_dir: str,
        config: OCRBoxGateConfig,
        prefix: str = "ocr_box_gate") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    image = _resize_pad_image(Image.open(image_path).convert("RGB"), config.image_size)
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for box in boxes:
        bbox = box.get("bbox")
        if bbox is None:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    overlay_path = os.path.join(output_dir, f"{prefix}_bbox_debug_overlay.png")
    overlay.save(overlay_path)

    gate_img = gate.detach().float().reshape(config.grid_size, config.grid_size)
    min_val = float(gate_img.min().item())
    max_val = float(gate_img.max().item())
    denom = max(max_val - min_val, 1e-8)
    norm = ((gate_img - min_val) / denom * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    gate_pil = Image.fromarray(norm, mode="L").resize((config.image_size, config.image_size), Image.NEAREST)
    gate_rgb = Image.new("RGB", gate_pil.size)
    gate_rgb.paste(gate_pil)
    gate_path = os.path.join(output_dir, f"{prefix}_gate_debug_16x16.png")
    gate_rgb.save(gate_path)
    return {"bbox_debug_overlay": overlay_path, "gate_debug_16x16": gate_path}


def _resize_pad_image(image: Image.Image, image_size: int, fill: int = 255) -> Image.Image:
    width, height = image.size
    scale = min(image_size / width, image_size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    image = image.resize((resized_width, resized_height), Image.BICUBIC)
    canvas = Image.new("RGB", (image_size, image_size), color=(fill, fill, fill))
    left = (image_size - resized_width) // 2
    top = (image_size - resized_height) // 2
    canvas.paste(image, (left, top))
    return canvas
