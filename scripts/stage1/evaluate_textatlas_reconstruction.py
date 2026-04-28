#!/usr/bin/env python3
"""Evaluate baseline vs HR tokenizer reconstruction on local TextAtlas images."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import unicodedata
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


TEXT_CONDITIONING_MISSING_PREFIXES = ("text_projection.",)
TEXT_CONDITIONING_MISSING_NAMES = {"text_type_embedding", "visual_type_embedding", "text_gate_logit"}


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
    parser.add_argument("--image-paths-json", type=Path, default=None, help="Optional JSON list of image paths. If omitted, paths are read from --manifest-jsonl.")
    parser.add_argument("--manifest-jsonl", type=Path, default=None, help="Optional materialized val manifest for subset labels and rendered text.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-backend", choices=("cuda", "npu", "cpu"), default="npu")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mixed-precision", choices=("none", "bf16", "fp16"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--pad-color", default="255,255,255")
    parser.add_argument("--max-images", type=int, default=10_000, help="0 means all images.")
    parser.add_argument("--grid-samples", type=int, default=24)
    parser.add_argument("--grid-seed", type=int, default=0)
    parser.add_argument("--save-per-sample", type=int, default=0, help="Save first N comparison rows as individual PNGs.")
    parser.add_argument("--ocr-backend", choices=("none", "pytesseract", "paddleocr", "easyocr"), default="none")
    parser.add_argument("--ocr-lang", default="en")
    parser.add_argument("--ocr-min-confidence", type=float, default=0.0)
    parser.add_argument("--ocr-jsonl", type=Path, default=None, help="Optional per-sample OCR prediction output JSONL.")
    parser.add_argument("--text-layer-pair-index", type=int, default=0, help="Deterministic [T5 layer, decoder layer] pair index for text-conditioned reconstruction.")
    parser.add_argument("--text-input-mode", choices=("correct", "empty", "shuffled"), default="correct", help="Text used by text-conditioned runs; use empty/shuffled for sensitivity checks.")
    parser.add_argument("--wrong-text-seed", type=int, default=0, help="Seed for --text-input-mode=shuffled.")
    parser.add_argument("--strip-punctuation", action="store_true", help="Remove punctuation before CER/NED/exact-match metrics.")
    parser.add_argument("--remove-spaces", action="store_true", help="Remove spaces before CER/NED/exact-match metrics.")
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


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
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


def autocast_context(device_backend: str, mixed_precision: str, dtype: Optional[torch.dtype] = None):
    if mixed_precision == "none" or device_backend == "cpu":
        return nullcontext()
    if device_backend == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    if device_backend == "npu":
        npu_amp = getattr(getattr(torch, "npu", None), "amp", None)
        if npu_amp is not None and hasattr(npu_amp, "autocast"):
            return npu_amp.autocast(dtype=dtype)
        return torch.autocast(device_type="npu", dtype=dtype)
    raise ValueError(f"unsupported device backend: {device_backend}")


def dtype_from_mixed_precision(mixed_precision: str) -> Optional[torch.dtype]:
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "fp16":
        return torch.float16
    return None


def load_manifest_metadata(manifest_jsonl: Path | None) -> Tuple[List[Path], Dict[str, Dict[str, str]]]:
    if manifest_jsonl is None or not manifest_jsonl.exists():
        return [], {}
    manifest_dir = manifest_jsonl.resolve().parent
    image_paths: List[Path] = []
    metadata_by_path: Dict[str, Dict[str, str]] = {}
    for row in iter_jsonl(manifest_jsonl):
        image_path = row.get("image_path")
        if not image_path:
            continue
        path = Path(str(image_path))
        if not path.is_absolute():
            path = manifest_dir / path
        image_paths.append(path)
        metadata_by_path[str(path.resolve())] = {
            "subset": str(row.get("subset") or "unknown"),
            "text": str(row.get("text") or ""),
            "source_key": str(row.get("source_key") or ""),
            "text_extraction_status": str(row.get("text_extraction_status") or ""),
        }
    return image_paths, metadata_by_path


def strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return dict(state_dict)
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def text_conditioning_missing_keys(model: torch.nn.Module) -> List[str]:
    missing_keys = []
    for name, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if name in TEXT_CONDITIONING_MISSING_NAMES or name.startswith(TEXT_CONDITIONING_MISSING_PREFIXES):
            missing_keys.append(name)
    return missing_keys


def load_text_encoder(config: Mapping[str, Any], device: torch.device):
    text_cfg = config.get("text_conditioning", {})
    if not text_cfg.get("enabled", False):
        return None, None, None, None

    try:
        from transformers import AutoTokenizer, T5EncoderModel
    except ImportError as exc:
        raise ImportError("text_conditioning.enabled=True requires transformers with T5EncoderModel.") from exc

    encoder_name = text_cfg.get("encoder_name", "google/t5-v1_1-xl")
    pretrained_kwargs = {
        "cache_dir": text_cfg.get("cache_dir", None),
        "local_files_only": bool(text_cfg.get("local_files_only", False)),
    }
    tokenizer = AutoTokenizer.from_pretrained(encoder_name, **pretrained_kwargs)
    encoder = T5EncoderModel.from_pretrained(encoder_name, **pretrained_kwargs)
    encoder.requires_grad_(False)
    encoder.eval().to(device)
    text_feature_dim = getattr(encoder.config, "d_model", None)
    text_num_layers = getattr(encoder.config, "num_layers", None)
    if text_feature_dim is None:
        raise ValueError(f"Cannot read d_model from T5 encoder config: {encoder_name}")
    return tokenizer, encoder, int(text_feature_dim), text_num_layers


def build_layer_pairs_from_cfg(
        cfg: Mapping[str, Any],
        decoder_num_layers: int,
        text_num_layers: Optional[int],
        config_name: str,
) -> List[Tuple[int, int]]:
    raw_pairs = cfg.get("layer_pairs", None)
    if not raw_pairs:
        shared_layers = min(decoder_num_layers, text_num_layers) if text_num_layers is not None else decoder_num_layers
        start = shared_layers // 3
        end = min(shared_layers, start + max(1, shared_layers // 3))
        raw_pairs = [[layer_idx, layer_idx] for layer_idx in range(start, end)]

    layer_pairs = []
    for pair in raw_pairs:
        if len(pair) != 2:
            raise ValueError(f"{config_name}.layer_pairs entries must be [t5_layer, decoder_layer], got {pair}")
        text_layer, decoder_layer = int(pair[0]), int(pair[1])
        if decoder_layer < 0 or decoder_layer >= decoder_num_layers:
            raise ValueError(
                f"decoder_layer={decoder_layer} is out of range for decoder_num_layers={decoder_num_layers}"
            )
        if text_num_layers is not None and (text_layer < 0 or text_layer >= text_num_layers):
            raise ValueError(f"text_layer={text_layer} is out of range for text_num_layers={text_num_layers}")
        layer_pairs.append((text_layer, decoder_layer))
    if not layer_pairs:
        raise ValueError("No text layer pairs selected.")
    return layer_pairs


def build_text_layer_pairs(config: Mapping[str, Any], decoder_num_layers: int, text_num_layers: Optional[int]) -> List[Tuple[int, int]]:
    return build_layer_pairs_from_cfg(config.get("text_hr", {}), decoder_num_layers, text_num_layers, "text_hr")


def build_text_recon_layer_pairs(config: Mapping[str, Any], decoder_num_layers: int, text_num_layers: Optional[int]) -> List[Tuple[int, int]]:
    text_recon_cfg = config.get("text_recon_conditioning", {})
    return build_layer_pairs_from_cfg(
        text_recon_cfg, decoder_num_layers, text_num_layers, "text_recon_conditioning")


def validate_text_recon_config(config: Mapping[str, Any]) -> None:
    text_recon_cfg = config.get("text_recon_conditioning", {})
    if not bool(text_recon_cfg.get("enabled", False)):
        return
    if str(text_recon_cfg.get("mode", "concat_memory")) != "concat_memory":
        raise NotImplementedError("Only text_recon_conditioning.mode=concat_memory is implemented.")
    if not text_recon_cfg.get("layers", None) and not text_recon_cfg.get("layer_pairs", None):
        raise ValueError("text_recon_conditioning.enabled=True requires non-empty layers or layer_pairs.")
    for block_name in ("visual_memory_mask", "residual", "adaln"):
        block_cfg = text_recon_cfg.get(block_name, {})
        if isinstance(block_cfg, Mapping) and bool(block_cfg.get("enabled", False)):
            raise NotImplementedError(f"text_recon_conditioning.{block_name} is planned but not implemented.")


class TextContext:
    def __init__(
            self,
            tokenizer: Any,
            encoder: torch.nn.Module,
            layer_pairs: Sequence[Tuple[int, int]],
            text_recon_layer_pairs: Optional[Sequence[Tuple[int, int]]],
            text_injection_layers: Optional[Sequence[int]],
            max_length: int,
            pair_index: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.layer_pairs = list(layer_pairs)
        self.text_recon_layer_pairs = list(text_recon_layer_pairs or [])
        self.text_injection_layers = list(text_injection_layers or [])
        self.max_length = max_length
        if pair_index < 0 or pair_index >= len(self.layer_pairs):
            raise ValueError(f"--text-layer-pair-index={pair_index} out of range for {len(self.layer_pairs)} pairs")
        self.pair_index = pair_index

    @property
    def selected_pair(self) -> Tuple[int, int]:
        return self.layer_pairs[self.pair_index]


def load_tokenizer_model(
        config_path: Path,
        ckpt_path: Path,
        device: torch.device,
        text_layer_pair_index: int,
) -> Tuple[torch.nn.Module, Mapping[str, Any], Optional[TextContext]]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_text_recon_config(config)

    tokenizer, text_encoder, text_feature_dim, text_num_layers = load_text_encoder(config, device)
    model = load_model_from_config(config)
    if config.get("text_conditioning", {}).get("enabled", False):
        text_recon_cfg = config.get("text_recon_conditioning", {})
        text_recon_on = bool(text_recon_cfg.get("enabled", False))
        text_gate_cfg = text_recon_cfg.get("text_gate", {}) if text_recon_on else {}
        model.configure_text_conditioning(
            text_feature_dim=text_feature_dim,
            text_projection=config.get("text_conditioning", {}).get("text_projection", "linear_layernorm"),
            text_type_embedding=config.get("text_conditioning", {}).get("text_type_embedding", True),
            visual_type_embedding=text_recon_on and bool(text_recon_cfg.get("visual_type_embedding", False)),
            text_gate_enabled=text_recon_on and bool(text_gate_cfg.get("enabled", False)),
            text_gate_init=float(text_gate_cfg.get("init", 0.1)),
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        weights = checkpoint["ema"]
    elif "model" in checkpoint:
        weights = checkpoint["model"]
    elif "state_dict" in checkpoint:
        weights = checkpoint["state_dict"]
    else:
        raise KeyError(f"cannot find model weights in {ckpt_path}")
    ignore_missing = text_conditioning_missing_keys(model) if config.get("text_conditioning", {}).get("enabled", False) else None
    custom_load(model, strip_module_prefix(weights), ignore_missing_keys=ignore_missing)
    del checkpoint
    model.to(device)
    model.eval()
    text_context = None
    if tokenizer is not None and text_encoder is not None:
        decoder_num_layers = int(getattr(model.s1to2decoder, "num_layers"))
        layer_pairs = build_text_layer_pairs(config, decoder_num_layers, text_num_layers)
        text_recon_layer_pairs = None
        text_injection_layers = None
        if bool(config.get("text_recon_conditioning", {}).get("enabled", False)):
            text_recon_layer_pairs = build_text_recon_layer_pairs(config, decoder_num_layers, text_num_layers)
            text_injection_layers = [decoder_layer for _, decoder_layer in text_recon_layer_pairs]
        text_context = TextContext(
            tokenizer=tokenizer,
            encoder=text_encoder,
            layer_pairs=layer_pairs,
            text_recon_layer_pairs=text_recon_layer_pairs,
            text_injection_layers=text_injection_layers,
            max_length=int(config.get("text_conditioning", {}).get("max_length", 128)),
            pair_index=text_layer_pair_index,
        )
    return model, config, text_context


def reconstruct_batch(
        model: torch.nn.Module,
        batch: torch.Tensor,
        texts: Optional[Sequence[str]],
        text_context: Optional[TextContext],
        device_backend: str,
        mixed_precision: str,
) -> torch.Tensor:
    ptdtype = dtype_from_mixed_precision(mixed_precision)
    with torch.inference_mode():
        if text_context is None:
            latent, _, info = model.encode(batch)
            indices = info[2]
            return model.decode_code(indices, latent.shape)
        if texts is None:
            raise ValueError("Text-conditioned reconstruction requires rendered text from --manifest-jsonl.")

        selected_text_layer, selected_decoder_layer = text_context.selected_pair
        text_inputs = text_context.tokenizer(
            [str(text) for text in texts],
            padding="max_length",
            truncation=True,
            max_length=text_context.max_length,
            return_tensors="pt",
        )
        input_ids = text_inputs["input_ids"].to(batch.device, non_blocking=True)
        text_attention_mask = text_inputs["attention_mask"].to(batch.device, non_blocking=True)
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            text_outputs = text_context.encoder(
                input_ids=input_ids,
                attention_mask=text_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        hidden_states = text_outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("T5 encoder did not return hidden_states.")
        t5_layer_states = hidden_states[1:]
        decoder_text_key_padding_mask = ~text_attention_mask.bool()
        decoder_text_features = None
        decoder_text_features_by_layer = None
        call_selected_decoder_layer = selected_decoder_layer
        if text_context.text_recon_layer_pairs:
            decoder_text_features_by_layer = {}
            for text_layer, decoder_layer in text_context.text_recon_layer_pairs:
                if text_layer >= len(t5_layer_states):
                    raise ValueError(
                        f"text_layer={text_layer} is out of range for T5 layer outputs={len(t5_layer_states)}"
                    )
                decoder_text_features_by_layer[int(decoder_layer)] = t5_layer_states[text_layer].detach()
            call_selected_decoder_layer = None
        else:
            if selected_text_layer >= len(t5_layer_states):
                raise ValueError(
                    f"selected_text_layer={selected_text_layer} is out of range for "
                    f"T5 layer outputs={len(t5_layer_states)}"
                )
            decoder_text_features = t5_layer_states[selected_text_layer].detach()
        with autocast_context(device_backend, mixed_precision, dtype=ptdtype):
            outputs = model(
                batch,
                selected_decoder_layer=call_selected_decoder_layer,
                decoder_text_features=decoder_text_features,
                decoder_text_features_by_layer=decoder_text_features_by_layer,
                decoder_text_key_padding_mask=decoder_text_key_padding_mask,
                text_injection_layers=text_context.text_injection_layers or None,
            )
        return outputs[0] if isinstance(outputs, (list, tuple)) else outputs


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


def normalize_text_for_metric(
        text: str,
        *,
        lowercase: bool,
        strip_punctuation: bool,
        remove_spaces: bool,
) -> str:
    text = unicodedata.normalize("NFKC", str(text)).strip()
    text = re.sub(r"\s+", " ", text)
    if strip_punctuation:
        text = "".join(ch for ch in text if ch.isalnum() or ch.isspace())
        text = re.sub(r"\s+", " ", text).strip()
    if remove_spaces:
        text = re.sub(r"\s+", "", text)
    if lowercase:
        text = text.lower()
    return text


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def text_metric_values(
        gt_text: str,
        pred_text: str,
        *,
        strip_punctuation: bool,
        remove_spaces: bool,
) -> Dict[str, float]:
    gt_exact = normalize_text_for_metric(
        gt_text,
        lowercase=False,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    pred_exact = normalize_text_for_metric(
        pred_text,
        lowercase=False,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    gt_ci = normalize_text_for_metric(
        gt_text,
        lowercase=True,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    pred_ci = normalize_text_for_metric(
        pred_text,
        lowercase=True,
        strip_punctuation=strip_punctuation,
        remove_spaces=remove_spaces,
    )
    edit = levenshtein_distance(pred_ci, gt_ci)
    cer = edit / max(1, len(gt_ci))
    ned_similarity = 1.0 - edit / max(1, len(gt_ci), len(pred_ci))
    return {
        "ocr_exact_acc": float(pred_exact == gt_exact),
        "ocr_exact_acc_ci": float(pred_ci == gt_ci),
        "ocr_cer": float(cer),
        "ocr_ned_similarity": float(ned_similarity),
        "ocr_edit_distance": float(edit),
        "ocr_gt_chars": float(len(gt_ci)),
        "ocr_pred_chars": float(len(pred_ci)),
    }


def extract_paddle_texts(result: Any, min_confidence: float) -> List[str]:
    texts: List[str] = []
    if result is None:
        return texts
    if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[0], str):
        score = float(result[1]) if isinstance(result[1], (int, float)) else 1.0
        if score >= min_confidence:
            texts.append(result[0])
        return texts
    if (
        isinstance(result, (list, tuple))
        and len(result) >= 2
        and isinstance(result[1], tuple)
        and len(result[1]) >= 2
        and isinstance(result[1][0], str)
    ):
        score = float(result[1][1]) if isinstance(result[1][1], (int, float)) else 1.0
        if score >= min_confidence:
            texts.append(result[1][0])
        return texts
    if isinstance(result, (list, tuple)):
        for item in result:
            texts.extend(extract_paddle_texts(item, min_confidence))
    return texts


class OCRReader:
    def __init__(self, backend: str, lang: str, min_confidence: float, device_backend: str) -> None:
        self.backend = backend
        self.lang = lang
        self.min_confidence = min_confidence
        self.reader = None
        if backend == "none":
            return
        if backend == "pytesseract":
            import pytesseract  # type: ignore

            self.reader = pytesseract
        elif backend == "paddleocr":
            from paddleocr import PaddleOCR  # type: ignore

            self.reader = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
        elif backend == "easyocr":
            import easyocr  # type: ignore

            self.reader = easyocr.Reader([lang], gpu=device_backend != "cpu")
        else:
            raise ValueError(f"Unsupported OCR backend: {backend}")

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def recognize(self, image: np.ndarray) -> str:
        if self.backend == "none":
            return ""
        if self.backend == "pytesseract":
            text = self.reader.image_to_string(Image.fromarray(image), lang=self.lang)  # type: ignore[union-attr]
            return " ".join(str(text).split())
        if self.backend == "paddleocr":
            result = self.reader.ocr(image, cls=True)  # type: ignore[union-attr]
            return " ".join(extract_paddle_texts(result, self.min_confidence))
        if self.backend == "easyocr":
            result = self.reader.readtext(image, detail=1, paragraph=False)  # type: ignore[union-attr]
            texts = []
            for item in result:
                if len(item) >= 3 and float(item[2]) >= self.min_confidence:
                    texts.append(str(item[1]))
            return " ".join(texts)
        raise ValueError(f"Unsupported OCR backend: {self.backend}")


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
    metadata_by_path: Mapping[str, Mapping[str, str]],
    text_context: Optional[TextContext],
    ocr_reader: OCRReader,
    ocr_jsonl: Optional[Path],
    device_backend: str,
    mixed_precision: str,
    text_input_mode: str,
    wrong_text_seed: int,
    strip_punctuation: bool,
    remove_spaces: bool,
) -> Tuple[Dict[str, Any], Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, str]]:
    overall = MetricAccumulator()
    by_subset: Dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)
    grid_recons: Dict[int, np.ndarray] = {}
    grid_gts: Dict[int, np.ndarray] = {}
    grid_subsets: Dict[int, str] = {}
    grid_set = set(grid_indices)
    correct_texts = [
        metadata_by_path.get(str(path.resolve()), {}).get("text", "")
        for path in image_paths
    ]
    if text_input_mode == "shuffled":
        text_inputs_for_model = list(correct_texts)
        random.Random(wrong_text_seed).shuffle(text_inputs_for_model)
        if len(text_inputs_for_model) > 1 and text_inputs_for_model == correct_texts:
            text_inputs_for_model = text_inputs_for_model[1:] + text_inputs_for_model[:1]
    elif text_input_mode == "empty":
        text_inputs_for_model = [""] * len(image_paths)
    else:
        text_inputs_for_model = correct_texts

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        pil_images = [resize_pad_image(Image.open(path), image_size, pad_color) for path in batch_paths]
        batch = torch.stack([pil_to_tensor(img) for img in pil_images]).to(device, non_blocking=True)
        batch_texts = text_inputs_for_model[start : start + len(batch_paths)]
        rec_batch = tensor_to_uint8(
            reconstruct_batch(
                model,
                batch,
                batch_texts if text_context is not None else None,
                text_context,
                device_backend=device_backend,
                mixed_precision=mixed_precision,
            )
        )
        gt_batch = [np.asarray(img, dtype=np.uint8) for img in pil_images]
        ocr_rows = []

        for offset, (path, gt, rec) in enumerate(zip(batch_paths, gt_batch, rec_batch)):
            index = start + offset
            metadata = metadata_by_path.get(str(path.resolve()), {})
            subset = metadata.get("subset", subset_by_path.get(str(path.resolve()), "unknown"))
            values = metric_values(gt, rec)
            gt_text = metadata.get("text", "")
            pred_text = ""
            if ocr_reader.enabled:
                pred_text = ocr_reader.recognize(rec)
                values.update(
                    text_metric_values(
                        gt_text,
                        pred_text,
                        strip_punctuation=strip_punctuation,
                        remove_spaces=remove_spaces,
                    )
                )
            overall.update(values)
            by_subset[subset].update(values)
            if ocr_jsonl is not None and ocr_reader.enabled:
                ocr_rows.append({
                    "run": run_name,
                    "index": index,
                    "image_path": str(path),
                    "subset": subset,
                    "source_key": metadata.get("source_key", ""),
                    "gt_text": gt_text,
                    "pred_text": pred_text,
                    "metrics": {key: values[key] for key in values if key.startswith("ocr_")},
                })
            if index in grid_set:
                grid_gts[index] = gt
                grid_recons[index] = rec
                grid_subsets[index] = subset

        if ocr_rows and ocr_jsonl is not None:
            append_jsonl(ocr_jsonl, ocr_rows)

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

    manifest_image_paths, metadata_by_path = load_manifest_metadata(args.manifest_jsonl)
    if args.image_paths_json is not None:
        image_paths = [Path(path) for path in read_json(args.image_paths_json)]
    else:
        image_paths = manifest_image_paths
    if args.ocr_backend != "none" and not metadata_by_path:
        raise SystemExit("--ocr-backend requires --manifest-jsonl with rendered text.")
    if args.image_paths_json is None and args.manifest_jsonl is None:
        raise SystemExit("Provide --image-paths-json or --manifest-jsonl.")
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit("no images to evaluate")

    missing = [str(path) for path in image_paths[:1000] if not path.exists()]
    if missing:
        raise SystemExit(f"missing image files, first missing: {missing[0]}")

    subset_by_path = {path: row.get("subset", "unknown") for path, row in metadata_by_path.items()}
    grid_indices = select_grid_indices(len(image_paths), args.grid_samples, args.grid_seed)
    ocr_reader = OCRReader(
        backend=args.ocr_backend,
        lang=args.ocr_lang,
        min_confidence=args.ocr_min_confidence,
        device_backend=args.device_backend,
    )
    ocr_jsonl = args.ocr_jsonl or (output_dir / "ocr_predictions.jsonl" if ocr_reader.enabled else None)
    if ocr_jsonl is not None and ocr_jsonl.exists():
        ocr_jsonl.unlink()

    metrics_by_run: Dict[str, Any] = {}
    grid_recons_by_run: Dict[str, Dict[int, np.ndarray]] = {}
    grid_gts: Dict[int, np.ndarray] = {}
    grid_subsets: Dict[int, str] = {}
    run_names: List[str] = []

    for run_name, config_path, ckpt_path in args.run:
        print(f"== evaluating {run_name} ==")
        run_names.append(run_name)
        model, config, text_context = load_tokenizer_model(
            config_path,
            ckpt_path,
            device,
            text_layer_pair_index=args.text_layer_pair_index,
        )
        if text_context is not None:
            print(f"[{run_name}] text-conditioned reconstruction uses pair {text_context.selected_pair}")
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
            metadata_by_path=metadata_by_path,
            text_context=text_context,
            ocr_reader=ocr_reader,
            ocr_jsonl=ocr_jsonl,
            device_backend=args.device_backend,
            mixed_precision=args.mixed_precision,
            text_input_mode=args.text_input_mode,
            wrong_text_seed=args.wrong_text_seed,
            strip_punctuation=args.strip_punctuation,
            remove_spaces=args.remove_spaces,
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
        "mixed_precision": args.mixed_precision,
        "batch_size": args.batch_size,
        "ocr_backend": args.ocr_backend,
        "ocr_lang": args.ocr_lang,
        "ocr_min_confidence": args.ocr_min_confidence,
        "ocr_jsonl": str(ocr_jsonl) if ocr_jsonl is not None else None,
        "text_layer_pair_index": args.text_layer_pair_index,
        "text_input_mode": args.text_input_mode,
        "wrong_text_seed": args.wrong_text_seed,
        "strip_punctuation": args.strip_punctuation,
        "remove_spaces": args.remove_spaces,
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
