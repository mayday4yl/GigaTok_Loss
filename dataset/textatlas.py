import json
import os
import re

from PIL import Image
from torch.utils.data import Dataset

from dataset.ocr_box_gate import (
    build_ocr_box_gate,
    load_ocr_box_index,
    ocr_box_gate_config_from_dict,
)


PROMPT_TEXT_PATTERNS = (
    r"displaying\s+the\s+text\s*[:：]",
    r"written\s+with\s+the\s+text\s*[:：]",
    r"showing\s+the\s+text\s*[:：]",
    r"containing\s+the\s+text\s*[:：]",
)
ALLOWED_EMPTY_TEXT_STATUSES = {
    "styled_no_explicit_text",
    "textvision_no_explicit_text",
}


def looks_like_raw_prompt(text):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROMPT_TEXT_PATTERNS)


class TextAtlasImageTextDataset(Dataset):
    def __init__(self, manifest_path, transform=None, max_images=0, ocr_box_gate_cfg=None, ocr_box_index=None):
        self.transform = transform
        self.manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.rows = []
        self.ocr_box_gate_config = ocr_box_gate_config_from_dict(ocr_box_gate_cfg)
        self.ocr_box_gate_enabled = bool(self.ocr_box_gate_config.enabled)
        if self.ocr_box_gate_enabled:
            self.ocr_box_index = ocr_box_index or load_ocr_box_index(self.ocr_box_gate_config.bbox_jsonl)
        else:
            self.ocr_box_index = None
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "image_path" not in row:
                    raise ValueError(f"{manifest_path}:{line_no}: missing image_path")
                if "text" not in row:
                    raise ValueError(f"{manifest_path}:{line_no}: missing text")
                text = str(row.get("text") or "").strip()
                status = str(row.get("text_extraction_status") or "")
                if not text and status not in ALLOWED_EMPTY_TEXT_STATUSES:
                    raise ValueError(f"{manifest_path}:{line_no}: empty text")
                if looks_like_raw_prompt(text):
                    raise ValueError(f"{manifest_path}:{line_no}: text looks like raw prompt, not rendered text")
                raw_annotation = str(row.get("raw_annotation") or "").strip()
                if (
                    raw_annotation and row.get("text_source") != "raw_text"
                    and text == raw_annotation and status != "plain_annotation"
                ):
                    raise ValueError(f"{manifest_path}:{line_no}: text equals raw_annotation for annotation-based row")
                if status.startswith("failed") or "_failed" in status:
                    raise ValueError(f"{manifest_path}:{line_no}: failed text extraction status={status!r}")
                self.rows.append(row)

        if max_images and max_images > 0:
            self.rows = self.rows[:max_images]

        if not self.rows:
            raise ValueError(f"No rows found in {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image_path = row["image_path"]
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.manifest_dir, image_path)
        image_path = os.path.abspath(image_path) if os.path.isabs(image_path) else image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        text = str(row.get("text") or "")
        if not self.ocr_box_gate_enabled:
            return image, text

        box_entry = self.ocr_box_index.get(image_path)
        if box_entry is None:
            raise KeyError(f"OCR bbox missing for image_path={image_path}")
        gate, gate_stats = build_ocr_box_gate(box_entry["boxes"], self.ocr_box_gate_config)
        return image, text, gate, gate_stats


def build_textatlas_image_text(args, transform):
    manifest_path = args.json_path or args.data_path
    if manifest_path is None:
        raise ValueError("textatlas_image_text requires --json-path or --data-path pointing to a materialized manifest JSONL.")
    ocr_box_gate_cfg = getattr(args, "ocr_box_gate_cfg", None)
    ocr_box_index = getattr(args, "ocr_box_index", None)
    return TextAtlasImageTextDataset(
        manifest_path,
        transform=transform,
        max_images=getattr(args, "max_images", 0),
        ocr_box_gate_cfg=ocr_box_gate_cfg,
        ocr_box_index=ocr_box_index,
    )
