import json
import os
import re

from PIL import Image
from torch.utils.data import Dataset


PROMPT_TEXT_PATTERNS = (
    r"displaying\s+the\s+text\s*[:：]",
    r"written\s+with\s+the\s+text\s*[:：]",
    r"showing\s+the\s+text\s*[:：]",
    r"containing\s+the\s+text\s*[:：]",
)


def looks_like_raw_prompt(text):
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in PROMPT_TEXT_PATTERNS)


class TextAtlasImageTextDataset(Dataset):
    def __init__(self, manifest_path, transform=None, max_images=0):
        self.transform = transform
        self.manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        self.rows = []
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
                if not text:
                    raise ValueError(f"{manifest_path}:{line_no}: empty text")
                if looks_like_raw_prompt(text):
                    raise ValueError(f"{manifest_path}:{line_no}: text looks like raw prompt, not rendered text")
                status = str(row.get("text_extraction_status") or "")
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
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, str(row.get("text") or "")


def build_textatlas_image_text(args, transform):
    manifest_path = args.json_path or args.data_path
    if manifest_path is None:
        raise ValueError("textatlas_image_text requires --json-path or --data-path pointing to a materialized manifest JSONL.")
    return TextAtlasImageTextDataset(
        manifest_path,
        transform=transform,
        max_images=getattr(args, "max_images", 0),
    )
