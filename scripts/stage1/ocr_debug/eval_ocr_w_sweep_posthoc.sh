#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch}"
export DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_ROOT="${SAVE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep}"
OUT_DIR="${OUT_DIR:-$SAVE_ROOT/post_ocr}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}"
DEEPSEEK_OCR_MODEL="${DEEPSEEK_OCR_MODEL:-/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR}"

RUN_GROUPS="${RUN_GROUPS:-noocr w002 w005}"
DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
DEVICE_ID="${DEVICE_ID:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MAX_IMAGES="${MAX_IMAGES:-50}"
GRID_SAMPLES="${GRID_SAMPLES:-16}"
TEXT_LAYER_PAIR_INDEX="${TEXT_LAYER_PAIR_INDEX:-0}"
DEEPSEEK_OCR_PROMPT="${DEEPSEEK_OCR_PROMPT:-<image>
<|grounding|>OCR this image.}"
DEEPSEEK_OCR_BASE_SIZE="${DEEPSEEK_OCR_BASE_SIZE:-1024}"
DEEPSEEK_OCR_IMAGE_SIZE="${DEEPSEEK_OCR_IMAGE_SIZE:-640}"
DEEPSEEK_OCR_DTYPE="${DEEPSEEK_OCR_DTYPE:-bf16}"
DEEPSEEK_OCR_ATTN_IMPLEMENTATION="${DEEPSEEK_OCR_ATTN_IMPLEMENTATION:-eager}"
DEEPSEEK_OCR_MAX_NEW_TOKENS="${DEEPSEEK_OCR_MAX_NEW_TOKENS:-1024}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

find_config() {
  local group="$1"
  local direct="$SAVE_ROOT/configs/${group}.yaml"
  if [[ -f "$direct" ]]; then
    printf '%s\n' "$direct"
    return 0
  fi
  local found
  found=$(find "$SAVE_ROOT" -maxdepth 4 -type f \( -name "${group}.yaml" -o -name "*${group}*.yaml" \) | sort | head -n 1 || true)
  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi
  return 1
}

find_checkpoint() {
  local group="$1"
  local direct="$SAVE_ROOT/$group/checkpoints/0000500.pt"
  if [[ -f "$direct" ]]; then
    printf '%s\n' "$direct"
    return 0
  fi
  local found
  found=$(find "$SAVE_ROOT/$group" -maxdepth 4 -type f \( -name "0000500.pt" -o -name "*0500*.pt" \) | sort | head -n 1 || true)
  if [[ -n "$found" ]]; then
    printf '%s\n' "$found"
    return 0
  fi
  return 1
}

write_paths_json() {
  local output="$1"
  local group="$2"
  local config="$3"
  local ckpt="$4"
  "$PYTHON_BIN" - "$output" "$group" "$config" "$ckpt" "$VAL_MANIFEST" "$DEEPSEEK_OCR_MODEL" <<'PY'
import json
import sys
from pathlib import Path

output, group, config, ckpt, manifest, ocr_model = sys.argv[1:]
data = {
    "run": group,
    "config_path": config,
    "checkpoint_path": ckpt,
    "manifest_jsonl": manifest,
    "deepseek_ocr_model": ocr_model,
    "checkpoint_step": 500,
    "text_input_mode": "correct",
    "ocr_backend": "deepseek_ocr",
}
Path(output).parent.mkdir(parents=True, exist_ok=True)
Path(output).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

require_file "$VAL_MANIFEST" "VAL_MANIFEST"
if [[ "$DEEPSEEK_OCR_MODEL" == /* ]]; then
  if [[ ! -e "$DEEPSEEK_OCR_MODEL" ]]; then
    echo "DEEPSEEK_OCR_MODEL not found: $DEEPSEEK_OCR_MODEL" >&2
    exit 1
  fi
fi
mkdir -p "$OUT_DIR"

echo "OCR posthoc eval"
echo "  save_root: $SAVE_ROOT"
echo "  out_dir: $OUT_DIR"
echo "  val_manifest: $VAL_MANIFEST"
echo "  groups: $RUN_GROUPS"
echo "  max_images: $MAX_IMAGES"
echo "  deepseek_ocr_model: $DEEPSEEK_OCR_MODEL"

for group in $RUN_GROUPS; do
  config=$(find_config "$group") || { echo "config not found for group: $group" >&2; exit 1; }
  ckpt=$(find_checkpoint "$group") || { echo "step500 checkpoint not found for group: $group" >&2; exit 1; }
  run_out="$OUT_DIR/$group"
  mkdir -p "$run_out/deepseek_ocr_tmp"
  write_paths_json "$run_out/eval_paths.json" "$group" "$config" "$ckpt"

  echo
  echo "===== posthoc OCR $group ====="
  echo "config: $config"
  echo "checkpoint: $ckpt"

  if [[ "$SKIP_EXISTING" == "1" && -f "$run_out/ocr_preds.jsonl" && -f "$run_out/ocr_summary.json" ]]; then
    echo "skip eval; existing outputs found: $run_out"
    continue
  fi

  "$PYTHON_BIN" scripts/stage1/evaluate_textatlas_reconstruction.py \
    --run "$group:$config:$ckpt" \
    --manifest-jsonl "$VAL_MANIFEST" \
    --output-dir "$run_out" \
    --device-backend "$DEVICE_BACKEND" \
    --device-id "$DEVICE_ID" \
    --mixed-precision "$MIXED_PRECISION" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-images "$MAX_IMAGES" \
    --grid-samples "$GRID_SAMPLES" \
    --text-layer-pair-index "$TEXT_LAYER_PAIR_INDEX" \
    --text-input-mode correct \
    --ocr-backend deepseek_ocr \
    --ocr-jsonl "$run_out/ocr_predictions.raw.jsonl" \
    --deepseek-ocr-model "$DEEPSEEK_OCR_MODEL" \
    --deepseek-ocr-prompt "$DEEPSEEK_OCR_PROMPT" \
    --deepseek-ocr-base-size "$DEEPSEEK_OCR_BASE_SIZE" \
    --deepseek-ocr-image-size "$DEEPSEEK_OCR_IMAGE_SIZE" \
    --deepseek-ocr-attn-implementation "$DEEPSEEK_OCR_ATTN_IMPLEMENTATION" \
    --deepseek-ocr-dtype "$DEEPSEEK_OCR_DTYPE" \
    --deepseek-ocr-output-dir "$run_out/deepseek_ocr_tmp" \
    --deepseek-ocr-max-new-tokens "$DEEPSEEK_OCR_MAX_NEW_TOKENS"
done

"$PYTHON_BIN" scripts/stage1/ocr_debug/summarize_ocr_w_sweep_posthoc.py \
  --save-root "$SAVE_ROOT" \
  --out-dir "$OUT_DIR" \
  --groups $RUN_GROUPS

echo
echo "posthoc summary: $OUT_DIR/summary/post_ocr_summary.md"
