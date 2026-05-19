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
OUT_DIR="${OUT_DIR:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_extra_val}"
MANIFEST_DIR="${MANIFEST_DIR:-$OUT_DIR/manifests}"
SOURCE_MANIFEST="${SOURCE_MANIFEST:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/textatlas_stage1_fixed_310k/manifest/val_materialized_manifest_v2text.jsonl}"
READABLE_ROOT="${READABLE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests}"
EXTRA_VAL_MANIFEST="${EXTRA_VAL_MANIFEST:-$MANIFEST_DIR/cleantextsynth_extra_val_30.jsonl}"
NUM_IMAGES="${NUM_IMAGES:-30}"
SEED="${SEED:-5000}"
MIN_TEXT_CHARS="${MIN_TEXT_CHARS:-20}"
REBUILD_MANIFEST="${REBUILD_MANIFEST:-0}"

ORIG_ROOT="${ORIG_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000}"
NOOCR_ROOT="${NOOCR_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000}"
W005_ROOT="${W005_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache}"

DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
DEVICE_ID="${DEVICE_ID:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRID_SAMPLES="${GRID_SAMPLES:-16}"
RUN_OCR="${RUN_OCR:-0}"
DEEPSEEK_OCR_MODEL="${DEEPSEEK_OCR_MODEL:-/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR}"
DEEPSEEK_OCR_PROMPT="${DEEPSEEK_OCR_PROMPT:-<image>
<|grounding|>OCR this image.}"
DEEPSEEK_OCR_BASE_SIZE="${DEEPSEEK_OCR_BASE_SIZE:-1024}"
DEEPSEEK_OCR_IMAGE_SIZE="${DEEPSEEK_OCR_IMAGE_SIZE:-640}"
DEEPSEEK_OCR_DTYPE="${DEEPSEEK_OCR_DTYPE:-bf16}"
DEEPSEEK_OCR_ATTN_IMPLEMENTATION="${DEEPSEEK_OCR_ATTN_IMPLEMENTATION:-eager}"
DEEPSEEK_OCR_MAX_NEW_TOKENS="${DEEPSEEK_OCR_MAX_NEW_TOKENS:-1024}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

orig_config="$ORIG_ROOT/configs/orig.yaml"
orig_ckpt="$ORIG_ROOT/orig/checkpoints/0005000.pt"
noocr_config="$NOOCR_ROOT/configs/noocr.yaml"
noocr_ckpt="$NOOCR_ROOT/noocr/checkpoints/0005000.pt"
w005_config="$W005_ROOT/configs/w005.yaml"
w005_ckpt="$W005_ROOT/w005/checkpoints/0005000.pt"

require_file "$SOURCE_MANIFEST" "SOURCE_MANIFEST"
require_file "$READABLE_ROOT/cleantextsynth_ocr_readable_50_train.jsonl" "readable50 train manifest"
require_file "$READABLE_ROOT/cleantextsynth_ocr_readable_50_val.jsonl" "readable50 val manifest"
require_file "$READABLE_ROOT/cleantextsynth_ocr_readable_50_holdout.jsonl" "readable50 holdout manifest"
require_file "$orig_config" "orig config"
require_file "$orig_ckpt" "orig checkpoint"
require_file "$noocr_config" "noocr config"
require_file "$noocr_ckpt" "noocr checkpoint"
require_file "$w005_config" "w005 config"
require_file "$w005_ckpt" "w005 checkpoint"

mkdir -p "$MANIFEST_DIR" "$OUT_DIR"

if [[ "$REBUILD_MANIFEST" == "1" || ! -f "$EXTRA_VAL_MANIFEST" ]]; then
  "$PYTHON_BIN" scripts/stage1/ocr_debug/make_extra_val_manifest.py \
    --source-manifest "$SOURCE_MANIFEST" \
    --exclude-manifest "$READABLE_ROOT/cleantextsynth_ocr_readable_50_train.jsonl" \
    --exclude-manifest "$READABLE_ROOT/cleantextsynth_ocr_readable_50_val.jsonl" \
    --exclude-manifest "$READABLE_ROOT/cleantextsynth_ocr_readable_50_holdout.jsonl" \
    --output-manifest "$EXTRA_VAL_MANIFEST" \
    --summary-json "$MANIFEST_DIR/cleantextsynth_extra_val_30_summary.json" \
    --contact-sheet "$MANIFEST_DIR/cleantextsynth_extra_val_30_contact_sheet.png" \
    --subset CleanTextSynth \
    --num-images "$NUM_IMAGES" \
    --seed "$SEED" \
    --min-text-chars "$MIN_TEXT_CHARS" \
    --require-existing-images
else
  echo "using existing extra val manifest: $EXTRA_VAL_MANIFEST"
fi

ocr_args=(--ocr-backend none)
if [[ "$RUN_OCR" == "1" ]]; then
  require_file "$DEEPSEEK_OCR_MODEL" "DEEPSEEK_OCR_MODEL"
  ocr_args=(
    --ocr-backend deepseek_ocr
    --ocr-jsonl "$OUT_DIR/ocr_predictions.jsonl"
    --deepseek-ocr-model "$DEEPSEEK_OCR_MODEL"
    --deepseek-ocr-prompt "$DEEPSEEK_OCR_PROMPT"
    --deepseek-ocr-base-size "$DEEPSEEK_OCR_BASE_SIZE"
    --deepseek-ocr-image-size "$DEEPSEEK_OCR_IMAGE_SIZE"
    --deepseek-ocr-attn-implementation "$DEEPSEEK_OCR_ATTN_IMPLEMENTATION"
    --deepseek-ocr-dtype "$DEEPSEEK_OCR_DTYPE"
    --deepseek-ocr-output-dir "$OUT_DIR/deepseek_ocr_tmp"
    --deepseek-ocr-max-new-tokens "$DEEPSEEK_OCR_MAX_NEW_TOKENS"
  )
fi

"$PYTHON_BIN" scripts/stage1/evaluate_textatlas_reconstruction.py \
  --run "orig:$orig_config:$orig_ckpt" \
  --run "noocr:$noocr_config:$noocr_ckpt" \
  --run "w005:$w005_config:$w005_ckpt" \
  --manifest-jsonl "$EXTRA_VAL_MANIFEST" \
  --output-dir "$OUT_DIR" \
  --device-backend "$DEVICE_BACKEND" \
  --device-id "$DEVICE_ID" \
  --mixed-precision "$MIXED_PRECISION" \
  --batch-size "$EVAL_BATCH_SIZE" \
  --max-images "$NUM_IMAGES" \
  --grid-samples "$GRID_SAMPLES" \
  --text-input-mode correct \
  "${ocr_args[@]}"

"$PYTHON_BIN" scripts/stage1/ocr_debug/summarize_extra_val_eval.py \
  --metrics-json "$OUT_DIR/metrics.json" \
  --output-dir "$OUT_DIR/summary"

echo
echo "extra val manifest: $EXTRA_VAL_MANIFEST"
echo "comparison grid: $OUT_DIR/comparison_grid.png"
echo "summary: $OUT_DIR/summary/summary.md"
