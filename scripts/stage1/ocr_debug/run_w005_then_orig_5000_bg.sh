#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

TOTAL_LOG="${TOTAL_LOG:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/run_w005_then_orig_5000.log}"
mkdir -p "$(dirname "$TOTAL_LOG")"

if [[ "${OCR5000_LOG_ACTIVE:-0}" != "1" ]]; then
  export OCR5000_LOG_ACTIVE=1
  exec > >(tee -a "$TOTAL_LOG") 2>&1
fi

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}"
TEXT_FEATURE_CACHE_PATH="${TEXT_FEATURE_CACHE_PATH:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_feature_cache/readable50_glyph_mapped/text_features.pt}"

COMMON_ENV=(
  ITERS=5000
  GLOBAL_BATCH_SIZE=8
  LOG_EVERY=10
  VAL_EVERY=100
  CKPT_EVERY=500
  GRID_EVERY=500
  RUN_POSTPROCESS=1
  RUN_GRIDS=1
  SKIP_EXISTING=1
  TRAIN_MANIFEST="$TRAIN_MANIFEST"
  VAL_MANIFEST="$VAL_MANIFEST"
)

echo "[$(date '+%F %T')] sequential run started"
echo "repo: $REPO_ROOT"
echo "total_log: $TOTAL_LOG"
echo "train_manifest: $TRAIN_MANIFEST"
echo "val_manifest: $VAL_MANIFEST"
echo "text_feature_cache: $TEXT_FEATURE_CACHE_PATH"
echo

echo "[$(date '+%F %T')] stage 1/2: w005 5000 cached text features"
env \
  "${COMMON_ENV[@]}" \
  SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache \
  RUN_GROUPS=w005 \
  TEXT_FEATURE_CACHE="$TEXT_FEATURE_CACHE_PATH" \
  bash scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh

echo
echo "[$(date '+%F %T')] stage 1/2 finished successfully"
echo "[$(date '+%F %T')] stage 2/2: orig 5000 pure original GigaTok"
env -u TEXT_FEATURE_CACHE \
  "${COMMON_ENV[@]}" \
  SAVE_ROOT=/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000 \
  RUN_GROUPS=orig \
  bash scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh

echo
echo "[$(date '+%F %T')] sequential run finished successfully"
