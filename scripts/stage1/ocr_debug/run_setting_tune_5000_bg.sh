#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

LOG_PATH="${LOG_PATH:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/run_setting_tune_5000.log}"
mkdir -p "$(dirname "$LOG_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch}"
export DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_ROOT="${SAVE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_feature_cache/readable50_glyph_mapped/text_features.pt}"

ORIG_ROOT="${ORIG_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000}"
NOOCR_ROOT="${NOOCR_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000}"
W005_ROOT="${W005_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache}"

ITERS="${ITERS:-5000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-100}"
CKPT_EVERY="${CKPT_EVERY:-500}"
GRID_SAMPLES="${GRID_SAMPLES:-16}"
SAVE_ALL_GRIDS="${SAVE_ALL_GRIDS:-0}"
RUN_GROUPS_TO_TRAIN="${RUN_GROUPS_TO_TRAIN:-w001 w002 w003}"

DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
DEVICE_ID="${DEVICE_ID:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SETTING_EVAL_DIR="${SETTING_EVAL_DIR:-$SAVE_ROOT/setting_eval}"
RUNS_JSON="${RUNS_JSON:-$SETTING_EVAL_DIR/runs.json}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

require_complete_run() {
  local root="$1"
  local group="$2"
  require_file "$root/configs/$group.yaml" "$group config"
  require_file "$root/$group/checkpoints/0005000.pt" "$group checkpoint 0005000"
}

echo "setting tune 5000"
echo "  log_path: $LOG_PATH"
echo "  save_root: $SAVE_ROOT"
echo "  train_manifest: $TRAIN_MANIFEST"
echo "  val_manifest: $VAL_MANIFEST"
echo "  text_feature_cache: $TEXT_FEATURE_CACHE"
echo "  visible_devices: $ASCEND_RT_VISIBLE_DEVICES"

require_file "$TRAIN_MANIFEST" "TRAIN_MANIFEST"
require_file "$VAL_MANIFEST" "VAL_MANIFEST"
require_file "$TEXT_FEATURE_CACHE" "TEXT_FEATURE_CACHE"
require_complete_run "$ORIG_ROOT" orig
require_complete_run "$NOOCR_ROOT" noocr
require_complete_run "$W005_ROOT" w005
mkdir -p "$SAVE_ROOT" "$SETTING_EVAL_DIR"

for group in $RUN_GROUPS_TO_TRAIN; do
  run_dir="$SAVE_ROOT/$group"
  final_ckpt="$run_dir/checkpoints/0005000.pt"
  if [[ -f "$final_ckpt" ]]; then
    echo "skip $group; complete checkpoint exists: $final_ckpt"
    continue
  fi
  if [[ -d "$run_dir" ]]; then
    echo "refusing to overwrite incomplete existing run directory: $run_dir" >&2
    echo "Move or remove it manually if you want to rerun $group." >&2
    exit 1
  fi

  echo
  echo "===== training $group ====="
  if [[ "$SAVE_ALL_GRIDS" == "1" ]]; then
    grid_env=(GRID_EVERY=500)
  else
    grid_env=(GRID_STEPS="5000")
  fi

  env \
    ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
    SAVE_ROOT="$SAVE_ROOT" \
    RUN_GROUPS="$group" \
    ITERS="$ITERS" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    LOG_EVERY="$LOG_EVERY" \
    VAL_EVERY="$VAL_EVERY" \
    CKPT_EVERY="$CKPT_EVERY" \
    TEXT_FEATURE_CACHE="$TEXT_FEATURE_CACHE" \
    RUN_POSTPROCESS=1 \
    RUN_GRIDS=1 \
    GRID_SAMPLES="$GRID_SAMPLES" \
    SKIP_EXISTING=0 \
    "${grid_env[@]}" \
    bash scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh
done

"$PYTHON_BIN" - "$RUNS_JSON" "$ORIG_ROOT" "$NOOCR_ROOT" "$SAVE_ROOT" "$W005_ROOT" <<'PY'
import json
import sys
from pathlib import Path

out, orig_root, noocr_root, tune_root, w005_root = [Path(arg) for arg in sys.argv[1:]]
runs = {
    "orig": {
        "ocr_weight": None,
        "text_injection": False,
        "ocr_loss": False,
        "config": str(orig_root / "configs" / "orig.yaml"),
        "ckpt": str(orig_root / "orig" / "checkpoints" / "0005000.pt"),
        "run_dir": str(orig_root / "orig"),
    },
    "noocr": {
        "ocr_weight": 0.0,
        "text_injection": True,
        "ocr_loss": False,
        "config": str(noocr_root / "configs" / "noocr.yaml"),
        "ckpt": str(noocr_root / "noocr" / "checkpoints" / "0005000.pt"),
        "run_dir": str(noocr_root / "noocr"),
    },
    "w001": {
        "ocr_weight": 0.01,
        "text_injection": True,
        "ocr_loss": True,
        "config": str(tune_root / "configs" / "w001.yaml"),
        "ckpt": str(tune_root / "w001" / "checkpoints" / "0005000.pt"),
        "run_dir": str(tune_root / "w001"),
    },
    "w002": {
        "ocr_weight": 0.02,
        "text_injection": True,
        "ocr_loss": True,
        "config": str(tune_root / "configs" / "w002.yaml"),
        "ckpt": str(tune_root / "w002" / "checkpoints" / "0005000.pt"),
        "run_dir": str(tune_root / "w002"),
    },
    "w003": {
        "ocr_weight": 0.03,
        "text_injection": True,
        "ocr_loss": True,
        "config": str(tune_root / "configs" / "w003.yaml"),
        "ckpt": str(tune_root / "w003" / "checkpoints" / "0005000.pt"),
        "run_dir": str(tune_root / "w003"),
    },
    "w005": {
        "ocr_weight": 0.05,
        "text_injection": True,
        "ocr_loss": True,
        "config": str(w005_root / "configs" / "w005.yaml"),
        "ckpt": str(w005_root / "w005" / "checkpoints" / "0005000.pt"),
        "run_dir": str(w005_root / "w005"),
    },
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"runs_json: {out}")
PY

"$PYTHON_BIN" scripts/stage1/ocr_debug/evaluate_setting_metrics.py \
  --runs-json "$RUNS_JSON" \
  --manifest-jsonl "$VAL_MANIFEST" \
  --output-dir "$SETTING_EVAL_DIR" \
  --step "$ITERS" \
  --device-backend "$DEVICE_BACKEND" \
  --device-id "$DEVICE_ID" \
  --mixed-precision "$MIXED_PRECISION" \
  --image-size 256 \
  --batch-size "$EVAL_BATCH_SIZE"

echo
echo "setting tune finished"
echo "setting metrics: $SETTING_EVAL_DIR/setting_metrics.csv"
echo "paper table: $SETTING_EVAL_DIR/paper_table.md"
echo "qual grid: $SETTING_EVAL_DIR/qual_grid_5000.png"
