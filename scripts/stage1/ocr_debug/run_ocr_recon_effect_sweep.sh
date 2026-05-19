#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch}"
export DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_ROOT="${SAVE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}"
BASE_CONFIG="${BASE_CONFIG:-configs/vq/VQ_BL256_dino_disc_glyph_byt5_residual_cross_attn_scaled_hr_w500_ocr_tf_warmup_l6_12_18_scale005_v1.yaml}"
VQ_CKPT="${VQ_CKPT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt}"

RUN_GROUPS="${RUN_GROUPS:-noocr w002 w005}"
ITERS="${ITERS:-500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-50}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-50}"
CONFIG_WORK_DIR="${CONFIG_WORK_DIR:-$SAVE_ROOT/configs}"
GRID_STEPS="${GRID_STEPS:-50 250 500}"
GRID_EVERY="${GRID_EVERY:-}"
GRID_SAMPLES="${GRID_SAMPLES:-16}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_POSTPROCESS="${RUN_POSTPROCESS:-1}"
RUN_GRIDS="${RUN_GRIDS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

ocr_weight_for_group() {
  case "$1" in
    orig) echo "" ;;
    noocr) echo "" ;;
    w0005) echo "0.005" ;;
    w001) echo "0.01" ;;
    w0015) echo "0.015" ;;
    w002) echo "0.02" ;;
    w0025) echo "0.025" ;;
    w003) echo "0.03" ;;
    w005) echo "0.05" ;;
    *)
      echo "unknown RUN_GROUP: $1" >&2
      exit 1
      ;;
  esac
}

make_config() {
  local group="$1"
  local output_config="$2"
  local weight
  weight=$(ocr_weight_for_group "$group")
  mkdir -p "$(dirname "$output_config")"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$output_config" "$group" "$weight" <<'PY'
import sys
import yaml

base_config, output_config, group, weight_text = sys.argv[1:5]
with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

trainer = cfg.setdefault("trainer", {})
trainer["hr_on"] = False
trainer["hr_loss_weight"] = 0.0

text_hr = cfg.setdefault("text_hr", {})
text_hr["enabled"] = False
text_hr["hr_loss_weight"] = 0.0

ocr_feature = cfg.setdefault("ocr_feature_loss", {})
ocr_feature["enabled"] = False
ocr_feature["weight"] = 0.0

ocr_tf = cfg.setdefault("ocr_teacher_forcing_loss", {})
if group == "orig":
    text_conditioning = cfg.setdefault("text_conditioning", {})
    text_conditioning["enabled"] = False

    text_recon = cfg.setdefault("text_recon_conditioning", {})
    text_recon["enabled"] = False

    ocr_tf["enabled"] = False
elif group == "noocr":
    text_conditioning = cfg.setdefault("text_conditioning", {})
    text_conditioning["enabled"] = True

    text_recon = cfg.setdefault("text_recon_conditioning", {})
    text_recon["enabled"] = True

    ocr_tf["enabled"] = False
else:
    text_conditioning = cfg.setdefault("text_conditioning", {})
    text_conditioning["enabled"] = True

    text_recon = cfg.setdefault("text_recon_conditioning", {})
    text_recon["enabled"] = True

    ocr_tf["enabled"] = True
    ocr_tf["weight"] = float(weight_text)
    ocr_tf["start_step"] = 0
    ocr_tf["start_weight"] = 0.005
    ocr_tf["warmup_steps"] = 50

with open(output_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
print(output_config)
PY
}

train_run() {
  local config="$1"
  local run_name="$2"
  local cache_args=()
  if [[ -n "$TEXT_FEATURE_CACHE" && "$run_name" == "orig" ]]; then
    echo "warning: TEXT_FEATURE_CACHE is set but ignored for pure original GigaTok group: $run_name" >&2
  elif [[ -n "$TEXT_FEATURE_CACHE" ]]; then
    require_file "$TEXT_FEATURE_CACHE" "TEXT_FEATURE_CACHE"
    cache_args+=(--text-feature-cache "$TEXT_FEATURE_CACHE")
  fi
  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --model-config "$config" \
    --dataset textatlas_image_text \
    --data-path "$TRAIN_MANIFEST" \
    --json-path "$TRAIN_MANIFEST" \
    --val-json-path "$VAL_MANIFEST" \
    --save-path "$SAVE_ROOT" \
    --sub-exp-dir "$run_name" \
    --vq-ckpt "$VQ_CKPT" \
    --finetune \
    --iterations "$ITERS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --global-seed "$GLOBAL_SEED" \
    --max-images 0 \
    --num-workers "$NUM_WORKERS" \
    --mixed-precision "$MIXED_PRECISION" \
    --device-backend "$DEVICE_BACKEND" \
    --log-every "$LOG_EVERY" \
    --val-every "$VAL_EVERY" \
    --val-max-images "$VAL_MAX_IMAGES" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --val-num-workers "$VAL_NUM_WORKERS" \
    --ckpt-every "$CKPT_EVERY" \
    --save-last \
    --val-compute-ssim \
    --no-wandb \
    "${cache_args[@]}"
}

require_file "$TRAIN_MANIFEST" "TRAIN_MANIFEST"
require_file "$VAL_MANIFEST" "VAL_MANIFEST"
require_file "$BASE_CONFIG" "BASE_CONFIG"
require_file "$VQ_CKPT" "VQ_CKPT"
mkdir -p "$SAVE_ROOT" "$CONFIG_WORK_DIR"

echo "OCR reconstruction-effect sweep"
echo "  save_root: $SAVE_ROOT"
echo "  base_config: $BASE_CONFIG"
echo "  train_manifest: $TRAIN_MANIFEST"
echo "  val_manifest: $VAL_MANIFEST"
echo "  groups: $RUN_GROUPS"
echo "  iterations: $ITERS"
echo "  global_batch_size: $GLOBAL_BATCH_SIZE"
echo "  log_every: $LOG_EVERY"
echo "  val_every: $VAL_EVERY"
echo "  ckpt_every: $CKPT_EVERY"
if [[ -n "$TEXT_FEATURE_CACHE" ]]; then
  echo "  text_feature_cache: $TEXT_FEATURE_CACHE"
fi
echo "  visible_devices: $ASCEND_RT_VISIBLE_DEVICES"

for group in $RUN_GROUPS; do
  config="$CONFIG_WORK_DIR/${group}.yaml"
  run_dir="$SAVE_ROOT/$group"
  ckpt="$run_dir/checkpoints/last.pt"
  make_config "$group" "$config"

  echo
  echo "===== $group ====="
  if [[ "$RUN_TRAIN" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt" ]]; then
      echo "skip training; checkpoint exists: $ckpt"
    else
      train_run "$config" "$group"
    fi
  fi
done

if [[ "$RUN_POSTPROCESS" == "1" ]]; then
  postprocess_grid_args=(--grid-steps $GRID_STEPS)
  if [[ -n "$GRID_EVERY" ]]; then
    postprocess_grid_args=(--grid-every "$GRID_EVERY")
  fi
  "$PYTHON_BIN" scripts/stage1/ocr_debug/plot_ocr_recon_effect_sweep.py \
    --save-root "$SAVE_ROOT" \
    --groups $RUN_GROUPS \
    --val-manifest "$VAL_MANIFEST" \
    --device-backend "$DEVICE_BACKEND" \
    --mixed-precision "$MIXED_PRECISION" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --val-max-images "$VAL_MAX_IMAGES" \
    "${postprocess_grid_args[@]}" \
    --grid-samples "$GRID_SAMPLES" \
    $(if [[ "$RUN_GRIDS" == "1" ]]; then echo "--run-grids"; fi)
fi
