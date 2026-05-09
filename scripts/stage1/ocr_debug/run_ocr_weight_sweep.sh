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
SAVE_ROOT="${SAVE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_ocr_weight_sweep_debug}"
MANIFEST_ROOT="${MANIFEST_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}"
HOLDOUT_MANIFEST="${HOLDOUT_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_holdout.jsonl}"
BASE_CONFIG="${BASE_CONFIG:-configs/vq/VQ_BL256_dino_disc_glyph_byt5_residual_cross_attn_scaled_hr_w500_ocr_tf_warmup_l6_12_18_scale005_v1.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/vq/VQ_BL256_dino_disc_matched_native_v1.yaml}"
BASELINE_CKPT="${BASELINE_CKPT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/cleantextsynth_dense_100_matched_native_500step_seed0/checkpoints/last.pt}"
VQ_CKPT="${VQ_CKPT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt}"
DEEPSEEK_OCR_MODEL="${DEEPSEEK_OCR_MODEL:-/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR}"

WEIGHTS="${WEIGHTS:-0.005 0.01 0.02 0.05}"
ITERS="${ITERS:-200}"
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
CKPT_EVERY="${CKPT_EVERY:-200}"
RUN_PREFIX="${RUN_PREFIX:-glyph_ocr_weight_sweep_readable50}"
CONFIG_WORK_DIR="${CONFIG_WORK_DIR:-$SAVE_ROOT/configs}"

RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_GRAD_PROBE="${RUN_GRAD_PROBE:-1}"
RUN_RECON_EVAL="${RUN_RECON_EVAL:-1}"
RUN_OCR_TF_EVAL="${RUN_OCR_TF_EVAL:-1}"
RUN_IMAGE_DIFF="${RUN_IMAGE_DIFF:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "$label not found: $path" >&2
    exit 1
  fi
}

weight_tag() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
text = f"{value:.6g}".replace(".", "p").replace("-", "m")
print(text)
PY
}

make_config() {
  local weight="$1"
  local output_config="$2"
  mkdir -p "$(dirname "$output_config")"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$output_config" "$weight" <<'PY'
import sys
import yaml

base_config, output_config, weight = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

ocr_cfg = cfg.setdefault("ocr_teacher_forcing_loss", {})
ocr_cfg["enabled"] = True
ocr_cfg["weight"] = weight

with open(output_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
print(output_config)
PY
}

train_run() {
  local config="$1"
  local run_name="$2"
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
    --no-wandb
}

run_recon_eval_mode() {
  local run_name="$1"
  local config="$2"
  local ckpt="$3"
  local mode="$4"
  local output_dir="$SAVE_ROOT/recon_eval/$run_name/$mode"
  mkdir -p "$output_dir"
  "$PYTHON_BIN" scripts/stage1/evaluate_textatlas_reconstruction.py \
    --run "matched_native:$BASELINE_CONFIG:$BASELINE_CKPT" \
    --run "$run_name:$config:$ckpt" \
    --manifest-jsonl "$HOLDOUT_MANIFEST" \
    --output-dir "$output_dir" \
    --device-backend "$DEVICE_BACKEND" \
    --mixed-precision "$MIXED_PRECISION" \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-images "$VAL_MAX_IMAGES" \
    --grid-samples 16 \
    --text-input-mode "$mode" \
    --wrong-text-seed "$GLOBAL_SEED"
}

run_ocr_tf_eval_mode() {
  local run_name="$1"
  local config="$2"
  local ckpt="$3"
  local mode="$4"
  local output_dir="$SAVE_ROOT/ocr_tf_eval/$run_name/$mode"
  mkdir -p "$output_dir"
  "$PYTHON_BIN" scripts/stage1/ocr_debug/probe_deepseek_ocr_teacher_forcing_loss.py \
    --run "$run_name:$config:$ckpt" \
    --manifest-jsonl "$HOLDOUT_MANIFEST" \
    --output-dir "$output_dir" \
    --deepseek-ocr-model "$DEEPSEEK_OCR_MODEL" \
    --device-backend "$DEVICE_BACKEND" \
    --mixed-precision "$MIXED_PRECISION" \
    --ocr-dtype bf16 \
    --batch-size 1 \
    --max-images "$VAL_MAX_IMAGES" \
    --text-input-mode "$mode" \
    --target-text-mode correct \
    --target-max-tokens 256 \
    --wrong-text-seed "$GLOBAL_SEED"
}

require_file "$TRAIN_MANIFEST" "TRAIN_MANIFEST"
require_file "$VAL_MANIFEST" "VAL_MANIFEST"
require_file "$HOLDOUT_MANIFEST" "HOLDOUT_MANIFEST"
require_file "$BASE_CONFIG" "BASE_CONFIG"
if [[ "$RUN_RECON_EVAL" == "1" ]]; then
  require_file "$BASELINE_CONFIG" "BASELINE_CONFIG"
  require_file "$BASELINE_CKPT" "BASELINE_CKPT"
fi
mkdir -p "$SAVE_ROOT" "$CONFIG_WORK_DIR"

echo "OCR weight sweep"
echo "  save_root: $SAVE_ROOT"
echo "  base_config: $BASE_CONFIG"
echo "  train_manifest: $TRAIN_MANIFEST"
echo "  val_manifest: $VAL_MANIFEST"
echo "  holdout_manifest: $HOLDOUT_MANIFEST"
echo "  weights: $WEIGHTS"
echo "  iterations: $ITERS"
echo "  global_batch_size: $GLOBAL_BATCH_SIZE"
echo "  visible_devices: $ASCEND_RT_VISIBLE_DEVICES"

for weight in $WEIGHTS; do
  tag=$(weight_tag "$weight")
  config="$CONFIG_WORK_DIR/${RUN_PREFIX}_ocrw${tag}.yaml"
  run_name="${RUN_PREFIX}_ocrw${tag}_${ITERS}step_gbs${GLOBAL_BATCH_SIZE}_seed${GLOBAL_SEED}"
  ckpt="$SAVE_ROOT/$run_name/checkpoints/last.pt"
  make_config "$weight" "$config"

  echo
  echo "===== OCR weight $weight ($run_name) ====="
  if [[ "$RUN_TRAIN" == "1" ]]; then
    if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt" ]]; then
      echo "skip training; checkpoint exists: $ckpt"
    else
      train_run "$config" "$run_name"
    fi
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "checkpoint not found after training step: $ckpt" >&2
    exit 1
  fi

  if [[ "$RUN_GRAD_PROBE" == "1" ]]; then
    "$PYTHON_BIN" scripts/stage1/ocr_debug/probe_ocr_tf_param_grads.py \
      --run "$run_name:$config:$ckpt" \
      --manifest-jsonl "$HOLDOUT_MANIFEST" \
      --output-dir "$SAVE_ROOT/grad_probe/$run_name" \
      --deepseek-ocr-model "$DEEPSEEK_OCR_MODEL" \
      --device-backend "$DEVICE_BACKEND" \
      --mixed-precision "$MIXED_PRECISION" \
      --ocr-dtype bf16 \
      --batch-size 1 \
      --max-images 1 \
      --ocr-loss-weight "$weight" \
      --global-step "$ITERS" \
      --target-max-tokens 256
  fi

  if [[ "$RUN_RECON_EVAL" == "1" ]]; then
    for mode in correct empty shuffled length_matched_wrong; do
      run_recon_eval_mode "$run_name" "$config" "$ckpt" "$mode"
    done
  fi

  if [[ "$RUN_OCR_TF_EVAL" == "1" ]]; then
    for mode in correct empty shuffled length_matched_wrong; do
      run_ocr_tf_eval_mode "$run_name" "$config" "$ckpt" "$mode"
    done
  fi

  if [[ "$RUN_IMAGE_DIFF" == "1" ]]; then
    "$PYTHON_BIN" scripts/stage1/ocr_debug/measure_text_mode_image_diff.py \
      --run "$run_name:$config:$ckpt" \
      --manifest-jsonl "$HOLDOUT_MANIFEST" \
      --output-dir "$SAVE_ROOT/image_diff/$run_name" \
      --device-backend "$DEVICE_BACKEND" \
      --mixed-precision "$MIXED_PRECISION" \
      --batch-size "$EVAL_BATCH_SIZE" \
      --max-images "$VAL_MAX_IMAGES" \
      --compare-modes empty,shuffled,length_matched_wrong \
      --wrong-text-seed "$GLOBAL_SEED"
  fi
done

"$PYTHON_BIN" scripts/stage1/ocr_debug/summarize_ocr_weight_sweep.py \
  --save-root "$SAVE_ROOT" \
  --run-prefix "$RUN_PREFIX" \
  --weights $WEIGHTS \
  --iterations "$ITERS" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --seed "$GLOBAL_SEED" \
  --output "$SAVE_ROOT/ocr_weight_sweep_summary.csv"
