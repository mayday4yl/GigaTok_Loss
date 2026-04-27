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

SAVE_ROOT="${SAVE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_single_debug}"
TAG="${TAG:-dense}"
ITERS="${ITERS:-1000}"
MODE="${MODE:-hr}"
HR_CONFIG="${HR_CONFIG:-configs/vq/VQ_BL256_dino_disc_text_hr_v2.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/vq/VQ_BL256_dino_disc_text_baseline_v2.yaml}"
ORIGINAL_CONFIG="${ORIGINAL_CONFIG:-configs/vq/VQ_BL256_dino_disc_stage1_baseline.yaml}"
if [[ -z "${CONFIG:-}" ]]; then
  case "$MODE" in
    hr)
      CONFIG="$HR_CONFIG"
      ;;
    baseline)
      CONFIG="$BASELINE_CONFIG"
      ;;
    original)
      CONFIG="$ORIGINAL_CONFIG"
      ;;
    *)
      echo "Unsupported MODE=$MODE. Use MODE=hr, MODE=baseline, or MODE=original." >&2
      exit 1
      ;;
  esac
fi
VQ_CKPT="${VQ_CKPT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt}"
MANIFEST="${MANIFEST:-$SAVE_ROOT/manifests/$TAG.jsonl}"
RUN_NAME="${RUN_NAME:-${TAG}_${MODE}_1img_${ITERS}step}"
DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-100}"
LR_NOTE="${LR_NOTE:-config_lr}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  echo "Run make_single_image_manifests.py first, or pass MANIFEST=/path/to/one.jsonl." >&2
  exit 1
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  echo "Pass CONFIG=/path/to/config.yaml, or set MODE=hr / MODE=baseline / MODE=original to use tracked defaults." >&2
  echo "On the server, use CONFIG=configs/vq/_pilot_text_hr_local.yaml if you need the local T5 path." >&2
  exit 1
fi

echo "Single-image overfit"
echo "  tag: $TAG"
echo "  mode: $MODE"
echo "  manifest: $MANIFEST"
echo "  config: $CONFIG"
echo "  ckpt: $VQ_CKPT"
echo "  save_root: $SAVE_ROOT"
echo "  run_name: $RUN_NAME"
echo "  iterations: $ITERS"
echo "  lr: $LR_NOTE"
echo "  visible_devices: $ASCEND_RT_VISIBLE_DEVICES"

torchrun --standalone --nproc_per_node=1 \
  tokenizer/tokenizer_image/vq/vq_train.py \
  --model-config "$CONFIG" \
  --dataset textatlas_image_text \
  --data-path "$MANIFEST" \
  --json-path "$MANIFEST" \
  --val-json-path "$MANIFEST" \
  --save-path "$SAVE_ROOT" \
  --sub-exp-dir "$RUN_NAME" \
  --vq-ckpt "$VQ_CKPT" \
  --finetune \
  --iterations "$ITERS" \
  --global-batch-size 1 \
  --max-images 0 \
  --num-workers 0 \
  --mixed-precision "$MIXED_PRECISION" \
  --device-backend "$DEVICE_BACKEND" \
  --log-every "$LOG_EVERY" \
  --val-every "$VAL_EVERY" \
  --val-max-images 1 \
  --eval-batch-size 1 \
  --val-num-workers 0 \
  --ckpt-every "$CKPT_EVERY" \
  --save-last \
  --no-wandb
