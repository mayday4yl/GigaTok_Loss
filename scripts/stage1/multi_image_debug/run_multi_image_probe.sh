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
SAVE_ROOT="${SAVE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$SAVE_ROOT/manifests}"
MANIFEST_NAME="${MANIFEST_NAME:-cleantextsynth_dense_100}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_ROOT/${MANIFEST_NAME}_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/${MANIFEST_NAME}_val.jsonl}"
MODE="${MODE:-glyph_r030}"
ITERS="${ITERS:-500}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-20}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-500}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
VQ_CKPT="${VQ_CKPT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt}"
CONFIG_WORK_DIR="${CONFIG_WORK_DIR:-$SAVE_ROOT/configs}"
TEXT_LAYER_TAG="${TEXT_LAYER_TAG:-l20_23}"

generate_glyph_probe_config() {
  local base_config="$1"
  local output_config="$2"
  local mode="$3"

  mkdir -p "$(dirname "$output_config")"
  "$PYTHON_BIN" - "$base_config" "$output_config" "$mode" <<'PY'
import sys
import yaml

base_config, output_config, mode = sys.argv[1:4]
with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

if mode == "glyph_r010":
    ratio = 0.1
    block_size = 2
elif mode == "glyph_r020":
    ratio = 0.2
    block_size = 2
elif mode == "glyph_r030":
    ratio = 0.3
    block_size = 2
elif mode == "glyph_r070":
    ratio = 0.7
    block_size = 4
else:
    raise SystemExit(f"unsupported glyph mode: {mode}")

layer_pairs = [[0, 20], [0, 23]]
text_cfg = cfg.setdefault("text_conditioning", {})
text_cfg["enabled"] = True

recon_cfg = cfg.setdefault("text_recon_conditioning", {})
recon_cfg["enabled"] = True
recon_cfg["mode"] = "residual_cross_attn_visual_mask"
recon_cfg["layers"] = [20, 23]
recon_cfg["layer_pairs"] = layer_pairs
recon_cfg["visual_type_embedding"] = False
recon_cfg.setdefault("text_gate", {})["enabled"] = False

mask_cfg = recon_cfg.setdefault("visual_memory_mask", {})
mask_cfg.update({
    "enabled": True,
    "ratio": ratio,
    "mode": "learned_mask_token",
    "strategy": "block_random",
    "block_size": block_size,
    "fixed_pattern": True,
    "seed": 0,
    "apply_in_eval": True,
})

hr_cfg = cfg.setdefault("text_hr", {})
hr_cfg["enabled"] = False
hr_cfg["hr_loss_weight"] = 0.0
hr_cfg["layer_pairs"] = layer_pairs

with open(output_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
print(output_config)
PY
}

if [[ -z "${CONFIG:-}" ]]; then
  case "$MODE" in
    glyph_r010)
      BASE_CONFIG="configs/vq/VQ_BL256_dino_disc_glyph_byt5_residual_cross_attn_single_sample_probe_r030_block_v1.yaml"
      CONFIG="$CONFIG_WORK_DIR/${MANIFEST_NAME}_${MODE}_${TEXT_LAYER_TAG}_evalmask.yaml"
      generate_glyph_probe_config "$BASE_CONFIG" "$CONFIG" "$MODE"
      ;;
    glyph_r020)
      BASE_CONFIG="configs/vq/VQ_BL256_dino_disc_glyph_byt5_residual_cross_attn_single_sample_probe_r030_block_v1.yaml"
      CONFIG="$CONFIG_WORK_DIR/${MANIFEST_NAME}_${MODE}_${TEXT_LAYER_TAG}_evalmask.yaml"
      generate_glyph_probe_config "$BASE_CONFIG" "$CONFIG" "$MODE"
      ;;
    glyph_r030)
      BASE_CONFIG="configs/vq/VQ_BL256_dino_disc_glyph_byt5_residual_cross_attn_single_sample_probe_r030_block_v1.yaml"
      CONFIG="$CONFIG_WORK_DIR/${MANIFEST_NAME}_${MODE}_${TEXT_LAYER_TAG}_evalmask.yaml"
      generate_glyph_probe_config "$BASE_CONFIG" "$CONFIG" "$MODE"
      ;;
    glyph_r070)
      BASE_CONFIG="configs/vq/VQ_BL256_dino_disc_glyph_byt5_residual_cross_attn_single_sample_oracle_r070_block_v1.yaml"
      CONFIG="$CONFIG_WORK_DIR/${MANIFEST_NAME}_${MODE}_${TEXT_LAYER_TAG}_evalmask.yaml"
      generate_glyph_probe_config "$BASE_CONFIG" "$CONFIG" "$MODE"
      ;;
    matched_native|baseline)
      CONFIG="configs/vq/VQ_BL256_dino_disc_matched_native_v1.yaml"
      ;;
    *)
      echo "Unsupported MODE=$MODE. Use glyph_r010, glyph_r020, glyph_r030, glyph_r070, or matched_native; or pass CONFIG=..." >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "$MODE" == glyph_r010 || "$MODE" == glyph_r020 || "$MODE" == glyph_r030 || "$MODE" == glyph_r070 ]]; then
    RUN_NAME="${MANIFEST_NAME}_${MODE}_${TEXT_LAYER_TAG}_${ITERS}step_seed${GLOBAL_SEED}"
  else
    RUN_NAME="${MANIFEST_NAME}_${MODE}_${ITERS}step_seed${GLOBAL_SEED}"
  fi
fi

if [[ ! -f "$TRAIN_MANIFEST" ]]; then
  echo "Training manifest not found: $TRAIN_MANIFEST" >&2
  echo "Run make_multi_image_manifests.py first, or pass TRAIN_MANIFEST=/path/to/train.jsonl." >&2
  exit 1
fi

if [[ ! -f "$VAL_MANIFEST" ]]; then
  echo "Validation manifest not found: $VAL_MANIFEST" >&2
  echo "Run make_multi_image_manifests.py with --val-source-manifest, or pass VAL_MANIFEST=/path/to/val.jsonl." >&2
  exit 1
fi

echo "Multi-image probe"
echo "  mode: $MODE"
echo "  config: $CONFIG"
echo "  train_manifest: $TRAIN_MANIFEST"
echo "  val_manifest: $VAL_MANIFEST"
echo "  save_root: $SAVE_ROOT"
echo "  run_name: $RUN_NAME"
echo "  iterations: $ITERS"
echo "  global_batch_size: $GLOBAL_BATCH_SIZE"
echo "  val_max_images: $VAL_MAX_IMAGES"
echo "  visible_devices: $ASCEND_RT_VISIBLE_DEVICES"
echo "  nproc_per_node: $NPROC_PER_NODE"

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
  tokenizer/tokenizer_image/vq/vq_train.py \
  --model-config "$CONFIG" \
  --dataset textatlas_image_text \
  --data-path "$TRAIN_MANIFEST" \
  --json-path "$TRAIN_MANIFEST" \
  --val-json-path "$VAL_MANIFEST" \
  --save-path "$SAVE_ROOT" \
  --sub-exp-dir "$RUN_NAME" \
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
