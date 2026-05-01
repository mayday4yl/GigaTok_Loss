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
MANIFEST_NAME="${MANIFEST_NAME:-cleantextsynth_dense_100}"
SOURCE_MANIFEST_ROOT="${SOURCE_MANIFEST_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$SOURCE_MANIFEST_ROOT/${MANIFEST_NAME}_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$SOURCE_MANIFEST_ROOT/${MANIFEST_NAME}_val.jsonl}"
HOLDOUT_MANIFEST="${HOLDOUT_MANIFEST:-$SOURCE_MANIFEST_ROOT/${MANIFEST_NAME}_holdout.jsonl}"

SAVE_ROOT="${SAVE_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_layer_sweep_debug}"
CONFIG_WORK_DIR="${CONFIG_WORK_DIR:-$SAVE_ROOT/configs}"
BASE_CONFIG="${BASE_CONFIG:-configs/vq/VQ_BL256_dino_disc_glyph_byt5_residual_cross_attn_single_sample_probe_r030_block_v1.yaml}"
MATCHED_CKPT="${MATCHED_CKPT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/cleantextsynth_dense_100_matched_native_500step_seed0/checkpoints/last.pt}"
MATCHED_CONFIG="${MATCHED_CONFIG:-configs/vq/VQ_BL256_dino_disc_matched_native_v1.yaml}"

SWEEP_ITERS="${SWEEP_ITERS:-200}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-20}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
TRAIN_EVAL_BATCH_SIZE="${TRAIN_EVAL_BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-50}"
CKPT_EVERY="${CKPT_EVERY:-$SWEEP_ITERS}"

MASK_RATIO="${MASK_RATIO:-0.2}"
MASK_TAG="${MASK_TAG:-r020}"
BLOCK_SIZE="${BLOCK_SIZE:-2}"
LAYER_SPECS="${LAYER_SPECS:-l20_23:20,23 l16_20_23:16,20,23 l12_16_20_23:12,16,20,23}"
TEXT_MODES="${TEXT_MODES:-correct empty shuffled wrong}"
EVAL_MASK_TAGS="${EVAL_MASK_TAGS:-evalmask}"
EVAL_TAG="${EVAL_TAG:-layer_sweep_${MASK_TAG}_${SWEEP_ITERS}step_seed${GLOBAL_SEED}}"
FIXED_WRONG_TEXT="${FIXED_WRONG_TEXT:-THIS IS A FIXED WRONG TEXT 0123456789}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$CONFIG_WORK_DIR" "$SAVE_ROOT/recon_eval"

for required in "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$HOLDOUT_MANIFEST" "$BASE_CONFIG"; do
  if [[ ! -f "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

generate_layer_config() {
  local output_config="$1"
  local layers_csv="$2"
  local apply_in_eval="$3"

  "$PYTHON_BIN" - "$BASE_CONFIG" "$output_config" "$layers_csv" "$MASK_RATIO" "$BLOCK_SIZE" "$apply_in_eval" <<'PY'
import sys
import yaml

base_config, output_config, layers_csv, ratio, block_size, apply_in_eval = sys.argv[1:7]
layers = [int(item) for item in layers_csv.split(",") if item.strip()]
if not layers:
    raise SystemExit("empty layers")
layer_pairs = [[0, layer] for layer in layers]

with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

text_cfg = cfg.setdefault("text_conditioning", {})
text_cfg["enabled"] = True

recon_cfg = cfg.setdefault("text_recon_conditioning", {})
recon_cfg["enabled"] = True
recon_cfg["mode"] = "residual_cross_attn_visual_mask"
recon_cfg["layers"] = layers
recon_cfg["layer_pairs"] = layer_pairs
recon_cfg["visual_type_embedding"] = False
recon_cfg.setdefault("text_gate", {})["enabled"] = False

mask_cfg = recon_cfg.setdefault("visual_memory_mask", {})
mask_cfg.update({
    "enabled": True,
    "ratio": float(ratio),
    "mode": "learned_mask_token",
    "strategy": "block_random",
    "block_size": int(block_size),
    "fixed_pattern": True,
    "seed": 0,
    "apply_in_eval": apply_in_eval.lower() == "true",
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

declare -a RUN_TAGS=()
declare -a RUN_NAMES=()
declare -a EVALMASK_CONFIGS=()
declare -a NOMASK_CONFIGS=()

echo "Layer sweep"
echo "  manifest: $MANIFEST_NAME"
echo "  save_root: $SAVE_ROOT"
echo "  mask: $MASK_TAG ratio=$MASK_RATIO block_size=$BLOCK_SIZE"
echo "  layer_specs: $LAYER_SPECS"
echo "  iters: $SWEEP_ITERS"
echo "  nproc_per_node: $NPROC_PER_NODE"

for spec in $LAYER_SPECS; do
  tag="${spec%%:*}"
  layers_csv="${spec#*:}"
  run_tag="glyph_${MASK_TAG}_${tag}"
  run_name="${MANIFEST_NAME}_${run_tag}_${SWEEP_ITERS}step_seed${GLOBAL_SEED}"
  evalmask_config="$CONFIG_WORK_DIR/${MANIFEST_NAME}_${run_tag}_evalmask.yaml"
  nomask_config="$CONFIG_WORK_DIR/${MANIFEST_NAME}_${run_tag}_nomask.yaml"

  generate_layer_config "$evalmask_config" "$layers_csv" true
  generate_layer_config "$nomask_config" "$layers_csv" false

  RUN_TAGS+=("$run_tag")
  RUN_NAMES+=("$run_name")
  EVALMASK_CONFIGS+=("$evalmask_config")
  NOMASK_CONFIGS+=("$nomask_config")

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] generated configs for $run_tag"
    continue
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$SAVE_ROOT/$run_name/checkpoints/last.pt" ]]; then
    echo "[skip train] existing checkpoint: $SAVE_ROOT/$run_name/checkpoints/last.pt"
    continue
  fi

  CONFIG="$evalmask_config" \
  MODE="$run_tag" \
  RUN_NAME="$run_name" \
  SAVE_ROOT="$SAVE_ROOT" \
  TRAIN_MANIFEST="$TRAIN_MANIFEST" \
  VAL_MANIFEST="$VAL_MANIFEST" \
  MANIFEST_NAME="$MANIFEST_NAME" \
  ITERS="$SWEEP_ITERS" \
  GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
  GLOBAL_SEED="$GLOBAL_SEED" \
  VAL_MAX_IMAGES="$VAL_MAX_IMAGES" \
  EVAL_BATCH_SIZE="$TRAIN_EVAL_BATCH_SIZE" \
  NUM_WORKERS="$NUM_WORKERS" \
  VAL_NUM_WORKERS="$VAL_NUM_WORKERS" \
  LOG_EVERY="$LOG_EVERY" \
  VAL_EVERY="$VAL_EVERY" \
  CKPT_EVERY="$CKPT_EVERY" \
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
  bash scripts/stage1/multi_image_debug/run_multi_image_probe.sh
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] skip training and evaluation"
  exit 0
fi

for mask_tag in $EVAL_MASK_TAGS; do
  for text_mode in $TEXT_MODES; do
    run_args=()
    if [[ -f "$MATCHED_CKPT" ]]; then
      run_args+=(--run "matched_native:$MATCHED_CONFIG:$MATCHED_CKPT")
    fi
    for i in "${!RUN_TAGS[@]}"; do
      ckpt="$SAVE_ROOT/${RUN_NAMES[$i]}/checkpoints/last.pt"
      if [[ ! -f "$ckpt" ]]; then
        echo "Missing checkpoint for eval: $ckpt" >&2
        exit 1
      fi
      if [[ "$mask_tag" == "nomask" ]]; then
        config="${NOMASK_CONFIGS[$i]}"
      else
        config="${EVALMASK_CONFIGS[$i]}"
      fi
      run_args+=(--run "${RUN_TAGS[$i]}:$config:$ckpt")
    done

    output_dir="$SAVE_ROOT/recon_eval/${mask_tag}_${text_mode}_${EVAL_TAG}"
    echo "[eval] mask=$mask_tag text=$text_mode output=$output_dir"
    "$PYTHON_BIN" scripts/stage1/evaluate_textatlas_reconstruction.py \
      "${run_args[@]}" \
      --manifest-jsonl "$HOLDOUT_MANIFEST" \
      --output-dir "$output_dir" \
      --device-backend npu \
      --mixed-precision bf16 \
      --batch-size "$EVAL_BATCH_SIZE" \
      --max-images 0 \
      --grid-samples 40 \
      --text-input-mode "$text_mode" \
      --wrong-text-seed 0 \
      --fixed-wrong-text "$FIXED_WRONG_TEXT"
  done
done

"$PYTHON_BIN" - "$SAVE_ROOT" "$EVAL_TAG" "$EVAL_MASK_TAGS" "$TEXT_MODES" "${RUN_TAGS[@]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

save_root = Path(sys.argv[1])
eval_tag = sys.argv[2]
mask_tags = sys.argv[3].split()
text_modes = sys.argv[4].split()
run_tags = ["matched_native"] + sys.argv[5:]
eval_root = save_root / "recon_eval"

def load(mask, text):
    path = eval_root / f"{mask}_{text}_{eval_tag}" / "metrics.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle).get("runs", {})

rows = []
for mask in mask_tags:
    loaded = {text: load(mask, text) for text in text_modes}
    for run in run_tags:
        if run not in loaded.get("correct", {}):
            continue
        correct = loaded["correct"][run]["overall"]
        row = {
            "mask": mask,
            "run": run,
            "correct_mse": correct.get("mse"),
            "correct_psnr": correct.get("psnr"),
            "correct_ssim": correct.get("ssim"),
        }
        for text in ("empty", "shuffled", "wrong"):
            if run in loaded.get(text, {}):
                row[f"{text}_minus_correct_mse"] = loaded[text][run]["overall"]["mse"] - correct["mse"]
        rows.append(row)

out_csv = save_root / f"layer_sweep_summary_{eval_tag}.csv"
if rows:
    fieldnames = [
        "mask", "run", "correct_mse", "correct_psnr", "correct_ssim",
        "empty_minus_correct_mse", "shuffled_minus_correct_mse", "wrong_minus_correct_mse",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

print(f"wrote {out_csv}")
print("mask,run,correct_mse,empty_gap,shuffled_gap,wrong_gap")
for row in rows:
    print(
        f"{row.get('mask')},{row.get('run')},"
        f"{row.get('correct_mse'):.6f},"
        f"{row.get('empty_minus_correct_mse', float('nan')):+.6f},"
        f"{row.get('shuffled_minus_correct_mse', float('nan')):+.6f},"
        f"{row.get('wrong_minus_correct_mse', float('nan')):+.6f}"
    )
PY
