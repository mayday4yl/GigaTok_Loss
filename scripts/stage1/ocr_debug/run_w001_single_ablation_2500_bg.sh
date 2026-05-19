#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(pwd)}
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch}"
export DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs}"
SAVE_ROOT="${SAVE_ROOT:-$OUTPUT_ROOT/ocr_two_new_ablation_readable50}"
TARGET_STEPS="${TARGET_STEPS:-2500}"
SETTING="${SETTING:-w001_hr_only}"
LOG="${LOG:-$SAVE_ROOT/run_${SETTING}_${TARGET_STEPS}.log}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
RUN_SMOKE="${RUN_SMOKE:-0}"
ALLOW_EXISTING_TRAINING_PROCESS="${ALLOW_EXISTING_TRAINING_PROCESS:-0}"

MANIFEST_ROOT="${MANIFEST_ROOT:-$OUTPUT_ROOT/text_hr_multi_debug/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-$OUTPUT_ROOT/text_feature_cache/readable50_glyph_mapped/text_features.pt}"
DEEPSEEK_OCR_MODEL="${DEEPSEEK_OCR_MODEL:-/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR}"
VQ_CKPT="${VQ_CKPT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt}"

BASELINE_CONFIG="${BASELINE_CONFIG:-$OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/configs/w001.yaml}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/w001}"
BASELINE_CKPT="${BASELINE_CKPT:-$BASELINE_RUN_DIR/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")}"

CONFIG_DIR="${CONFIG_DIR:-$SAVE_ROOT/configs}"
RUN_CONFIG="${RUN_CONFIG:-$CONFIG_DIR/${SETTING}.yaml}"
RUN_DIR="${RUN_DIR:-$SAVE_ROOT/$SETTING}"
RUN_CKPT="$RUN_DIR/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")"

SMOKE_STEPS="${SMOKE_STEPS:-2}"
SMOKE_NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-1}"
SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-1}"
SMOKE_SAVE_ROOT="${SMOKE_SAVE_ROOT:-$SAVE_ROOT/smoke}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-${SETTING}_smoke}"
SMOKE_RUN_DIR="$SMOKE_SAVE_ROOT/$SMOKE_RUN_NAME"
SMOKE_LAST_CKPT="$SMOKE_RUN_DIR/checkpoints/last.pt"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-100}"
CKPT_EVERY="${CKPT_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-50}"

case "$SETTING" in
  w001_hr_only|w001_local_only) ;;
  *)
    echo "[ERROR] SETTING must be w001_hr_only or w001_local_only, got: $SETTING" >&2
    exit 1
    ;;
esac

mkdir -p "$SAVE_ROOT" "$CONFIG_DIR"
exec > >(tee -a "$LOG") 2>&1

stage() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] $label not found: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] $label not found: $path" >&2
    exit 1
  fi
}

safe_bool() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

check_no_training_process() {
  local matches
  matches=$(ps -ef | grep -E "vq_train.py|torchrun" | grep -v grep || true)
  if [[ -n "$matches" ]]; then
    echo "[ERROR] existing torchrun/vq_train.py process found; refusing to start another training job." >&2
    echo "$matches" >&2
    echo "Set ALLOW_EXISTING_TRAINING_PROCESS=1 only if you intentionally want to share devices." >&2
    exit 1
  fi
}

precheck() {
  stage "[START] precheck"
  echo "repo: $REPO_ROOT"
  echo "save_root: $SAVE_ROOT"
  echo "setting: $SETTING"
  echo "target_steps: $TARGET_STEPS"
  echo "prepare_only: $PREPARE_ONLY"
  echo "run_smoke: $RUN_SMOKE"
  echo "run_config: $RUN_CONFIG"
  echo "run_dir: $RUN_DIR"
  echo "vq_ckpt: $VQ_CKPT"
  df -h /home/ma-user/work || true

  require_file "$REPO_ROOT/tokenizer/tokenizer_image/vq/vq_train.py" "vq_train.py"
  require_file "$BASELINE_CONFIG" "baseline w001 config"
  require_file "$BASELINE_CKPT" "baseline w001 checkpoint at target step"
  require_file "$BASELINE_RUN_DIR/val.csv" "baseline w001 val.csv"
  require_file "$TRAIN_MANIFEST" "train manifest"
  require_file "$VAL_MANIFEST" "val manifest"
  require_file "$TEXT_FEATURE_CACHE" "text feature cache"
  require_file "$VQ_CKPT" "initial GigaTok checkpoint"
  require_dir "$DEEPSEEK_OCR_MODEL" "DeepSeek-OCR model"

  if ! safe_bool "$PREPARE_ONLY" && ! safe_bool "$ALLOW_EXISTING_TRAINING_PROCESS"; then
    check_no_training_process
  fi

  "$PYTHON_BIN" -m py_compile \
    tokenizer/tokenizer_image/vq/vq_train.py \
    tokenizer/tokenizer_image/vq/vq_vit_model.py \
    tokenizer/tokenizer_image/vq/blocks.py \
    tokenizer/tokenizer_image/vq/vq_loss.py

  "$PYTHON_BIN" - "$VAL_MANIFEST" "$BASELINE_RUN_DIR/val.csv" "$TARGET_STEPS" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
val_csv = Path(sys.argv[2])
target_steps = int(sys.argv[3])

rows = []
missing = 0
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        image_path = Path(str(row.get("image_path") or row.get("path") or row.get("file_name") or ""))
        if not image_path.exists():
            missing += 1
print(f"manifest rows = {len(rows)}")
print(f"missing images = {missing}")
if len(rows) != 50 or missing != 0:
    raise SystemExit("readable50 val manifest check failed")

found = None
with val_csv.open("r", encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(handle):
        if int(float(row.get("step", "-1"))) == target_steps:
            found = row
            break
if found is None:
    raise SystemExit(f"baseline val.csv does not contain required step={target_steps}")
print(f"baseline w001@{target_steps} val row = {found}")
PY
  stage "[DONE] precheck"
}

generate_config() {
  stage "[START] generate config $RUN_CONFIG"
  "$PYTHON_BIN" - "$BASELINE_CONFIG" "$RUN_CONFIG" "$SETTING" "$DEEPSEEK_OCR_MODEL" <<'PY'
import copy
import sys
import yaml

base_config, out_config, setting, deepseek_model = sys.argv[1:5]

with open(base_config, "r", encoding="utf-8") as handle:
    base = yaml.safe_load(handle)

text_recon_base = base.get("text_recon_conditioning") or {}
layers = list(text_recon_base.get("layers") or [])
layer_pairs = list(text_recon_base.get("layer_pairs") or [])
if [int(x) for x in layers] != [6, 12, 18]:
    raise SystemExit(f"baseline text_recon_conditioning.layers must be [6,12,18], got {layers}")

def make_w001_base(src):
    cfg = copy.deepcopy(src)
    cfg.setdefault("trainer", {})["hr_on"] = False
    cfg.setdefault("trainer", {})["hr_loss_weight"] = 0.0
    cfg.setdefault("trainer", {})["hr_random_one_layer"] = True
    cfg.setdefault("trainer", {})["freeze_encoder"] = True
    cfg.setdefault("trainer", {})["freeze_quantizer"] = True
    cfg.setdefault("trainer", {})["freeze_codebook"] = True
    cfg.setdefault("text_conditioning", {})["enabled"] = True
    text_recon = cfg.setdefault("text_recon_conditioning", {})
    text_recon["enabled"] = True
    text_recon["mode"] = "residual_cross_attn_visual_mask"
    text_recon["layers"] = layers
    text_recon["layer_pairs"] = layer_pairs
    ocr_tf = cfg.setdefault("ocr_teacher_forcing_loss", {})
    ocr_tf["enabled"] = True
    ocr_tf["backend"] = "deepseek_ocr"
    ocr_tf["model_path"] = deepseek_model
    ocr_tf["weight"] = 0.01
    ocr_tf["start_weight"] = 0.005
    ocr_tf["start_step"] = 0
    ocr_tf["warmup_steps"] = 50
    ocr_feature = cfg.setdefault("ocr_feature_loss", {})
    ocr_feature["enabled"] = False
    ocr_feature["weight"] = 0.0
    ocr_vis = cfg.setdefault("ocr_visual_alignment_loss", {})
    ocr_vis["enabled"] = False
    ocr_vis["weight"] = 0.0
    return cfg

def make_w001_hr_only(src):
    cfg = make_w001_base(src)
    text_hr = cfg.setdefault("text_hr", {})
    text_hr["enabled"] = True
    text_hr["hr_loss_weight"] = 500.0
    text_hr["svd_mode"] = "gram_scaled_identity"
    text_hr["image_token_len"] = 0
    text_hr["attention_source"] = "residual_cross_attn_post_softmax"
    text_hr["layer_strategy"] = "explicit_pairs"
    text_hr["layer_pairs"] = layer_pairs
    text_hr["random_one_pair_per_step"] = True
    text_hr["mask_padding_before_svd"] = True
    text_hr["skip_if_valid_tokens_lt"] = int(text_hr.get("skip_if_valid_tokens_lt") or 2)
    local = cfg.setdefault("local_similarity_loss", {})
    local["enabled"] = False
    local["weight"] = 0.0
    return cfg

def make_w001_local_only(src):
    cfg = make_w001_base(src)
    text_hr = cfg.setdefault("text_hr", {})
    text_hr["enabled"] = False
    text_hr["hr_loss_weight"] = 0.0
    text_hr["image_token_len"] = 0
    text_hr["svd_mode"] = "gram_scaled_identity"
    text_hr["attention_source"] = "residual_cross_attn_post_softmax"
    text_hr["layer_strategy"] = "explicit_pairs"
    text_hr["layer_pairs"] = layer_pairs
    text_hr["random_one_pair_per_step"] = True
    text_hr["mask_padding_before_svd"] = True
    text_hr["skip_if_valid_tokens_lt"] = int(text_hr.get("skip_if_valid_tokens_lt") or 2)
    local = cfg.setdefault("local_similarity_loss", {})
    local["enabled"] = True
    local["weight"] = 1.0e-4
    local["distance_type"] = "positional"
    local["average_heads_before_loss"] = True
    local["eps"] = 1.0e-8
    return cfg

if setting == "w001_hr_only":
    cfg = make_w001_hr_only(base)
elif setting == "w001_local_only":
    cfg = make_w001_local_only(base)
else:
    raise SystemExit(f"unknown setting: {setting}")

with open(out_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
print(out_config)
PY
  stage "[DONE] generate config"
}

assert_generated_config() {
  stage "[START] assert generated config"
  "$PYTHON_BIN" - "$RUN_CONFIG" "$SETTING" <<'PY'
import sys
import yaml
from pathlib import Path

path = Path(sys.argv[1])
setting = sys.argv[2]
with path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

def require(condition, message):
    if not condition:
        raise SystemExit(message)

def bool_enabled(block):
    return bool((block or {}).get("enabled", False))

trainer = cfg.get("trainer") or {}
text_recon = cfg.get("text_recon_conditioning") or {}
text_hr = cfg.get("text_hr") or {}
local = cfg.get("local_similarity_loss") or {}
ocr_tf = cfg.get("ocr_teacher_forcing_loss") or {}
ocr_feature = cfg.get("ocr_feature_loss") or {}
ocr_vis = cfg.get("ocr_visual_alignment_loss") or {}

checks = []
checks.append(("trainer.freeze_encoder", trainer.get("freeze_encoder"), True))
checks.append(("trainer.freeze_quantizer", trainer.get("freeze_quantizer"), True))
checks.append(("trainer.freeze_codebook", trainer.get("freeze_codebook"), True))
checks.append(("ocr_teacher_forcing_loss.enabled", bool_enabled(ocr_tf), True))
checks.append(("ocr_teacher_forcing_loss.weight", float(ocr_tf.get("weight", -1)), 0.01))
checks.append(("ocr_teacher_forcing_loss.start_weight", float(ocr_tf.get("start_weight", -1)), 0.005))
checks.append(("ocr_visual_alignment_loss.enabled", bool_enabled(ocr_vis), False))
checks.append(("ocr_feature_loss.enabled", bool_enabled(ocr_feature), False))
checks.append(("text_recon_conditioning.layers", [int(x) for x in (text_recon.get("layers") or [])], [6, 12, 18]))
checks.append(("text_recon_conditioning.mode", text_recon.get("mode"), "residual_cross_attn_visual_mask"))

if setting == "w001_hr_only":
    checks.extend([
        ("text_hr.enabled", bool_enabled(text_hr), True),
        ("text_hr.hr_loss_weight", float(text_hr.get("hr_loss_weight", -1)), 500.0),
        ("text_hr.image_token_len", int(text_hr.get("image_token_len", -1)), 0),
        ("text_hr.svd_mode", text_hr.get("svd_mode"), "gram_scaled_identity"),
        ("local_similarity_loss.enabled", bool_enabled(local), False),
        ("local_similarity_loss.weight", float(local.get("weight", -1)), 0.0),
    ])
elif setting == "w001_local_only":
    checks.extend([
        ("text_hr.enabled", bool_enabled(text_hr), False),
        ("text_hr.hr_loss_weight", float(text_hr.get("hr_loss_weight", -1)), 0.0),
        ("text_hr.image_token_len", int(text_hr.get("image_token_len", -1)), 0),
        ("local_similarity_loss.enabled", bool_enabled(local), True),
        ("local_similarity_loss.weight", float(local.get("weight", -1)), 1.0e-4),
        ("local_similarity_loss.distance_type", local.get("distance_type"), "positional"),
        ("local_similarity_loss.average_heads_before_loss", bool(local.get("average_heads_before_loss", False)), True),
    ])
else:
    raise SystemExit(f"unknown setting: {setting}")

for name, got, expected in checks:
    print(f"[ASSERT] {setting} {name} = {got!r} expected {expected!r}")
    require(got == expected, f"{setting} {name}: got {got!r}, expected {expected!r}")

print(f"[OK] {setting} generated config hard assertions passed")
PY
  stage "[DONE] assert generated config"
}

train_formal() {
  stage "[START] train $SETTING"
  if [[ -f "$RUN_CKPT" ]]; then
    echo "[INFO] skip $SETTING; target checkpoint exists: $RUN_CKPT"
    stage "[DONE] train $SETTING"
    return 0
  fi
  if [[ -d "$RUN_DIR" ]]; then
    echo "[ERROR] run dir exists but target checkpoint missing; refusing to overwrite partial run: $RUN_DIR" >&2
    exit 1
  fi

  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --model-config "$RUN_CONFIG" \
    --dataset textatlas_image_text \
    --data-path "$TRAIN_MANIFEST" \
    --json-path "$TRAIN_MANIFEST" \
    --val-json-path "$VAL_MANIFEST" \
    --save-path "$SAVE_ROOT" \
    --sub-exp-dir "$SETTING" \
    --vq-ckpt "$VQ_CKPT" \
    --finetune \
    --iterations "$TARGET_STEPS" \
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
    --text-feature-cache "$TEXT_FEATURE_CACHE"

  require_file "$RUN_CKPT" "$SETTING target checkpoint"
  stage "[DONE] train $SETTING"
}

train_smoke() {
  stage "[START] smoke $SMOKE_RUN_NAME"
  if [[ -d "$SMOKE_RUN_DIR" ]]; then
    echo "[ERROR] smoke run dir already exists; refusing to reuse stale smoke output: $SMOKE_RUN_DIR" >&2
    exit 1
  fi
  mkdir -p "$SMOKE_SAVE_ROOT"

  torchrun --standalone --nproc_per_node="$SMOKE_NPROC_PER_NODE" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --model-config "$RUN_CONFIG" \
    --dataset textatlas_image_text \
    --data-path "$TRAIN_MANIFEST" \
    --json-path "$TRAIN_MANIFEST" \
    --val-json-path "$VAL_MANIFEST" \
    --save-path "$SMOKE_SAVE_ROOT" \
    --sub-exp-dir "$SMOKE_RUN_NAME" \
    --vq-ckpt "$VQ_CKPT" \
    --finetune \
    --iterations "$SMOKE_STEPS" \
    --global-batch-size "$SMOKE_GLOBAL_BATCH_SIZE" \
    --global-seed "$GLOBAL_SEED" \
    --max-images 0 \
    --num-workers 0 \
    --mixed-precision "$MIXED_PRECISION" \
    --device-backend "$DEVICE_BACKEND" \
    --log-every 1 \
    --val-every "$SMOKE_STEPS" \
    --val-max-images 1 \
    --eval-batch-size 1 \
    --val-num-workers 0 \
    --ckpt-every "$SMOKE_STEPS" \
    --save-last \
    --val-compute-ssim \
    --no-wandb \
    --text-feature-cache "$TEXT_FEATURE_CACHE"

  require_file "$SMOKE_LAST_CKPT" "$SMOKE_RUN_NAME last checkpoint"
  stage "[DONE] smoke $SMOKE_RUN_NAME"
}

main() {
  precheck
  generate_config
  assert_generated_config
  if safe_bool "$PREPARE_ONLY"; then
    stage "[DONE] prepare only"
    exit 0
  fi
  if safe_bool "$RUN_SMOKE"; then
    train_smoke
    stage "[DONE] smoke only"
    exit 0
  fi
  train_formal
}

main "$@"
