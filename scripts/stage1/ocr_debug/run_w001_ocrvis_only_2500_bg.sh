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
LOG="${LOG:-$SAVE_ROOT/run_w001_ocrvis_only_${TARGET_STEPS}.log}"
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
RUN_NAME="${RUN_NAME:-w001_ocrvis_only}"
RUN_CONFIG="${RUN_CONFIG:-$CONFIG_DIR/${RUN_NAME}.yaml}"
RUN_DIR="$SAVE_ROOT/$RUN_NAME"
RUN_CKPT="$RUN_DIR/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")"
SMOKE_STEPS="${SMOKE_STEPS:-2}"
SMOKE_SAVE_ROOT="${SMOKE_SAVE_ROOT:-$SAVE_ROOT/smoke}"
SMOKE_RUN_NAME="${SMOKE_RUN_NAME:-${RUN_NAME}_smoke}"
SMOKE_RUN_DIR="$SMOKE_SAVE_ROOT/$SMOKE_RUN_NAME"
SMOKE_LAST_CKPT="$SMOKE_RUN_DIR/checkpoints/last.pt"

W001_HR_LOCAL_CONFIG="${W001_HR_LOCAL_CONFIG:-$CONFIG_DIR/w001_hr_local.yaml}"
W001_HR_LOCAL_DIR="${W001_HR_LOCAL_DIR:-$SAVE_ROOT/w001_hr_local}"
W001_HR_LOCAL_CKPT="${W001_HR_LOCAL_CKPT:-$W001_HR_LOCAL_DIR/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")}"

OCRVIS_HR_LOCAL_CONFIG="${OCRVIS_HR_LOCAL_CONFIG:-$CONFIG_DIR/ocrvis_hr_local.yaml}"
OCRVIS_HR_LOCAL_DIR="${OCRVIS_HR_LOCAL_DIR:-$SAVE_ROOT/ocrvis_hr_local}"
OCRVIS_HR_LOCAL_CKPT="${OCRVIS_HR_LOCAL_CKPT:-$OCRVIS_HR_LOCAL_DIR/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")}"

RUNS_JSON="${RUNS_JSON:-$SAVE_ROOT/runs_${TARGET_STEPS}_four_methods.json}"
EVAL_DIR="${EVAL_DIR:-$SAVE_ROOT/setting_eval_${TARGET_STEPS}_four_methods}"

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
GRID_SAMPLES="${GRID_SAMPLES:-12}"

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
  echo "target_steps: $TARGET_STEPS"
  echo "prepare_only: $PREPARE_ONLY"
  echo "run_smoke: $RUN_SMOKE"
  echo "run_name: $RUN_NAME"
  echo "eval_dir: $EVAL_DIR"
  df -h /home/ma-user/work || true

  require_file "$REPO_ROOT/tokenizer/tokenizer_image/vq/vq_train.py" "vq_train.py"
  require_file "$REPO_ROOT/scripts/stage1/ocr_debug/evaluate_two_ablation_readable50.py" "evaluate_two_ablation_readable50.py"
  require_file "$BASELINE_CONFIG" "baseline w001 config"
  require_file "$BASELINE_CKPT" "baseline w001 checkpoint at target step"
  require_file "$BASELINE_RUN_DIR/val.csv" "baseline w001 val.csv"
  require_file "$W001_HR_LOCAL_CONFIG" "w001_hr_local config"
  require_file "$W001_HR_LOCAL_CKPT" "w001_hr_local checkpoint at target step"
  require_file "$OCRVIS_HR_LOCAL_CONFIG" "ocrvis_hr_local config"
  require_file "$OCRVIS_HR_LOCAL_CKPT" "ocrvis_hr_local checkpoint at target step"
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
    tokenizer/tokenizer_image/vq/vq_loss.py \
    scripts/stage1/ocr_debug/evaluate_setting_metrics.py \
    scripts/stage1/ocr_debug/evaluate_two_ablation_readable50.py

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
  "$PYTHON_BIN" - "$BASELINE_CONFIG" "$RUN_CONFIG" "$DEEPSEEK_OCR_MODEL" <<'PY'
import sys
import yaml

base_config, out_config, deepseek_model = sys.argv[1:4]
with open(base_config, "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

text_recon = cfg.setdefault("text_recon_conditioning", {})
layers = list(text_recon.get("layers") or [])
if not layers:
    raise SystemExit("baseline config has empty text_recon_conditioning.layers")
if 12 not in [int(x) for x in layers]:
    raise SystemExit("ocr_visual_alignment decoder_layer=12 is not in text_recon_conditioning.layers")

cfg.setdefault("trainer", {})["hr_on"] = False
cfg.setdefault("trainer", {})["hr_loss_weight"] = 0.0
cfg.setdefault("trainer", {})["freeze_encoder"] = True
cfg.setdefault("trainer", {})["freeze_quantizer"] = True
cfg.setdefault("trainer", {})["freeze_codebook"] = True
cfg.setdefault("text_conditioning", {})["enabled"] = True
text_recon["enabled"] = True
text_recon["layers"] = layers

text_hr = cfg.setdefault("text_hr", {})
text_hr["enabled"] = False
text_hr["hr_loss_weight"] = 0.0

local = cfg.setdefault("local_similarity_loss", {})
local["enabled"] = False
local["weight"] = 0.0

feature = cfg.setdefault("ocr_feature_loss", {})
feature["enabled"] = False
feature["weight"] = 0.0

ocr_tf = cfg.setdefault("ocr_teacher_forcing_loss", {})
ocr_tf["enabled"] = False
ocr_tf["weight"] = 0.0
ocr_tf["start_weight"] = 0.0

ocr_vis = cfg.setdefault("ocr_visual_alignment_loss", {})
ocr_vis.update({
    "enabled": True,
    "backend": "deepseek_ocr",
    "model_path": deepseek_model,
    "weight": 0.05,
    "start_step": 0,
    "dtype": "bf16",
    "image_size": 512,
    "attn_implementation": "eager",
    "decoder_layer": 12,
    "loss_type": "cosine",
    "mlp_hidden_mult": 4.0,
})

with open(out_config, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
print(out_config)
PY
  stage "[DONE] generate config"
}

assert_generated_config() {
  stage "[START] assert generated config"
  "$PYTHON_BIN" - "$RUN_CONFIG" <<'PY'
import sys
import yaml
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)

def require(condition, message):
    if not condition:
        raise SystemExit(message)

trainer = cfg.get("trainer") or {}
text = cfg.get("text_conditioning") or {}
text_recon = cfg.get("text_recon_conditioning") or {}
text_hr = cfg.get("text_hr") or {}
local = cfg.get("local_similarity_loss") or {}
ocr_tf = cfg.get("ocr_teacher_forcing_loss") or {}
ocr_feature = cfg.get("ocr_feature_loss") or {}
ocr_vis = cfg.get("ocr_visual_alignment_loss") or {}

require(trainer.get("freeze_encoder") is True, "freeze_encoder must be true")
require(trainer.get("freeze_quantizer") is True, "freeze_quantizer must be true")
require(trainer.get("freeze_codebook") is True, "freeze_codebook must be true")
require(text.get("enabled") is True, "text_conditioning.enabled must be true")
require(text_recon.get("enabled") is True, "text_recon_conditioning.enabled must be true")
require(text_recon.get("mode") == "residual_cross_attn_visual_mask", "text_recon_conditioning.mode mismatch")
require([int(x) for x in (text_recon.get("layers") or [])] == [6, 12, 18], "text_recon_conditioning.layers must be [6,12,18]")
require(text_hr.get("enabled") is False, "text_hr.enabled must be false")
require(float(text_hr.get("hr_loss_weight", 0.0)) == 0.0, "text_hr.hr_loss_weight must be 0")
require(local.get("enabled") is False, "local_similarity_loss.enabled must be false")
require(float(local.get("weight", 0.0)) == 0.0, "local_similarity_loss.weight must be 0")
require(ocr_tf.get("enabled") is False, "ocr_teacher_forcing_loss.enabled must be false")
require(float(ocr_tf.get("weight", 0.0)) == 0.0, "ocr_teacher_forcing_loss.weight must be 0")
require(float(ocr_tf.get("start_weight", 0.0)) == 0.0, "ocr_teacher_forcing_loss.start_weight must be 0")
require(ocr_feature.get("enabled") is False, "ocr_feature_loss.enabled must be false")
require(float(ocr_feature.get("weight", 0.0)) == 0.0, "ocr_feature_loss.weight must be 0")
require(ocr_vis.get("enabled") is True, "ocr_visual_alignment_loss.enabled must be true")
require(ocr_vis.get("backend") == "deepseek_ocr", "ocr_visual_alignment_loss.backend mismatch")
require(float(ocr_vis.get("weight", 0.0)) == 0.05, "ocr_visual_alignment_loss.weight must be 0.05")
require(int(ocr_vis.get("start_step", -1)) == 0, "ocr_visual_alignment_loss.start_step must be 0")
require(str(ocr_vis.get("dtype")) == "bf16", "ocr_visual_alignment_loss.dtype must be bf16")
require(int(ocr_vis.get("image_size", 0)) == 512, "ocr_visual_alignment_loss.image_size must be 512")
require(str(ocr_vis.get("attn_implementation")) == "eager", "ocr_visual_alignment_loss.attn_implementation must be eager")
require(int(ocr_vis.get("decoder_layer", -1)) == 12, "ocr_visual_alignment_loss.decoder_layer must be 12")
require(str(ocr_vis.get("loss_type")) == "cosine", "ocr_visual_alignment_loss.loss_type must be cosine")
require(float(ocr_vis.get("mlp_hidden_mult", 0.0)) == 4.0, "ocr_visual_alignment_loss.mlp_hidden_mult must be 4.0")

print("[OK] generated config hard assertions passed")
PY
  stage "[DONE] assert generated config"
}

train_run() {
  stage "[START] train $RUN_NAME"
  if [[ -f "$RUN_CKPT" ]]; then
    echo "[INFO] skip $RUN_NAME; target checkpoint exists: $RUN_CKPT"
    stage "[DONE] train $RUN_NAME"
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
    --sub-exp-dir "$RUN_NAME" \
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

  require_file "$RUN_CKPT" "$RUN_NAME target checkpoint"
  stage "[DONE] train $RUN_NAME"
}

train_smoke() {
  stage "[START] smoke $SMOKE_RUN_NAME"
  if [[ -d "$SMOKE_RUN_DIR" ]]; then
    echo "[ERROR] smoke run dir already exists; refusing to reuse stale smoke output: $SMOKE_RUN_DIR" >&2
    exit 1
  fi
  mkdir -p "$SMOKE_SAVE_ROOT"

  torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
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
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --global-seed "$GLOBAL_SEED" \
    --max-images 0 \
    --num-workers "$NUM_WORKERS" \
    --mixed-precision "$MIXED_PRECISION" \
    --device-backend "$DEVICE_BACKEND" \
    --log-every 1 \
    --val-every "$VAL_EVERY" \
    --val-max-images "$VAL_MAX_IMAGES" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --val-num-workers "$VAL_NUM_WORKERS" \
    --ckpt-every "$SMOKE_STEPS" \
    --save-last \
    --val-compute-ssim \
    --no-wandb \
    --text-feature-cache "$TEXT_FEATURE_CACHE"

  require_file "$SMOKE_LAST_CKPT" "$SMOKE_RUN_NAME last checkpoint"
  stage "[DONE] smoke $SMOKE_RUN_NAME"
}

write_runs_json_common() {
  local label="$1"
  stage "[START] $label"
  "$PYTHON_BIN" - "$RUNS_JSON" "$TARGET_STEPS" \
    "$BASELINE_CONFIG" "$BASELINE_CKPT" "$BASELINE_RUN_DIR" \
    "$W001_HR_LOCAL_CONFIG" "$W001_HR_LOCAL_CKPT" "$W001_HR_LOCAL_DIR" \
    "$OCRVIS_HR_LOCAL_CONFIG" "$OCRVIS_HR_LOCAL_CKPT" "$OCRVIS_HR_LOCAL_DIR" \
    "$RUN_CONFIG" "$RUN_CKPT" "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

(
    out_path, target_steps,
    w001_config, w001_ckpt, w001_run,
    a_config, a_ckpt, a_run,
    b_config, b_ckpt, b_run,
    c_config, c_ckpt, c_run,
) = sys.argv[1:15]

step = int(target_steps)
data = {
    "w001": {
        "ocr_weight": 0.01,
        "text_injection": True,
        "ocr_loss": True,
        "config": w001_config,
        "ckpt": w001_ckpt,
        "run_dir": w001_run,
        "step": step,
        "train_loss_type": "text + OCR CE(0.01)",
        "high_rank": "no",
        "local_sim": "no",
        "note": "strong reference reused at target step",
    },
    "w001_hr_local": {
        "ocr_weight": 0.01,
        "text_injection": True,
        "ocr_loss": True,
        "config": a_config,
        "ckpt": a_ckpt,
        "run_dir": a_run,
        "step": step,
        "train_loss_type": "text + OCR CE(0.01) + HR + local-sim",
        "high_rank": "yes",
        "local_sim": "yes",
        "note": "combined route A",
    },
    "ocrvis_hr_local": {
        "ocr_weight": None,
        "text_injection": True,
        "ocr_loss": False,
        "config": b_config,
        "ckpt": b_ckpt,
        "run_dir": b_run,
        "step": step,
        "train_loss_type": "text + HR + local-sim + OCR visual alignment",
        "high_rank": "yes",
        "local_sim": "yes",
        "note": "combined route B; OCR CE disabled",
    },
    "w001_ocrvis_only": {
        "ocr_weight": None,
        "text_injection": True,
        "ocr_loss": False,
        "config": c_config,
        "ckpt": c_ckpt,
        "run_dir": c_run,
        "step": step,
        "train_loss_type": "text + OCR visual alignment",
        "high_rank": "no",
        "local_sim": "no",
        "note": "single ablation; OCR CE disabled",
    },
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out_path)
PY
  stage "[DONE] $label"
}

write_runs_json() {
  require_file "$RUN_CKPT" "$RUN_NAME checkpoint at target step"
  write_runs_json_common "write four-method runs.json"
}

write_planned_runs_json() {
  write_runs_json_common "write planned four-method runs.json"
}

evaluate_four_methods() {
  stage "[START] evaluate four methods"
  write_runs_json
  "$PYTHON_BIN" scripts/stage1/ocr_debug/evaluate_two_ablation_readable50.py \
    --runs-json "$RUNS_JSON" \
    --manifest-jsonl "$VAL_MANIFEST" \
    --output-dir "$EVAL_DIR" \
    --step "$TARGET_STEPS" \
    --device-backend "$DEVICE_BACKEND" \
    --device-id 0 \
    --mixed-precision "$MIXED_PRECISION" \
    --image-size 256 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --grid-samples "$GRID_SAMPLES" \
    --python-bin "$PYTHON_BIN" \
    --deepseek-ocr-model "$DEEPSEEK_OCR_MODEL" \
    --thirdparty-ocr-status "unavailable"

  "$PYTHON_BIN" - "$EVAL_DIR" "$TARGET_STEPS" <<'PY'
import csv
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
target_steps = int(sys.argv[2])
runs = ["w001", "w001_hr_local", "ocrvis_hr_local", "w001_ocrvis_only"]
total = 0
for run in runs:
    count = len(list((eval_dir / "reconstruction" / run).glob("*.png")))
    print(f"{run}: {count}")
    if count != 50:
        raise SystemExit(f"{run} expected 50 reconstruction PNGs, got {count}")
    total += count
print(f"total: {total}")
if total != 200:
    raise SystemExit(f"expected total 200 reconstruction PNGs, got {total}")
rows = list(csv.DictReader((eval_dir / "setting_metrics.csv").open("r", encoding="utf-8")))
if len(rows) != 4:
    raise SystemExit(f"expected 4 rows, got {len(rows)}")
for row in rows:
    if int(float(row.get("step", "-1"))) != target_steps:
        raise SystemExit(f"row has wrong step: {row}")
    if not row.get("psnr") or not row.get("ssim"):
        raise SystemExit(f"missing train-val PSNR/SSIM: {row}")
print(f"paper_table: {eval_dir / 'paper_table.md'}")
print(f"setting_metrics: {eval_dir / 'setting_metrics.csv'}")
print(f"qual_grid: {eval_dir / f'qual_grid_{target_steps}.png'}")
print(f"comparison_grid: {eval_dir / 'reconstruction' / 'comparison_grid.png'}")
PY
  stage "[DONE] evaluate four methods"
}

precheck
generate_config
assert_generated_config
if safe_bool "$PREPARE_ONLY"; then
  write_planned_runs_json
  stage "[DONE] prepare_only; no training launched"
  exit 0
fi
if safe_bool "$RUN_SMOKE"; then
  train_smoke
  stage "[DONE] smoke_only; no formal training launched"
  exit 0
fi
train_run
evaluate_four_methods
stage "[DONE] run_w001_ocrvis_only_${TARGET_STEPS}"
