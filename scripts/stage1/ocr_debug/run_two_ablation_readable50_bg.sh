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
RUN_SMOKE="${RUN_SMOKE:-0}"
LOG="${LOG:-$SAVE_ROOT/run_two_ablation.log}"

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
RUNS_JSON="${RUNS_JSON:-$SAVE_ROOT/runs_${TARGET_STEPS}.json}"
EVAL_DIR="${EVAL_DIR:-$SAVE_ROOT/setting_eval_${TARGET_STEPS}}"

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
GLOBAL_SEED="${GLOBAL_SEED:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
VAL_EVERY="${VAL_EVERY:-100}"
CKPT_EVERY="${CKPT_EVERY:-500}"
GRID_EVERY="${GRID_EVERY:-500}"
NUM_WORKERS="${NUM_WORKERS:-2}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
VAL_MAX_IMAGES="${VAL_MAX_IMAGES:-50}"
GRID_SAMPLES="${GRID_SAMPLES:-12}"

SMOKE_STEPS="${SMOKE_STEPS:-2}"
SMOKE_GLOBAL_BATCH_SIZE="${SMOKE_GLOBAL_BATCH_SIZE:-1}"
SMOKE_NPROC_PER_NODE="${SMOKE_NPROC_PER_NODE:-1}"
SMOKE_SAVE_ROOT="${SMOKE_SAVE_ROOT:-$SAVE_ROOT/smoke}"

mkdir -p "$SAVE_ROOT"
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

old_task_running() {
  local pid_file="$OUTPUT_ROOT/run_rebuild_then_weight_sweep_then_eval.pid"
  [[ -f "$pid_file" ]] || return 1
  local pid
  pid=$(cat "$pid_file" 2>/dev/null || true)
  [[ -n "$pid" ]] || return 1
  ps -p "$pid" >/dev/null 2>&1
}

precheck_static() {
  stage "[START] precheck"
  echo "repo: $REPO_ROOT"
  echo "save_root: $SAVE_ROOT"
  echo "target_steps: $TARGET_STEPS"
  echo "run_smoke: $RUN_SMOKE"
  echo "train_manifest: $TRAIN_MANIFEST"
  echo "val_manifest: $VAL_MANIFEST"
  echo "text_feature_cache: $TEXT_FEATURE_CACHE"
  echo "deepseek_ocr_model: $DEEPSEEK_OCR_MODEL"
  df -h /home/ma-user/work || true

  if old_task_running; then
    local real_repo
    real_repo=$(cd "$REPO_ROOT" && pwd -P)
    if [[ "$real_repo" == "/home/ma-user/work/GigaTok_hr/GigaTok_Loss" ]]; then
      echo "[ERROR] old sequential task is still running and REPO_ROOT is the main repo."
      echo "Use /home/ma-user/work/GigaTok_hr/GigaTok_Loss_two_ablation as an independent worktree."
      exit 1
    fi
    echo "[INFO] old sequential task is still running; current repo is separate: $real_repo"
  fi

  require_file "$REPO_ROOT/tokenizer/tokenizer_image/vq/vq_train.py" "vq_train.py"
  require_file "$BASELINE_CONFIG" "baseline w001 config"
  require_file "$BASELINE_CKPT" "baseline w001 checkpoint at target step"
  require_file "$BASELINE_RUN_DIR/val.csv" "baseline w001 val.csv"
  require_file "$TRAIN_MANIFEST" "train manifest"
  require_file "$VAL_MANIFEST" "val manifest"
  require_file "$TEXT_FEATURE_CACHE" "text feature cache"
  require_file "$VQ_CKPT" "initial GigaTok checkpoint"
  require_dir "$DEEPSEEK_OCR_MODEL" "DeepSeek-OCR model"

  "$PYTHON_BIN" -m py_compile \
    tokenizer/tokenizer_image/vq/vq_train.py \
    tokenizer/tokenizer_image/vq/vq_vit_model.py \
    tokenizer/tokenizer_image/vq/blocks.py \
    tokenizer/tokenizer_image/vq/vq_loss.py \
    scripts/stage1/ocr_debug/evaluate_setting_metrics.py \
    scripts/stage1/ocr_debug/evaluate_two_ablation_readable50.py

  "$PYTHON_BIN" - "$VAL_MANIFEST" "$BASELINE_CKPT" "$BASELINE_RUN_DIR/val.csv" "$TARGET_STEPS" <<'PY'
import csv
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
ckpt = Path(sys.argv[2])
val_csv = Path(sys.argv[3])
target_steps = int(sys.argv[4])

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
print(f"baseline w001@{target_steps} ckpt = {ckpt}")
print(f"baseline w001@{target_steps} val row = {found}")
PY

  "$PYTHON_BIN" - <<'PY'
try:
    import lpips  # noqa: F401
    print("lpips available")
except Exception as exc:
    print(f"lpips unavailable: {exc!r}")
PY
  stage "[DONE] precheck"
}

generate_configs() {
  mkdir -p "$CONFIG_DIR"
  "$PYTHON_BIN" - "$BASELINE_CONFIG" "$CONFIG_DIR/w001_hr_local.yaml" "$CONFIG_DIR/ocrvis_hr_local.yaml" <<'PY'
import copy
import sys
import yaml

base_config, a_config, b_config = sys.argv[1:4]
with open(base_config, "r", encoding="utf-8") as handle:
    base = yaml.safe_load(handle)

layers = list((base.get("text_recon_conditioning") or {}).get("layers") or [])
layer_pairs = list((base.get("text_recon_conditioning") or {}).get("layer_pairs") or [])
if not layers:
    raise SystemExit("baseline config has empty text_recon_conditioning.layers")
if 12 not in [int(x) for x in layers]:
    raise SystemExit("ocr_visual_alignment decoder_layer=12 is not in baseline text_recon_conditioning.layers")

def common(cfg):
    cfg.setdefault("trainer", {})["hr_on"] = False
    cfg.setdefault("trainer", {})["hr_loss_weight"] = 0.0
    cfg.setdefault("trainer", {})["hr_random_one_layer"] = True
    cfg.setdefault("trainer", {})["freeze_encoder"] = True
    cfg.setdefault("trainer", {})["freeze_quantizer"] = True
    cfg.setdefault("trainer", {})["freeze_codebook"] = True
    cfg.setdefault("text_conditioning", {})["enabled"] = True
    cfg.setdefault("text_recon_conditioning", {})["enabled"] = True
    cfg.setdefault("text_recon_conditioning", {})["layers"] = layers
    cfg.setdefault("text_recon_conditioning", {})["layer_pairs"] = layer_pairs
    text_hr = cfg.setdefault("text_hr", {})
    text_hr["enabled"] = True
    text_hr["hr_loss_weight"] = float(text_hr.get("hr_loss_weight") or 500.0)
    text_hr["svd_mode"] = "gram_scaled_identity"
    text_hr["image_token_len"] = 0
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
    feature = cfg.setdefault("ocr_feature_loss", {})
    feature["enabled"] = False
    feature["weight"] = 0.0
    cfg.setdefault("text_hr", {})

a = copy.deepcopy(base)
common(a)
a.setdefault("ocr_visual_alignment_loss", {})["enabled"] = False
ocr_tf = a.setdefault("ocr_teacher_forcing_loss", {})
ocr_tf["enabled"] = True
ocr_tf["weight"] = 0.01
ocr_tf["start_step"] = 0
ocr_tf["start_weight"] = 0.005
ocr_tf["warmup_steps"] = 50

b = copy.deepcopy(base)
common(b)
b_tf = b.setdefault("ocr_teacher_forcing_loss", {})
b_tf["enabled"] = False
vis = b.setdefault("ocr_visual_alignment_loss", {})
vis.update({
    "enabled": True,
    "backend": "deepseek_ocr",
    "model_path": "/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR",
    "weight": 0.05,
    "start_step": 0,
    "dtype": "bf16",
    "image_size": 512,
    "attn_implementation": "eager",
    "decoder_layer": 12,
    "loss_type": "cosine",
    "mlp_hidden_mult": 4.0,
})

for path, cfg in ((a_config, a), (b_config, b)):
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    print(path)
PY
}

write_runs_json() {
  local a_ckpt="$SAVE_ROOT/w001_hr_local/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")"
  local b_ckpt="$SAVE_ROOT/ocrvis_hr_local/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")"
  require_file "$a_ckpt" "w001_hr_local checkpoint at target step"
  require_file "$b_ckpt" "ocrvis_hr_local checkpoint at target step"
  "$PYTHON_BIN" - "$RUNS_JSON" "$TARGET_STEPS" "$BASELINE_CONFIG" "$BASELINE_CKPT" "$BASELINE_RUN_DIR" "$CONFIG_DIR/w001_hr_local.yaml" "$a_ckpt" "$SAVE_ROOT/w001_hr_local" "$CONFIG_DIR/ocrvis_hr_local.yaml" "$b_ckpt" "$SAVE_ROOT/ocrvis_hr_local" <<'PY'
import json
import sys
from pathlib import Path

(
    out_path, target_steps, w001_config, w001_ckpt, w001_run,
    a_config, a_ckpt, a_run, b_config, b_ckpt, b_run
) = sys.argv[1:12]
data = {
    "w001": {
        "ocr_weight": 0.01,
        "text_injection": True,
        "ocr_loss": True,
        "config": w001_config,
        "ckpt": w001_ckpt,
        "run_dir": w001_run,
        "step": int(target_steps),
    },
    "w001_hr_local": {
        "ocr_weight": 0.01,
        "text_injection": True,
        "ocr_loss": True,
        "config": a_config,
        "ckpt": a_ckpt,
        "run_dir": a_run,
        "step": int(target_steps),
    },
    "ocrvis_hr_local": {
        "ocr_weight": None,
        "text_injection": True,
        "ocr_loss": False,
        "config": b_config,
        "ckpt": b_ckpt,
        "run_dir": b_run,
        "step": int(target_steps),
    },
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out_path)
PY
}

validate_config_summary() {
  "$PYTHON_BIN" - "$CONFIG_DIR/w001_hr_local.yaml" "$CONFIG_DIR/ocrvis_hr_local.yaml" "$SAVE_ROOT/config_summary.txt" <<'PY'
import sys
import yaml
from pathlib import Path

out_lines = []
for name, path in (("w001_hr_local", sys.argv[1]), ("ocrvis_hr_local", sys.argv[2])):
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    text_hr = cfg.get("text_hr") or {}
    text_recon = cfg.get("text_recon_conditioning") or {}
    local = cfg.get("local_similarity_loss") or {}
    ocr_tf = cfg.get("ocr_teacher_forcing_loss") or {}
    ocr_vis = cfg.get("ocr_visual_alignment_loss") or {}
    if not bool(text_hr.get("enabled")):
        raise SystemExit(f"{name}: text_hr.enabled is not true")
    if float(text_hr.get("hr_loss_weight", 0.0)) <= 0.0:
        raise SystemExit(f"{name}: text_hr.hr_loss_weight must be > 0")
    if not bool(local.get("enabled")) or float(local.get("weight", 0.0)) <= 0.0:
        raise SystemExit(f"{name}: local_similarity_loss must be enabled with positive weight")
    selected = f"random_one_pair_per_step over decoder layers {[pair[1] for pair in text_recon.get('layer_pairs', [])]}"
    out_lines.extend([
        f"[{name}]",
        f"text_hr.enabled={text_hr.get('enabled')}",
        f"text_hr.hr_loss_weight={text_hr.get('hr_loss_weight')}",
        f"text_hr.svd_mode={text_hr.get('svd_mode')}",
        f"text_hr.image_token_len={text_hr.get('image_token_len')}",
        f"text_hr.attention_source={text_hr.get('attention_source', 'selected_residual_cross_attn')}",
        f"text_recon_conditioning.layers={text_recon.get('layers')}",
        f"text_recon_conditioning.layer_pairs={text_recon.get('layer_pairs')}",
        f"selected_decoder_layer_for_hr_local={selected}",
        f"ocr_visual_alignment_decoder_layer={ocr_vis.get('decoder_layer', '')}",
        f"ocr_teacher_forcing_loss.enabled={ocr_tf.get('enabled')}",
        f"ocr_teacher_forcing_loss.weight={ocr_tf.get('weight', '')}",
        f"ocr_visual_alignment_loss.enabled={ocr_vis.get('enabled', False)}",
        "",
    ])
Path(sys.argv[3]).write_text("\n".join(out_lines), encoding="utf-8")
print("\n".join(out_lines))
PY
}

train_one() {
  local name="$1"
  local config="$2"
  local save_root="$3"
  local iterations="$4"
  local global_batch="$5"
  local nproc="$6"
  local ckpt_every="$7"
  local val_every="$8"
  local run_dir="$save_root/$name"
  local target_ckpt="$run_dir/checkpoints/$(printf "%07d.pt" "$iterations")"

  if [[ -f "$target_ckpt" ]]; then
    echo "[INFO] skip $name; target checkpoint exists: $target_ckpt"
    return 0
  fi
  if [[ -d "$run_dir" && "$RUN_SMOKE" != "1" ]]; then
    echo "[ERROR] run dir exists but target checkpoint missing; refusing to overwrite partial run: $run_dir" >&2
    exit 1
  fi

  torchrun --standalone --nproc_per_node="$nproc" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --model-config "$config" \
    --dataset textatlas_image_text \
    --data-path "$TRAIN_MANIFEST" \
    --json-path "$TRAIN_MANIFEST" \
    --val-json-path "$VAL_MANIFEST" \
    --save-path "$save_root" \
    --sub-exp-dir "$name" \
    --vq-ckpt "$VQ_CKPT" \
    --finetune \
    --iterations "$iterations" \
    --global-batch-size "$global_batch" \
    --global-seed "$GLOBAL_SEED" \
    --max-images 0 \
    --num-workers "$NUM_WORKERS" \
    --mixed-precision "$MIXED_PRECISION" \
    --device-backend "$DEVICE_BACKEND" \
    --log-every "$LOG_EVERY" \
    --val-every "$val_every" \
    --val-max-images "$VAL_MAX_IMAGES" \
    --eval-batch-size "$EVAL_BATCH_SIZE" \
    --val-num-workers "$VAL_NUM_WORKERS" \
    --ckpt-every "$ckpt_every" \
    --save-last \
    --val-compute-ssim \
    --no-wandb \
    --text-feature-cache "$TEXT_FEATURE_CACHE"
}

generate_bbox_smoke() {
  "$PYTHON_BIN" - "$VAL_MANIFEST" "$SAVE_ROOT/text_bbox_debug_grid_smoke.png" <<'PY'
import sys
from pathlib import Path
from PIL import Image, ImageDraw
from scripts.stage1.ocr_debug.evaluate_two_ablation_readable50 import (
    estimate_text_bbox,
    iter_jsonl,
    resize_pad_image,
    row_image_path,
)

manifest = Path(sys.argv[1])
out_path = Path(sys.argv[2])
cells = []
for idx, row in enumerate(iter_jsonl(manifest)):
    if idx >= 4:
        break
    img = resize_pad_image(Image.open(row_image_path(row)), 256)
    bbox, failed, _ = estimate_text_bbox(img)
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(255, 0, 0), width=3)
    draw.text((4, 4), f"#{idx} failed={int(failed)}", fill=(255, 0, 0))
    cells.append(img)
canvas = Image.new("RGB", (512, 512), (255, 255, 255))
for idx, cell in enumerate(cells):
    canvas.paste(cell, ((idx % 2) * 256, (idx // 2) * 256))
out_path.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out_path)
print(out_path)
PY
}

write_smoke_summary() {
  local a_dir="$SMOKE_SAVE_ROOT/w001_hr_local_smoke"
  local b_dir="$SMOKE_SAVE_ROOT/ocrvis_hr_local_smoke"
  local out="$SAVE_ROOT/smoke_summary.md"
  "$PYTHON_BIN" - "$SAVE_ROOT/config_summary.txt" "$a_dir" "$b_dir" "$out" <<'PY'
import re
import sys
from pathlib import Path

import torch

config_summary = Path(sys.argv[1]).read_text(encoding="utf-8")
a_dir = Path(sys.argv[2])
b_dir = Path(sys.argv[3])
out = Path(sys.argv[4])

def read_text(path):
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""

def collect_logs(run_dir):
    parts = []
    for path in sorted(run_dir.rglob("*.txt")) + sorted(run_dir.rglob("*.log")):
        if path.is_file() and path.stat().st_size < 20_000_000:
            parts.append(f"\n--- {path} ---\n{read_text(path)}")
    return "\n".join(parts)

a_logs = collect_logs(a_dir)
b_logs = collect_logs(b_dir)

required_a = ["text_hr_loss", "local_similarity_loss", "weighted_local_similarity_loss"]
required_b = [
    "text_hr_loss",
    "local_similarity_loss",
    "ocr_visual_align_loss",
    "weighted_ocr_visual_align_loss",
    "OCR visual alignment grad check",
]

missing_a = [item for item in required_a if item not in a_logs]
missing_b = [item for item in required_b if item not in b_logs]

b_ckpt = b_dir / "checkpoints" / "last.pt"
ckpt_contains = False
if b_ckpt.exists():
    ckpt = torch.load(b_ckpt, map_location="cpu")
    model = ckpt.get("model") or ckpt.get("state_dict") or {}
    ckpt_contains = any(str(key).startswith("ocr_visual_alignment_mlp.") for key in model.keys())

grad_match = re.findall(r"ocr_visual_mlp_grad_norm=([0-9.eE+-]+)", b_logs)
encoder_grad_match = re.findall(r"ocr_visual_encoder_grad_param_count=([0-9]+)", b_logs)
grad_norm = float(grad_match[-1]) if grad_match else None
encoder_grad_count = int(encoder_grad_match[-1]) if encoder_grad_match else None

lines = [
    "# Two Ablation Smoke Summary",
    "",
    "Preliminary 2500-step quick screening.",
    "Do not compare directly with 5000-step final tables.",
    "",
    "## Config hard checks",
    "",
    "```text",
    config_summary.strip(),
    "```",
    "",
    "## Smoke results",
    "",
    f"w001_hr_local smoke checkpoint exists: {(a_dir / 'checkpoints' / 'last.pt').exists()}",
    f"ocrvis_hr_local smoke checkpoint exists: {b_ckpt.exists()}",
    f"w001_hr_local missing log keys: {missing_a}",
    f"ocrvis_hr_local missing log keys: {missing_b}",
    f"checkpoint_contains_ocr_visual_mlp: {ckpt_contains}",
    f"ocr_visual_mlp_grad_norm: {grad_norm}",
    f"ocr_visual_encoder_grad_param_count: {encoder_grad_count}",
    "",
    "## Required interpretation",
    "",
    "This is a two combined routes quick comparison, not a full causal ablation.",
    "A change in performance cannot be attributed to high-rank, local-similarity, or OCR visual alignment individually.",
]
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)

if missing_a or missing_b:
    raise SystemExit("smoke log key check failed")
if not ckpt_contains:
    raise SystemExit("checkpoint does not contain ocr_visual_alignment_mlp")
if grad_norm is None or grad_norm <= 0:
    raise SystemExit("ocr_visual_mlp_grad_norm missing or non-positive")
if encoder_grad_count != 0:
    raise SystemExit("ocr_visual_encoder_grad_param_count must be 0")
PY
}

run_smoke() {
  stage "[START] smoke"
  generate_configs
  validate_config_summary
  generate_bbox_smoke
  train_one "w001_hr_local_smoke" "$CONFIG_DIR/w001_hr_local.yaml" "$SMOKE_SAVE_ROOT" "$SMOKE_STEPS" "$SMOKE_GLOBAL_BATCH_SIZE" "$SMOKE_NPROC_PER_NODE" "$SMOKE_STEPS" 999999
  train_one "ocrvis_hr_local_smoke" "$CONFIG_DIR/ocrvis_hr_local.yaml" "$SMOKE_SAVE_ROOT" "$SMOKE_STEPS" "$SMOKE_GLOBAL_BATCH_SIZE" "$SMOKE_NPROC_PER_NODE" "$SMOKE_STEPS" 999999
  write_smoke_summary
  stage "[DONE] smoke"
}

run_formal() {
  stage "[START] train w001_hr_local"
  train_one "w001_hr_local" "$CONFIG_DIR/w001_hr_local.yaml" "$SAVE_ROOT" "$TARGET_STEPS" "$GLOBAL_BATCH_SIZE" "$NPROC_PER_NODE" "$CKPT_EVERY" "$VAL_EVERY"
  stage "[DONE] train w001_hr_local"

  stage "[START] train ocrvis_hr_local"
  train_one "ocrvis_hr_local" "$CONFIG_DIR/ocrvis_hr_local.yaml" "$SAVE_ROOT" "$TARGET_STEPS" "$GLOBAL_BATCH_SIZE" "$NPROC_PER_NODE" "$CKPT_EVERY" "$VAL_EVERY"
  stage "[DONE] train ocrvis_hr_local"

  stage "[START] evaluate w001 + w001_hr_local + ocrvis_hr_local"
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
runs = ["w001", "w001_hr_local", "ocrvis_hr_local"]
total = 0
for run in runs:
    count = len(list((eval_dir / "reconstruction" / run).glob("*.png")))
    print(f"{run}: {count}")
    if count != 50:
        raise SystemExit(f"{run} expected 50 reconstruction PNGs, got {count}")
    total += count
print(f"total: {total}")
if total != 150:
    raise SystemExit(f"expected total 150 reconstruction PNGs, got {total}")
rows = list(csv.DictReader((eval_dir / "setting_metrics.csv").open("r", encoding="utf-8")))
for row in rows:
    if int(float(row.get("step", "-1"))) != target_steps:
        raise SystemExit(f"row has wrong step: {row}")
    if row.get("image_metric_source") != "training validation CSV":
        raise SystemExit(f"image metrics are not from training validation CSV: {row}")
    if not row.get("psnr") or not row.get("ssim"):
        raise SystemExit(f"missing train-val PSNR/SSIM: {row}")
print(f"paper_table: {eval_dir / 'paper_table.md'}")
print(f"setting_metrics: {eval_dir / 'setting_metrics.csv'}")
print(f"qual_grid: {eval_dir / f'qual_grid_{target_steps}.png'}")
PY
  stage "[DONE] evaluate"
}

precheck_static
generate_configs
validate_config_summary

if safe_bool "$RUN_SMOKE"; then
  run_smoke
else
  run_formal
fi

stage "[DONE] run_two_ablation"
