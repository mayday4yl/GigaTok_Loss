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
TORCH_RUN="${TORCH_RUN:-torchrun}"

PERSIST="${PERSIST:-/home/ma-user/work/GigaTok_hr/gigatok_persist}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PERSIST/outputs}"
SAVE_ROOT="${SAVE_ROOT:-$OUTPUT_ROOT/ocr_box_gate/pipeline_runs}"
TARGET_STEPS="${TARGET_STEPS:-2500}"
TRAIN_TOTAL_STEPS="${TRAIN_TOTAL_STEPS:-5000}"
PIPELINE_NAME="${PIPELINE_NAME:-ocr_box_gate_w001_${TARGET_STEPS}}"
PIPELINE_DIR="${PIPELINE_DIR:-$SAVE_ROOT/$PIPELINE_NAME}"
STATE_DIR="$PIPELINE_DIR/state"
CONFIG_DIR="$PIPELINE_DIR/configs"
LOG="${LOG:-$PIPELINE_DIR/pipeline.log}"

BASELINE_CONFIG="${BASELINE_CONFIG:-$OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/configs/w001.yaml}"
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:-$OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/w001}"
BASELINE_CKPT="${BASELINE_CKPT:-$BASELINE_RUN_DIR/checkpoints/$(printf "%07d.pt" "$TARGET_STEPS")}"
BASE_CONFIG="${BASE_CONFIG:-$BASELINE_CONFIG}"
BBOX_JSONL="${BBOX_JSONL:-$OUTPUT_ROOT/ocr_box_gate/ocr_bbox_readable50_train_val.jsonl}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$OUTPUT_ROOT/text_hr_multi_debug/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-$OUTPUT_ROOT/text_feature_cache/readable50_glyph_mapped/text_features.pt}"
VQ_CKPT="${VQ_CKPT:-$PERSIST/checkpoints/VQ_BL256_dino_disc.pt}"

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

INSIDE_VALUE="${INSIDE_VALUE:-1.0}"
EDGE_VALUE="${EDGE_VALUE:-0.3}"
OUTSIDE_VALUE="${OUTSIDE_VALUE:-0.05}"
DILATE_TOKENS="${DILATE_TOKENS:-1}"
LOW_CONF_ALPHA="${LOW_CONF_ALPHA:-0.5}"
LOW_CONF_MAX="${LOW_CONF_MAX:-1.2}"

# auto: run v2 only when bbox JSONL has non-default confidence.
# always: run v2 even if confidence is all default 1.0.
# skip: never run v2.
RUN_V2="${RUN_V2:-auto}"

# Defaults preserve the w001 baseline training口径; set these only for debug smokes.
DISABLE_OCR_TEACHER_FORCING="${DISABLE_OCR_TEACHER_FORCING:-0}"
DISABLE_OCR_FEATURE_LOSS="${DISABLE_OCR_FEATURE_LOSS:-0}"
DISABLE_DISTILL_LOSS="${DISABLE_DISTILL_LOSS:-0}"

ALLOW_EXISTING_TRAINING_PROCESS="${ALLOW_EXISTING_TRAINING_PROCESS:-0}"

mkdir -p "$PIPELINE_DIR" "$STATE_DIR" "$CONFIG_DIR"
exec > >(tee -a "$LOG") 2>&1

stage() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

require_file() {
  local path="$1"
  local label="$2"
  [[ -f "$path" ]] || die "$label not found: $path"
}

check_no_training_process() {
  if [[ "$ALLOW_EXISTING_TRAINING_PROCESS" == "1" ]]; then
    return 0
  fi
  local matches
  matches=$(ps -ef | grep -E "vq_train.py|torchrun" | grep -v grep || true)
  if [[ -n "$matches" ]]; then
    echo "$matches" >&2
    die "existing torchrun/vq_train.py process found. Set ALLOW_EXISTING_TRAINING_PROCESS=1 only if intentional."
  fi
}

target_ckpt_for() {
  local run_name="$1"
  printf "%s/%s/checkpoints/%07d.pt" "$PIPELINE_DIR" "$run_name" "$TARGET_STEPS"
}

has_real_confidence() {
  local status="${1:-UNKNOWN}"
  "$PYTHON_BIN" - "$BBOX_JSONL" "$status" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = sys.argv[2]
total = 0
real = 0
values = []
for line in path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    for box in row.get("boxes", []):
        total += 1
        conf = box.get("confidence")
        source = str(box.get("confidence_source", ""))
        if conf is not None:
            values.append(float(conf))
        if conf is not None and abs(float(conf) - 1.0) > 1e-6:
            real += 1
        elif source and not source.startswith("default_"):
            real += 1
unique = sorted(set(values))
print(json.dumps({
    "boxes": total,
    "confidence_values": len(values),
    "confidence_min": min(values) if values else None,
    "confidence_max": max(values) if values else None,
    "confidence_unique_count": len(unique),
    "non_default_confidence_count": real,
    "v2_status": status,
}))
raise SystemExit(0 if real > 0 else 1)
PY
}

precheck() {
  stage "[START] precheck"
  echo "repo=$REPO_ROOT"
  echo "pipeline_dir=$PIPELINE_DIR"
  echo "target_steps=$TARGET_STEPS"
  echo "train_total_steps=$TRAIN_TOTAL_STEPS"
  echo "run_v2=$RUN_V2"
  echo "bbox_jsonl=$BBOX_JSONL"
  echo "train_manifest=$TRAIN_MANIFEST"
  echo "val_manifest=$VAL_MANIFEST"
  df -h /home/ma-user/work || true

  require_file "$BASE_CONFIG" "base config"
  require_file "$BASELINE_CONFIG" "baseline w001 config"
  require_file "$BASELINE_CKPT" "baseline w001 target checkpoint"
  require_file "$BASELINE_RUN_DIR/val.csv" "baseline w001 val.csv"
  require_file "$BBOX_JSONL" "OCR bbox JSONL"
  require_file "$TRAIN_MANIFEST" "train manifest"
  require_file "$VAL_MANIFEST" "val manifest"
  require_file "$TEXT_FEATURE_CACHE" "text feature cache"
  require_file "$VQ_CKPT" "GigaTok base checkpoint"
  require_file "tokenizer/tokenizer_image/vq/vq_train.py" "vq_train.py"

  "$PYTHON_BIN" -m py_compile \
    dataset/ocr_box_gate.py \
    dataset/textatlas.py \
    tokenizer/tokenizer_image/vq/blocks.py \
    tokenizer/tokenizer_image/vq/vq_vit_model.py \
    tokenizer/tokenizer_image/vq/vq_train.py

  "$PYTHON_BIN" - "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$BBOX_JSONL" <<'PY'
import json
import sys
from pathlib import Path

train = Path(sys.argv[1])
val = Path(sys.argv[2])
bbox = Path(sys.argv[3])

def read_paths(path):
    return [json.loads(line)["image_path"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

train_paths = read_paths(train)
val_paths = read_paths(val)
bbox_rows = [json.loads(line) for line in bbox.read_text(encoding="utf-8").splitlines() if line.strip()]
bbox_paths = [row["image_path"] for row in bbox_rows]
bbox_set = set(bbox_paths)
missing_train = [p for p in train_paths if p not in bbox_set]
missing_val = [p for p in val_paths if p not in bbox_set]
empty = [row["image_path"] for row in bbox_rows if not row.get("boxes")]
summary = {
    "train_rows": len(train_paths),
    "val_rows": len(val_paths),
    "train_unique": len(set(train_paths)),
    "val_unique": len(set(val_paths)),
    "unique_train_val": len(set(train_paths) | set(val_paths)),
    "train_val_identical_order": train_paths == val_paths,
    "bbox_rows": len(bbox_rows),
    "bbox_unique": len(set(bbox_paths)),
    "bbox_empty_rows": len(empty),
    "missing_train": len(missing_train),
    "missing_val": len(missing_val),
}
print(json.dumps(summary, indent=2))
if missing_train or missing_val or empty:
    raise SystemExit("bbox preflight failed")
PY
  "$PYTHON_BIN" - "$BASE_CONFIG" <<'PY'
import json
import sys
import yaml
from pathlib import Path

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "ocr_teacher_forcing_loss.enabled": cfg.get("ocr_teacher_forcing_loss", {}).get("enabled"),
    "ocr_teacher_forcing_loss.weight": cfg.get("ocr_teacher_forcing_loss", {}).get("weight"),
    "text_hr.enabled": cfg.get("text_hr", {}).get("enabled"),
    "local_similarity_loss.enabled": (cfg.get("local_similarity_loss") or {}).get("enabled") if cfg.get("local_similarity_loss") is not None else None,
    "ocr_visual_alignment_loss.enabled": (cfg.get("ocr_visual_alignment_loss") or {}).get("enabled") if cfg.get("ocr_visual_alignment_loss") is not None else None,
}
print(json.dumps(checks, indent=2))
if checks["ocr_teacher_forcing_loss.enabled"] is not True:
    raise SystemExit("base config must have ocr_teacher_forcing_loss.enabled=true")
if abs(float(checks["ocr_teacher_forcing_loss.weight"]) - 0.01) > 1e-12:
    raise SystemExit("base config must have ocr_teacher_forcing_loss.weight=0.01")
if checks["text_hr.enabled"] is not False:
    raise SystemExit("base config must have text_hr.enabled=false")
if checks["local_similarity_loss.enabled"] not in (None, False):
    raise SystemExit("base config must have local_similarity_loss disabled/absent")
if checks["ocr_visual_alignment_loss.enabled"] not in (None, False):
    raise SystemExit("base config must have ocr_visual_alignment_loss disabled/absent")
PY
  stage "[DONE] precheck"
}

write_config() {
  local name="$1"
  local low_conf_enabled="$2"
  local out="$CONFIG_DIR/${name}.yaml"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$out" "$BBOX_JSONL" "$low_conf_enabled" <<PY
import sys
import yaml
from pathlib import Path

base = Path(sys.argv[1])
out = Path(sys.argv[2])
bbox = sys.argv[3]
low_conf_enabled = sys.argv[4].lower() == "true"

cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
cfg.setdefault("ocr_box_gate", {})
cfg["ocr_box_gate"].update({
    "enabled": True,
    "bbox_jsonl": bbox,
    "grid_size": 16,
    "inside_value": float("$INSIDE_VALUE"),
    "edge_value": float("$EDGE_VALUE"),
    "outside_value": float("$OUTSIDE_VALUE"),
    "dilate_tokens": int("$DILATE_TOKENS"),
    "missing_policy": "error",
    "empty_boxes_policy": "error",
    "low_conf_enabled": low_conf_enabled,
    "low_conf_alpha": float("$LOW_CONF_ALPHA"),
    "low_conf_max": float("$LOW_CONF_MAX"),
    "apply_layers": [6, 12, 18],
    "log_debug": True,
})
if "$DISABLE_OCR_TEACHER_FORCING" == "1":
    cfg.setdefault("ocr_teacher_forcing_loss", {})["enabled"] = False
if "$DISABLE_OCR_FEATURE_LOSS" == "1":
    cfg.setdefault("ocr_feature_loss", {})["enabled"] = False
    cfg["ocr_feature_loss"]["weight"] = 0.0
if "$DISABLE_DISTILL_LOSS" == "1":
    cfg.setdefault("trainer", {})["distill_loss"] = False
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out)
PY
}

train_one() {
  local name="$1"
  local config="$2"
  local done="$STATE_DIR/${name}.done"
  local run_dir="$PIPELINE_DIR/$name"
  local target_ckpt
  local resume_args=()
  target_ckpt="$(target_ckpt_for "$name")"

  if [[ -f "$target_ckpt" ]]; then
    echo "$target_ckpt" > "$done"
    stage "[SKIP] $name already done: $target_ckpt"
    return 0
  fi

  stage "[START] train $name"
  mkdir -p "$run_dir"
  if compgen -G "$run_dir/checkpoints/[0-9]*.pt" >/dev/null; then
    stage "[RESUME] $name will auto-resume from latest checkpoint under $run_dir/checkpoints"
  else
    resume_args=(--vq-ckpt "$VQ_CKPT" --finetune)
    stage "[INIT] $name will initialize from $VQ_CKPT"
  fi
  "$TORCH_RUN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --model-config "$config" \
    --dataset textatlas_image_text \
    --data-path "$TRAIN_MANIFEST" \
    --json-path "$TRAIN_MANIFEST" \
    --val-json-path "$VAL_MANIFEST" \
    --save-path "$PIPELINE_DIR" \
    --sub-exp-dir "$name" \
    "${resume_args[@]}" \
    --iterations "$TRAIN_TOTAL_STEPS" \
    --early-stop-iter "$TARGET_STEPS" \
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

  require_file "$target_ckpt" "$name target checkpoint"
  echo "$target_ckpt" > "$done"
  stage "[DONE] train $name"
}

write_runs_json() {
  local runs_json="$PIPELINE_DIR/runs_${TARGET_STEPS}.json"
  "$PYTHON_BIN" - "$PIPELINE_DIR" "$CONFIG_DIR" "$TARGET_STEPS" "$runs_json" "$BASELINE_CONFIG" "$BASELINE_CKPT" "$BASELINE_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

pipeline = Path(sys.argv[1])
configs = Path(sys.argv[2])
steps = int(sys.argv[3])
out = Path(sys.argv[4])
baseline_config = Path(sys.argv[5])
baseline_ckpt = Path(sys.argv[6])
baseline_run_dir = Path(sys.argv[7])
runs = {
    "w001": {
        "config": str(baseline_config),
        "ckpt": str(baseline_ckpt),
        "run_dir": str(baseline_run_dir),
        "note": "baseline w001 reused at target step",
    }
}
for name in ("ocr_box_gate_v1", "ocr_box_gate_v2_conf"):
    run_dir = pipeline / name
    ckpt = run_dir / "checkpoints" / f"{steps:07d}.pt"
    config = configs / f"{name}.yaml"
    if ckpt.exists() and config.exists():
        runs[name] = {
            "config": str(config),
            "ckpt": str(ckpt),
            "run_dir": str(run_dir),
            "note": "OCR bbox gate pipeline",
        }
out.write_text(json.dumps(runs, indent=2), encoding="utf-8")
print(out)
PY
}

write_next_eval_commands() {
  local runs_json="$PIPELINE_DIR/runs_${TARGET_STEPS}.json"
  local eval_dir="$PIPELINE_DIR/eval_${TARGET_STEPS}"
  local eval_sh="$PIPELINE_DIR/next_eval_commands.sh"
  cat > "$eval_sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:\\${PYTHONPATH:-}"
source /home/ma-user/anaconda3/bin/activate PyTorch-2.1.0

python scripts/stage1/ocr_debug/evaluate_setting_metrics.py \\
  --runs-json "$runs_json" \\
  --manifest-jsonl "$VAL_MANIFEST" \\
  --output-dir "$eval_dir" \\
  --step "$TARGET_STEPS" \\
  --image-metric-source train_val_csv \\
  --device-backend "$DEVICE_BACKEND" \\
  --device-id 0 \\
  --mixed-precision "$MIXED_PRECISION" \\
  --image-size 256 \\
  --batch-size "$EVAL_BATCH_SIZE" \\
  --max-images "$VAL_MAX_IMAGES" \\
  --grid-samples 12 \\
  --checkpoint-weight-key model \\
  --deepseek-ocr-model "$PERSIST/models/DeepSeek-OCR" \\
  --thirdparty-ocr-backend none

echo "setting_metrics.csv: $eval_dir/setting_metrics.csv"
echo "paper_table.md: $eval_dir/paper_table.md"
echo "qual_grid: $eval_dir/qual_grid_5000.png"
EOF
  chmod +x "$eval_sh"
  echo "$eval_sh"
}

main() {
  check_no_training_process
  precheck
  local v1_config
  local v2_config
  v1_config="$(write_config ocr_box_gate_v1 false | tail -n 1)"
  v2_config="$(write_config ocr_box_gate_v2_conf true | tail -n 1)"
  train_one ocr_box_gate_v1 "$v1_config"

  case "$RUN_V2" in
    skip)
      stage "[SKIP] v2 disabled by RUN_V2=skip"
      echo "RUN_V2=skip" > "$STATE_DIR/ocr_box_gate_v2_conf.skipped"
      ;;
    always)
      stage "[START] v2 forced by RUN_V2=always"
      has_real_confidence "FORCED_RUN_V2_ALWAYS" > "$STATE_DIR/confidence_summary.json" || true
      cat "$STATE_DIR/confidence_summary.json"
      train_one ocr_box_gate_v2_conf "$v2_config"
      ;;
    auto)
      if has_real_confidence "RUN_V2" > "$STATE_DIR/confidence_summary.json"; then
        cat "$STATE_DIR/confidence_summary.json"
        train_one ocr_box_gate_v2_conf "$v2_config"
      else
        has_real_confidence "SKIPPED_NO_REAL_CONFIDENCE" > "$STATE_DIR/confidence_summary.json" || true
        cat "$STATE_DIR/confidence_summary.json"
        stage "[SKIP] v2 skipped: bbox JSONL has no real confidence; confidence is default 1.0."
        echo "SKIPPED_NO_REAL_CONFIDENCE" > "$STATE_DIR/ocr_box_gate_v2_conf.skipped"
      fi
      ;;
    *)
      die "RUN_V2 must be auto, always, or skip. Got: $RUN_V2"
      ;;
  esac

  write_runs_json
  local eval_commands
  eval_commands="$(write_next_eval_commands)"
  stage "[DONE] pipeline"
  echo "PIPELINE_DIR=$PIPELINE_DIR"
  echo "LOG=$LOG"
  echo "NEXT_EVAL_COMMANDS=$eval_commands"
}

main "$@"
