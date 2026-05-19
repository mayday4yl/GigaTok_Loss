#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ma-user/work/GigaTok_hr/GigaTok_Loss_two_ablation}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch}"
export DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_RUN="${TORCH_RUN:-torchrun}"

PERSIST="${PERSIST:-/home/ma-user/work/GigaTok_hr/gigatok_persist}"
PERSIST_OUTPUT_ROOT="$PERSIST/outputs"
PIPELINE_ROOT="${PIPELINE_ROOT:-$PERSIST_OUTPUT_ROOT/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500}"
STATE_DIR="$PIPELINE_ROOT/state"
CONFIG_DIR="$PIPELINE_ROOT/configs"
EVAL_DIR="$PIPELINE_ROOT/eval_2500"
PIPELINE_LOG="${PIPELINE_LOG:-$PIPELINE_ROOT/pipeline.log}"

TARGET_STEPS="${TARGET_STEPS:-2500}"
TRAIN_TOTAL_STEPS="${TRAIN_TOTAL_STEPS:-5000}"
RUN_V1="${RUN_V1:-1}"
RUN_PROXY_CONF="${RUN_PROXY_CONF:-0}"
RUN_V2="${RUN_V2:-auto}"
RUN_EVAL="${RUN_EVAL:-1}"
AUTO_EVAL="${AUTO_EVAL:-1}"
FORCE="${FORCE:-0}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"

BASELINE_CONFIG="$PERSIST_OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/configs/w001.yaml"
BASELINE_RUN_DIR="$PERSIST_OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/w001"
BASELINE_CKPT="$BASELINE_RUN_DIR/checkpoints/0002500.pt"
BASELINE_VAL_CSV="$BASELINE_RUN_DIR/val.csv"
BASE_CONFIG="$BASELINE_CONFIG"
VQ_CKPT="$PERSIST/checkpoints/VQ_BL256_dino_disc.pt"
TRAIN_MANIFEST="$PERSIST_OUTPUT_ROOT/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_train.jsonl"
VAL_MANIFEST="$PERSIST_OUTPUT_ROOT/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_val.jsonl"
TEXT_FEATURE_CACHE="$PERSIST_OUTPUT_ROOT/text_feature_cache/readable50_glyph_mapped/text_features.pt"
BASE_BBOX_JSONL="$PERSIST_OUTPUT_ROOT/ocr_box_gate/ocr_bbox_readable50_train_val.jsonl"
PROXY_BBOX_JSONL="$PERSIST_OUTPUT_ROOT/ocr_box_gate/ocr_bbox_readable50_train_val_with_proxy_conf.jsonl"
PROXY_SUMMARY_JSON="$PERSIST_OUTPUT_ROOT/ocr_box_gate/proxy_confidence_summary.json"
PROXY_DEBUG_DIR="$PERSIST_OUTPUT_ROOT/ocr_box_gate/token_logprob_proxy_probe_full"
PROXY_ACTUAL_SUMMARY="$PROXY_DEBUG_DIR/token_confidence_summary.json"
DEEPSEEK_MODEL="$PERSIST/models/DeepSeek-OCR"

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

V1_NAME="ocr_box_gate_v1"
V2_NAME="ocr_box_gate_v2_conf_proxy"
V1_RUN_DIR="$PIPELINE_ROOT/$V1_NAME"
V2_RUN_DIR="$PIPELINE_ROOT/$V2_NAME"
V1_CONFIG="$CONFIG_DIR/$V1_NAME.yaml"
V2_CONFIG="$CONFIG_DIR/$V2_NAME.yaml"

mkdir -p "$PIPELINE_ROOT" "$STATE_DIR" "$CONFIG_DIR" "$EVAL_DIR"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

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

target_ckpt_for() {
  local run_name="$1"
  printf "%s/%s/checkpoints/%07d.pt" "$PIPELINE_ROOT" "$run_name" "$TARGET_STEPS"
}

write_key_values() {
  cat <<EOF
PIPELINE_ROOT=$PIPELINE_ROOT
PIPELINE_LOG=$PIPELINE_LOG
RUN_V1=$RUN_V1
RUN_PROXY_CONF=$RUN_PROXY_CONF
RUN_V2=$RUN_V2
RUN_EVAL=$RUN_EVAL
AUTO_EVAL=$AUTO_EVAL
FORCE=$FORCE
BASE_BBOX_JSONL=$BASE_BBOX_JSONL
PROXY_BBOX_JSONL=$PROXY_BBOX_JSONL
PROXY_SUMMARY_JSON=$PROXY_SUMMARY_JSON
V1_CONFIG=$V1_CONFIG
V2_CONFIG=$V2_CONFIG
V1_RUN_DIR=$V1_RUN_DIR
V2_RUN_DIR=$V2_RUN_DIR
EVAL_DIR=$EVAL_DIR
EOF
}

check_no_training_process() {
  if [[ "$FORCE" == "1" ]]; then
    stage "[WARN] FORCE=1, allowing existing torchrun/vq_train.py process."
    return 0
  fi
  local matches
  matches="$(ps -ef | grep -E '[v]q_train.py|[t]orchrun' || true)"
  if [[ -n "$matches" ]]; then
    echo "$matches" >&2
    die "existing torchrun/vq_train.py process found. Use FORCE=1 only if this is intentional."
  fi
}

wait_for_existing_proxy() {
  local matches
  while true; do
    matches="$(ps -ef | grep '[p]robe_deepseek_ocr_token_confidence.py' || true)"
    if [[ -z "$matches" ]]; then
      return 0
    fi
    stage "[WAIT] existing proxy confidence process is still running; not starting a duplicate."
    tail -n 80 "$PERSIST/logs/probe_deepseek_ocr_token_confidence_full.log" 2>/dev/null \
      | grep -E '\[INFO\]|token_score_proxy_feasible|recommended_v2_action|confidence_|\[OK\]|\[ERROR\]' \
      | tail -n 25 || true
    sleep 300
  done
}

run_proxy_if_needed() {
  wait_for_existing_proxy
  if [[ -f "$PROXY_BBOX_JSONL" && -f "$PROXY_ACTUAL_SUMMARY" ]]; then
    stage "[SKIP] proxy confidence already exists."
    return 0
  fi
  if [[ "$RUN_PROXY_CONF" != "1" ]]; then
    stage "[SKIP] RUN_PROXY_CONF=$RUN_PROXY_CONF and proxy output is incomplete; v2 may be skipped."
    return 0
  fi
  stage "[START] generating proxy confidence for readable50."
  mkdir -p "$PROXY_DEBUG_DIR" "$PERSIST/logs"
  ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES%%,*}" "$PYTHON_BIN" \
    scripts/stage1/ocr_debug/probe_deepseek_ocr_token_confidence.py \
    --manifest "$VAL_MANIFEST" \
    --model-path "$DEEPSEEK_MODEL" \
    --input-bbox-jsonl "$BASE_BBOX_JSONL" \
    --output-jsonl "$PROXY_BBOX_JSONL" \
    --output-dir "$PROXY_DEBUG_DIR" \
    --max-images 0 \
    --device-backend "$DEVICE_BACKEND" \
    --device-id 0 \
    --dtype fp32 \
    --attn-implementation eager \
    --image-size 256 \
    --ocr-image-size 640 \
    --base-size 1024 \
    --max-new-tokens 512 \
    --allow-overwrite \
    | tee "$PERSIST/logs/probe_deepseek_ocr_token_confidence_full.log"
}

standardize_proxy_confidence() {
  stage "[START] standardize proxy confidence summary and bbox JSONL."
  "$PYTHON_BIN" - "$PROXY_BBOX_JSONL" "$PROXY_ACTUAL_SUMMARY" "$PROXY_SUMMARY_JSON" "$STATE_DIR" <<'PY'
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path

proxy_bbox = Path(sys.argv[1])
actual_summary = Path(sys.argv[2])
fixed_summary = Path(sys.argv[3])
state_dir = Path(sys.argv[4])
state_dir.mkdir(parents=True, exist_ok=True)

def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    frac = pos - lo
    return values[lo] * (1 - frac) + values[hi] * frac

def write_summary(data):
    fixed_summary.parent.mkdir(parents=True, exist_ok=True)
    fixed_summary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

if not proxy_bbox.exists():
    source_summary = {}
    if actual_summary.exists():
        source_summary = json.loads(actual_summary.read_text(encoding="utf-8"))
    write_summary({
        "proxy_feasible": False,
        "v2_allowed": False,
        "recommended_v2_action": "skip",
        "reason": f"proxy bbox missing: {proxy_bbox}",
        "source_summary": source_summary,
        "boxes_total": 0,
        "boxes_with_proxy_conf": 0,
        "failed_alignment_count": source_summary.get("failed_alignment_count"),
        "confidence_source_counts": {},
    })
    raise SystemExit(0)

source_summary = {}
if actual_summary.exists():
    source_summary = json.loads(actual_summary.read_text(encoding="utf-8"))

rows = []
logprobs = []
boxes_total = 0
boxes_with_proxy_conf = 0
failed_alignment_count = int(source_summary.get("failed_alignment_count") or 0)
for line in proxy_bbox.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    for box in row.get("boxes", []):
        boxes_total += 1
        value = box.get("mean_logprob")
        valid = value is not None and box.get("token_span") is not None
        box["proxy_conf_valid"] = bool(valid)
        if valid:
            boxes_with_proxy_conf += 1
            logprobs.append(float(value))
    rows.append(row)

if logprobs:
    p10_logprob = percentile(logprobs, 0.10)
    p90_logprob = percentile(logprobs, 0.90)
    denom = (p90_logprob - p10_logprob) + 1e-8
else:
    p10_logprob = None
    p90_logprob = None
    denom = None

for row in rows:
    for box in row.get("boxes", []):
        value = box.get("mean_logprob")
        if value is None or denom is None or box.get("token_span") is None:
            box["confidence"] = 0.0
            box["proxy_conf_valid"] = False
        else:
            conf = (float(value) - p10_logprob) / denom
            box["confidence"] = max(0.0, min(1.0, float(conf)))
            box["proxy_conf_valid"] = True
        box["confidence_source"] = "deepseek_token_logprob_proxy"

backup_done = state_dir / "proxy_bbox_backup.done"
if not backup_done.exists() and proxy_bbox.exists():
    backup = proxy_bbox.with_suffix(proxy_bbox.suffix + ".before_robust_confidence")
    if not backup.exists():
        shutil.copy2(proxy_bbox, backup)
    backup_done.write_text(str(backup) + "\n", encoding="utf-8")

tmp = proxy_bbox.with_suffix(proxy_bbox.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
os.replace(tmp, proxy_bbox)

confidences = [float(box["confidence"]) for row in rows for box in row.get("boxes", [])]
source_counts = {}
for row in rows:
    for box in row.get("boxes", []):
        source = box.get("confidence_source", "<missing>")
        source_counts[source] = source_counts.get(source, 0) + 1

if confidences:
    conf_min = min(confidences)
    conf_max = max(confidences)
    conf_stats = {
        "confidence_min": conf_min,
        "confidence_p10": percentile(confidences, 0.10),
        "confidence_p25": percentile(confidences, 0.25),
        "confidence_mean": statistics.mean(confidences),
        "confidence_median": statistics.median(confidences),
        "confidence_p75": percentile(confidences, 0.75),
        "confidence_p90": percentile(confidences, 0.90),
        "confidence_max": conf_max,
        "confidence_unique_count": len(set(round(v, 8) for v in confidences)),
        "confidence_max_minus_min": conf_max - conf_min,
    }
else:
    conf_stats = {
        "confidence_min": None,
        "confidence_p10": None,
        "confidence_p25": None,
        "confidence_mean": None,
        "confidence_median": None,
        "confidence_p75": None,
        "confidence_p90": None,
        "confidence_max": None,
        "confidence_unique_count": 0,
        "confidence_max_minus_min": 0.0,
    }

coverage = (boxes_with_proxy_conf / boxes_total) if boxes_total else 0.0
source_feasible = bool(source_summary.get("token_score_proxy_feasible", True))
v2_allowed = (
    source_feasible
    and boxes_total > 0
    and coverage >= 0.9
    and int(conf_stats["confidence_unique_count"]) > 1
    and float(conf_stats["confidence_max_minus_min"]) >= 0.05
)
summary = {
    "proxy_feasible": bool(v2_allowed),
    "v2_allowed": bool(v2_allowed),
    "recommended_v2_action": "run" if v2_allowed else "skip",
    "boxes_total": boxes_total,
    "boxes_with_proxy_conf": boxes_with_proxy_conf,
    "proxy_conf_coverage": coverage,
    "failed_alignment_count": failed_alignment_count,
    "logprob_p10": p10_logprob,
    "logprob_p90": p90_logprob,
    **conf_stats,
    "confidence_source_counts": source_counts,
    "source_summary_path": str(actual_summary) if actual_summary.exists() else None,
    "proxy_bbox_jsonl": str(proxy_bbox),
    "normalization": "robust_p10_p90_from_mean_logprob",
}
write_summary(summary)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
  stage "[DONE] proxy standardization."
}

proxy_v2_allowed() {
  "$PYTHON_BIN" - "$PROXY_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(1)
data = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("v2_allowed") is True else 1)
PY
}

precheck() {
  stage "[START] precheck."
  write_key_values
  require_file "$BASELINE_CONFIG" "w001 config"
  require_file "$BASELINE_CKPT" "w001@2500 checkpoint"
  require_file "$BASELINE_VAL_CSV" "w001 val.csv"
  require_file "$VQ_CKPT" "base VQ checkpoint"
  require_file "$TRAIN_MANIFEST" "train manifest"
  require_file "$VAL_MANIFEST" "val manifest"
  require_file "$TEXT_FEATURE_CACHE" "text feature cache"
  require_file "$BASE_BBOX_JSONL" "base bbox JSONL"
  require_file "$DEEPSEEK_MODEL/config.json" "DeepSeek-OCR model"
  require_file "tokenizer/tokenizer_image/vq/vq_train.py" "vq_train.py"

  "$PYTHON_BIN" - "$BASELINE_VAL_CSV" <<'PY'
import csv
import sys
from pathlib import Path
path = Path(sys.argv[1])
rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
if not any(str(row.get("step", "")).strip() == "2500" for row in rows):
    raise SystemExit(f"{path} does not contain step=2500")
print(f"w001_val_step_2500=true rows={len(rows)}")
PY

  mapfile -t OCR_DEBUG_PY < <(find scripts/stage1/ocr_debug -maxdepth 1 -type f -name "*.py" ! -name "._*" | sort)
  "$PYTHON_BIN" -m py_compile \
    dataset/ocr_box_gate.py \
    dataset/textatlas.py \
    tokenizer/tokenizer_image/vq/vq_train.py \
    tokenizer/tokenizer_image/vq/vq_vit_model.py \
    tokenizer/tokenizer_image/vq/blocks.py \
    tokenizer/tokenizer_image/vq/vq_loss.py \
    "${OCR_DEBUG_PY[@]}"

  "$PYTHON_BIN" - "$TRAIN_MANIFEST" "$VAL_MANIFEST" "$BASE_BBOX_JSONL" "$PROXY_BBOX_JSONL" <<'PY'
import json
import os
import sys
from pathlib import Path

train, val, base_bbox, proxy_bbox = [Path(p) for p in sys.argv[1:]]

def read_manifest(path):
    paths = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        paths.append(os.path.abspath(json.loads(line)["image_path"]))
    return paths

def read_bbox(path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row["image_path"] = os.path.abspath(row["image_path"])
        rows.append(row)
    return rows

train_paths = read_manifest(train)
val_paths = read_manifest(val)
manifest_union = set(train_paths) | set(val_paths)

def summarize_bbox(path, label, required):
    if not path.exists():
        if required:
            raise SystemExit(f"{label} missing: {path}")
        return {"label": label, "exists": False}
    rows = read_bbox(path)
    paths = [row["image_path"] for row in rows]
    path_set = set(paths)
    empty = [row["image_path"] for row in rows if not row.get("boxes")]
    summary = {
        "label": label,
        "exists": True,
        "bbox_rows": len(rows),
        "bbox_unique": len(path_set),
        "bbox_empty_rows": len(empty),
        "missing_train": len([p for p in train_paths if p not in path_set]),
        "missing_val": len([p for p in val_paths if p not in path_set]),
        "extra_bbox_not_in_train_val": len([p for p in path_set if p not in manifest_union]),
    }
    print(json.dumps(summary, indent=2))
    if required and (summary["missing_train"] or summary["missing_val"] or summary["bbox_empty_rows"]):
        raise SystemExit(f"{label} bbox coverage preflight failed")
    return summary

print(json.dumps({
    "train_rows": len(train_paths),
    "val_rows": len(val_paths),
    "train_unique": len(set(train_paths)),
    "val_unique": len(set(val_paths)),
    "unique_train_val": len(manifest_union),
    "train_val_identical_order": train_paths == val_paths,
}, indent=2))
summarize_bbox(base_bbox, "base_bbox", required=True)
summarize_bbox(proxy_bbox, "proxy_bbox", required=False)
PY

  "$PYTHON_BIN" - "$BASE_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
checks = {
    "ocr_teacher_forcing_loss.enabled": cfg.get("ocr_teacher_forcing_loss", {}).get("enabled"),
    "ocr_teacher_forcing_loss.weight": cfg.get("ocr_teacher_forcing_loss", {}).get("weight"),
    "text_hr.enabled": cfg.get("text_hr", {}).get("enabled"),
    "local_similarity_loss.enabled": (cfg.get("local_similarity_loss") or {}).get("enabled") if cfg.get("local_similarity_loss") is not None else None,
    "ocr_visual_alignment_loss.enabled": (cfg.get("ocr_visual_alignment_loss") or {}).get("enabled") if cfg.get("ocr_visual_alignment_loss") is not None else None,
    "ocr_feature_loss.enabled": (cfg.get("ocr_feature_loss") or {}).get("enabled") if cfg.get("ocr_feature_loss") is not None else None,
}
print(json.dumps(checks, indent=2))
if checks["ocr_teacher_forcing_loss.enabled"] is not True:
    raise SystemExit("w001 base must have ocr_teacher_forcing_loss.enabled=true")
if abs(float(checks["ocr_teacher_forcing_loss.weight"]) - 0.01) > 1e-12:
    raise SystemExit("w001 base must have ocr_teacher_forcing_loss.weight=0.01")
if checks["text_hr.enabled"] is not False:
    raise SystemExit("w001 base must have text_hr.enabled=false")
if checks["local_similarity_loss.enabled"] not in (None, False):
    raise SystemExit("local_similarity_loss must be disabled/absent")
if checks["ocr_visual_alignment_loss.enabled"] not in (None, False):
    raise SystemExit("ocr_visual_alignment_loss must be disabled/absent")
if checks["ocr_feature_loss.enabled"] not in (None, False):
    raise SystemExit("ocr_feature_loss must be disabled/absent")
PY
  stage "[DONE] precheck."
}

write_one_config() {
  local out="$1"
  local bbox="$2"
  local low_conf_enabled="$3"
  local low_conf_alpha="$4"
  local low_conf_max="$5"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$out" "$bbox" "$low_conf_enabled" "$low_conf_alpha" "$low_conf_max" <<'PY'
import sys
from pathlib import Path
import yaml

base, out, bbox = [Path(sys.argv[i]) for i in range(1, 4)]
low_conf_enabled = sys.argv[4].lower() == "true"
low_conf_alpha = float(sys.argv[5])
low_conf_max = float(sys.argv[6])
cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
cfg.setdefault("ocr_feature_loss", {})
cfg["ocr_feature_loss"]["enabled"] = False
cfg["ocr_feature_loss"]["weight"] = 0.0
cfg.setdefault("ocr_box_gate", {})
cfg["ocr_box_gate"].update({
    "enabled": True,
    "bbox_jsonl": str(bbox),
    "grid_size": 16,
    "inside_value": 1.0,
    "edge_value": 0.3,
    "outside_value": 0.05,
    "dilate_tokens": 1,
    "missing_policy": "error",
    "empty_boxes_policy": "error",
    "low_conf_enabled": low_conf_enabled,
    "low_conf_alpha": low_conf_alpha,
    "low_conf_max": low_conf_max,
    "apply_layers": [6, 12, 18],
    "log_debug": True,
})
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out)
PY
}

write_configs() {
  stage "[START] write v1/v2 configs."
  write_one_config "$V1_CONFIG" "$BASE_BBOX_JSONL" false 1.0 1.2
  write_one_config "$V2_CONFIG" "$PROXY_BBOX_JSONL" true 0.5 1.1
  "$PYTHON_BIN" - "$V1_CONFIG" "$V2_CONFIG" "$BASE_BBOX_JSONL" "$PROXY_BBOX_JSONL" <<'PY'
import sys
from pathlib import Path
import yaml
v1, v2, base_bbox, proxy_bbox = [Path(p) for p in sys.argv[1:]]
cfg1 = yaml.safe_load(v1.read_text(encoding="utf-8"))
cfg2 = yaml.safe_load(v2.read_text(encoding="utf-8"))
assert cfg1["ocr_box_gate"]["bbox_jsonl"] == str(base_bbox), cfg1["ocr_box_gate"]["bbox_jsonl"]
assert cfg2["ocr_box_gate"]["bbox_jsonl"] == str(proxy_bbox), cfg2["ocr_box_gate"]["bbox_jsonl"]
assert cfg1["ocr_box_gate"]["low_conf_enabled"] is False
assert cfg2["ocr_box_gate"]["low_conf_enabled"] is True
assert abs(float(cfg2["ocr_box_gate"]["low_conf_alpha"]) - 0.5) < 1e-12
assert abs(float(cfg2["ocr_box_gate"]["low_conf_max"]) - 1.1) < 1e-12
print(f"v1_config={v1}")
print(f"v2_config={v2}")
PY
  stage "[DONE] write configs."
}

determine_v2() {
  local will_run="false"
  local reason=""
  if [[ "$RUN_V2" == "0" || "$RUN_V2" == "skip" ]]; then
    will_run="false"
    reason="disabled_by_RUN_V2"
  elif proxy_v2_allowed; then
    will_run="true"
    reason="valid_proxy_confidence"
  else
    will_run="false"
    reason="invalid_or_missing_proxy_confidence"
  fi
  "$PYTHON_BIN" - "$STATE_DIR/v2_decision.json" "$will_run" "$reason" "$RUN_V2" "$PROXY_SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
will_run = sys.argv[2] == "true"
reason = sys.argv[3]
run_v2 = sys.argv[4]
summary_path = Path(sys.argv[5])
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
data = {
    "v2_will_run": will_run,
    "reason": reason,
    "RUN_V2": run_v2,
    "proxy_summary": summary,
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(data, indent=2, ensure_ascii=False))
PY
}

v2_will_run() {
  "$PYTHON_BIN" - "$STATE_DIR/v2_decision.json" <<'PY'
import json
import sys
from pathlib import Path
data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("v2_will_run") else 1)
PY
}

train_one() {
  local name="$1"
  local config="$2"
  local run_dir="$PIPELINE_ROOT/$name"
  local target_ckpt
  local running="$STATE_DIR/${name}.running"
  local done="$STATE_DIR/${name}.done"
  local failed="$STATE_DIR/${name}.failed"
  local resume_args=()
  target_ckpt="$(target_ckpt_for "$name")"

  if [[ -f "$target_ckpt" ]]; then
    echo "$target_ckpt" > "$done"
    rm -f "$running"
    stage "[SKIP] $name target checkpoint exists: $target_ckpt"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "$(date '+%Y-%m-%d %H:%M:%S') $name" > "$running"
  rm -f "$failed"

  if compgen -G "$run_dir/checkpoints/[0-9]*.pt" >/dev/null; then
    stage "[RESUME] $name from latest checkpoint under $run_dir/checkpoints."
  else
    resume_args=(--vq-ckpt "$VQ_CKPT" --finetune)
    stage "[INIT] $name from base VQ checkpoint: $VQ_CKPT"
  fi

  set +e
  "$TORCH_RUN" --standalone --nproc_per_node="$NPROC_PER_NODE" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --model-config "$config" \
    --dataset textatlas_image_text \
    --data-path "$TRAIN_MANIFEST" \
    --json-path "$TRAIN_MANIFEST" \
    --val-json-path "$VAL_MANIFEST" \
    --save-path "$PIPELINE_ROOT" \
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
  local rc=$?
  set -e

  if [[ "$rc" != "0" ]]; then
    echo "exit_code=$rc" > "$failed"
    rm -f "$running"
    die "$name training failed with exit code $rc"
  fi
  require_file "$target_ckpt" "$name target checkpoint"
  echo "$target_ckpt" > "$done"
  rm -f "$running"
  stage "[DONE] train $name."
}

write_runs_json() {
  local runs_json="$EVAL_DIR/runs_2500.json"
  "$PYTHON_BIN" - "$runs_json" "$PIPELINE_ROOT" "$V1_CONFIG" "$V2_CONFIG" "$BASELINE_CONFIG" "$BASELINE_CKPT" "$BASELINE_RUN_DIR" "$V1_NAME" "$V2_NAME" "$TARGET_STEPS" <<'PY'
import json
import sys
from pathlib import Path

runs_json = Path(sys.argv[1])
pipeline = Path(sys.argv[2])
v1_config = Path(sys.argv[3])
v2_config = Path(sys.argv[4])
baseline_config = Path(sys.argv[5])
baseline_ckpt = Path(sys.argv[6])
baseline_run_dir = Path(sys.argv[7])
v1_name = sys.argv[8]
v2_name = sys.argv[9]
step = int(sys.argv[10])
runs = {
    "w001": {
        "config": str(baseline_config),
        "ckpt": str(baseline_ckpt),
        "run_dir": str(baseline_run_dir),
        "step": step,
        "text_injection": True,
        "ocr_loss": True,
        "ocr_weight": 0.01,
        "train_loss_type": "text + OCR CE(0.01)",
        "note": "existing baseline, not retrained",
    }
}
v1_ckpt = pipeline / v1_name / "checkpoints" / f"{step:07d}.pt"
if v1_ckpt.exists():
    runs[v1_name] = {
        "config": str(v1_config),
        "ckpt": str(v1_ckpt),
        "run_dir": str(pipeline / v1_name),
        "step": step,
        "text_injection": True,
        "ocr_loss": True,
        "ocr_weight": 0.01,
        "train_loss_type": "text + OCR CE(0.01) + OCR-box gate",
        "note": "bbox gate only, low_conf disabled",
    }
v2_ckpt = pipeline / v2_name / "checkpoints" / f"{step:07d}.pt"
if v2_ckpt.exists():
    runs[v2_name] = {
        "config": str(v2_config),
        "ckpt": str(v2_ckpt),
        "run_dir": str(pipeline / v2_name),
        "step": step,
        "text_injection": True,
        "ocr_loss": True,
        "ocr_weight": 0.01,
        "train_loss_type": "text + OCR CE(0.01) + OCR-box gate + proxy-confidence weighting",
        "note": "confidence_source=deepseek_token_logprob_proxy",
    }
runs_json.parent.mkdir(parents=True, exist_ok=True)
runs_json.write_text(json.dumps(runs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(runs_json)
PY
}

run_eval() {
  if [[ "$RUN_EVAL" != "1" || "$AUTO_EVAL" != "1" ]]; then
    stage "[SKIP] eval disabled: RUN_EVAL=$RUN_EVAL AUTO_EVAL=$AUTO_EVAL"
    return 0
  fi
  stage "[START] eval."
  local runs_json
  runs_json="$(write_runs_json | tail -n 1)"
  "$PYTHON_BIN" scripts/stage1/ocr_debug/evaluate_setting_metrics.py \
    --runs-json "$runs_json" \
    --manifest-jsonl "$VAL_MANIFEST" \
    --output-dir "$EVAL_DIR" \
    --step "$TARGET_STEPS" \
    --image-metric-source train_val_csv \
    --device-backend "$DEVICE_BACKEND" \
    --device-id 0 \
    --mixed-precision "$MIXED_PRECISION" \
    --image-size 256 \
    --batch-size "$EVAL_BATCH_SIZE" \
    --max-images "$VAL_MAX_IMAGES" \
    --grid-samples 12 \
    --checkpoint-weight-key model \
    --deepseek-ocr-model "$DEEPSEEK_MODEL" \
    --thirdparty-ocr-backend none

  if [[ -f "$EVAL_DIR/qual_grid_5000.png" && ! -f "$EVAL_DIR/qual_grid_2500.png" ]]; then
    cp "$EVAL_DIR/qual_grid_5000.png" "$EVAL_DIR/qual_grid_2500.png"
  fi
  if [[ -f "$EVAL_DIR/setting_metrics.md" && ! -f "$EVAL_DIR/summary.md" ]]; then
    cp "$EVAL_DIR/setting_metrics.md" "$EVAL_DIR/summary.md"
  fi
  "$PYTHON_BIN" - "$EVAL_DIR/text_region_metrics_summary.json" <<'PY'
import json
import sys
from pathlib import Path
out = Path(sys.argv[1])
if not out.exists():
    out.write_text(json.dumps({
        "status": "not_computed",
        "reason": "No dedicated text-region metric evaluator is implemented in the current repo; qualitative grid and OCR diagnostics were generated separately.",
    }, indent=2) + "\n", encoding="utf-8")
PY
  stage "[DONE] eval."
}

write_final_summary() {
  "$PYTHON_BIN" - "$PIPELINE_ROOT" "$PROXY_SUMMARY_JSON" "$EVAL_DIR/setting_metrics.csv" "$STATE_DIR/v2_decision.json" "$V1_NAME" "$V2_NAME" <<'PY'
import csv
import json
import sys
from pathlib import Path

pipeline = Path(sys.argv[1])
proxy_summary = Path(sys.argv[2])
metrics_csv = Path(sys.argv[3])
v2_decision = Path(sys.argv[4])
v1_name = sys.argv[5]
v2_name = sys.argv[6]

def read_metric(run, field):
    if not metrics_csv.exists():
        return None
    with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("run") == run:
                try:
                    return float(row.get(field, ""))
                except Exception:
                    return None
    return None

proxy = json.loads(proxy_summary.read_text(encoding="utf-8")) if proxy_summary.exists() else {}
v2 = json.loads(v2_decision.read_text(encoding="utf-8")) if v2_decision.exists() else {}
w001_psnr = read_metric("w001", "psnr")
v1_psnr = read_metric(v1_name, "psnr")
v2_psnr = read_metric(v2_name, "psnr")

if v1_psnr is None or w001_psnr is None:
    v1_conclusion = "v1 result unavailable; check training/eval logs."
elif v1_psnr > w001_psnr + 0.05:
    v1_conclusion = "bbox gate has positive signal on PSNR in this closed-set quick screening."
elif v1_psnr >= w001_psnr - 0.05:
    v1_conclusion = "bbox gate is not clearly harmful; readable50 is text-dense so effect may be limited."
else:
    v1_conclusion = "bbox gate may be suppressing useful text injection."

if not v2.get("v2_will_run"):
    v2_conclusion = "confidence weighting not evaluated because valid proxy confidence was unavailable or RUN_V2 disabled."
elif v2_psnr is None or v1_psnr is None:
    v2_conclusion = "v2 result unavailable; check training/eval logs."
elif v2_psnr > v1_psnr + 0.05:
    v2_conclusion = "proxy confidence weighting adds positive signal on PSNR under this proxy."
else:
    v2_conclusion = "proxy confidence weighting has no clear PSNR gain under this proxy."

lines = [
    "# OCR Box Gate Overnight Summary",
    "",
    f"- start_time: {(pipeline / 'state' / 'start_time.txt').read_text().strip() if (pipeline / 'state' / 'start_time.txt').exists() else ''}",
    f"- end_time: {(pipeline / 'state' / 'end_time.txt').read_text().strip() if (pipeline / 'state' / 'end_time.txt').exists() else ''}",
    f"- proxy confidence status: {'valid' if proxy.get('v2_allowed') else 'invalid/skipped'}",
    f"- confidence stats: min={proxy.get('confidence_min')}, mean={proxy.get('confidence_mean')}, max={proxy.get('confidence_max')}, unique={proxy.get('confidence_unique_count')}",
    "- w001 baseline used: existing w001@2500, not retrained",
    f"- v1 status: {'done' if (pipeline / 'state' / f'{v1_name}.done').exists() else 'not_done'}",
    f"- v2 status: {'done' if (pipeline / 'state' / f'{v2_name}.done').exists() else ('skipped' if (pipeline / 'state' / f'{v2_name}.skipped').exists() else 'not_done')}",
    f"- v1 checkpoint: {pipeline / v1_name / 'checkpoints' / '0002500.pt'}",
    f"- v2 checkpoint: {pipeline / v2_name / 'checkpoints' / '0002500.pt'}",
    f"- eval output: {pipeline / 'eval_2500'}",
    f"- paper_table: {pipeline / 'eval_2500' / 'paper_table.md'}",
    f"- setting_metrics: {pipeline / 'eval_2500' / 'setting_metrics.csv'}",
    f"- qual_grid: {pipeline / 'eval_2500' / 'qual_grid_2500.png'}",
    "",
    "## Conclusions",
    "",
    f"- {v1_conclusion}",
    f"- {v2_conclusion}",
    "",
    "## Warning",
    "",
    "- readable50 train/val are identical 50 images; this is closed-set quick screening, not independent validation.",
]
(pipeline / "final_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(pipeline / "final_summary.md")
PY
}

main() {
  date '+%Y-%m-%d %H:%M:%S' > "$STATE_DIR/start_time.txt"
  stage "[START] OCR box gate full pipeline."
  run_proxy_if_needed
  standardize_proxy_confidence
  check_no_training_process
  precheck
  write_configs
  determine_v2
  stage "[PLAN]"
  write_key_values
  cat "$PROXY_SUMMARY_JSON" || true
  cat "$STATE_DIR/v2_decision.json" || true
  if [[ "$PRECHECK_ONLY" == "1" ]]; then
    stage "[DONE] PRECHECK_ONLY=1; not starting training."
    return 0
  fi

  if [[ "$RUN_V1" == "1" ]]; then
    train_one "$V1_NAME" "$V1_CONFIG"
  else
    stage "[SKIP] v1 disabled by RUN_V1=$RUN_V1."
  fi

  if v2_will_run; then
    train_one "$V2_NAME" "$V2_CONFIG"
  else
    stage "[SKIP] v2 skipped."
    echo "SKIPPED_NO_VALID_PROXY_CONFIDENCE_OR_DISABLED" > "$STATE_DIR/${V2_NAME}.skipped"
  fi

  run_eval
  date '+%Y-%m-%d %H:%M:%S' > "$STATE_DIR/end_time.txt"
  write_final_summary
  stage "[DONE] OCR box gate full pipeline."
}

main "$@"
