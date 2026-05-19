#!/usr/bin/env bash
set -euo pipefail
trap 'echo "[ERROR] failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

REPO="/home/ma-user/work/GigaTok_hr/GigaTok_Loss_two_ablation"
BASE="/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_two_new_ablation_readable50"
TARGET_STEPS="2500"
UNDERLYING="$BASE/run_w001_single_ablation_2500_bg.sh"
RUNS_JSON="$BASE/runs_2500_single_and_context.json"
EVAL_DIR="$BASE/setting_eval_2500_single_and_context"
RUN_FINAL_EVAL="${RUN_FINAL_EVAL:-0}"

OUTPUT_ROOT="/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs"
W001_CONFIG="$OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/configs/w001.yaml"
W001_RUN_DIR="$OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000/w001"
W001_CKPT="$W001_RUN_DIR/checkpoints/0002500.pt"
VQ_CKPT="/home/ma-user/work/GigaTok_hr/gigatok_persist/checkpoints/VQ_BL256_dino_disc.pt"
TRAIN_MANIFEST="$OUTPUT_ROOT/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_train.jsonl"
VAL_MANIFEST="$OUTPUT_ROOT/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_val.jsonl"
TEXT_FEATURE_CACHE="$OUTPUT_ROOT/text_feature_cache/readable50_glyph_mapped/text_features.pt"
DEEPSEEK_OCR_MODEL="/home/ma-user/work/GigaTok_hr/gigatok_persist/models/DeepSeek-OCR"

export RUN_SMOKE=0
export TARGET_STEPS=2500
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NPROC_PER_NODE=2
export DEVICE_BACKEND=npu
export MIXED_PRECISION=bf16
export GLOBAL_BATCH_SIZE=8
export GLOBAL_SEED=0
export VQ_CKPT="$VQ_CKPT"

stage() {
  echo
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

require_dir() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || die "$label not found: $path"
}

ensure_no_training_process() {
  local matches
  matches=$(ps -ef | grep -E "torchrun|vq_train.py" | grep -v grep || true)
  if [[ -n "$matches" ]]; then
    echo "$matches" >&2
    die "existing torchrun/vq_train.py process found; refusing to start another training job."
  fi
}

check_val_step() {
  local label="$1"
  shift
  python3 - "$TARGET_STEPS" "$label" "$@" <<'PY'
import csv
import sys
from pathlib import Path

target = int(sys.argv[1])
label = sys.argv[2]
paths = [Path(p) for p in sys.argv[3:]]
for path in paths:
    if not path.exists():
        continue
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hits = [r for r in rows if int(float(r.get("step", -1))) == target]
    if hits:
        print(f"[VAL] {label}: path={path}")
        print(f"[VAL] {label}: step={target} row={hits[-1]}")
        raise SystemExit(0)
raise SystemExit(f"{label}: no val CSV row for step={target}; checked={paths}")
PY
}

check_manifest_50() {
  python3 - "$VAL_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
rows = []
missing = []
with manifest.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        image_path = Path(str(row.get("image_path") or row.get("path") or row.get("file_name") or ""))
        if not image_path.exists():
            missing.append(str(image_path))
print(f"[CHECK] val manifest rows={len(rows)}")
print(f"[CHECK] val manifest missing_images={len(missing)}")
if len(rows) != 50:
    raise SystemExit(f"val manifest must contain exactly 50 rows, got {len(rows)}")
if missing:
    raise SystemExit("missing image paths:\n" + "\n".join(missing[:20]))
PY
}

check_run_dir_safety() {
  local setting="$1"
  local run_dir="$BASE/$setting"
  local target_ckpt="$run_dir/checkpoints/0002500.pt"
  if [[ -f "$target_ckpt" ]]; then
    echo "[INFO] $setting target checkpoint exists; training will be skipped and verification will still run."
    return 0
  fi
  if [[ -d "$run_dir" ]]; then
    die "$setting run dir exists but 0002500.pt is missing; refusing to overwrite partial run: $run_dir"
  fi
  echo "[OK] $setting formal run dir absent."
}

hard_assert_configs() {
  stage "[START] hard config assert"
  python3 - "$BASE/configs/w001_hr_only.yaml" "$BASE/configs/w001_local_only.yaml" <<'PY'
import sys
import yaml
from pathlib import Path

hr_path = Path(sys.argv[1])
local_path = Path(sys.argv[2])

def load(path):
    if not path.exists():
        raise SystemExit(f"missing config: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def enabled(block):
    return bool((block or {}).get("enabled", False))

def require(name, got, expected):
    print(f"[ASSERT] {name} = {got!r} expected {expected!r}")
    if got != expected:
        raise SystemExit(f"{name}: got {got!r}, expected {expected!r}")

hr = load(hr_path)
loc = load(local_path)

def common_assert(prefix, cfg):
    text_recon = cfg.get("text_recon_conditioning") or {}
    ocr_tf = cfg.get("ocr_teacher_forcing_loss") or {}
    ocr_vis = cfg.get("ocr_visual_alignment_loss") or {}
    ocr_feature = cfg.get("ocr_feature_loss") or {}
    require(f"{prefix}.ocr_teacher_forcing_loss.enabled", enabled(ocr_tf), True)
    require(f"{prefix}.ocr_teacher_forcing_loss.weight", float(ocr_tf.get("weight", -1)), 0.01)
    require(f"{prefix}.ocr_teacher_forcing_loss.start_weight", float(ocr_tf.get("start_weight", -1)), 0.005)
    require(f"{prefix}.ocr_visual_alignment_loss.enabled", enabled(ocr_vis), False)
    require(f"{prefix}.ocr_feature_loss.enabled", enabled(ocr_feature), False)
    require(f"{prefix}.text_recon_conditioning.layers", [int(x) for x in text_recon.get("layers", [])], [6, 12, 18])
    require(f"{prefix}.text_recon_conditioning.mode", text_recon.get("mode"), "residual_cross_attn_visual_mask")

common_assert("w001_hr_only", hr)
text_hr = hr.get("text_hr") or {}
local = hr.get("local_similarity_loss") or {}
require("w001_hr_only.text_hr.enabled", enabled(text_hr), True)
require("w001_hr_only.text_hr.hr_loss_weight", float(text_hr.get("hr_loss_weight", -1)), 500.0)
require("w001_hr_only.text_hr.image_token_len", int(text_hr.get("image_token_len", -1)), 0)
require("w001_hr_only.text_hr.svd_mode", text_hr.get("svd_mode"), "gram_scaled_identity")
require("w001_hr_only.local_similarity_loss.enabled", enabled(local), False)
require("w001_hr_only.local_similarity_loss.weight", float(local.get("weight", -1)), 0.0)

common_assert("w001_local_only", loc)
text_hr = loc.get("text_hr") or {}
local = loc.get("local_similarity_loss") or {}
require("w001_local_only.text_hr.enabled", enabled(text_hr), False)
require("w001_local_only.text_hr.hr_loss_weight", float(text_hr.get("hr_loss_weight", -1)), 0.0)
require("w001_local_only.text_hr.image_token_len", int(text_hr.get("image_token_len", -1)), 0)
require("w001_local_only.local_similarity_loss.enabled", enabled(local), True)
require("w001_local_only.local_similarity_loss.weight", float(local.get("weight", -1)), 1.0e-4)
require("w001_local_only.local_similarity_loss.distance_type", local.get("distance_type"), "positional")
require("w001_local_only.local_similarity_loss.average_heads_before_loss", bool(local.get("average_heads_before_loss", False)), True)
print("[OK] hard config assertions passed")
PY
  stage "[DONE] hard config assert"
}

preflight() {
  stage "[START] preflight"
  require_dir "$REPO" "repo"
  mkdir -p "$BASE"
  require_file "$UNDERLYING" "underlying single-ablation script"
  if ! grep -Eq "w001_hr_only|w001_local_only" "$UNDERLYING"; then
    die "underlying script does not advertise w001_hr_only / w001_local_only support."
  fi
  require_file "$BASE/configs/w001_hr_only.yaml" "w001_hr_only config"
  require_file "$BASE/configs/w001_local_only.yaml" "w001_local_only config"
  require_file "$W001_CONFIG" "w001 baseline config"
  require_file "$W001_CKPT" "w001@2500 baseline checkpoint"
  require_file "$W001_RUN_DIR/val.csv" "w001 baseline val.csv"
  check_val_step "w001@2500" "$W001_RUN_DIR/val.csv" "$W001_RUN_DIR/metrics/val_metrics.csv"
  require_file "$VQ_CKPT" "initial VQ checkpoint"
  require_file "$TRAIN_MANIFEST" "train manifest"
  require_file "$VAL_MANIFEST" "val manifest"
  check_manifest_50
  require_file "$TEXT_FEATURE_CACHE" "text feature cache"
  require_dir "$DEEPSEEK_OCR_MODEL" "DeepSeek-OCR model"
  cd "$REPO"
  python3 -m py_compile \
    tokenizer/tokenizer_image/vq/vq_train.py \
    tokenizer/tokenizer_image/vq/vq_vit_model.py \
    tokenizer/tokenizer_image/vq/blocks.py \
    tokenizer/tokenizer_image/vq/vq_loss.py \
    scripts/stage1/ocr_debug/evaluate_two_ablation_readable50.py \
    scripts/stage1/ocr_debug/evaluate_setting_metrics.py
  ensure_no_training_process
  check_run_dir_safety "w001_hr_only"
  check_run_dir_safety "w001_local_only"
  hard_assert_configs
  stage "[DONE] preflight"
}

require_log_positive() {
  local log="$1"
  local pattern="$2"
  local label="$3"
  grep -E "$pattern" "$log" >/dev/null 2>&1 || die "missing expected active metric in $log: $label"
  echo "[OK] expected metric found: $label"
  grep -E "$pattern" "$log" | tail -n 2
}

require_log_forbidden_absent() {
  local log="$1"
  local pattern="$2"
  local label="$3"
  local hits
  hits=$(grep -E "$pattern" "$log" || true)
  if [[ -n "$hits" ]]; then
    echo "$hits" | tail -n 10 >&2
    die "forbidden active metric appeared in training metric lines: $label"
  fi
  echo "[OK] forbidden metric absent: $label"
}

verify_setting() {
  local setting="$1"
  local run_dir="$BASE/$setting"
  local ckpt="$run_dir/checkpoints/0002500.pt"
  local log="$BASE/run_${setting}_2500.log"
  stage "[START] verify $setting"
  require_file "$ckpt" "$setting checkpoint"
  check_val_step "$setting@2500" "$run_dir/val.csv" "$run_dir/metrics/val_metrics.csv"
  require_file "$log" "$setting formal log"
  if [[ "$setting" == "w001_hr_only" ]]; then
    require_log_positive "$log" "\\(Generator\\).*text_hr_loss:" "text_hr_loss"
    require_log_positive "$log" "\\(Generator\\).*weighted_text_hr_loss" "weighted_text_hr_loss"
    require_log_positive "$log" "\\(Generator\\).*ocr_tf_ce_loss" "ocr_tf_ce_loss"
    require_log_positive "$log" "\\(Generator\\).*weighted_ocr_tf_loss" "weighted_ocr_tf_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*local_similarity_loss" "local_similarity_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*weighted_local_similarity_loss" "weighted_local_similarity_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*ocr_visual_align_loss" "ocr_visual_align_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*weighted_ocr_visual_align_loss" "weighted_ocr_visual_align_loss"
  elif [[ "$setting" == "w001_local_only" ]]; then
    require_log_positive "$log" "\\(Generator\\).*local_similarity_loss" "local_similarity_loss"
    require_log_positive "$log" "\\(Generator\\).*weighted_local_similarity_loss" "weighted_local_similarity_loss"
    require_log_positive "$log" "\\(Generator\\).*ocr_tf_ce_loss" "ocr_tf_ce_loss"
    require_log_positive "$log" "\\(Generator\\).*weighted_ocr_tf_loss" "weighted_ocr_tf_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*text_hr_loss:" "text_hr_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*weighted_text_hr_loss" "weighted_text_hr_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*ocr_visual_align_loss" "ocr_visual_align_loss"
    require_log_forbidden_absent "$log" "\\(Generator\\).*weighted_ocr_visual_align_loss" "weighted_ocr_visual_align_loss"
  else
    die "unknown setting for verify: $setting"
  fi
  stage "[DONE] verify $setting"
}

run_setting() {
  local setting="$1"
  local ckpt="$BASE/$setting/checkpoints/0002500.pt"
  local log="$BASE/run_${setting}_2500.log"
  stage "[START] formal run $setting"
  echo "setting=$setting"
  echo "config=$BASE/configs/$setting.yaml"
  echo "target_ckpt=$ckpt"
  echo "formal_log=$log"
  echo "vq_ckpt(initial)=$VQ_CKPT"
  ensure_no_training_process
  check_run_dir_safety "$setting"
  if [[ -f "$ckpt" ]]; then
    echo "[INFO] skip formal training for $setting because target checkpoint already exists."
  else
    cd "$REPO"
    RUN_SMOKE=0 \
    SETTING="$setting" \
    TARGET_STEPS=2500 \
    LOG="$log" \
    ASCEND_RT_VISIBLE_DEVICES=0,1 \
    NPROC_PER_NODE=2 \
    DEVICE_BACKEND=npu \
    MIXED_PRECISION=bf16 \
    GLOBAL_BATCH_SIZE=8 \
    GLOBAL_SEED=0 \
    VQ_CKPT="$VQ_CKPT" \
      bash "$UNDERLYING"
  fi
  verify_setting "$setting"
  stage "[DONE] formal run $setting"
}

build_runs_json() {
  stage "[START] build runs json"
  python3 - "$RUNS_JSON" "$BASE" "$TARGET_STEPS" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
base = Path(sys.argv[2])
step = int(sys.argv[3])
root = Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs")
runs = {
    "w001": {"config": str(root / "ocr_w_sweep_setting_tune_5000/configs/w001.yaml"), "ckpt": str(root / "ocr_w_sweep_setting_tune_5000/w001/checkpoints/0002500.pt"), "run_dir": str(root / "ocr_w_sweep_setting_tune_5000/w001"), "step": step, "text_injection": True, "ocr_loss": True, "ocr_weight": 0.01, "high_rank": False, "local_similarity": False, "local_sim": "no", "ocr_visual_alignment": False, "train_loss_type": "text + OCR CE(0.01)", "note": "w001 baseline at 2500; do not rerun"},
    "w001_ocrvis_only": {"config": str(base / "configs/w001_ocrvis_only.yaml"), "ckpt": str(base / "w001_ocrvis_only/checkpoints/0002500.pt"), "run_dir": str(base / "w001_ocrvis_only"), "step": step, "text_injection": True, "ocr_loss": False, "ocr_weight": 0.0, "high_rank": False, "local_similarity": False, "local_sim": "no", "ocr_visual_alignment": True, "train_loss_type": "OCR visual projected feature alignment", "note": "single OCR CE -> OCR visual projected feature alignment"},
    "w001_hr_only": {"config": str(base / "configs/w001_hr_only.yaml"), "ckpt": str(base / "w001_hr_only/checkpoints/0002500.pt"), "run_dir": str(base / "w001_hr_only"), "step": step, "text_injection": True, "ocr_loss": True, "ocr_weight": 0.01, "high_rank": True, "local_similarity": False, "local_sim": "no", "ocr_visual_alignment": False, "train_loss_type": "text + OCR CE(0.01) + HR only", "note": "single high-rank ablation"},
    "w001_local_only": {"config": str(base / "configs/w001_local_only.yaml"), "ckpt": str(base / "w001_local_only/checkpoints/0002500.pt"), "run_dir": str(base / "w001_local_only"), "step": step, "text_injection": True, "ocr_loss": True, "ocr_weight": 0.01, "high_rank": False, "local_similarity": True, "local_sim": "yes", "ocr_visual_alignment": False, "train_loss_type": "text + OCR CE(0.01) + local-sim only", "note": "single local-sim ablation"},
    "w001_hr_local": {"config": str(base / "configs/w001_hr_local.yaml"), "ckpt": str(base / "w001_hr_local/checkpoints/0002500.pt"), "run_dir": str(base / "w001_hr_local"), "step": step, "text_injection": True, "ocr_loss": True, "ocr_weight": 0.01, "high_rank": True, "local_similarity": True, "local_sim": "yes", "ocr_visual_alignment": False, "train_loss_type": "text + OCR CE(0.01) + HR + local-sim", "note": "combined route context only"},
    "ocrvis_hr_local": {"config": str(base / "configs/ocrvis_hr_local.yaml"), "ckpt": str(base / "ocrvis_hr_local/checkpoints/0002500.pt"), "run_dir": str(base / "ocrvis_hr_local"), "step": step, "text_injection": True, "ocr_loss": False, "ocr_weight": 0.0, "high_rank": True, "local_similarity": True, "local_sim": "yes", "ocr_visual_alignment": True, "train_loss_type": "OCR visual projected alignment + HR + local-sim", "note": "combined route context only; OCR CE disabled"},
}
missing = []
for name, item in runs.items():
    for key in ("config", "ckpt", "run_dir"):
        path = Path(item[key])
        ok = path.is_dir() if key == "run_dir" else path.is_file()
        if not ok:
            missing.append(f"{name}.{key}: {path}")
if missing:
    raise SystemExit("required run artifact missing; refusing eval:\n" + "\n".join(missing))
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as handle:
    json.dump(runs, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
print(f"[OK] wrote runs json: {out}")
PY
  stage "[DONE] build runs json"
}

run_eval() {
  stage "[START] final eval"
  cd "$REPO"
  python3 scripts/stage1/ocr_debug/evaluate_two_ablation_readable50.py \
    --runs-json "$RUNS_JSON" \
    --manifest-jsonl "$VAL_MANIFEST" \
    --output-dir "$BASE/setting_eval_2500_single_and_context" \
    --step 2500 \
    --device-backend npu \
    --device-id 0 \
    --mixed-precision bf16 \
    --image-size 256 \
    --batch-size 4 \
    --grid-samples 12 \
    --python-bin python3 \
    --deepseek-ocr-model "$DEEPSEEK_OCR_MODEL" \
    --thirdparty-ocr-status unavailable
  stage "[DONE] final eval"
}

print_final_summary() {
  stage "[FINAL SUMMARY]"
  echo "hr_only checkpoint: $BASE/w001_hr_only/checkpoints/0002500.pt"
  echo "local_only checkpoint: $BASE/w001_local_only/checkpoints/0002500.pt"
  check_val_step "w001_hr_only@2500" "$BASE/w001_hr_only/val.csv" "$BASE/w001_hr_only/metrics/val_metrics.csv"
  check_val_step "w001_local_only@2500" "$BASE/w001_local_only/val.csv" "$BASE/w001_local_only/metrics/val_metrics.csv"
  echo "runs json: $RUNS_JSON"
  echo "paper_table.md: $BASE/setting_eval_2500_single_and_context/paper_table.md"
  echo "setting_metrics.csv: $BASE/setting_eval_2500_single_and_context/setting_metrics.csv"
  echo "summary.md: $BASE/setting_eval_2500_single_and_context/summary.md"
  echo "qual_grid_2500.png: $BASE/setting_eval_2500_single_and_context/qual_grid_2500.png"
  echo "comparison_grid.png: $BASE/setting_eval_2500_single_and_context/reconstruction/comparison_grid.png"
  cat <<'EOF'

Interpretation reminder:
- w001_hr_only causal conclusion only: w001@2500 vs w001_hr_only@2500.
- w001_local_only causal conclusion only: w001@2500 vs w001_local_only@2500.
- w001_ocrvis_only causal conclusion only: w001@2500 vs w001_ocrvis_only@2500.
- w001_hr_local / ocrvis_hr_local are combined-route context only; do not attribute their changes to a single module.
- Do not compare this 2500-step quick screening directly with 5000-step final tables.
- High-rank is Frobenius-normalized scaled Gram identity, not raw ||A^T A - I||.
- Local-sim is positional local-similarity, not feature-cosine topology.
EOF
}

main() {
  stage "[PIPELINE START]"
  echo "RUN_FINAL_EVAL=$RUN_FINAL_EVAL"
  preflight
  run_setting "w001_hr_only"
  run_setting "w001_local_only"
  if [[ "$RUN_FINAL_EVAL" == "1" ]]; then
    build_runs_json
    run_eval
  else
    echo "[SKIP] final eval disabled because RUN_FINAL_EVAL=$RUN_FINAL_EVAL."
  fi
  print_final_summary
  stage "[PIPELINE DONE]"
}

main "$@"
