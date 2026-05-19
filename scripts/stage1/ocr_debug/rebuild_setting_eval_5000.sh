#!/usr/bin/env bash
set -euo pipefail

# Rebuild setting-eval artifacts from existing 5000-step checkpoints only.
# This script does not train, does not modify run directories, and does not
# delete previous outputs. If the target rebuilt directory already exists, set
# ARCHIVE_EXISTING=1 to move it aside with a timestamp before rebuilding.

REPO_ROOT=${REPO_ROOT:-/home/ma-user/work/GigaTok_hr/GigaTok_Loss}
BASE=${BASE:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000}
OUT_DIR=${OUT_DIR:-"$BASE/setting_eval_rebuilt"}
FINAL_LINK=${FINAL_LINK:-"$BASE/setting_eval_final"}
MANIFEST=${MANIFEST:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_val.jsonl}
PYTHON_BIN=${PYTHON_BIN:-python3}

DEVICE_BACKEND=${DEVICE_BACKEND:-npu}
DEVICE_ID=${DEVICE_ID:-0}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
BATCH_SIZE=${BATCH_SIZE:-8}
IMAGE_SIZE=${IMAGE_SIZE:-256}
GRID_SAMPLES=${GRID_SAMPLES:-12}
STEP=${STEP:-5000}

RUN_ORDER=(orig noocr w001 w002 w003 w005)

check_file() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] missing file: $path" >&2
    exit 1
  fi
}

check_dir() {
  local path=$1
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] missing directory: $path" >&2
    exit 1
  fi
}

copy_deepseek_recon_images() {
  local src_dir=$1
  local dst_root=$2
  local expected_per_run=50
  local run_index=0

  check_dir "$src_dir"

  for run in "${RUN_ORDER[@]}"; do
    mkdir -p "$dst_root/$run"
    for idx in $(seq 0 $((expected_per_run - 1))); do
      local src_num=$((run_index * expected_per_run + idx + 1))
      local src
      src=$(printf "%s/recon_%06d.png" "$src_dir" "$src_num")
      local dst
      dst=$(printf "%s/%s/%03d.png" "$dst_root" "$run" "$idx")
      check_file "$src"
      cp "$src" "$dst"
    done
    run_index=$((run_index + 1))
  done
}

count_recon_images() {
  local dst_root=$1
  local total=0
  local summary=$2
  : > "$summary"

  for run in "${RUN_ORDER[@]}"; do
    local count
    count=$(find "$dst_root/$run" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
    echo "$run: $count" | tee -a "$summary"
    total=$((total + count))
  done
  echo "total: $total" | tee -a "$summary"

  if [[ "$total" -ne 300 ]]; then
    echo "[ERROR] expected 300 reconstruction images, got $total" >&2
    exit 1
  fi
}

cd "$REPO_ROOT"

check_file "$MANIFEST"

check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/configs/orig.yaml
check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/orig/checkpoints/0005000.pt
check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/orig/val.csv

check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/configs/noocr.yaml
check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/noocr/checkpoints/0005000.pt
check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/noocr/val.csv

check_file "$BASE/configs/w001.yaml"
check_file "$BASE/w001/checkpoints/0005000.pt"
check_file "$BASE/w001/val.csv"

check_file "$BASE/configs/w002.yaml"
check_file "$BASE/w002/checkpoints/0005000.pt"
check_file "$BASE/w002/val.csv"

check_file "$BASE/configs/w003.yaml"
check_file "$BASE/w003/checkpoints/0005000.pt"
check_file "$BASE/w003/val.csv"

check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/configs/w005.yaml
check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/w005/checkpoints/0005000.pt
check_file /home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/w005/val.csv

if [[ -e "$OUT_DIR" ]]; then
  if [[ "${ARCHIVE_EXISTING:-0}" != "1" ]]; then
    echo "[ERROR] output directory already exists: $OUT_DIR" >&2
    echo "        To keep it and rebuild cleanly, rerun with ARCHIVE_EXISTING=1." >&2
    exit 2
  fi
  archived="${OUT_DIR}.prev.$(date +%Y%m%d_%H%M%S)"
  echo "[INFO] archiving existing rebuilt directory without deleting: $archived"
  mv "$OUT_DIR" "$archived"
fi

mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/runs.json" <<'JSON'
{
  "orig": {
    "ocr_weight": null,
    "text_injection": false,
    "ocr_loss": false,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/configs/orig.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/orig/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/orig"
  },
  "noocr": {
    "ocr_weight": 0.0,
    "text_injection": true,
    "ocr_loss": false,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/configs/noocr.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/noocr/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/noocr"
  },
  "w001": {
    "ocr_weight": 0.01,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w001.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w001/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w001"
  },
  "w002": {
    "ocr_weight": 0.02,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w002.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w002/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w002"
  },
  "w003": {
    "ocr_weight": 0.03,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w003.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w003/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w003"
  },
  "w005": {
    "ocr_weight": 0.05,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/configs/w005.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/w005/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/w005"
  }
}
JSON

"$PYTHON_BIN" -m json.tool "$OUT_DIR/runs.json" >/dev/null

"$PYTHON_BIN" scripts/stage1/ocr_debug/evaluate_setting_metrics.py \
  --runs-json "$OUT_DIR/runs.json" \
  --manifest-jsonl "$MANIFEST" \
  --output-dir "$OUT_DIR" \
  --step "$STEP" \
  --image-metric-source train_val_csv \
  --device-backend "$DEVICE_BACKEND" \
  --device-id "$DEVICE_ID" \
  --mixed-precision "$MIXED_PRECISION" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --grid-samples "$GRID_SAMPLES" \
  --checkpoint-weight-key model \
  --thirdparty-ocr-backend none

check_file "$OUT_DIR/setting_metrics.csv"
check_file "$OUT_DIR/setting_metrics.md"
check_file "$OUT_DIR/paper_table.md"
check_file "$OUT_DIR/qual_grid_5000.png"
check_file "$OUT_DIR/ocr_predictions_deepseek.jsonl"

raw_ocr="$OUT_DIR/deepseek_eval/ocr_raw.jsonl"
check_file "$raw_ocr"
raw_count=$(wc -l < "$raw_ocr" | tr -d ' ')
flat_count=$(wc -l < "$OUT_DIR/ocr_predictions_deepseek.jsonl" | tr -d ' ')
if [[ "$raw_count" -ne 300 || "$flat_count" -ne 300 ]]; then
  echo "[ERROR] expected 300 DeepSeek OCR rows, got raw=$raw_count flat=$flat_count" >&2
  exit 1
fi

copy_deepseek_recon_images "$OUT_DIR/deepseek_eval/deepseek_ocr_tmp" "$OUT_DIR/reconstruction"
count_recon_images "$OUT_DIR/reconstruction" "$OUT_DIR/reconstruction_count_summary.txt"

if ! grep -q "training val.csv" "$OUT_DIR/setting_metrics.csv"; then
  echo "[ERROR] setting_metrics.csv does not record image_metric_source=training val.csv" >&2
  exit 1
fi

if grep -qE ',2[0-9]\\.[0-9]+,' "$OUT_DIR/setting_metrics.csv"; then
  echo "[WARN] setting_metrics.csv contains a 20.x numeric field; inspect before using paper_table.md" >&2
fi

ln -sfn "$OUT_DIR" "$FINAL_LINK"

echo "[OK] rebuilt setting eval without retraining."
echo "[OK] final link: $FINAL_LINK"
echo "[OK] view:"
echo "  $FINAL_LINK/paper_table.md"
echo "  $FINAL_LINK/qual_grid_5000.png"
echo "  $FINAL_LINK/setting_metrics.csv"
