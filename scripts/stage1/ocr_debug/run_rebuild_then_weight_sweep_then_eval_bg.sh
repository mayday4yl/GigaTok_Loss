#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ma-user/work/GigaTok_hr/GigaTok_Loss}
OUTPUT_ROOT=${OUTPUT_ROOT:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs}
SETTING_ROOT=${SETTING_ROOT:-$OUTPUT_ROOT/ocr_w_sweep_setting_tune_5000}
WEIGHT_ROOT=${WEIGHT_ROOT:-$OUTPUT_ROOT/ocr_w_sweep_weight_finetune_5000}
LOG=${LOG:-$OUTPUT_ROOT/run_rebuild_then_weight_sweep_then_eval.log}

MANIFEST_ROOT=${MANIFEST_ROOT:-$OUTPUT_ROOT/text_hr_multi_debug/manifests}
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_train.jsonl}
VAL_MANIFEST=${VAL_MANIFEST:-$MANIFEST_ROOT/cleantextsynth_ocr_readable_50_val.jsonl}
TEXT_FEATURE_CACHE=${TEXT_FEATURE_CACHE:-$OUTPUT_ROOT/text_feature_cache/readable50_glyph_mapped/text_features.pt}

PYTHON_BIN=${PYTHON_BIN:-python3}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1}
DEVICE_BACKEND=${DEVICE_BACKEND:-npu}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
BATCH_SIZE=${BATCH_SIZE:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
IMAGE_SIZE=${IMAGE_SIZE:-256}
GRID_SAMPLES=${GRID_SAMPLES:-12}

OLD_RUN_ORDER=(orig noocr w001 w002 w003 w005)
ALL_RUN_ORDER=(orig noocr w0005 w001 w0015 w002 w0025 w003 w005)
NEW_WEIGHT_GROUPS=(w0005 w0015 w0025)

PHASE_A_OUT="$SETTING_ROOT/setting_eval_rebuilt"
PHASE_A_FINAL="$SETTING_ROOT/setting_eval_final"
PHASE_C_OUT="$WEIGHT_ROOT/setting_eval_all"
PHASE_D_OUT="$WEIGHT_ROOT/setting_eval_all_posthoc"
RUN_PHASE_D_POSTHOC=${RUN_PHASE_D_POSTHOC:-0}

mkdir -p "$OUTPUT_ROOT"
exec > >(tee -a "$LOG") 2>&1

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

stage() {
  echo "[$(timestamp)] $*"
}

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

safe_remove_phase_a_eval() {
  local expected="$SETTING_ROOT/setting_eval_rebuilt"
  if [[ "$PHASE_A_OUT" != "$expected" ]]; then
    echo "[ERROR] refusing to remove unexpected Phase A path: $PHASE_A_OUT" >&2
    exit 1
  fi
  if [[ "$PHASE_A_OUT" == "$SETTING_ROOT" || "$PHASE_A_OUT" == "$OUTPUT_ROOT" || "$PHASE_A_OUT" == "/" ]]; then
    echo "[ERROR] refusing to remove unsafe Phase A path: $PHASE_A_OUT" >&2
    exit 1
  fi
  if [[ -e "$PHASE_A_OUT" ]]; then
    stage "[INFO] removing allowed derived eval directory: $PHASE_A_OUT"
    rm -rf -- "$PHASE_A_OUT"
  fi
}

write_phase_a_runs_json() {
  mkdir -p "$PHASE_A_OUT"
  cat > "$PHASE_A_OUT/runs.json" <<'JSON'
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
  "$PYTHON_BIN" -m json.tool "$PHASE_A_OUT/runs.json" >/dev/null
}

write_phase_c_runs_json() {
  mkdir -p "$PHASE_C_OUT"
  cat > "$PHASE_C_OUT/runs.json" <<'JSON'
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
  "w0005": {
    "ocr_weight": 0.005,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/configs/w0005.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/w0005/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/w0005"
  },
  "w001": {
    "ocr_weight": 0.01,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w001.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w001/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w001"
  },
  "w0015": {
    "ocr_weight": 0.015,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/configs/w0015.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/w0015/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/w0015"
  },
  "w002": {
    "ocr_weight": 0.02,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w002.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w002/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w002"
  },
  "w0025": {
    "ocr_weight": 0.025,
    "text_injection": true,
    "ocr_loss": true,
    "config": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/configs/w0025.yaml",
    "ckpt": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/w0025/checkpoints/0005000.pt",
    "run_dir": "/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_weight_finetune_5000/w0025"
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
  "$PYTHON_BIN" -m json.tool "$PHASE_C_OUT/runs.json" >/dev/null
}

copy_deepseek_recon_images() {
  local src_dir=$1
  local dst_root=$2
  local expected_per_run=$3
  shift 3
  local runs=("$@")
  local run_index=0

  check_dir "$src_dir"
  for run in "${runs[@]}"; do
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
  local expected_total=$2
  local summary=$3
  shift 3
  local runs=("$@")
  local total=0
  : > "$summary"

  for run in "${runs[@]}"; do
    local count
    count=$(find "$dst_root/$run" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d ' ')
    echo "$run: $count" | tee -a "$summary"
    total=$((total + count))
  done
  echo "total: $total" | tee -a "$summary"

  if [[ "$total" -ne "$expected_total" ]]; then
    echo "[ERROR] expected $expected_total reconstruction images, got $total" >&2
    exit 1
  fi
}

run_precheck() {
  stage "[START] precheck"
  cd "$REPO_ROOT"

  echo "repo: $REPO_ROOT"
  echo "setting_root: $SETTING_ROOT"
  echo "weight_root: $WEIGHT_ROOT"
  echo "train_manifest: $TRAIN_MANIFEST"
  echo "val_manifest: $VAL_MANIFEST"
  echo "text_feature_cache: $TEXT_FEATURE_CACHE"
  echo "disk space:"
  df -h /home/ma-user/work

  check_file "$TRAIN_MANIFEST"
  check_file "$VAL_MANIFEST"
  check_file "$TEXT_FEATURE_CACHE"
  check_file scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh
  check_file scripts/stage1/ocr_debug/evaluate_setting_metrics.py
  check_file scripts/stage1/evaluate_textatlas_reconstruction.py

  if [[ -e "$PHASE_C_OUT" ]]; then
    echo "[ERROR] Phase C output already exists; refusing to overwrite: $PHASE_C_OUT" >&2
    exit 1
  fi
  if [[ "$RUN_PHASE_D_POSTHOC" == "1" && -e "$PHASE_D_OUT" ]]; then
    echo "[ERROR] Phase D output already exists; refusing to overwrite: $PHASE_D_OUT" >&2
    exit 1
  fi
  for group in "${NEW_WEIGHT_GROUPS[@]}"; do
    if [[ -e "$WEIGHT_ROOT/$group" ]]; then
      echo "[ERROR] new run directory already exists; refusing to overwrite: $WEIGHT_ROOT/$group" >&2
      exit 1
    fi
    if [[ -e "$WEIGHT_ROOT/configs/$group.yaml" ]]; then
      echo "[ERROR] new config already exists; refusing to overwrite: $WEIGHT_ROOT/configs/$group.yaml" >&2
      exit 1
    fi
  done

  "$PYTHON_BIN" - <<'PY'
import csv
import gc
import json
from pathlib import Path

import torch
import yaml

runs = {
    "orig": {
        "config": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/configs/orig.yaml"),
        "ckpt": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/orig/checkpoints/0005000.pt"),
        "run_dir": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_orig_5000/orig"),
        "expect_text": False,
        "expect_ocr": False,
        "expect_weight": None,
    },
    "noocr": {
        "config": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/configs/noocr.yaml"),
        "ckpt": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/noocr/checkpoints/0005000.pt"),
        "run_dir": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_noocr_5000/noocr"),
        "expect_text": True,
        "expect_ocr": False,
        "expect_weight": None,
    },
    "w001": {
        "config": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w001.yaml"),
        "ckpt": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w001/checkpoints/0005000.pt"),
        "run_dir": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w001"),
        "expect_text": True,
        "expect_ocr": True,
        "expect_weight": 0.01,
    },
    "w002": {
        "config": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w002.yaml"),
        "ckpt": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w002/checkpoints/0005000.pt"),
        "run_dir": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w002"),
        "expect_text": True,
        "expect_ocr": True,
        "expect_weight": 0.02,
    },
    "w003": {
        "config": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/configs/w003.yaml"),
        "ckpt": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w003/checkpoints/0005000.pt"),
        "run_dir": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/w003"),
        "expect_text": True,
        "expect_ocr": True,
        "expect_weight": 0.03,
    },
    "w005": {
        "config": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/configs/w005.yaml"),
        "ckpt": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/w005/checkpoints/0005000.pt"),
        "run_dir": Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_long_w005_5000_cache/w005"),
        "expect_text": True,
        "expect_ocr": True,
        "expect_weight": 0.05,
    },
}
val_manifest = Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_val.jsonl")

print("== old run files and val final rows ==")
for name, spec in runs.items():
    for key in ("config", "ckpt"):
        path = spec[key]
        print(f"{name}.{key}: exists={path.exists()} path={path}")
        if not path.exists():
            raise SystemExit(f"missing {name}.{key}: {path}")
    val_csv = spec["run_dir"] / "val.csv"
    print(f"{name}.val_csv: exists={val_csv.exists()} path={val_csv}")
    if not val_csv.exists():
        raise SystemExit(f"missing {name}.val_csv: {val_csv}")
    with val_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty val.csv: {val_csv}")
    print(f"{name}.val_final: {rows[-1]}")

rows = []
missing = []
with val_manifest.open("r", encoding="utf-8") as handle:
    for idx, line in enumerate(handle):
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        image_path = row.get("image_path") or row.get("path") or row.get("file_name") or row.get("image")
        if not image_path or not Path(str(image_path)).exists():
            missing.append((idx, image_path))
print(f"manifest rows = {len(rows)}")
print(f"missing images = {len(missing)}")
if len(rows) != 50 or missing:
    raise SystemExit("manifest check failed")

print("== checkpoint metadata ==")
for name, spec in runs.items():
    ckpt = torch.load(spec["ckpt"], map_location="cpu")
    steps = ckpt.get("steps")
    has_ema = "ema" in ckpt
    print(f"{name}: steps={steps}, has_ema={has_ema}, has_model={'model' in ckpt}")
    if int(steps) != 5000 or has_ema:
        raise SystemExit(f"checkpoint metadata mismatch for {name}")
    del ckpt
    gc.collect()

print("== config key fields ==")
for name, spec in runs.items():
    with spec["config"].open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    text_on = bool((cfg.get("text_conditioning") or {}).get("enabled", False))
    ocr_cfg = cfg.get("ocr_teacher_forcing_loss") or {}
    ocr_on = bool(ocr_cfg.get("enabled", False))
    weight = ocr_cfg.get("weight")
    text_hr = bool((cfg.get("text_hr") or {}).get("enabled", False))
    trainer = cfg.get("trainer") or {}
    hr_on = bool(trainer.get("hr_on", False))
    ocr_feature = bool((cfg.get("ocr_feature_loss") or {}).get("enabled", False))
    print(
        f"{name}: text_conditioning={text_on}, "
        f"ocr_teacher_forcing={ocr_on}, weight={weight}, "
        f"text_hr.enabled={text_hr}, trainer.hr_on={hr_on}, "
        f"ocr_feature_loss.enabled={ocr_feature}"
    )
    if text_on != spec["expect_text"] or ocr_on != spec["expect_ocr"]:
        raise SystemExit(f"config boolean mismatch for {name}")
    if spec["expect_weight"] is not None and abs(float(weight) - float(spec["expect_weight"])) > 1e-12:
        raise SystemExit(f"config OCR weight mismatch for {name}: {weight}")
    if text_hr or hr_on or ocr_feature:
        raise SystemExit(f"forbidden HR/OCR-feature setting enabled for {name}")
print("[OK] python precheck passed")
PY
  stage "[DONE] precheck"
}

run_eval() {
  local out_dir=$1
  local expected_total=$2
  local image_metric_source=$3
  shift 3
  local runs=("$@")

  "$PYTHON_BIN" scripts/stage1/ocr_debug/evaluate_setting_metrics.py \
    --runs-json "$out_dir/runs.json" \
    --manifest-jsonl "$VAL_MANIFEST" \
    --output-dir "$out_dir" \
    --step 5000 \
    --image-metric-source "$image_metric_source" \
    --device-backend "$DEVICE_BACKEND" \
    --device-id 0 \
    --mixed-precision "$MIXED_PRECISION" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --grid-samples "$GRID_SAMPLES" \
    --checkpoint-weight-key model \
    --thirdparty-ocr-backend none

  check_file "$out_dir/setting_metrics.csv"
  check_file "$out_dir/setting_metrics.md"
  check_file "$out_dir/paper_table.md"
  check_file "$out_dir/qual_grid_5000.png"
  check_file "$out_dir/ocr_predictions_deepseek.jsonl"
  check_file "$out_dir/tradeoff_psnr_vs_cer.png"
  check_file "$out_dir/compare_core_metrics.png"
  check_file "$out_dir/deepseek_eval/ocr_raw.jsonl"

  local raw_count
  local flat_count
  raw_count=$(wc -l < "$out_dir/deepseek_eval/ocr_raw.jsonl" | tr -d ' ')
  flat_count=$(wc -l < "$out_dir/ocr_predictions_deepseek.jsonl" | tr -d ' ')
  echo "DeepSeek OCR raw rows: $raw_count"
  echo "DeepSeek OCR flat rows: $flat_count"
  if [[ "$raw_count" -ne "$expected_total" || "$flat_count" -ne "$expected_total" ]]; then
    echo "[ERROR] expected $expected_total DeepSeek OCR rows" >&2
    exit 1
  fi

  copy_deepseek_recon_images "$out_dir/deepseek_eval/deepseek_ocr_tmp" "$out_dir/reconstruction" 50 "${runs[@]}"
  count_recon_images "$out_dir/reconstruction" "$expected_total" "$out_dir/reconstruction_count_summary.txt" "${runs[@]}"

  "$PYTHON_BIN" - "$out_dir/setting_metrics.csv" "$image_metric_source" <<'PY'
import csv
import sys

path, image_metric_source = sys.argv[1:3]
expected_source = "training val.csv" if image_metric_source == "train_val_csv" else "setting_eval reconstruction"
with open(path, "r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    run = row["run"]
    source = row.get("image_metric_source")
    psnr = float(row["psnr"])
    cer = row.get("deepseek_ocr_cer")
    third = row.get("thirdparty_ocr_backend", "")
    print(f"{run}: source={source}, psnr={psnr:.4f}, deepseek_cer={cer}, thirdparty={third!r}")
    if source != expected_source:
        raise SystemExit(f"wrong image metric source for {run}: {source}; expected {expected_source}")
    if image_metric_source == "train_val_csv" and 20.0 <= psnr < 30.0:
        raise SystemExit(f"post-hoc 20.x PSNR detected for {run}: {psnr}")
if image_metric_source == "setting_eval":
    print("[OK] post-hoc diagnostic metric source confirmed; post-hoc 20.x PSNR is allowed here")
else:
    print("[OK] metric source and PSNR range check passed")
PY
}

write_candidate_summary() {
  local out_dir=$1
  "$PYTHON_BIN" - "$out_dir/setting_metrics.csv" "$out_dir/candidate_summary.md" <<'PY'
import csv
import sys

metrics_path, out_path = sys.argv[1:3]
with open(metrics_path, "r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
by_run = {row["run"]: row for row in rows}

def f(row, key):
    try:
        value = row.get(key, "")
        return None if value == "" else float(value)
    except ValueError:
        return None

noocr = by_run.get("noocr", {})
noocr_psnr = f(noocr, "psnr")
noocr_ssim = f(noocr, "ssim")
trade_candidates = []
for row in rows:
    run = row["run"]
    if run in {"orig", "noocr", "w005"}:
        continue
    if row.get("ocr_loss") not in {"True", "true", "1"}:
        continue
    psnr = f(row, "psnr")
    ssim = f(row, "ssim")
    cer = f(row, "deepseek_ocr_cer")
    ned = f(row, "deepseek_ocr_ned") or 0.0
    if cer is None:
        continue
    if noocr_psnr is not None and psnr is not None and noocr_psnr - psnr > 0.3:
        continue
    if noocr_ssim is not None and ssim is not None and noocr_ssim - ssim > 0.005:
        continue
    trade_candidates.append((cer, -ned, run))

trade_candidates.sort()
tradeoff = trade_candidates[0][2] if trade_candidates else "noocr"
lines = [
    "# Candidate Summary",
    "",
    "- conservative candidate = noocr",
    f"- best trade-off candidate = {tradeoff}",
    "- OCR-aggressive candidate = w005",
    "",
    "Selection rule: trade-off candidate is the OCR-loss setting with the best DeepSeek CER among runs passing the noocr image-quality gate.",
]
with open(out_path, "w", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
}

run_phase_a() {
  stage "[START] phase A rebuild existing eval"
  safe_remove_phase_a_eval
  write_phase_a_runs_json
  run_eval "$PHASE_A_OUT" 300 train_val_csv "${OLD_RUN_ORDER[@]}"
  ln -sfn "$PHASE_A_OUT" "$PHASE_A_FINAL"
  echo "Phase A files:"
  echo "$PHASE_A_FINAL/paper_table.md"
  echo "$PHASE_A_FINAL/qual_grid_5000.png"
  echo "$PHASE_A_FINAL/setting_metrics.csv"
  stage "[DONE] phase A rebuild existing eval"
}

train_new_group() {
  local group=$1
  stage "[START] phase B train $group"
  cd "$REPO_ROOT"
  ASCEND_RT_VISIBLE_DEVICES="$ASCEND_RT_VISIBLE_DEVICES" \
  SAVE_ROOT="$WEIGHT_ROOT" \
  RUN_GROUPS="$group" \
  ITERS=5000 \
  GLOBAL_BATCH_SIZE=8 \
  LOG_EVERY=10 \
  VAL_EVERY=100 \
  CKPT_EVERY=500 \
  GRID_EVERY=500 \
  TEXT_FEATURE_CACHE="$TEXT_FEATURE_CACHE" \
  RUN_POSTPROCESS=1 \
  SKIP_EXISTING=0 \
  bash scripts/stage1/ocr_debug/run_ocr_recon_effect_sweep.sh

  check_file "$WEIGHT_ROOT/$group/checkpoints/0005000.pt"
  check_file "$WEIGHT_ROOT/$group/val.csv"
  check_file "$WEIGHT_ROOT/configs/$group.yaml"

  "$PYTHON_BIN" - "$WEIGHT_ROOT/$group/checkpoints/0005000.pt" "$group" <<'PY'
import sys
import torch

path, group = sys.argv[1:3]
ckpt = torch.load(path, map_location="cpu")
steps = ckpt.get("steps")
has_ema = "ema" in ckpt
print(f"{group}: steps={steps}, has_ema={has_ema}, has_model={'model' in ckpt}")
if int(steps) != 5000 or has_ema:
    raise SystemExit(f"checkpoint metadata mismatch for {group}")
PY
  stage "[DONE] phase B train $group"
}

run_phase_b() {
  train_new_group w0005
  train_new_group w0015
  train_new_group w0025
}

run_phase_c() {
  stage "[START] phase C eval all settings"
  if [[ -e "$PHASE_C_OUT" ]]; then
    echo "[ERROR] Phase C output already exists; refusing to overwrite: $PHASE_C_OUT" >&2
    exit 1
  fi
  write_phase_c_runs_json
  run_eval "$PHASE_C_OUT" 450 train_val_csv "${ALL_RUN_ORDER[@]}"
  write_candidate_summary "$PHASE_C_OUT"
  echo "Phase C files:"
  echo "$PHASE_C_OUT/paper_table.md"
  echo "$PHASE_C_OUT/qual_grid_5000.png"
  echo "$PHASE_C_OUT/setting_metrics.csv"
  stage "[DONE] phase C eval all settings"
}

run_phase_d() {
  if [[ "$RUN_PHASE_D_POSTHOC" != "1" ]]; then
    stage "[SKIP] phase D post-hoc diagnostic; set RUN_PHASE_D_POSTHOC=1 to enable"
    return
  fi

  stage "[START] phase D post-hoc diagnostic eval"
  if [[ -e "$PHASE_D_OUT" ]]; then
    echo "[ERROR] Phase D output already exists; refusing to overwrite: $PHASE_D_OUT" >&2
    exit 1
  fi
  check_file "$PHASE_C_OUT/runs.json"
  mkdir -p "$PHASE_D_OUT"
  cp "$PHASE_C_OUT/runs.json" "$PHASE_D_OUT/runs.json"
  run_eval "$PHASE_D_OUT" 450 setting_eval "${ALL_RUN_ORDER[@]}"
  write_candidate_summary "$PHASE_D_OUT"
  cat > "$PHASE_D_OUT/protocol_note.md" <<'EOF'
# Protocol Note

Phase C is for current reporting.
Phase D is for post-hoc diagnostic / future paper protocol check.
Do not mix the PSNR columns between the two.

- Current report table image metrics: training val.csv final row.
- Post-hoc diagnostic table image metrics: recomputed from checkpoints.
EOF
  echo "Current report table:"
  echo "$PHASE_C_OUT/paper_table.md"
  echo "Post-hoc diagnostic table:"
  echo "$PHASE_D_OUT/paper_table.md"
  echo "Phase C is for current reporting."
  echo "Phase D is for post-hoc diagnostic / future paper protocol check."
  echo "Do not mix the PSNR columns between the two."
  stage "[DONE] phase D post-hoc diagnostic eval"
}

run_precheck
run_phase_a
run_phase_b
run_phase_c
run_phase_d
