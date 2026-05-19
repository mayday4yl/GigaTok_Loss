#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/home/ma-user/work/GigaTok_hr/GigaTok_Loss}
BASE=${BASE:-/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000}
OUT_DIR=${OUT_DIR:-"$BASE/setting_eval_rebuilt"}
FINAL_LINK=${FINAL_LINK:-"$BASE/setting_eval_final"}
LOG=${LOG:-"$BASE/run_setting_eval_rebuilt.log"}
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

mkdir -p "$BASE"
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

write_runs_json() {
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

run_precheck() {
  stage "[START] precheck"
  cd "$REPO_ROOT"

  if ps -ef | grep '[e]valuate_setting_metrics.py' >/dev/null; then
    echo "[ERROR] another evaluate_setting_metrics.py process is already running" >&2
    ps -ef | grep '[e]valuate_setting_metrics.py' >&2 || true
    exit 1
  fi

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
manifest = Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_val.jsonl")
out_dir = Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/setting_eval_rebuilt")

print("== file existence ==")
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

print("== val.csv final rows ==")
for name, spec in runs.items():
    val_csv = spec["run_dir"] / "val.csv"
    with val_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty val.csv: {val_csv}")
    print(f"{name}: {rows[-1]}")

print("== manifest ==")
if not manifest.exists():
    raise SystemExit(f"missing manifest: {manifest}")
rows = []
missing = []
with manifest.open("r", encoding="utf-8") as handle:
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
if missing[:5]:
    print(f"first missing = {missing[:5]}")
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

print("== output directory safety ==")
print(f"setting_eval_rebuilt exists = {out_dir.exists()}")
if out_dir.exists():
    dangerous_existing = [
        out_dir / "deepseek_eval" / "ocr_raw.jsonl",
        out_dir / "thirdparty_eval" / "ocr_raw.jsonl",
    ]
    for path in dangerous_existing:
        print(f"existing file that evaluator would refresh: {path} exists={path.exists()}")
        if path.exists():
            raise SystemExit(
                "output dir contains existing OCR raw jsonl; stop to avoid implicit unlink/overwrite"
            )

print("[OK] precheck passed")
PY
  stage "[DONE] precheck"
}

run_eval() {
  stage "[START] write runs.json"
  write_runs_json
  stage "[DONE] write runs.json"

  cd "$REPO_ROOT"
  stage "[START] rebuild recon images"
  stage "[START] DeepSeek OCR"
  stage "[START] build paper table"
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
  stage "[DONE] evaluator"

  stage "[START] organize 300 reconstruction PNGs"
  check_file "$OUT_DIR/setting_metrics.csv"
  check_file "$OUT_DIR/setting_metrics.md"
  check_file "$OUT_DIR/paper_table.md"
  check_file "$OUT_DIR/qual_grid_5000.png"
  check_file "$OUT_DIR/ocr_predictions_deepseek.jsonl"
  check_file "$OUT_DIR/deepseek_eval/ocr_raw.jsonl"

  raw_count=$(wc -l < "$OUT_DIR/deepseek_eval/ocr_raw.jsonl" | tr -d ' ')
  flat_count=$(wc -l < "$OUT_DIR/ocr_predictions_deepseek.jsonl" | tr -d ' ')
  echo "DeepSeek OCR raw rows: $raw_count"
  echo "DeepSeek OCR flat rows: $flat_count"
  if [[ "$raw_count" -ne 300 || "$flat_count" -ne 300 ]]; then
    echo "[ERROR] expected 300 DeepSeek OCR rows" >&2
    exit 1
  fi

  copy_deepseek_recon_images "$OUT_DIR/deepseek_eval/deepseek_ocr_tmp" "$OUT_DIR/reconstruction"
  count_recon_images "$OUT_DIR/reconstruction" "$OUT_DIR/reconstruction_count_summary.txt"
  stage "[DONE] organize 300 reconstruction PNGs"

  stage "[START] final metric checks"
  "$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

path = Path("/home/ma-user/work/GigaTok_hr/gigatok_persist/outputs/ocr_w_sweep_setting_tune_5000/setting_eval_rebuilt/setting_metrics.csv")
expected = {
    "orig": 31.685,
    "noocr": 31.670,
    "w001": 31.735,
    "w002": 31.440,
    "w003": 31.142,
    "w005": 30.686,
}
with path.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
for row in rows:
    run = row["run"]
    source = row.get("image_metric_source")
    psnr = float(row["psnr"])
    cer = row.get("deepseek_ocr_cer")
    third = row.get("thirdparty_ocr_backend", "")
    print(f"{run}: source={source}, psnr={psnr:.4f}, deepseek_cer={cer}, thirdparty={third!r}")
    if source != "training val.csv":
        raise SystemExit(f"wrong image metric source for {run}: {source}")
    if run in expected and abs(psnr - expected[run]) > 0.05:
        raise SystemExit(f"PSNR mismatch for {run}: {psnr}")
    if 20.0 <= psnr < 30.0:
        raise SystemExit(f"post-hoc 20.x PSNR detected for {run}: {psnr}")
print("[OK] final metric checks passed")
PY

  echo "== paper_table.md =="
  cat "$OUT_DIR/paper_table.md"
  stage "[DONE] final metric checks"

  stage "[START] create final symlink"
  ln -sfn "$OUT_DIR" "$FINAL_LINK"
  stage "[DONE] setting_eval_rebuilt"
  echo "Final files:"
  echo "$FINAL_LINK/paper_table.md"
  echo "$FINAL_LINK/qual_grid_5000.png"
  echo "$FINAL_LINK/setting_metrics.csv"
}

run_precheck

if [[ "${PRECHECK_ONLY:-0}" == "1" ]]; then
  stage "[DONE] PRECHECK_ONLY=1"
  exit 0
fi

run_eval
