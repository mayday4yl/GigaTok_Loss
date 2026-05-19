#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/ma-user/work/GigaTok_hr/GigaTok_Loss_two_ablation}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCH_HOME="${TORCH_HOME:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch}"
export DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-/home/ma-user/work/GigaTok_hr/gigatok_persist/cache/torch/hub/facebookresearch_dinov2_main}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"

PERSIST="${PERSIST:-/home/ma-user/work/GigaTok_hr/gigatok_persist}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-$PERSIST/outputs/ocr_box_gate/batch_probe_$STAMP}"
CONFIG="${CONFIG:-$PERSIST/outputs/ocr_box_gate/pipeline_runs/ocr_box_gate_w001_v1_v2_eval_2500/configs/ocr_box_gate_v2_conf_proxy.yaml}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$PERSIST/outputs/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_train.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$PERSIST/outputs/text_hr_multi_debug/manifests/cleantextsynth_ocr_readable_50_val.jsonl}"
VQ_CKPT="${VQ_CKPT:-$PERSIST/checkpoints/VQ_BL256_dino_disc.pt}"
TEXT_FEATURE_CACHE="${TEXT_FEATURE_CACHE:-$PERSIST/outputs/text_feature_cache/readable50_glyph_mapped/text_features.pt}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
DEVICE_BACKEND="${DEVICE_BACKEND:-npu}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
PROBE_STEPS="${PROBE_STEPS:-5}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_EVERY="${LOG_EVERY:-1}"

if [[ "$#" -gt 0 ]]; then
  BATCHES=("$@")
else
  BATCHES=(8 16 32 48 64)
fi

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/batch_probe_summary.tsv"
printf "global_batch_size\tper_npu_batch\tstatus\tsteps_per_sec\tlog\n" > "$SUMMARY"

log_msg() {
  echo "[$(date '+%F %T')] $*" | tee -a "$OUT_ROOT/probe.log"
}

log_msg "OUT_ROOT=$OUT_ROOT"
log_msg "SUMMARY=$SUMMARY"
npu-smi info | sed -n "1,70p" >> "$OUT_ROOT/probe.log" || true

for gbs in "${BATCHES[@]}"; do
  per="NA"
  if [[ "$NPROC_PER_NODE" -gt 0 ]]; then
    per=$((gbs / NPROC_PER_NODE))
  fi
  name="gbs_${gbs}"
  log="$OUT_ROOT/${name}.log"
  log_msg "testing global_batch_size=$gbs per_npu_batch=$per"
  set +e
  timeout "$TIMEOUT_SEC" torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" \
    tokenizer/tokenizer_image/vq/vq_train.py \
    --model-config "$CONFIG" \
    --dataset textatlas_image_text \
    --data-path "$TRAIN_MANIFEST" \
    --json-path "$TRAIN_MANIFEST" \
    --val-json-path "$VAL_MANIFEST" \
    --save-path "$OUT_ROOT" \
    --sub-exp-dir "$name" \
    --vq-ckpt "$VQ_CKPT" \
    --finetune \
    --iterations "$PROBE_STEPS" \
    --early-stop-iter "$PROBE_STEPS" \
    --global-batch-size "$gbs" \
    --global-seed 0 \
    --max-images 0 \
    --num-workers "$NUM_WORKERS" \
    --mixed-precision "$MIXED_PRECISION" \
    --device-backend "$DEVICE_BACKEND" \
    --log-every "$LOG_EVERY" \
    --val-every 999999 \
    --val-max-images 0 \
    --eval-batch-size 4 \
    --val-num-workers 1 \
    --ckpt-every 999999 \
    --no-wandb \
    --text-feature-cache "$TEXT_FEATURE_CACHE" \
    > "$log" 2>&1
  rc=$?
  set -e

  if [[ "$rc" == "0" ]]; then
    sps="$(grep -oE 'Train Steps/Sec: [0-9.]+' "$log" | tail -n 1 | awk '{print $3}')"
    sps="${sps:-NA}"
    printf "%s\t%s\tOK\t%s\t%s\n" "$gbs" "$per" "$sps" "$log" >> "$SUMMARY"
    log_msg "OK global_batch_size=$gbs steps_per_sec=$sps"
  else
    printf "%s\t%s\tFAIL(rc=%s)\tNA\t%s\n" "$gbs" "$per" "$rc" "$log" >> "$SUMMARY"
    log_msg "FAIL global_batch_size=$gbs rc=$rc"
    grep -Ei "out of memory|oom|memory|hbm|error|runtimeerror|traceback" "$log" | tail -n 100 > "$OUT_ROOT/${name}_error_tail.txt" || true
    break
  fi
  npu-smi info | sed -n "1,70p" >> "$OUT_ROOT/probe.log" || true
  sleep 5
done

log_msg "DONE"
cat "$SUMMARY"
