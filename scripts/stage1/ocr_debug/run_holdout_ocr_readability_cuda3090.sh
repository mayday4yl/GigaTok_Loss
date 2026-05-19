#!/usr/bin/env bash
set -euo pipefail

PERSIST=/public/sdc/yl_GigaTok_Loss/gigatok_persist
REPO=/data/duoduo_25/yl_GigaTok_Loss/repo/GigaTok_Loss_two_ablation
EVAL_DIR=${PERSIST}/outputs/eval_unified_realworld40_5k/holdout_200_full_20260518_032938
MANIFEST=${PERSIST}/outputs/text_rich_realworld_upweighted_manifests/holdout_200_realworld40_text_rich.newserver.jsonl
DEEPSEEK=${PERSIST}/models/DeepSeek-OCR
STAMP=${OCR_READABILITY_STAMP:-$(date +%Y%m%d_%H%M%S)}
OUT=${EVAL_DIR}/ocr_readability_cuda3090_${STAMP}
SMOKE=${OUT}/smoke
FULL=${OUT}/full
STATE=${OUT}/ocr_readability_pipeline.state
N_SHARDS=${OCR_READABILITY_N_SHARDS:-8}

mkdir -p "${SMOKE}" "${FULL}"
echo "STARTED $(date -Is)" > "${STATE}"
echo "${OUT}" > "${EVAL_DIR}/ocr_readability_cuda3090_latest_path.txt"

cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export HF_HOME=${PERSIST}/cache/huggingface
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_HUB_CACHE=${HF_HOME}/hub
export TORCH_HOME=${PERSIST}/cache/torch
export TOKENIZERS_PARALLELISM=false

PY=/data/duoduo_25/anaconda3/envs/RiemDiff_img/bin/python
COMMON=(
  scripts/stage1/ocr_debug/evaluate_saved_recon_ocr_readability.py
  --eval-dir "${EVAL_DIR}"
  --manifest-jsonl "${MANIFEST}"
  --runs main_best main_last baseline_last
  --include-gt
  --device-backend cuda
  --ocr-backend deepseek_ocr
  --deepseek-ocr-model "${DEEPSEEK}"
  --deepseek-ocr-attn-implementation eager
  --deepseek-ocr-dtype bf16
  --deepseek-ocr-max-new-tokens 1024
)

echo "SMOKE_STARTED $(date -Is)" | tee -a "${STATE}"
CUDA_VISIBLE_DEVICES=0 "${PY}" "${COMMON[@]}" \
  --output-dir "${SMOKE}" \
  --max-images 10 \
  --sample-seed 0 \
  --shard-index 0 \
  --num-shards 1 \
  --device-id 0 \
  > "${SMOKE}/smoke.log" 2>&1
"${PY}" "${COMMON[@]}" --output-dir "${SMOKE}" --summarize-only \
  > "${SMOKE}/summarize.log" 2>&1
echo "SMOKE_DONE $(date -Is)" | tee -a "${STATE}"

echo "FULL_STARTED $(date -Is)" | tee -a "${STATE}"
for shard in $(seq 0 $((N_SHARDS - 1))); do
  shard_padded=$(printf "%02d" "${shard}")
  (
    set +e
    echo "RUNNING $(date -Is)" > "${FULL}/shard_${shard_padded}.state"
    CUDA_VISIBLE_DEVICES="${shard}" "${PY}" "${COMMON[@]}" \
      --output-dir "${FULL}" \
      --shard-index "${shard}" \
      --num-shards "${N_SHARDS}" \
      --device-id 0 \
      > "${FULL}/shard_${shard_padded}.log" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then
      echo "DONE $(date -Is)" > "${FULL}/shard_${shard_padded}.state"
    else
      echo "FAILED status=${status} $(date -Is)" > "${FULL}/shard_${shard_padded}.state"
    fi
    exit ${status}
  ) &
  echo $! > "${FULL}/shard_${shard_padded}.pid"
done

failed=0
for shard in $(seq 0 $((N_SHARDS - 1))); do
  shard_padded=$(printf "%02d" "${shard}")
  pid=$(cat "${FULL}/shard_${shard_padded}.pid")
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ ${failed} -ne 0 ]]; then
  echo "FULL_FAILED $(date -Is)" | tee -a "${STATE}"
  exit 1
fi

"${PY}" "${COMMON[@]}" --output-dir "${FULL}" --summarize-only \
  > "${FULL}/summarize.log" 2>&1
echo "DONE $(date -Is)" | tee -a "${STATE}"
