#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/scripts/dev/source_stage1_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/scripts/dev/source_stage1_env.sh"
fi

GPUS="${GPUS:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
PORT="${PORT:-55565}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
TORCH_RUN_PATH="${TORCH_RUN_PATH:-torchrun}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

"${TORCH_RUN_PATH}" \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${PORT}" \
  tokenizer/tokenizer_image/vq/vq_train.py \
  --device-backend npu \
  "$@"
