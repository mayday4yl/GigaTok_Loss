#!/usr/bin/env bash
# Source this file before stage-1 smoke or finetune commands.
# It only exports environment variables; it does not start training.

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _STAGE1_ENV_SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [[ -n "${ZSH_VERSION:-}" ]]; then
  _STAGE1_ENV_SCRIPT_PATH="${(%):-%x}"
else
  _STAGE1_ENV_SCRIPT_PATH="$0"
fi

_STAGE1_ENV_SCRIPT_DIR="$(cd "$(dirname "${_STAGE1_ENV_SCRIPT_PATH}")" && pwd)"
_STAGE1_ENV_REPO_ROOT="$(cd "${_STAGE1_ENV_SCRIPT_DIR}/../.." && pwd)"
_STAGE1_ENV_REPO_PARENT="$(cd "${_STAGE1_ENV_REPO_ROOT}/.." && pwd)"

export PROJECT_ROOT="${PROJECT_ROOT:-${_STAGE1_ENV_REPO_ROOT}}"
export PERSIST_ROOT="${PERSIST_ROOT:-${_STAGE1_ENV_REPO_PARENT}/gigatok_persist}"

export CKPT_DIR="${CKPT_DIR:-${PERSIST_ROOT}/checkpoints}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PERSIST_ROOT}/outputs}"
export DATA_ROOT="${DATA_ROOT:-${PERSIST_ROOT}/datasets}"

export TORCH_HOME="${TORCH_HOME:-${PERSIST_ROOT}/cache/torch}"
export HF_HOME="${HF_HOME:-${PERSIST_ROOT}/cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PERSIST_ROOT}/cache/xdg}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${PERSIST_ROOT}/cache/pip}"

export DINOV2_REPO_DIR="${DINOV2_REPO_DIR:-${TORCH_HOME}/hub/facebookresearch_dinov2_main}"
export DINOV2_EXPECTED_COMMIT="${DINOV2_EXPECTED_COMMIT:-4d83ffd4f4b39df112cef7ed7833de7ebf3a202e}"

export VQ_CKPT="${VQ_CKPT:-${CKPT_DIR}/VQ_BL256_dino_disc.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"

if [[ -z "${TORCH_RUN_PATH:-}" ]]; then
  if command -v torchrun >/dev/null 2>&1; then
    export TORCH_RUN_PATH="$(command -v torchrun)"
  else
    export TORCH_RUN_PATH="torchrun"
  fi
else
  export TORCH_RUN_PATH
fi

case ":${PYTHONPATH:-}:" in
  *":${PROJECT_ROOT}:"*) ;;
  *) export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" ;;
esac
