#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-gigatok-hr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.9}"
INSTALL_TORCH="${INSTALL_TORCH:-auto}"
INSTALL_TF="${INSTALL_TF:-0}"
PERSIST_ROOT="${PERSIST_ROOT:-${HOME}/gigatok_persist}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[stage1-env] conda not found. Please load conda first." >&2
  exit 2
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
  echo "[stage1-env] creating conda env ${CONDA_ENV_NAME} with python=${PYTHON_VERSION}"
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
else
  echo "[stage1-env] conda env ${CONDA_ENV_NAME} already exists"
fi

conda activate "${CONDA_ENV_NAME}"

mkdir -p \
  "${PERSIST_ROOT}/checkpoints" \
  "${PERSIST_ROOT}/outputs" \
  "${PERSIST_ROOT}/datasets" \
  "${PERSIST_ROOT}/cache/torch" \
  "${PERSIST_ROOT}/cache/huggingface" \
  "${PERSIST_ROOT}/cache/xdg"

export TORCH_HOME="${PERSIST_ROOT}/cache/torch"
export HF_HOME="${PERSIST_ROOT}/cache/huggingface"
export XDG_CACHE_HOME="${PERSIST_ROOT}/cache/xdg"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

install_torch() {
  echo "[stage1-env] installing PyTorch with CUDA 12.1"
  conda install -y \
    pytorch==2.1.2 \
    torchvision==0.16.2 \
    pytorch-cuda=12.1 \
    -c pytorch \
    -c nvidia
}

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  install_torch
elif [[ "${INSTALL_TORCH}" == "auto" ]]; then
  if ! python - <<'PY' >/dev/null 2>&1
import torch
print(torch.__version__)
PY
  then
    install_torch
  else
    echo "[stage1-env] existing torch detected; skip torch install"
  fi
else
  echo "[stage1-env] INSTALL_TORCH=${INSTALL_TORCH}; skip torch install"
fi

python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  numpy==1.23.5 \
  scipy \
  scikit-image \
  pyyaml \
  timm==0.9.12 \
  einops \
  wandb \
  matplotlib \
  tqdm \
  requests \
  pillow \
  pandas \
  gdown \
  transformers \
  open_clip_torch

if [[ "${INSTALL_TF}" == "1" ]]; then
  python -m pip install "tensorflow-cpu==2.12.*"
fi

python - <<'PY'
import torch
import torchvision
import timm
import yaml
import einops
import skimage
import scipy
from PIL import Image

print("[stage1-env] python import check passed")
print("[stage1-env] torch:", torch.__version__)
print("[stage1-env] torch cuda:", torch.version.cuda)
print("[stage1-env] cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[stage1-env] gpu:", torch.cuda.get_device_name(0))
PY

cat <<EOF

[stage1-env] done.
Run this in future shells:

conda activate ${CONDA_ENV_NAME}
export PROJECT_ROOT=${REPO_ROOT}
export PERSIST_ROOT=${PERSIST_ROOT}
export CKPT_DIR=\$PERSIST_ROOT/checkpoints
export OUTPUT_DIR=\$PERSIST_ROOT/outputs
export DATA_ROOT=\$PERSIST_ROOT/datasets
export TORCH_HOME=\$PERSIST_ROOT/cache/torch
export HF_HOME=\$PERSIST_ROOT/cache/huggingface
export XDG_CACHE_HOME=\$PERSIST_ROOT/cache/xdg
export PYTHONPATH=\$PROJECT_ROOT:\${PYTHONPATH:-}
export TORCH_RUN_PATH=\$(which torchrun)
export CUDA_VISIBLE_DEVICES=0
cd \$PROJECT_ROOT

EOF
