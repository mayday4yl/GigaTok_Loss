#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/source_stage1_env.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DINO1_SHA256_FILE="${DINO1_SHA256_FILE:-${PERSIST_ROOT}/versions/dino_deitsmall16_pretrain.sha256}"

print_commit_context() {
  local actual="${1:-unknown}"
  echo "[dino-check] expected commit: ${DINOV2_EXPECTED_COMMIT}" >&2
  echo "[dino-check] actual commit:   ${actual}" >&2
}

fail() {
  local message="$1"
  local actual="${2:-unknown}"
  echo "[dino-check] ERROR: ${message}" >&2
  print_commit_context "${actual}"
  exit 1
}

echo "[dino-check] DINOV2_REPO_DIR=${DINOV2_REPO_DIR}"

if [[ -z "${DINOV2_REPO_DIR}" ]]; then
  fail "DINOV2_REPO_DIR is empty" "unknown"
fi

if [[ ! -d "${DINOV2_REPO_DIR}" ]]; then
  fail "DINOv2 repo directory does not exist: ${DINOV2_REPO_DIR}" "missing"
fi

actual_commit="$(git -C "${DINOV2_REPO_DIR}" rev-parse HEAD 2>/dev/null || true)"
if [[ -z "${actual_commit}" ]]; then
  fail "failed to read DINOv2 repo HEAD" "unknown"
fi

echo "[dino-check] expected commit: ${DINOV2_EXPECTED_COMMIT}"
echo "[dino-check] actual commit:   ${actual_commit}"

if [[ "${actual_commit}" != "${DINOV2_EXPECTED_COMMIT}" ]]; then
  fail "DINOv2 commit mismatch" "${actual_commit}"
fi

git_status="$(git -C "${DINOV2_REPO_DIR}" status --short 2>/dev/null || true)"
if [[ -n "${git_status}" ]]; then
  echo "[dino-check] git status:" >&2
  printf "%s\n" "${git_status}" >&2
  fail "DINOv2 repo is not clean" "${actual_commit}"
fi
echo "[dino-check] git status: clean"

if [[ ! -f "${DINO1_SHA256_FILE}" ]]; then
  fail "DINOv1 checksum file does not exist: ${DINO1_SHA256_FILE}" "${actual_commit}"
fi

if ! sha256sum -c "${DINO1_SHA256_FILE}"; then
  fail "DINOv1 dino-disc checkpoint checksum failed" "${actual_commit}"
fi

export DINOV2_REPO_DIR
if ! "${PYTHON_BIN}" - <<'PY'
import os
import torch

repo = os.environ["DINOV2_REPO_DIR"]
model = torch.hub.load(repo, "dinov2_vitb14", source="local")
print(f"[dino-check] embed_dim: {model.embed_dim}")
if model.embed_dim != 768:
    raise SystemExit(f"[dino-check] ERROR: expected embed_dim=768, got {model.embed_dim}")
print("[dino-check] local DINOv2 torch.hub.load passed")
PY
then
  fail "DINOv2 local torch.hub.load failed" "${actual_commit}"
fi

echo "[dino-check] DINO dependency check passed"
