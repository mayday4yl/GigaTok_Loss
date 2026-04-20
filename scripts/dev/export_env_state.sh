#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dev/export_env_state.sh [--output-dir DIR] [--manifest-root DIR]

Export stage-1 migration state from the current server. The script records
environment variables, git state, Python package state, checkpoint hashes,
DINO state, manifest metadata, and disk usage. It does not copy image data.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

OUTPUT_DIR_ARG=""
MANIFEST_ROOT_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR_ARG="$2"
      shift 2
      ;;
    --manifest-root)
      MANIFEST_ROOT_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source "${REPO_ROOT}/scripts/dev/source_stage1_env.sh"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
STATE_DIR="${OUTPUT_DIR_ARG:-${PERSIST_ROOT}/migration_state/${STAMP}}"
MANIFEST_ROOT="${MANIFEST_ROOT_ARG:-${OUTPUT_DIR}/textatlas_stage1_pilot/manifest}"

mkdir -p "${STATE_DIR}"

write_cmd() {
  local outfile="$1"
  shift
  {
    echo "\$ $*"
    "$@"
  } > "${outfile}" 2>&1 || true
}

write_shell() {
  local outfile="$1"
  shift
  {
    echo "\$ $*"
    bash -lc "$*"
  } > "${outfile}" 2>&1 || true
}

{
  echo "exported_at_utc=${STAMP}"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "PERSIST_ROOT=${PERSIST_ROOT}"
  echo "CKPT_DIR=${CKPT_DIR}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "DATA_ROOT=${DATA_ROOT}"
  echo "TORCH_HOME=${TORCH_HOME}"
  echo "HF_HOME=${HF_HOME}"
  echo "XDG_CACHE_HOME=${XDG_CACHE_HOME}"
  echo "DINOV2_REPO_DIR=${DINOV2_REPO_DIR}"
  echo "DINOV2_EXPECTED_COMMIT=${DINOV2_EXPECTED_COMMIT}"
  echo "VQ_CKPT=${VQ_CKPT}"
  echo "MANIFEST_ROOT=${MANIFEST_ROOT}"
} > "${STATE_DIR}/stage1_paths.env"

write_shell "${STATE_DIR}/git_state.txt" \
  "git status --short --branch && git rev-parse HEAD && git log --oneline --decorate -10 && git remote -v"
write_shell "${STATE_DIR}/python_state.txt" \
  "which python && python --version && python - <<'PY'
import sys
print('executable', sys.executable)
for name in ['torch', 'torchvision', 'datasets', 'huggingface_hub', 'PIL']:
    try:
        module = __import__(name)
        print(name, getattr(module, '__version__', 'unknown'))
    except Exception as exc:
        print(name, 'MISSING', repr(exc))
PY"
write_cmd "${STATE_DIR}/pip_freeze.txt" python -m pip freeze
if command -v conda >/dev/null 2>&1; then
  write_cmd "${STATE_DIR}/conda_info.txt" conda info
  write_cmd "${STATE_DIR}/conda_env_export.yml" conda env export --no-builds
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  write_cmd "${STATE_DIR}/nvidia_smi.txt" nvidia-smi
fi
write_shell "${STATE_DIR}/disk_state.txt" \
  "df -h '${PROJECT_ROOT}' '${PERSIST_ROOT}' 2>/dev/null; du -sh '${PROJECT_ROOT}' '${PERSIST_ROOT}'/* 2>/dev/null"

{
  echo "VQ_CKPT=${VQ_CKPT}"
  if [[ -f "${VQ_CKPT}" ]]; then
    sha256sum "${VQ_CKPT}"
  else
    echo "MISSING ${VQ_CKPT}"
  fi
} > "${STATE_DIR}/checkpoint_sha256.txt"

{
  echo "DINOV2_REPO_DIR=${DINOV2_REPO_DIR}"
  echo "DINOV2_EXPECTED_COMMIT=${DINOV2_EXPECTED_COMMIT}"
  if [[ -d "${DINOV2_REPO_DIR}/.git" ]]; then
    git -C "${DINOV2_REPO_DIR}" rev-parse HEAD
    git -C "${DINOV2_REPO_DIR}" status --short
  else
    echo "MISSING ${DINOV2_REPO_DIR}"
  fi
  for path in \
    "${TORCH_HOME}/hub/checkpoints/dinov2_vitb14_pretrain.pth" \
    "${TORCH_HOME}/hub/checkpoints/dino_deitsmall16_pretrain.pth"; do
    if [[ -f "${path}" ]]; then
      sha256sum "${path}"
    else
      echo "MISSING ${path}"
    fi
  done
} > "${STATE_DIR}/dino_state.txt"

{
  echo "MANIFEST_ROOT=${MANIFEST_ROOT}"
  if [[ -d "${MANIFEST_ROOT}" ]]; then
    find "${MANIFEST_ROOT}" -maxdepth 1 -type f -print | sort
    echo
    echo "sha256 files:"
    cat "${MANIFEST_ROOT}/manifest.sha256" 2>/dev/null || true
    cat "${MANIFEST_ROOT}/image_paths.sha256" 2>/dev/null || true
  else
    echo "MISSING ${MANIFEST_ROOT}"
  fi
} > "${STATE_DIR}/manifest_state.txt"

if [[ -d "${MANIFEST_ROOT}" ]]; then
  mkdir -p "${STATE_DIR}/manifest_metadata"
  for file in \
    counts.json \
    build_config.json \
    manifest.sha256 \
    image_paths.sha256 \
    bad_rows.jsonl; do
    if [[ -f "${MANIFEST_ROOT}/${file}" ]]; then
      cp "${MANIFEST_ROOT}/${file}" "${STATE_DIR}/manifest_metadata/${file}"
    fi
  done
fi

cat > "${STATE_DIR}/README.txt" <<EOF
Stage-1 migration state export.

This directory records reproducibility metadata only. It does not include
materialized images, checkpoints, or Hugging Face/DINO cache contents.

Suggested transfer roots:
- code: git clone/fetch this repository at the recorded commit
- persistent data: ${PERSIST_ROOT}
- pilot manifest: ${MANIFEST_ROOT}
- pilot images: see build_config.json image_root or rich manifest image_path values
EOF

echo "[export-env-state] wrote ${STATE_DIR}"
