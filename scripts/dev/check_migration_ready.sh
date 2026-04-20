#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dev/check_migration_ready.sh [options]

Options:
  --manifest-root DIR              Manifest directory to validate.
  --expected-train-per-subset N    Default: 10000.
  --expected-val-per-subset N      Default: 500.
  --max-image-checks N             Passed to check_textatlas_manifest.py. Default: 0.
  --forbid-prefix PREFIX           Fail if manifest files still contain this old path prefix.
  --run-dino-preflight             Also run scripts/dev/check_dino_deps.sh.

This script checks migration readiness on the new server. It does not train.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

EXPECTED_TRAIN=10000
EXPECTED_VAL=500
MAX_IMAGE_CHECKS=0
MANIFEST_ROOT_ARG=""
FORBID_PREFIX=""
RUN_DINO_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest-root)
      MANIFEST_ROOT_ARG="$2"
      shift 2
      ;;
    --expected-train-per-subset)
      EXPECTED_TRAIN="$2"
      shift 2
      ;;
    --expected-val-per-subset)
      EXPECTED_VAL="$2"
      shift 2
      ;;
    --max-image-checks)
      MAX_IMAGE_CHECKS="$2"
      shift 2
      ;;
    --forbid-prefix)
      FORBID_PREFIX="$2"
      shift 2
      ;;
    --run-dino-preflight)
      RUN_DINO_PREFLIGHT=1
      shift
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

MANIFEST_ROOT="${MANIFEST_ROOT_ARG:-${OUTPUT_DIR}/textatlas_stage1_pilot/manifest}"
STATUS=0

fail() {
  echo "[migration-check] ERROR: $*" >&2
  STATUS=1
}

note() {
  echo "[migration-check] $*"
}

note "PROJECT_ROOT=${PROJECT_ROOT}"
note "PERSIST_ROOT=${PERSIST_ROOT}"
note "MANIFEST_ROOT=${MANIFEST_ROOT}"
note "git_head=$(git rev-parse HEAD)"

if [[ -n "$(git status --short)" ]]; then
  fail "git working tree is not clean"
fi

for path in \
  "${REPO_ROOT}/scripts/stage1/build_textatlas_image_manifest.py" \
  "${REPO_ROOT}/scripts/stage1/check_textatlas_manifest.py" \
  "${REPO_ROOT}/scripts/dev/source_stage1_env.sh"; do
  [[ -f "${path}" ]] || fail "required file missing: ${path}"
done

python -m py_compile \
  scripts/stage1/build_textatlas_image_manifest.py \
  scripts/stage1/check_textatlas_manifest.py \
  scripts/dev/rewrite_manifest_paths.py || STATUS=1

python - <<'PY' || STATUS=1
import sys
print("[migration-check] python", sys.executable)
for name in ("torch", "PIL"):
    try:
        module = __import__(name)
        print("[migration-check]", name, getattr(module, "__version__", "ok"))
    except Exception as exc:
        raise SystemExit(f"missing required python dependency {name}: {exc!r}")
PY

if [[ ! -f "${VQ_CKPT}" ]]; then
  fail "VQ_CKPT missing: ${VQ_CKPT}"
else
  note "VQ_CKPT present: ${VQ_CKPT}"
fi

if [[ ! -d "${MANIFEST_ROOT}" ]]; then
  fail "manifest root missing: ${MANIFEST_ROOT}"
else
  if [[ -n "${FORBID_PREFIX}" ]]; then
    if grep -R -F -l "${FORBID_PREFIX}" \
      "${MANIFEST_ROOT}/train_image_paths.json" \
      "${MANIFEST_ROOT}/val_image_paths.json" \
      "${MANIFEST_ROOT}/train_manifest.jsonl" \
      "${MANIFEST_ROOT}/val_manifest.jsonl" \
      "${MANIFEST_ROOT}/build_config.json" >/tmp/gigatok_migration_forbidden_paths.txt 2>/dev/null; then
      fail "manifest still contains forbidden old prefix ${FORBID_PREFIX}; see /tmp/gigatok_migration_forbidden_paths.txt"
    fi
  fi

  python scripts/stage1/check_textatlas_manifest.py \
    --manifest-root "${MANIFEST_ROOT}" \
    --expected-train-per-subset "${EXPECTED_TRAIN}" \
    --expected-val-per-subset "${EXPECTED_VAL}" \
    --max-image-checks "${MAX_IMAGE_CHECKS}" || STATUS=1
fi

if [[ "${RUN_DINO_PREFLIGHT}" == "1" ]]; then
  bash scripts/dev/check_dino_deps.sh || STATUS=1
else
  note "skip DINO load preflight; pass --run-dino-preflight to enable"
fi

if [[ "${STATUS}" -eq 0 ]]; then
  note "migration readiness check passed"
else
  fail "migration readiness check failed"
fi

exit "${STATUS}"
