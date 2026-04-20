#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dev/cleanup_stage1_artifacts.sh [options]

Clean stage-1 bulky artifacts according to the disk-control policy.
Default mode is dry-run. Pass --execute to delete.

Options:
  --stage1-root PATH         Stage-1 run root. Defaults to $OUTPUT_DIR/stage1_runs.
  --persist-root PATH        Persist root. Defaults to $PERSIST_ROOT or ~/gigatok_persist.
  --output-root PATH         Output root. Defaults to $OUTPUT_DIR or $PERSIST_ROOT/outputs.
  --keep-full-resume N       Keep newest N full resume checkpoints. Default: 2.
  --keep-model-only N        Keep newest N model-only checkpoints. Default: 3.
  --max-eval-samples N       Max non-grid eval sample images to keep. Default: 32.
  --max-eval-dir-size-gb GB  Max eval directory size after pruning. Default: 1.
  --include-cache            Also consider safe cache temp directories. Default: false.
  --execute                  Actually delete. Default is dry-run.
  -h, --help                 Show this help.

Protected by policy:
  manifests, sha256 files, run_state, fixed val, latest/best/final checkpoints,
  current chunk, prefetch chunk, DINO pinned records, and start checkpoints.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

persist_root="${PERSIST_ROOT:-${HOME}/gigatok_persist}"
output_root="${OUTPUT_DIR:-}"
stage1_root=""
keep_full_resume=2
keep_model_only=3
max_eval_samples=32
max_eval_dir_size_gb=1
include_cache=0
execute=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage1-root)
      stage1_root="$2"
      shift 2
      ;;
    --persist-root)
      persist_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --keep-full-resume)
      keep_full_resume="$2"
      shift 2
      ;;
    --keep-model-only)
      keep_model_only="$2"
      shift 2
      ;;
    --max-eval-samples)
      max_eval_samples="$2"
      shift 2
      ;;
    --max-eval-dir-size-gb)
      max_eval_dir_size_gb="$2"
      shift 2
      ;;
    --include-cache)
      include_cache=1
      shift
      ;;
    --execute)
      execute=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ -z "${output_root}" ]]; then
  output_root="${persist_root}/outputs"
fi
if [[ -z "${stage1_root}" ]]; then
  stage1_root="${output_root}/stage1_runs"
fi

mode="DRY-RUN"
if [[ "${execute}" -eq 1 ]]; then
  mode="EXECUTE"
fi

echo "== cleanup_stage1_artifacts (${mode}) =="
echo "stage1_root: ${stage1_root}"
echo "persist_root: ${persist_root}"

if [[ ! -d "${stage1_root}" ]]; then
  echo "stage1_root missing; nothing to clean"
  exit 0
fi

safe_under_root() {
  local path="$1"
  local root="$2"
  local real_path
  local real_root
  real_path="$(cd "$(dirname "${path}")" 2>/dev/null && pwd)/$(basename "${path}")"
  real_root="$(cd "${root}" 2>/dev/null && pwd)"
  case "${real_path}" in
    "${real_root}"/*) return 0 ;;
    *) return 1 ;;
  esac
}

delete_path() {
  local path="$1"
  if [[ ! -e "${path}" && ! -L "${path}" ]]; then
    return 0
  fi
  if ! safe_under_root "${path}" "${stage1_root}"; then
    echo "skip outside stage1_root: ${path}" >&2
    return 0
  fi
  if [[ "${execute}" -eq 1 ]]; then
    rm -rf "${path}"
    log_cleanup "delete" "${path}"
    echo "delete ${path}"
  else
    echo "would delete ${path}"
  fi
}

log_cleanup() {
  local action="$1"
  local path="$2"
  local log_dir="${stage1_root}/run_state"
  local log_file="${log_dir}/cleanup_history.jsonl"
  mkdir -p "${log_dir}"
  python3 - "$action" "$path" "$log_file" <<'PY'
import json
import sys
from datetime import datetime

action, path, log_file = sys.argv[1], sys.argv[2], sys.argv[3]
record = {
    "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "action": action,
    "path": path,
}
with open(log_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

echo
echo "== protected paths =="
for protected in \
  "${stage1_root}/manifests" \
  "${stage1_root}/run_state/run_state.json" \
  "${stage1_root}/chunks/current" \
  "${stage1_root}/chunks/prefetch" \
  "${stage1_root}/val" \
  "${stage1_root}/checkpoints/full_resume/latest.pt" \
  "${stage1_root}/checkpoints/model_only/latest.pt" \
  "${stage1_root}/checkpoints/model_only/best.pt" \
  "${stage1_root}/checkpoints/model_only/final.pt"; do
  echo "protect ${protected}"
done

echo
echo "== enforce single chunk lifecycle =="
chunks_root="${stage1_root}/chunks"
if [[ -d "${chunks_root}" ]]; then
  while IFS= read -r -d '' item; do
    name="$(basename "${item}")"
    case "${name}" in
      current|prefetch)
        echo "protect ${item}"
        ;;
      *)
        delete_path "${item}"
        ;;
    esac
  done < <(find "${chunks_root}" -mindepth 1 -maxdepth 1 -print0)
else
  echo "missing ${chunks_root}"
fi

echo
echo "== prune eval outputs =="
eval_root="${stage1_root}/eval"
if [[ -d "${eval_root}" ]]; then
  prune_args=(
    "--eval-root" "${eval_root}"
    "--max-samples" "${max_eval_samples}"
    "--max-dir-size-gb" "${max_eval_dir_size_gb}"
  )
  if [[ "${execute}" -eq 1 ]]; then
    prune_args+=("--execute")
  fi
  python3 "${script_dir}/prune_stage1_eval_outputs.py" "${prune_args[@]}"
else
  echo "missing ${eval_root}"
fi

echo
echo "== rotate checkpoints =="
rotate_args=(
  "--full-resume-dir" "${stage1_root}/checkpoints/full_resume"
  "--model-only-dir" "${stage1_root}/checkpoints/model_only"
  "--keep-full" "${keep_full_resume}"
  "--keep-model" "${keep_model_only}"
)
if [[ "${execute}" -eq 1 ]]; then
  rotate_args+=("--execute")
fi
python3 "${script_dir}/rotate_stage1_checkpoints.py" "${rotate_args[@]}"

if [[ "${include_cache}" -eq 1 ]]; then
  echo
  echo "== cache cleanup candidates =="
  for cache_dir in \
    "${HF_HOME:-${persist_root}/cache/huggingface}/datasets/downloads" \
    "${PIP_CACHE_DIR:-${persist_root}/cache/pip}"; do
    if [[ -e "${cache_dir}" ]]; then
      if ! safe_under_root "${cache_dir}" "${persist_root}"; then
        echo "skip cache outside persist_root: ${cache_dir}" >&2
        continue
      fi
      if [[ "${execute}" -eq 1 ]]; then
        rm -rf "${cache_dir}"
        log_cleanup "delete_cache" "${cache_dir}"
        echo "delete ${cache_dir}"
      else
        echo "would delete ${cache_dir}"
      fi
    else
      echo "missing ${cache_dir}"
    fi
  done
fi

echo
echo "cleanup complete (${mode})"
