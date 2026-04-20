#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dev/check_disk_budget.sh [options]

Print stage-1 disk usage and check configurable disk watermarks.

Options:
  --persist-root PATH        Persist root. Defaults to $PERSIST_ROOT or ~/gigatok_persist.
  --output-root PATH         Output root. Defaults to $OUTPUT_DIR or $PERSIST_ROOT/outputs.
  --stage1-root PATH         Stage-1 run root. Defaults to $OUTPUT_ROOT/stage1_runs.
  --min-free-gb GB           Warning free-space floor. Default: 350.
  --max-used-pct PCT         Warning used-percent ceiling. Default: 75.
  --hard-min-free-gb GB      Hard free-space floor. Default: 200.
  --hard-max-used-pct PCT    Hard used-percent ceiling. Default: 85.
  --emergency-min-free-gb GB Emergency free-space floor. Default: 100.
  --emergency-max-used-pct PCT
                              Emergency used-percent ceiling. Default: 90.
  --fail-on-warning          Exit non-zero on warning status.
  -h, --help                 Show this help.

Exit codes:
  0 normal or warning without --fail-on-warning
  1 warning with --fail-on-warning
  2 hard limit
  3 emergency limit
EOF
}

persist_root="${PERSIST_ROOT:-${HOME}/gigatok_persist}"
output_root="${OUTPUT_DIR:-}"
stage1_root=""
min_free_gb=350
max_used_pct=75
hard_min_free_gb=200
hard_max_used_pct=85
emergency_min_free_gb=100
emergency_max_used_pct=90
fail_on_warning=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --persist-root)
      persist_root="$2"
      shift 2
      ;;
    --output-root)
      output_root="$2"
      shift 2
      ;;
    --stage1-root)
      stage1_root="$2"
      shift 2
      ;;
    --min-free-gb)
      min_free_gb="$2"
      shift 2
      ;;
    --max-used-pct)
      max_used_pct="$2"
      shift 2
      ;;
    --hard-min-free-gb)
      hard_min_free_gb="$2"
      shift 2
      ;;
    --hard-max-used-pct)
      hard_max_used_pct="$2"
      shift 2
      ;;
    --emergency-min-free-gb)
      emergency_min_free_gb="$2"
      shift 2
      ;;
    --emergency-max-used-pct)
      emergency_max_used_pct="$2"
      shift 2
      ;;
    --fail-on-warning)
      fail_on_warning=1
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

if [[ -d "${stage1_root}" ]]; then
  probe_path="${stage1_root}"
elif [[ -d "${output_root}" ]]; then
  probe_path="${output_root}"
elif [[ -d "${persist_root}" ]]; then
  probe_path="${persist_root}"
else
  probe_path="$(pwd)"
fi

df_line="$(df -Pk "${probe_path}" | awk 'NR==2 {print $2, $3, $4, $5, $6}')"
read -r total_k used_k avail_k used_pct_raw mount_point <<<"${df_line}"
used_pct="${used_pct_raw%\%}"
avail_gb="$(awk -v kb="${avail_k}" 'BEGIN {printf "%.1f", kb / 1024 / 1024}')"

echo "== Stage-1 disk budget check =="
echo "probe_path: ${probe_path}"
echo "mount: ${mount_point}"
echo "used_pct: ${used_pct}%"
echo "free_gb: ${avail_gb}"
echo

echo "== df -h =="
df -h "${probe_path}"
echo

echo "== key directory sizes =="
print_du() {
  local path="$1"
  local label="$2"
  if [[ -e "${path}" ]]; then
    du -sh "${path}" 2>/dev/null | awk -v label="${label}" '{print label "\t" $0}'
  else
    printf "%s\t%s\n" "${label}" "MISSING ${path}"
  fi
}

print_du "${persist_root}" "persist_root"
print_du "${output_root}" "output_root"
print_du "${stage1_root}" "stage1_root"
print_du "${CKPT_DIR:-${persist_root}/checkpoints}" "checkpoints"
print_du "${HF_HOME:-${persist_root}/cache/huggingface}" "hf_cache"
print_du "${TORCH_HOME:-${persist_root}/cache/torch}" "torch_cache"
print_du "${XDG_CACHE_HOME:-${persist_root}/cache/xdg}" "xdg_cache"
print_du "${PIP_CACHE_DIR:-${persist_root}/cache/pip}" "pip_cache"
echo

status="normal"
exit_code=0

free_lt_emergency="$(awk -v a="${avail_gb}" -v b="${emergency_min_free_gb}" 'BEGIN {print (a < b) ? 1 : 0}')"
free_lt_hard="$(awk -v a="${avail_gb}" -v b="${hard_min_free_gb}" 'BEGIN {print (a < b) ? 1 : 0}')"
free_lt_warning="$(awk -v a="${avail_gb}" -v b="${min_free_gb}" 'BEGIN {print (a < b) ? 1 : 0}')"

if [[ "${used_pct}" -ge "${emergency_max_used_pct}" || "${free_lt_emergency}" -eq 1 ]]; then
  status="emergency"
  exit_code=3
elif [[ "${used_pct}" -ge "${hard_max_used_pct}" || "${free_lt_hard}" -eq 1 ]]; then
  status="hard"
  exit_code=2
elif [[ "${used_pct}" -ge "${max_used_pct}" || "${free_lt_warning}" -eq 1 ]]; then
  status="warning"
  if [[ "${fail_on_warning}" -eq 1 ]]; then
    exit_code=1
  fi
fi

echo "== status =="
echo "status: ${status}"
echo "warning: used_pct>=${max_used_pct}% OR free_gb<${min_free_gb}"
echo "hard: used_pct>=${hard_max_used_pct}% OR free_gb<${hard_min_free_gb}"
echo "emergency: used_pct>=${emergency_max_used_pct}% OR free_gb<${emergency_min_free_gb}"

exit "${exit_code}"
