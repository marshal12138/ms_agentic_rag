#!/usr/bin/env bash

# Shared GPU wait helpers for training/evaluation task launchers.
# Can be sourced as a library or executed directly as a small CLI.

gpu_wait_is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

gpu_wait_csv_to_lines() {
  local csv="$1"
  printf '%s\n' "${csv//,/ }" | tr ' ' '\n' | sed '/^$/d' | sort -n | uniq
}

gpu_wait_validate_gpu_list() {
  local gpu_csv="$1"
  local gpu
  if [[ -z "${gpu_csv//[[:space:],]/}" ]]; then
    echo "ERROR: GPU wait list is empty." >&2
    return 2
  fi
  while IFS= read -r gpu; do
    if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: invalid GPU id in wait list: ${gpu}" >&2
      return 2
    fi
  done < <(gpu_wait_csv_to_lines "${gpu_csv}")
}

gpu_wait_cleanup_tmp() {
  local tmp_dir="$1"
  if [[ -n "${tmp_dir}" && -d "${tmp_dir}" ]]; then
    rm -rf "${tmp_dir}"
  fi
}

wait_for_gpu_release() {
  local gpu_csv="$1"
  local interval="${2:-${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}}"
  local timeout="${3:-${WAIT_FOR_GPU_TIMEOUT_SECONDS:-0}}"
  local label="${4:-${WAIT_FOR_GPU_LABEL:-GPU wait}}"

  gpu_wait_validate_gpu_list "${gpu_csv}"
  if ! [[ "${interval}" =~ ^[0-9]+$ ]] || (( interval < 1 )); then
    echo "ERROR: GPU wait interval must be a positive integer; got ${interval}" >&2
    return 2
  fi
  if ! [[ "${timeout}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: GPU wait timeout must be a non-negative integer; got ${timeout}" >&2
    return 2
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi is required for GPU wait." >&2
    return 2
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local gpu_file="${tmp_dir}/gpus.txt"
  local gpu_info_file="${tmp_dir}/gpu_info.txt"
  local uuid_file="${tmp_dir}/uuids.txt"
  local missing_file="${tmp_dir}/missing.txt"
  local busy_file="${tmp_dir}/busy.txt"
  local start_ts now_ts elapsed

  gpu_wait_csv_to_lines "${gpu_csv}" > "${gpu_file}"
  nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits > "${gpu_info_file}"
  awk -F', ' 'NR==FNR {want[$1]=1; next} {seen[$1]=1} END {for (gpu in want) if (!(gpu in seen)) print gpu}' "${gpu_file}" "${gpu_info_file}" > "${missing_file}"
  if [[ -s "${missing_file}" ]]; then
    echo "ERROR: ${label}: requested GPU ids do not exist:" >&2
    cat "${missing_file}" >&2
    gpu_wait_cleanup_tmp "${tmp_dir}"
    return 2
  fi
  awk -F', ' 'NR==FNR {want[$1]=1; next} ($1 in want) {print $2}' "${gpu_file}" "${gpu_info_file}" > "${uuid_file}"
  start_ts="$(date +%s)"

  echo "${label}: waiting for GPUs to be free: ${gpu_csv}"
  while true; do
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits \
      | awk -F', ' 'NR==FNR {want[$1]=1; next} ($1 in want) {print $0}' "${uuid_file}" - > "${busy_file}" || true

    if [[ ! -s "${busy_file}" ]]; then
      echo "${label}: target GPUs are free."
      gpu_wait_cleanup_tmp "${tmp_dir}"
      return 0
    fi

    now_ts="$(date +%s)"
    elapsed=$((now_ts - start_ts))
    if (( timeout > 0 && elapsed >= timeout )); then
      echo "ERROR: ${label}: timed out after ${elapsed}s waiting for GPUs: ${gpu_csv}" >&2
      echo "Busy processes:" >&2
      cat "${busy_file}" >&2
      gpu_wait_cleanup_tmp "${tmp_dir}"
      return 124
    fi

    echo "${label}: target GPUs still busy; checking again in ${interval}s. Busy processes:"
    cat "${busy_file}"
    sleep "${interval}"
  done
}

wait_for_gpus_if_enabled() {
  local enabled="${1:-${WAIT_FOR_GPU_RELEASE:-0}}"
  local gpu_csv="${2:-${WAIT_FOR_GPUS:-}}"
  local interval="${3:-${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}}"
  local timeout="${4:-${WAIT_FOR_GPU_TIMEOUT_SECONDS:-0}}"
  local label="${5:-${WAIT_FOR_GPU_LABEL:-GPU wait}}"

  if ! gpu_wait_is_truthy "${enabled}"; then
    echo "${label}: skipped because WAIT_FOR_GPU_RELEASE=${enabled}."
    return 0
  fi
  wait_for_gpu_release "${gpu_csv}" "${interval}" "${timeout}" "${label}"
}

gpu_wait_usage() {
  cat <<'USAGE'
Usage:
  wait_for_gpus.sh --gpus "0,1,2" [--interval 30] [--timeout 0] [--label "GPU wait"]

Options:
  --gpus       Comma/space separated GPU ids to wait for. Required.
  --interval   Poll interval in seconds. Default: 30.
  --timeout    Max wait seconds. 0 means no timeout. Default: 0.
  --label      Prefix used in log messages. Default: GPU wait.
  --help       Show this help.
USAGE
}

gpu_wait_main() {
  local gpus="${WAIT_FOR_GPUS:-}"
  local interval="${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}"
  local timeout="${WAIT_FOR_GPU_TIMEOUT_SECONDS:-0}"
  local label="${WAIT_FOR_GPU_LABEL:-GPU wait}"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --gpus)
        gpus="${2:-}"
        shift 2
        ;;
      --interval)
        interval="${2:-}"
        shift 2
        ;;
      --timeout)
        timeout="${2:-}"
        shift 2
        ;;
      --label)
        label="${2:-}"
        shift 2
        ;;
      --help|-h)
        gpu_wait_usage
        return 0
        ;;
      *)
        echo "ERROR: unknown argument: $1" >&2
        gpu_wait_usage >&2
        return 2
        ;;
    esac
  done

  wait_for_gpu_release "${gpus}" "${interval}" "${timeout}" "${label}"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  gpu_wait_main "$@"
fi
