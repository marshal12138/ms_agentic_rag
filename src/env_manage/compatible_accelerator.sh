#!/usr/bin/env bash

# Accelerator compatibility helpers.
#
# Default behavior stays CUDA/NVIDIA. When CUDA tooling is unavailable and
# Ascend NPU tooling is present, callers can use these helpers to switch shell
# commands and visibility variables to NPU equivalents.

co_accel_detect() {
  if [[ -n "${COSEARCH_ACCELERATOR:-}" ]]; then
    printf '%s\n' "${COSEARCH_ACCELERATOR}"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'gpu\n'
    return 0
  fi
  if command -v npu-smi >/dev/null 2>&1 || compgen -G "/dev/davinci[0-9]*" >/dev/null; then
    printf 'npu\n'
    return 0
  fi
  printf 'cpu\n'
}

export COSEARCH_ACCELERATOR="${COSEARCH_ACCELERATOR:-$(co_accel_detect)}"

_co_accel_source_if_exists() {
  local path="$1"
  local had_nounset=0
  shift || true
  [[ -f "${path}" ]] || return 0
  case "$-" in
    *u*) had_nounset=1; set +u ;;
  esac
  # shellcheck disable=SC1090
  source "${path}" "$@"
  if [[ "${had_nounset}" == "1" ]]; then
    set -u
  fi
}

_co_accel_source_ascend_runtime_env() {
  co_accel_is_npu || return 0

  local cann_set_env="${COSEARCH_ASCEND_CANN_SET_ENV:-/usr/local/Ascend/cann/set_env.sh}"
  local atb_set_env="${COSEARCH_ASCEND_ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
  local atb_cxx_abi="${COSEARCH_ASCEND_ATB_CXX_ABI:-1}"

  _co_accel_source_if_exists "${cann_set_env}"
  if [[ -f "${atb_set_env}" ]]; then
    _co_accel_source_if_exists "${atb_set_env}" "--cxx_abi=${atb_cxx_abi}"
    export COSEARCH_ASCEND_ATB_HOME_PATH="${ATB_HOME_PATH:-}"
  fi

  export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
  export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
  export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
  export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
  export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
}

co_accel_is_gpu() {
  [[ "${COSEARCH_ACCELERATOR}" == "gpu" || "${COSEARCH_ACCELERATOR}" == "cuda" ]]
}

co_accel_is_npu() {
  [[ "${COSEARCH_ACCELERATOR}" == "npu" || "${COSEARCH_ACCELERATOR}" == "ascend" ]]
}

co_accel_visible_devices_var() {
  if co_accel_is_npu; then
    printf 'ASCEND_RT_VISIBLE_DEVICES\n'
  else
    printf 'CUDA_VISIBLE_DEVICES\n'
  fi
}

co_accel_device_prefix() {
  if co_accel_is_npu; then
    printf 'npu\n'
  else
    printf 'cuda\n'
  fi
}

co_accel_device_spec() {
  local index="${1:-}"
  local prefix
  prefix="$(co_accel_device_prefix)"
  if [[ -n "${index}" ]]; then
    printf '%s:%s\n' "${prefix}" "${index}"
  else
    printf '%s\n' "${prefix}"
  fi
}

co_accel_normalize_device_spec() {
  local device="${1:-}"
  local suffix=""

  if [[ -z "${device}" ]]; then
    co_accel_device_spec
    return 0
  fi

  case "${device}" in
    cuda:*)
      suffix=":${device#cuda:}"
      if co_accel_is_npu; then
        printf 'npu%s\n' "${suffix}"
      else
        printf 'cuda%s\n' "${suffix}"
      fi
      ;;
    cuda)
      co_accel_device_prefix
      ;;
    npu:*)
      suffix=":${device#npu:}"
      if co_accel_is_gpu; then
        printf 'cuda%s\n' "${suffix}"
      else
        printf 'npu%s\n' "${suffix}"
      fi
      ;;
    npu)
      co_accel_device_prefix
      ;;
    *)
      printf '%s\n' "${device}"
      ;;
  esac
}

co_accel_export_visible_devices() {
  local ids="$1"
  if co_accel_is_npu; then
    export ASCEND_RT_VISIBLE_DEVICES="${ids}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${ids}}"
  else
    export CUDA_VISIBLE_DEVICES="${ids}"
  fi
}

co_accel_env_visible_devices_cmd() {
  local ids="$1"
  if co_accel_is_npu; then
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s CUDA_VISIBLE_DEVICES=%s' "${ids}" "${ids}"
  else
    printf 'CUDA_VISIBLE_DEVICES=%s' "${ids}"
  fi
}

co_accel_count() {
  if co_accel_is_gpu; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l
      return 0
    fi
    printf '0\n'
    return 0
  fi
  if co_accel_is_npu; then
    if command -v npu-smi >/dev/null 2>&1; then
      npu-smi info -m 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && $2 == 0 && $3 ~ /^[0-9]+$/ {count++} END {print count+0}'
      return 0
    fi
    compgen -G "/dev/davinci[0-9]*" | sed -E 's#.*/davinci##' | sort -n | wc -l
    return 0
  fi
  printf '0\n'
}

co_accel_device_ids() {
  if co_accel_is_gpu; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | awk '{gsub(/[[:space:]]/, ""); if ($0 != "") print $0}'
    return 0
  fi
  if co_accel_is_npu; then
    if command -v npu-smi >/dev/null 2>&1; then
      npu-smi info -m 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && $2 == 0 && $3 ~ /^[0-9]+$/ {print $3}'
      return 0
    fi
    compgen -G "/dev/davinci[0-9]*" | sed -E 's#.*/davinci##' | sort -n
    return 0
  fi
}

co_accel_processes() {
  if co_accel_is_gpu; then
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true
    return 0
  fi
  if co_accel_is_npu; then
    npu-smi info 2>/dev/null | awk '
      BEGIN {in_proc=0}
      /Process id/ {in_proc=1; next}
      in_proc && /No running processes found in NPU/ {
        if (match($0, /NPU [0-9]+/)) {
          npu=substr($0, RSTART + 4, RLENGTH - 4)
          print npu ", -, -, 0"
        }
      }
      in_proc && $0 ~ /^\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+/ {
        gsub(/\|/, " ", $0)
        n=split($0, f, /[[:space:]]+/)
        compact_count=0
        for (i=1; i<=n; i++) {
          if (f[i] != "") {
            compact[++compact_count]=f[i]
          }
        }
        if (compact_count >= 5) {
          print compact[1] ", " compact[3] ", " compact[4] ", " compact[5]
        }
      }
    ' | grep -v ', -, -, 0' || true
    return 0
  fi
}

co_accel_python_available_expr() {
  if co_accel_is_npu; then
    cat <<'PY'
import torch
try:
    import torch_npu  # noqa: F401
except Exception:
    pass
raise SystemExit(0 if hasattr(torch, "npu") and torch.npu.is_available() else 1)
PY
  else
    cat <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  fi
}

co_accel_print_summary() {
  echo "COSEARCH_ACCELERATOR=${COSEARCH_ACCELERATOR}"
  echo "VISIBLE_DEVICES_VAR=$(co_accel_visible_devices_var)"
  echo "DEVICE_PREFIX=$(co_accel_device_prefix)"
  echo "DEVICE_COUNT=$(co_accel_count)"
}

_co_accel_source_ascend_runtime_env
