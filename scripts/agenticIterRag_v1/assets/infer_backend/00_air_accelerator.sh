#!/usr/bin/env bash

# AgenticIterRag v1 accelerator compatibility helpers.
# 该文件是 AIR 自有实现，使用 AIR_ACCELERATOR 作为唯一对外变量。

air_accel_detect() {
  if [[ -n "${AIR_ACCELERATOR:-}" ]]; then
    printf '%s\n' "${AIR_ACCELERATOR}"
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

export AIR_ACCELERATOR="${AIR_ACCELERATOR:-$(air_accel_detect)}"

air_accel_source_if_exists() {
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

air_accel_is_gpu() {
  [[ "${AIR_ACCELERATOR}" == "gpu" || "${AIR_ACCELERATOR}" == "cuda" ]]
}

air_accel_is_npu() {
  [[ "${AIR_ACCELERATOR}" == "npu" || "${AIR_ACCELERATOR}" == "ascend" ]]
}

air_accel_source_ascend_runtime_env() {
  air_accel_is_npu || return 0

  local cann_set_env="${AIR_ASCEND_CANN_SET_ENV:-/usr/local/Ascend/cann/set_env.sh}"
  local atb_set_env="${AIR_ASCEND_ATB_SET_ENV:-/usr/local/Ascend/nnal/atb/set_env.sh}"
  local atb_cxx_abi="${AIR_ASCEND_ATB_CXX_ABI:-1}"

  air_accel_source_if_exists "${cann_set_env}"
  if [[ -f "${atb_set_env}" ]]; then
    air_accel_source_if_exists "${atb_set_env}" "--cxx_abi=${atb_cxx_abi}"
    export AIR_ASCEND_ATB_HOME_PATH="${ATB_HOME_PATH:-}"
  fi

  export VLLM_ASCEND_ENABLE_NZ="${VLLM_ASCEND_ENABLE_NZ:-0}"
  export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1500}"
  export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-1800}"
  export HCCL_HOST_SOCKET_PORT_RANGE="${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}"
  export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}"
}

air_accel_visible_devices_var() {
  if air_accel_is_npu; then
    printf 'ASCEND_RT_VISIBLE_DEVICES\n'
  else
    printf 'CUDA_VISIBLE_DEVICES\n'
  fi
}

air_accel_device_prefix() {
  if air_accel_is_npu; then
    printf 'npu\n'
  else
    printf 'cuda\n'
  fi
}

air_accel_device_spec() {
  local index="${1:-}"
  local prefix
  prefix="$(air_accel_device_prefix)"
  if [[ -n "${index}" ]]; then
    printf '%s:%s\n' "${prefix}" "${index}"
  else
    printf '%s\n' "${prefix}"
  fi
}

air_accel_env_visible_devices_cmd() {
  local ids="$1"
  if air_accel_is_npu; then
    printf 'ASCEND_RT_VISIBLE_DEVICES=%s CUDA_VISIBLE_DEVICES=%s' "${ids}" "${ids}"
  else
    printf 'CUDA_VISIBLE_DEVICES=%s' "${ids}"
  fi
}

air_accel_python_available_expr() {
  if air_accel_is_npu; then
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

air_accel_source_ascend_runtime_env
