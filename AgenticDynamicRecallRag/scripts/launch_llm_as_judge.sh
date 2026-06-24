#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_ROOT}/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PY_CONFIG="${PY_CONFIG:-/data04/envs/ms/ms_cosearch_official/bin/python}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "ERROR: LLM judge service config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

read_config() {
  local key="$1"
  "${PY_CONFIG}" - "${CONFIG_PATH}" "${key}" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

path = Path(sys.argv[1])
key = sys.argv[2]
cfg = OmegaConf.load(path)
value = OmegaConf.select(cfg, key)
if value is None:
    sys.exit(3)
print(value)
PY
}

cfg_or_default() {
  local key="$1"
  local default="$2"
  read_config "${key}" 2>/dev/null || printf '%s\n' "${default}"
}

MODEL_PATH="${LLM_JUDGE_MODEL_PATH:-$(cfg_or_default model.model_path "")}"
SERVED_MODEL_NAME="${LLM_JUDGE_SERVED_MODEL_NAME:-$(cfg_or_default model.served_model_name DeepSeek-V4-Flash)}"
HOST="${LLM_JUDGE_HOST:-$(cfg_or_default server.host 0.0.0.0)}"
PORT="${LLM_JUDGE_PORT:-$(cfg_or_default server.port 8067)}"
PY="${LLM_JUDGE_PYTHON:-$(cfg_or_default runtime.python /data04/envs/ms/deepseek_v4/bin/python)}"
VLLM="${LLM_JUDGE_VLLM:-$(cfg_or_default runtime.vllm /data04/envs/ms/deepseek_v4/bin/vllm)}"
ENV_SITE_PACKAGES="${LLM_JUDGE_ENV_SITE_PACKAGES:-$(cfg_or_default runtime.env_site_packages /data04/envs/ms/deepseek_v4/lib/python3.11/site-packages)}"
GPU_IDS="${LLM_JUDGE_GPU_IDS:-$(cfg_or_default runtime.cuda_visible_devices 6,7)}"
TP_SIZE="${LLM_JUDGE_TENSOR_PARALLEL_SIZE:-$(cfg_or_default runtime.tensor_parallel_size 2)}"
GPU_MEMORY_UTILIZATION="${LLM_JUDGE_GPU_MEMORY_UTILIZATION:-$(cfg_or_default runtime.gpu_memory_utilization 0.95)}"
MAX_MODEL_LEN="${LLM_JUDGE_MAX_MODEL_LEN:-$(cfg_or_default runtime.max_model_len 11000)}"
KV_CACHE_DTYPE="${LLM_JUDGE_KV_CACHE_DTYPE:-$(cfg_or_default runtime.kv_cache_dtype fp8)}"
LINEAR_BACKEND="${LLM_JUDGE_LINEAR_BACKEND:-$(cfg_or_default runtime.linear_backend auto)}"
MOE_BACKEND="${LLM_JUDGE_MOE_BACKEND:-$(cfg_or_default runtime.moe_backend auto)}"
DISABLE_CUSTOM_ALL_REDUCE="${LLM_JUDGE_DISABLE_CUSTOM_ALL_REDUCE:-$(cfg_or_default runtime.disable_custom_all_reduce true)}"

DEFAULT_LOG_DIR="${LLM_JUDGE_LOG_DIR:-${PROJECT_ROOT}/../log/llm_judge}"
LOG_DIR="$(cfg_or_default logs.log_dir "")"
if [[ -z "${LOG_DIR}" || "${LOG_DIR}" == "null" ]]; then
  LOG_DIR="${DEFAULT_LOG_DIR}"
fi
LOG_FILE_NAME="$(cfg_or_default logs.log_file vllm_gpu06_07_8067.log)"
PID_FILE_NAME="$(cfg_or_default logs.pid_file vllm_gpu06_07_8067.pid)"
LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}"
PID_FILE="${LOG_DIR}/${PID_FILE_NAME}"

mkdir -p "${LOG_DIR}"

export LD_LIBRARY_PATH="${ENV_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${ENV_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"

ready() {
  "${PY}" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5).status < 500 else 1)" >/dev/null 2>&1
}

pid_is_live() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; LLM judge service config resolved"
  echo "CONFIG_PATH=${CONFIG_PATH}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
  echo "HOST=${HOST}"
  echo "PORT=${PORT}"
  echo "PY=${PY}"
  echo "VLLM=${VLLM}"
  echo "GPU_IDS=${GPU_IDS}"
  echo "TP_SIZE=${TP_SIZE}"
  echo "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
  echo "MAX_MODEL_LEN=${MAX_MODEL_LEN}"
  echo "LOG_FILE=${LOG_FILE}"
  echo "PID_FILE=${PID_FILE}"
  exit 0
fi

if ready; then
  echo "vLLM already ready: http://127.0.0.1:${PORT}"
  exit 0
fi

if pid_is_live; then
  echo "vLLM process exists but is not ready yet: pid=$(cat "${PID_FILE}")"
  exit 0
fi

if [[ ! -x "${VLLM}" ]]; then
  echo "ERROR: vLLM executable not found: ${VLLM}" >&2
  exit 2
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path not found: ${MODEL_PATH}" >&2
  exit 2
fi

cmd=(
  env CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  "${VLLM}" serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP_SIZE}"
  --host "${HOST}"
  --port "${PORT}"
  --trust-remote-code
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --linear-backend "${LINEAR_BACKEND}"
  --moe-backend "${MOE_BACKEND}"
)

case "${DISABLE_CUSTOM_ALL_REDUCE}" in
  1|true|True|TRUE|yes|YES|on|ON)
  cmd+=(--disable-custom-all-reduce)
    ;;
esac

echo "starting vLLM judge: model=${MODEL_PATH}, gpus=${GPU_IDS}, port=${PORT}, log=${LOG_FILE}"
setsid "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "pid=${pid}"
