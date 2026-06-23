#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${PIPELINE_DIR}/logs" "${PIPELINE_DIR}/run"

PY="${PY:-/data04/envs/ms/deepseek_v4/bin/python}"
VLLM="${VLLM:-/data04/envs/ms/deepseek_v4/bin/vllm}"
ENV_SITE_PACKAGES="${ENV_SITE_PACKAGES:-/data04/envs/ms/deepseek_v4/lib/python3.11/site-packages}"
export LD_LIBRARY_PATH="${ENV_SITE_PACKAGES}/nvidia/cuda_runtime/lib:${ENV_SITE_PACKAGES}/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-1}"
MODEL_PATH="${MODEL_PATH:-/data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-DeepSeek-V4-Flash}"
GPU_IDS="${GPU_IDS:-6,7}"
PORT="${PORT:-8067}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-11000}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
MOE_BACKEND="${MOE_BACKEND:-auto}"
LOG_FILE="${LOG_FILE:-${PIPELINE_DIR}/logs/vllm_gpu06_07_${PORT}.log}"
PID_FILE="${PID_FILE:-${PIPELINE_DIR}/run/vllm_gpu06_07_${PORT}.pid}"

ready() {
  "${PY}" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5).status < 500 else 1)" >/dev/null 2>&1
}

pid_is_live() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

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
  --disable-custom-all-reduce
)

echo "starting vLLM: model=${MODEL_PATH}, gpus=${GPU_IDS}, port=${PORT}, log=${LOG_FILE}"
setsid "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
pid=$!
echo "${pid}" > "${PID_FILE}"
echo "pid=${pid}"
