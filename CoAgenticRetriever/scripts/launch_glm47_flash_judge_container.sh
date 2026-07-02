#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"

IMAGE="${GLM47_VLLM_IMAGE:-m.daocloud.io/quay.io/ascend/vllm-ascend:v0.21.0rc1}"
CONTAINER_NAME="${GLM47_CONTAINER_NAME:-glm47_vllm_8067}"
MODEL_PATH="${GLM47_MODEL_PATH:-/data01/ms_wksp/agent_up_to_date/models/llm/GLM-4.7-Flash}"
SERVED_MODEL_NAME="${GLM47_SERVED_MODEL_NAME:-GLM-4.7-Flash}"
HOST="${GLM47_HOST:-0.0.0.0}"
PORT="${GLM47_PORT:-8067}"
NPU_IDS="${GLM47_NPU_IDS:-6,7}"
TP_SIZE="${GLM47_TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GLM47_GPU_MEMORY_UTILIZATION:-0.95}"
MAX_MODEL_LEN="${GLM47_MAX_MODEL_LEN:-32000}"
KV_CACHE_DTYPE="${GLM47_KV_CACHE_DTYPE:-auto}"
MOE_BACKEND="${GLM47_MOE_BACKEND:-auto}"
LOG_DIR="${GLM47_LOG_DIR:-${ROOT}/log/llm_judge}"
LOG_FILE="${GLM47_LOG_FILE:-${LOG_DIR}/vllm_glm_4_7_flash_container_8067.log}"

ACTION="start"
PULL_IMAGE=0

usage() {
  cat <<USAGE
Usage: $0 [--pull] [--restart] [--stop] [--status] [--logs]

Environment overrides:
  GLM47_VLLM_IMAGE=${IMAGE}
  GLM47_MODEL_PATH=${MODEL_PATH}
  GLM47_NPU_IDS=${NPU_IDS}
  GLM47_PORT=${PORT}
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --pull)
      PULL_IMAGE=1
      shift
      ;;
    --restart)
      ACTION="restart"
      shift
      ;;
    --stop)
      ACTION="stop"
      shift
      ;;
    --status)
      ACTION="status"
      shift
      ;;
    --logs)
      ACTION="logs"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ready() {
  curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1
}

container_exists() {
  docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

container_running() {
  docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"
}

status() {
  docker ps -a --format '{{.ID}} {{.Names}} {{.Status}}' | grep -F "${CONTAINER_NAME}" || true
  if ready; then
    echo "ready=http://127.0.0.1:${PORT}"
    curl -fsS "http://127.0.0.1:${PORT}/v1/models"
    echo
  else
    echo "ready=false"
  fi
}

mkdir -p "${LOG_DIR}"

case "${ACTION}" in
  stop)
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    echo "stopped ${CONTAINER_NAME}"
    exit 0
    ;;
  status)
    status
    exit 0
    ;;
  logs)
    docker logs -f "${CONTAINER_NAME}"
    exit 0
    ;;
  restart)
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    ;;
esac

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path not found: ${MODEL_PATH}" >&2
  exit 2
fi

if [[ "${PULL_IMAGE}" == "1" ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker pull "${IMAGE}"
fi

if ready; then
  echo "vLLM already ready: http://127.0.0.1:${PORT}"
  status
  exit 0
fi

if container_exists; then
  if container_running; then
    echo "container exists but service is not ready yet: ${CONTAINER_NAME}"
    status
    exit 0
  fi
  docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

if ss -ltn "sport = :${PORT}" | grep -q ":${PORT}"; then
  echo "ERROR: port ${PORT} is already in use but /v1/models is not ready" >&2
  ss -ltnp | grep ":${PORT}" >&2 || true
  exit 2
fi

echo "starting GLM-4.7-Flash judge container: name=${CONTAINER_NAME}, image=${IMAGE}, npus=${NPU_IDS}, port=${PORT}"
docker run -d --name "${CONTAINER_NAME}" --privileged --net=host --ipc=host \
  -e ASCEND_RT_VISIBLE_DEVICES="${NPU_IDS}" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v /data01:/data01 \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  "${IMAGE}" \
  bash -lc "exec vllm serve '${MODEL_PATH}' \
    --served-model-name '${SERVED_MODEL_NAME}' \
    --host '${HOST}' \
    --port '${PORT}' \
    --trust-remote-code \
    --tensor-parallel-size '${TP_SIZE}' \
    --gpu-memory-utilization '${GPU_MEMORY_UTILIZATION}' \
    --max-model-len '${MAX_MODEL_LEN}' \
    --kv-cache-dtype '${KV_CACHE_DTYPE}' \
    --moe-backend '${MOE_BACKEND}' \
    --disable-custom-all-reduce"

echo "log=${LOG_FILE}"
nohup docker logs -f "${CONTAINER_NAME}" > "${LOG_FILE}" 2>&1 &
