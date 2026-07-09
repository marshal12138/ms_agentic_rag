#!/usr/bin/env bash
set -euo pipefail

# AgenticIterRag v1 recall retriever 服务启动器。
# 该脚本属于 AIR infer backend，只调用仓库公共 retriever server 源码，不调用 CAR launcher。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/00_project_paths.sh"
setup_agent_iteration_paths "${ROOT}"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
source "${SCRIPT_DIR}/00_air_accelerator.sh"

# 检索资产路径；默认复用仓库公共 retrieval 数据目录。
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}"
RETRIEVER_SRC_DIR="${RETRIEVER_SRC_DIR:-${ROOT}/src/retrievers}"
GPU_DENSE_RETRIEVER_SERVER="${GPU_DENSE_RETRIEVER_SERVER:-${RETRIEVER_SRC_DIR}/gpu_dense_retriever_server.py}"
VERIFY_RETRIEVAL_ASSETS="${VERIFY_RETRIEVAL_ASSETS:-${RETRIEVER_SRC_DIR}/verify_official_retrieval_assets.py}"

# 服务和资源配置；由 AIR pipeline runner 注入，以下值仅作为直接调试兜底。
PORT="${PORT:-8030}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-$(air_accel_device_prefix)}"
RECALL_GPU_ID="${RECALL_GPU_ID:-5}"
RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS:-${GPU_ID:-${RECALL_GPU_ID}}}"
RECALL_FINAL_TOP_N="${RECALL_FINAL_TOP_N:-50}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
DOC_DTYPE="${DOC_DTYPE:-float16}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
DRY_RUN="${DRY_RUN:-0}"
# retriever 资产预检默认关闭；训练/推理编排层可以对第一个 backend 显式打开。
# 即使跳过 verifier，脚本仍会做必要的文件存在性检查，并在 server 启动时真实加载 index/corpus。
SKIP_RETRIEVAL_ASSET_VERIFY="${SKIP_RETRIEVAL_ASSET_VERIFY:-1}"

is_truthy() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

cd "${ROOT}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OMP_NUM_THREADS MKL_NUM_THREADS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  cat <<EOF
DRY_RUN=1
ROOT=${ROOT}
RETRIEVER_SRC_DIR=${RETRIEVER_SRC_DIR}
GPU_DENSE_RETRIEVER_SERVER=${GPU_DENSE_RETRIEVER_SERVER}
VERIFY_RETRIEVAL_ASSETS=${VERIFY_RETRIEVAL_ASSETS}
INDEX_FILE=${INDEX_FILE}
CORPUS_FILE=${CORPUS_FILE}
RETRIEVER_MODEL=${RETRIEVER_MODEL}
AIR_ACCELERATOR=${AIR_ACCELERATOR}
$(air_accel_visible_devices_var)=${RETRIEVER_GPU_IDS}
DEVICE=${DEVICE}
DOC_DTYPE=${DOC_DTYPE}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE}
HOST=${HOST}
PORT=${PORT}
RECALL_FINAL_TOP_N=${RECALL_FINAL_TOP_N}
SKIP_RETRIEVAL_ASSET_VERIFY=${SKIP_RETRIEVAL_ASSET_VERIFY}
EOF
  exit 0
fi

if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "ERROR: AIR retrieval index not found: ${INDEX_FILE}" >&2
  echo "Expected retrieval assets under ${RETRIEVAL_DATA_DIR}." >&2
  exit 2
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "ERROR: AIR retrieval corpus not found: ${CORPUS_FILE}" >&2
  echo "Expected wiki-18.jsonl corpus under ${RETRIEVAL_DATA_DIR}." >&2
  exit 2
fi

if [[ ! -f "${GPU_DENSE_RETRIEVER_SERVER}" ]]; then
  echo "ERROR: AIR GPU dense retrieval server source not found: ${GPU_DENSE_RETRIEVER_SERVER}" >&2
  exit 2
fi

if [[ "${DEVICE}" != "$(air_accel_device_prefix)"* ]]; then
  echo "ERROR: AIR dense retrieval server requires DEVICE prefix $(air_accel_device_prefix); got DEVICE=${DEVICE}" >&2
  exit 2
fi

if ! env $(air_accel_env_visible_devices_cmd "${RETRIEVER_GPU_IDS}") "${PY}" - <<PY
$(air_accel_python_available_expr)
PY
then
  echo "ERROR: ${AIR_ACCELERATOR} accelerator is not visible to PyTorch for AIR dense retrieval server; refusing to run on CPU." >&2
  exit 2
fi

if is_truthy "${SKIP_RETRIEVAL_ASSET_VERIFY}"; then
  echo "Skipping AIR retrieval asset verifier for this backend: SKIP_RETRIEVAL_ASSET_VERIFY=${SKIP_RETRIEVAL_ASSET_VERIFY}" >&2
else
  if [[ ! -f "${VERIFY_RETRIEVAL_ASSETS}" ]]; then
    echo "ERROR: AIR retrieval asset verifier not found: ${VERIFY_RETRIEVAL_ASSETS}" >&2
    exit 2
  fi
  "${PY}" "${VERIFY_RETRIEVAL_ASSETS}" \
    --index "${INDEX_FILE}" \
    --corpus "${CORPUS_FILE}"
fi

echo "Starting AIR GPU dense retriever from ${GPU_DENSE_RETRIEVER_SERVER}" >&2
echo "  $(air_accel_visible_devices_var)=${RETRIEVER_GPU_IDS}" >&2
echo "  device=${DEVICE}; doc embeddings will be loaded into ${AIR_ACCELERATOR} memory as ${DOC_DTYPE}" >&2
echo "  retrieval endpoint=http://${HOST}:${PORT}/retrieve, topk=${RECALL_FINAL_TOP_N}" >&2

exec env $(air_accel_env_visible_devices_cmd "${RETRIEVER_GPU_IDS}") "${PY}" "${GPU_DENSE_RETRIEVER_SERVER}" \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk "${RECALL_FINAL_TOP_N}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --retriever_model "${RETRIEVER_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --doc_dtype "${DOC_DTYPE}"
