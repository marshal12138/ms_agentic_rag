#!/usr/bin/env bash
set -euo pipefail

# AgenticIterRag v1 CPU recall retriever 服务启动器。
# 该脚本只启动单个 CPU backend；多实例和负载均衡由 02_air_infer_launcher
# 或 TrainingServiceManager 统一编排，避免 agent/continuation 直接感知多个 backend。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${SCRIPT_DIR}/00_project_paths.sh"
setup_agent_iteration_paths "${ROOT}"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"

# 检索资产路径；默认复用仓库公共 retrieval 数据目录。
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}"
RETRIEVER_SRC_DIR="${RETRIEVER_SRC_DIR:-${ROOT}/src/retrievers}"
CPU_DENSE_RETRIEVER_SERVER="${CPU_DENSE_RETRIEVER_SERVER:-${RETRIEVER_SRC_DIR}/cpu_dense_retriever_server.py}"
VERIFY_RETRIEVAL_ASSETS="${VERIFY_RETRIEVAL_ASSETS:-${RETRIEVER_SRC_DIR}/verify_official_retrieval_assets.py}"

# 服务和 CPU 配置；由 AIR pipeline runner 或 service manager 注入。
PORT="${PORT:-8030}"
HOST="${HOST:-0.0.0.0}"
RECALL_FINAL_TOP_N="${RECALL_FINAL_TOP_N:-50}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
DOC_DTYPE="${DOC_DTYPE:-float32}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-8}"
CPU_THREADS_PER_INSTANCE="${CPU_THREADS_PER_INSTANCE:-8}"
DRY_RUN="${DRY_RUN:-0}"
# 资产预检默认关闭；8 个 CPU backend 同时启动时不应该重复扫描大文件。
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
# OMP/MKL/torch/faiss 线程数保持一致，避免每个 backend 默认抢满全机 CPU。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${CPU_THREADS_PER_INSTANCE}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${CPU_THREADS_PER_INSTANCE}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  cat <<EOF
DRY_RUN=1
ROOT=${ROOT}
CPU_DENSE_RETRIEVER_SERVER=${CPU_DENSE_RETRIEVER_SERVER}
VERIFY_RETRIEVAL_ASSETS=${VERIFY_RETRIEVAL_ASSETS}
INDEX_FILE=${INDEX_FILE}
CORPUS_FILE=${CORPUS_FILE}
RETRIEVER_MODEL=${RETRIEVER_MODEL}
DEVICE=cpu
DOC_DTYPE=${DOC_DTYPE}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE}
CPU_THREADS_PER_INSTANCE=${CPU_THREADS_PER_INSTANCE}
OMP_NUM_THREADS=${OMP_NUM_THREADS}
MKL_NUM_THREADS=${MKL_NUM_THREADS}
HOST=${HOST}
PORT=${PORT}
RECALL_FINAL_TOP_N=${RECALL_FINAL_TOP_N}
SKIP_RETRIEVAL_ASSET_VERIFY=${SKIP_RETRIEVAL_ASSET_VERIFY}
EOF
  exit 0
fi

if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "ERROR: AIR retrieval index not found: ${INDEX_FILE}" >&2
  exit 2
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "ERROR: AIR retrieval corpus not found: ${CORPUS_FILE}" >&2
  exit 2
fi

if [[ ! -f "${CPU_DENSE_RETRIEVER_SERVER}" ]]; then
  echo "ERROR: AIR CPU dense retrieval server source not found: ${CPU_DENSE_RETRIEVER_SERVER}" >&2
  exit 2
fi

if [[ "${DOC_DTYPE}" != "float32" ]]; then
  echo "ERROR: CPU retriever currently supports DOC_DTYPE=float32 only; got ${DOC_DTYPE}" >&2
  exit 2
fi

if is_truthy "${SKIP_RETRIEVAL_ASSET_VERIFY}"; then
  echo "Skipping AIR retrieval asset verifier for this CPU backend: SKIP_RETRIEVAL_ASSET_VERIFY=${SKIP_RETRIEVAL_ASSET_VERIFY}" >&2
else
  if [[ ! -f "${VERIFY_RETRIEVAL_ASSETS}" ]]; then
    echo "ERROR: AIR retrieval asset verifier not found: ${VERIFY_RETRIEVAL_ASSETS}" >&2
    exit 2
  fi
  "${PY}" "${VERIFY_RETRIEVAL_ASSETS}" \
    --index "${INDEX_FILE}" \
    --corpus "${CORPUS_FILE}"
fi

echo "Starting AIR CPU dense retriever from ${CPU_DENSE_RETRIEVER_SERVER}" >&2
echo "  device=cpu; cpu_threads_per_instance=${CPU_THREADS_PER_INSTANCE}; query_batch_size=${QUERY_BATCH_SIZE}" >&2
echo "  retrieval endpoint=http://${HOST}:${PORT}/retrieve, topk=${RECALL_FINAL_TOP_N}" >&2

exec "${PY}" "${CPU_DENSE_RETRIEVER_SERVER}" \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk "${RECALL_FINAL_TOP_N}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --retriever_model "${RETRIEVER_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --cpu_threads "${CPU_THREADS_PER_INSTANCE}" \
  --doc_dtype "${DOC_DTYPE}"
