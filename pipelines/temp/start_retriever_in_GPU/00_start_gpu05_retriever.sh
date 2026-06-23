#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PIPELINE_DIR}/../../.." && pwd)"
PARENT_ROOT="$(cd "${ROOT}/.." && pwd)"

mkdir -p "${PIPELINE_DIR}/logs" "${PIPELINE_DIR}/run"

PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-8050}"
DEVICE="${DEVICE:-cuda}"
FAISS_GPU="${FAISS_GPU:-0}"
DOC_DTYPE="${DOC_DTYPE:-float16}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"
LOG_FILE="${LOG_FILE:-${PIPELINE_DIR}/logs/retriever_gpu${GPU_ID}_${PORT}.log}"
PID_FILE="${PID_FILE:-${PIPELINE_DIR}/run/retriever_gpu${GPU_ID}_${PORT}.pid}"
START_BACKGROUND="${START_BACKGROUND:-1}"

is_ready() {
  "${PY}" -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:${PORT}/docs', timeout=5).status < 500 else 1)" >/dev/null 2>&1
}

pid_is_live() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" >/dev/null 2>&1
}

if is_ready; then
  echo "retriever already ready: http://127.0.0.1:${PORT}"
  exit 0
fi

if pid_is_live; then
  echo "retriever process exists but not ready yet: pid=$(cat "${PID_FILE}")"
  exit 0
fi

export PY
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export RETRIEVER_GPU_IDS="${GPU_ID}"
export GPU_ID
export PORT
export DEVICE
export FAISS_GPU
export DOC_DTYPE
export QUERY_BATCH_SIZE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

EXTERNAL_MODEL_ROOT="${EXTERNAL_MODEL_ROOT:-${PARENT_ROOT}/models}"
EXTERNAL_RETRIEVAL_ROOT="${EXTERNAL_RETRIEVAL_ROOT:-${ROOT}/data/retrieval}"
COSEARCH_PROJECT_ROOT="${COSEARCH_PROJECT_ROOT:-${ROOT}/CoSearch}"
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}"
GPU_RETRIEVAL_SERVER="${GPU_RETRIEVAL_SERVER:-${ROOT}/src/retrievers/gpu_dense_retriever_server.py}"

for path in "${INDEX_FILE}" "${CORPUS_FILE}" "${RETRIEVER_MODEL}" "${GPU_RETRIEVAL_SERVER}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path not found: ${path}" >&2
    exit 2
  fi
done

cmd=(
  env CUDA_VISIBLE_DEVICES="${GPU_ID}"
  "${PY}" "${GPU_RETRIEVAL_SERVER}"
  --index_path "${INDEX_FILE}"
  --corpus_path "${CORPUS_FILE}"
  --topk 50
  --retriever_name e5
  --retriever_model "${RETRIEVER_MODEL}"
  --host 0.0.0.0
  --port "${PORT}"
  --device "${DEVICE}"
  --query_batch_size "${QUERY_BATCH_SIZE}"
  --doc_dtype "${DOC_DTYPE}"
)

if [[ "${START_BACKGROUND}" == "1" || "${START_BACKGROUND}" == "true" ]]; then
  echo "starting retriever on GPU ${GPU_ID}, port ${PORT}; log=${LOG_FILE}"
  setsid "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
  pid=$!
  echo "${pid}" > "${PID_FILE}"
  echo "pid=${pid}"
else
  echo "starting retriever in foreground on GPU ${GPU_ID}, port ${PORT}"
  exec "${cmd[@]}"
fi
