#!/usr/bin/env bash
set -euo pipefail

# Shared dense retriever server launcher.
# Default mode is CPU/Search-R1 native FAISS. GPU mode uses the local torch
# GPU-resident flat retriever because the current faiss install lacks complete
# GPU resource APIs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARENT_ROOT="$(cd "${ROOT}/.." && pwd)"
source "${ROOT}/src/env_manage/compatible_accelerator.sh"

PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
PORT="${PORT:-8010}"
MODE="${MODE:-cpu}"
GPU_ID="${GPU_ID:-5}"
DOC_DTYPE="${DOC_DTYPE:-float16}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"

usage() {
  cat <<'EOF'
Usage: start_dense_retriever_server.sh [options]

Options:
  --mode cpu|gpu              Retrieval serving mode. Default: cpu.
  --port PORT                 Server port. Default: 8010.
  --gpu-id GPU_ID             GPU id for --mode gpu. Default: 5.
  --doc-dtype float16|float32 GPU doc embedding dtype. Default: float16.
  --query-batch-size N        Internal GPU query batch size. Default: 32.
  -h, --help                  Show this help.

Environment variables still work: MODE, PORT, GPU_ID, DOC_DTYPE,
QUERY_BATCH_SIZE, PY, INDEX_FILE, CORPUS_FILE, RETRIEVER_MODEL.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --gpu-id)
      GPU_ID="$2"
      shift 2
      ;;
    --doc-dtype)
      DOC_DTYPE="$2"
      shift 2
      ;;
    --query-batch-size)
      QUERY_BATCH_SIZE="$2"
      shift 2
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

case "${MODE}" in
  cpu|gpu) ;;
  *)
    echo "ERROR: --mode must be cpu or gpu, got: ${MODE}" >&2
    exit 2
    ;;
esac

case "${DOC_DTYPE}" in
  float16|float32) ;;
  *)
    echo "ERROR: --doc-dtype must be float16 or float32, got: ${DOC_DTYPE}" >&2
    exit 2
    ;;
esac

EXTERNAL_MODEL_ROOT="${EXTERNAL_MODEL_ROOT:-${PARENT_ROOT}/models}"
EXTERNAL_RETRIEVAL_ROOT="${EXTERNAL_RETRIEVAL_ROOT:-${ROOT}/data/retrieval}"
if [[ -z "${COSEARCH_PROJECT_ROOT:-}" ]]; then
  if [[ -d "${ROOT}/CoSearch" ]]; then
    COSEARCH_PROJECT_ROOT="${ROOT}/CoSearch"
  else
    COSEARCH_PROJECT_ROOT="${ROOT}/CoAgenticRetriever"
  fi
fi

RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}"
SEARCH_R1_RETRIEVAL_SERVER="${SEARCH_R1_RETRIEVAL_SERVER:-${COSEARCH_PROJECT_ROOT}/Search-R1/search_r1/search/retrieval_server.py}"
GPU_RETRIEVAL_SERVER="${GPU_RETRIEVAL_SERVER:-${ROOT}/src/retrievers/gpu_dense_retriever_server.py}"

DEVICE="${DEVICE:-cpu}"
if [[ "${MODE}" == "gpu" ]]; then
  DEVICE="$(co_accel_device_prefix)"
fi
RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS:-${GPU_ID}}"
FAISS_GPU="${FAISS_GPU:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "${ROOT}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OMP_NUM_THREADS MKL_NUM_THREADS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ ! -f "${INDEX_FILE}" ]]; then
  echo "ERROR: paper retrieval index not found: ${INDEX_FILE}" >&2
  echo "Download or build the official Search-R1 retrieval assets under ${RETRIEVAL_DATA_DIR}." >&2
  exit 2
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "ERROR: paper retrieval corpus not found: ${CORPUS_FILE}" >&2
  echo "Expected Search-R1 wiki-18.jsonl corpus." >&2
  exit 2
fi

if [[ "${MODE}" == "cpu" ]]; then
  if [[ ! -f "${SEARCH_R1_RETRIEVAL_SERVER}" ]]; then
    echo "ERROR: Search-R1 retrieval server not found: ${SEARCH_R1_RETRIEVAL_SERVER}" >&2
    exit 2
  fi
else
  if [[ ! -f "${GPU_RETRIEVAL_SERVER}" ]]; then
    echo "ERROR: GPU retrieval server not found: ${GPU_RETRIEVAL_SERVER}" >&2
    exit 2
  fi
fi

"${PY}" "${ROOT}/src/retrievers/verify_official_retrieval_assets.py" \
  --index "${INDEX_FILE}" \
  --corpus "${CORPUS_FILE}"

if [[ "${MODE}" == "gpu" ]]; then
  echo "Starting ${COSEARCH_ACCELERATOR} dense retriever: devices=${RETRIEVER_GPU_IDS}, doc_dtype=${DOC_DTYPE}, port=${PORT}" >&2
  exec env $(co_accel_env_visible_devices_cmd "${RETRIEVER_GPU_IDS}") "${PY}" "${GPU_RETRIEVAL_SERVER}" \
    --index_path "${INDEX_FILE}" \
    --corpus_path "${CORPUS_FILE}" \
    --topk 50 \
    --retriever_name e5 \
    --retriever_model "${RETRIEVER_MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --device "${DEVICE}" \
    --query_batch_size "${QUERY_BATCH_SIZE}" \
    --doc_dtype "${DOC_DTYPE}"
fi

FAISS_GPU_ARG=()
if [[ "${FAISS_GPU}" == "1" || "${FAISS_GPU}" == "true" || "${FAISS_GPU}" == "yes" ]]; then
  if "${PY}" - <<'PY'
import faiss
raise SystemExit(0 if hasattr(faiss, "GpuMultipleClonerOptions") else 1)
PY
  then
    FAISS_GPU_ARG=(--faiss_gpu)
  else
    echo "WARNING: FAISS_GPU=${FAISS_GPU} requested, but this Python environment has faiss-cpu without GPU APIs; falling back to CPU FAISS." >&2
    FAISS_GPU_ARG=()
    if [[ "${DEVICE}" == "cuda" ]]; then
      DEVICE="cpu"
    fi
  fi
fi

echo "Starting CPU dense retriever: port=${PORT}" >&2
exec env $(co_accel_env_visible_devices_cmd "${RETRIEVER_GPU_IDS}") "${PY}" "${SEARCH_R1_RETRIEVAL_SERVER}" \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk 50 \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device "${DEVICE}" \
  "${FAISS_GPU_ARG[@]}"
