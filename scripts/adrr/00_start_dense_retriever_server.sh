#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# =========================
# External resource dependencies
# =========================
# These paths intentionally point to shared assets outside this lightweight
# project. Edit this block when moving models or retrieval data.
COAGENTIC_LEARN_ROOT="${COAGENTIC_LEARN_ROOT:-$(cd "${ROOT}/.." && pwd)/CoSearch_derevitives}"
EXTERNAL_MODEL_ROOT="${EXTERNAL_MODEL_ROOT:-$(cd "${ROOT}/.." && pwd)/models}"
EXTERNAL_RETRIEVAL_ROOT="${EXTERNAL_RETRIEVAL_ROOT:-${COAGENTIC_LEARN_ROOT}/data/retrieval}"
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"
BM25_FILE="${BM25_FILE:-${RETRIEVAL_DATA_DIR}/bm25/bm25}"
GRAPH_FILE="${GRAPH_FILE:-${SCRIPT_DIR}/wiki_graph.graphml}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${EXTERNAL_MODEL_ROOT}/retriever/e5-base-v2}"
RETRIEVER_SRC_DIR="${RETRIEVER_SRC_DIR:-${SCRIPT_DIR}/src/retrievers}"
GPU_DENSE_RETRIEVER_SERVER="${GPU_DENSE_RETRIEVER_SERVER:-${RETRIEVER_SRC_DIR}/gpu_dense_retriever_server_weight.py}"
VERIFY_RETRIEVAL_ASSETS="${VERIFY_RETRIEVAL_ASSETS:-${RETRIEVER_SRC_DIR}/verify_official_retrieval_assets.py}"

PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
PORT="${PORT:-8030}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cuda}"
RECALL_GPU_ID="${RECALL_GPU_ID:-5}"
RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS:-${GPU_ID:-${RECALL_GPU_ID}}}"
RECALL_TOP_K="${RECALL_TOP_K:-50}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
DOC_DTYPE="${DOC_DTYPE:-float16}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
DRY_RUN="${DRY_RUN:-0}"

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
BM25_FILE=${BM25_FILE}
GRAPH_FILE=${GRAPH_FILE}
RETRIEVER_MODEL=${RETRIEVER_MODEL}
CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU_IDS}
DEVICE=${DEVICE}
DOC_DTYPE=${DOC_DTYPE}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE}
HOST=${HOST}
PORT=${PORT}
RECALL_TOP_K=${RECALL_TOP_K}
EOF
  exit 0
fi

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

if [[ ! -f "${GPU_DENSE_RETRIEVER_SERVER}" ]]; then
  echo "ERROR: GPU dense retrieval server not found: ${GPU_DENSE_RETRIEVER_SERVER}" >&2
  exit 2
fi

if [[ ! -f "${VERIFY_RETRIEVAL_ASSETS}" ]]; then
  echo "ERROR: retrieval asset verifier not found: ${VERIFY_RETRIEVAL_ASSETS}" >&2
  exit 2
fi

if [[ "${DEVICE}" != cuda* ]]; then
  echo "ERROR: CoAgenticRetriever dense retrieval server requires DEVICE=cuda; got DEVICE=${DEVICE}" >&2
  exit 2
fi

if ! env CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU_IDS}" "${PY}" - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "ERROR: CUDA is not visible to PyTorch for dense retrieval server; refusing to run on CPU." >&2
  exit 2
fi

"${PY}" "${VERIFY_RETRIEVAL_ASSETS}" \
  --index "${INDEX_FILE}" \
  --corpus "${CORPUS_FILE}"

echo "Starting GPU dense retriever from ${GPU_DENSE_RETRIEVER_SERVER}" >&2
echo "  CUDA_VISIBLE_DEVICES=${RETRIEVER_GPU_IDS}" >&2
echo "  device=${DEVICE}; doc embeddings will be loaded into GPU memory as ${DOC_DTYPE}" >&2
echo "  retrieval endpoint=http://${HOST}:${PORT}/retrieve, topk=${RECALL_TOP_K}" >&2

exec env CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU_IDS}" "${PY}" "${GPU_DENSE_RETRIEVER_SERVER}" \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --bm25_path "${BM25_FILE}" \
  --graph_path "${GRAPH_FILE}" \
  --topk "${RECALL_TOP_K}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --retriever_model "${RETRIEVER_MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --doc_dtype "${DOC_DTYPE}"
