#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PIPELINE_DIR}/../../.." && pwd)"
PARENT_ROOT="$(cd "${ROOT}/.." && pwd)"

PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
GPU_ID="${GPU_ID:-7}"
PORT="${PORT:-8053}"
DOC_DTYPE="${DOC_DTYPE:-float16}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-32}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

OLD_INDEX_FILE="${OLD_INDEX_FILE:-${ROOT}/data/retrieval/wiki-18/e5_Flat.index}"
if [[ -n "${ENCODE_OUT_DIR:-}" && -f "${ENCODE_OUT_DIR}/e5_Flat.index" ]]; then
  DEFAULT_NEW_INDEX_FILE="${ENCODE_OUT_DIR}/e5_Flat.index"
else
  DEFAULT_NEW_INDEX_FILE="${ROOT}/data/retrieval/wiki-18/e5_Flat.index"
fi
INDEX_FILE="${INDEX_FILE:-${DEFAULT_NEW_INDEX_FILE}}"
CORPUS_FILE="${CORPUS_FILE:-${ROOT}/data/retrieval/wiki-18/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-${PARENT_ROOT}/models/retriever/e5-base-v2}"
GPU_RETRIEVAL_SERVER="${GPU_RETRIEVAL_SERVER:-${ROOT}/src/retrievers/gpu_dense_retriever_server.py}"
OUT_DIR="${OUT_DIR:-${PIPELINE_DIR}/results/${RUN_ID}/load_gpu07}"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

"${PY}" "${PIPELINE_DIR}/measure_retriever_load.py" \
  --gpu-id "${GPU_ID}" \
  --port "${PORT}" \
  --old-index-path "${OLD_INDEX_FILE}" \
  --index-path "${INDEX_FILE}" \
  --corpus-path "${CORPUS_FILE}" \
  --retriever-model "${RETRIEVER_MODEL}" \
  --server-path "${GPU_RETRIEVAL_SERVER}" \
  --out-dir "${OUT_DIR}" \
  --doc-dtype "${DOC_DTYPE}" \
  --query-batch-size "${QUERY_BATCH_SIZE}"
