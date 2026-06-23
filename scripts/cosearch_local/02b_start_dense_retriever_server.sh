#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
PORT="${PORT:-8010}"
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-data/retrieval/wiki-18}"
INDEX_FILE="${INDEX_FILE:-${RETRIEVAL_DATA_DIR}/e5_Flat.index}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2}"
DEVICE="${DEVICE:-cpu}"
RETRIEVER_GPU_IDS="${RETRIEVER_GPU_IDS:-${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}}"
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
  echo "Run scripts/cosearch_local/01b_download_e5_and_build_dense_retriever.sh to download and merge official Search-R1 assets." >&2
  exit 2
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "ERROR: paper retrieval corpus not found: ${CORPUS_FILE}" >&2
  echo "Expected Search-R1 wiki-18.jsonl corpus." >&2
  exit 2
fi

"${PY}" scripts/cosearch_local/verify_official_retrieval_assets.py \
  --index "${INDEX_FILE}" \
  --corpus "${CORPUS_FILE}"

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

exec env CUDA_VISIBLE_DEVICES="${RETRIEVER_GPU_IDS}" "${PY}" CoSearch/Search-R1/search_r1/search/retrieval_server.py \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk 50 \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --device "${DEVICE}" \
  "${FAISS_GPU_ARG[@]}"
