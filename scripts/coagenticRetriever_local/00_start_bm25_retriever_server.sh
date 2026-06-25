#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =========================
# External resource dependencies
# =========================
# This launcher intentionally uses CoAgenticRetriever/Search-R1's Python
# retrieval server and a dedicated text-retriever environment.
COAGENTIC_LEARN_ROOT="${COAGENTIC_LEARN_ROOT:-${ROOT}}"
EXTERNAL_RETRIEVAL_ROOT="${EXTERNAL_RETRIEVAL_ROOT:-${COAGENTIC_LEARN_ROOT}/data/retrieval}"
RETRIEVAL_DATA_DIR="${RETRIEVAL_DATA_DIR:-${EXTERNAL_RETRIEVAL_ROOT}/wiki-18}"
INDEX_DIR="${INDEX_DIR:-${BM25_INDEX_DIR:-${RETRIEVAL_DATA_DIR}/bm25/bm25}}"
CORPUS_FILE="${CORPUS_FILE:-${RETRIEVAL_DATA_DIR}/wiki-18.jsonl}"

COAGENTIC_SEARCH_R1_ROOT="${COAGENTIC_SEARCH_R1_ROOT:-${COAGENTIC_LEARN_ROOT}/CoAgenticRetriever/Search-R1}"
RETRIEVAL_SERVER="${RETRIEVAL_SERVER:-${COAGENTIC_SEARCH_R1_ROOT}/search_r1/search/retrieval_server.py}"

PY="${PY:-/data04/envs/ms/ms_txt_retriever/bin/python}"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"

PORT="${PORT:-8030}"
HOST="${HOST:-0.0.0.0}"
DEVICE="${DEVICE:-cpu}"
RECALL_TOP_K="${RECALL_TOP_K:-50}"
RETRIEVER_NAME="${RETRIEVER_NAME:-bm25}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
DRY_RUN="${DRY_RUN:-0}"

cd "${COAGENTIC_SEARCH_R1_ROOT}"
# BM25 Search-R1 loads wiki-18.jsonl through HuggingFace datasets. Reusing the
# shared cache avoids rebuilding a 14GB JSONL cache every service start.
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OMP_NUM_THREADS MKL_NUM_THREADS
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ "${DRY_RUN}" == "1" || "${DRY_RUN}" == "true" || "${DRY_RUN}" == "yes" ]]; then
  cat <<EOF
DRY_RUN=1
ROOT=${ROOT}
SCRIPT_DIR=${SCRIPT_DIR}
COAGENTIC_SEARCH_R1_ROOT=${COAGENTIC_SEARCH_R1_ROOT}
RETRIEVAL_SERVER=${RETRIEVAL_SERVER}
PY=${PY}
INDEX_DIR=${INDEX_DIR}
CORPUS_FILE=${CORPUS_FILE}
RETRIEVER_NAME=${RETRIEVER_NAME}
DEVICE=${DEVICE}
HOST=${HOST}
PORT=${PORT}
RECALL_TOP_K=${RECALL_TOP_K}
HF_HOME=${HF_HOME}
HF_DATASETS_CACHE=${HF_DATASETS_CACHE}
OMP_NUM_THREADS=${OMP_NUM_THREADS}
MKL_NUM_THREADS=${MKL_NUM_THREADS}
EOF
  exit 0
fi

if [[ "${RETRIEVER_NAME}" != "bm25" ]]; then
  echo "ERROR: this launcher is BM25-only; got RETRIEVER_NAME=${RETRIEVER_NAME}" >&2
  exit 2
fi

if [[ ! -x "${PY}" ]]; then
  echo "ERROR: Python executable not found or not executable: ${PY}" >&2
  echo "Expected the dedicated BM25 environment at /data04/envs/ms/ms_txt_retriever." >&2
  exit 2
fi

if [[ ! -f "${RETRIEVAL_SERVER}" ]]; then
  echo "ERROR: CoAgenticRetriever Search-R1 retrieval server not found: ${RETRIEVAL_SERVER}" >&2
  exit 2
fi

if [[ ! -d "${INDEX_DIR}" ]]; then
  echo "ERROR: BM25 index directory not found: ${INDEX_DIR}" >&2
  echo "Expected the full wiki-18 Lucene index, e.g. ${RETRIEVAL_DATA_DIR}/bm25/bm25." >&2
  exit 2
fi

if [[ ! -f "${INDEX_DIR}/segments_1" && -z "$(find "${INDEX_DIR}" -maxdepth 1 -name 'segments_*' -print -quit)" ]]; then
  echo "ERROR: BM25 index directory does not look like a Lucene index: ${INDEX_DIR}" >&2
  exit 2
fi

if [[ ! -f "${CORPUS_FILE}" ]]; then
  echo "ERROR: wiki-18 corpus not found: ${CORPUS_FILE}" >&2
  exit 2
fi

"${PY}" - <<'PY'
import importlib
missing = []
for name in ("faiss", "pyserini", "jnius", "fastapi", "uvicorn", "datasets"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc!r}")
if missing:
    raise SystemExit("Missing BM25 retrieval dependencies:\n" + "\n".join(missing))
PY

echo "Starting BM25 retriever from ${RETRIEVAL_SERVER}" >&2
echo "  python=${PY}" >&2
echo "  index=${INDEX_DIR}" >&2
echo "  corpus=${CORPUS_FILE}" >&2
echo "  retrieval endpoint=http://${HOST}:${PORT}/retrieve, topk=${RECALL_TOP_K}" >&2
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}" >&2

exec "${PY}" "${RETRIEVAL_SERVER}" \
  --index_path "${INDEX_DIR}" \
  --corpus_path "${CORPUS_FILE}" \
  --topk "${RECALL_TOP_K}" \
  --retriever_name "${RETRIEVER_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --device "${DEVICE}"
