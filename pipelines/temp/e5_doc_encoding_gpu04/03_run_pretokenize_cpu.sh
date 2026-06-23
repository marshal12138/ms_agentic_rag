#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PIPELINE_DIR}/../../.." && pwd)"
PARENT_ROOT="$(cd "${ROOT}/.." && pwd)"

PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
MAX_LENGTH="${MAX_LENGTH:-256}"
LIMIT="${LIMIT:-0}"
NUM_DOCS="${NUM_DOCS:-21015324}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

CORPUS="${CORPUS:-${ROOT}/data/retrieval/wiki-18/wiki-18.jsonl}"
MODEL="${MODEL:-${PARENT_ROOT}/models/retriever/e5-base-v2}"
OUT_DIR="${OUT_DIR:-${PIPELINE_DIR}/results/${RUN_ID}/tokens_cpu}"

mkdir -p "${OUT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-64}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

echo "run_id=${RUN_ID}"
echo "out_dir=${OUT_DIR}"
echo "batch_size=${BATCH_SIZE}"
echo "tokenizers_parallelism=${TOKENIZERS_PARALLELISM}"
echo "rayon_num_threads=${RAYON_NUM_THREADS}"

exec "${PY}" "${PIPELINE_DIR}/pretokenize_wiki18_e5.py" \
  --corpus "${CORPUS}" \
  --model "${MODEL}" \
  --out-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --max-length "${MAX_LENGTH}" \
  --limit "${LIMIT}" \
  --num-docs "${NUM_DOCS}"
