#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PIPELINE_DIR}/../../.." && pwd)"
PARENT_ROOT="$(cd "${ROOT}/.." && pwd)"

PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
GPU_ID="${GPU_ID:-6}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
MODEL="${MODEL:-${PARENT_ROOT}/models/retriever/e5-base-v2}"
TOKENS_META="${TOKENS_META:?set TOKENS_META=/path/to/tokens_meta.json from 03_run_pretokenize_cpu.sh}"
OUT_DIR="${OUT_DIR:-${PIPELINE_DIR}/results/${RUN_ID}/encode_from_tokens_gpu06}"

mkdir -p "${OUT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

echo "run_id=${RUN_ID}"
echo "gpu_id=${GPU_ID}"
echo "out_dir=${OUT_DIR}"
echo "tokens_meta=${TOKENS_META}"
echo "batch_size=${BATCH_SIZE}"

exec "${PY}" "${PIPELINE_DIR}/encode_wiki18_e5_from_tokens.py" \
  --tokens-meta "${TOKENS_META}" \
  --model "${MODEL}" \
  --out-dir "${OUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --gpu-id "${GPU_ID}"
