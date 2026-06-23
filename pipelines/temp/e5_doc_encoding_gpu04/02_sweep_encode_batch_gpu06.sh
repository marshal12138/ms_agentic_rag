#!/usr/bin/env bash
set -euo pipefail

PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_ID="${GPU_ID:-6}"
LIMIT="${LIMIT:-131072}"
MAX_LENGTH="${MAX_LENGTH:-256}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZES="${BATCH_SIZES:-256 1024 2048 4096}"

for bs in ${BATCH_SIZES}; do
  echo "sweep batch_size=${bs}, limit=${LIMIT}, gpu=${GPU_ID}"
  OUT_DIR="${PIPELINE_DIR}/results/${RUN_ID}/sweep_bs${bs}" \
  GPU_ID="${GPU_ID}" \
  BATCH_SIZE="${bs}" \
  MAX_LENGTH="${MAX_LENGTH}" \
  LIMIT="${LIMIT}" \
  RUN_ID="${RUN_ID}" \
  TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}" \
  RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-32}" \
  bash "${PIPELINE_DIR}/00_run_encode_gpu06.sh"
done
