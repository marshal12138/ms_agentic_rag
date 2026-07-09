#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JUDGE_SCRIPT="${SCRIPT_DIR}/llm_judge_dpskv4f.sh"
CONFIG_PATH="${LLM_JUDGE_CONFIG:-${PROJECT_ROOT}/CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml}"

INPUT_JSONL="${INPUT_JSONL:-${PROJECT_ROOT}/data/llm_judge/chunk_ranking/examples/chunk_ranking_judge_examples_100.jsonl}"
WORK_DIR="${WORK_DIR:-${SCRIPT_DIR}/run_examples}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${WORK_DIR}/batch_results.jsonl}"
CONCURRENCY="${CONCURRENCY:-2}"
LIMIT="${LIMIT:-25}"

mkdir -p "${WORK_DIR}"

DRY_RUN_ARGS=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_ARGS=(--dry-run)
fi

"${JUDGE_SCRIPT}" batch \
  --config "${CONFIG_PATH}" \
  --input "${INPUT_JSONL}" \
  --output "${OUTPUT_JSONL}" \
  --concurrency "${CONCURRENCY}" \
  --limit "${LIMIT}" \
  "${DRY_RUN_ARGS[@]}"

echo "batch result: ${OUTPUT_JSONL}"
