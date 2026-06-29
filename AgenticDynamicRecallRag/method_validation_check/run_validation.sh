#!/usr/bin/env bash
# set -euo pipefail

# Launch the alpha-fusion validation pipeline.
# Resolves the project Python the same way the other launch scripts do, then
# runs the validation module from the repo root.
#
# Override anything via env or pass extra CLI flags through, e.g.:
#   LLM_ENDPOINT=http://127.0.0.1:8067/v1/chat/completions \
#   LLM_MODEL=DeepSeek-V4-Flash \
#   RETRIEVER_ENDPOINT=http://localhost:9011/retrieve \
#   bash run_validation.sh --num-samples 100

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../.." && pwd)"

LLM_ENDPOINT="${LLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}"
LLM_MODEL="${LLM_MODEL:-qwen3-4b}"
RETRIEVER_ENDPOINT="${RETRIEVER_ENDPOINT:-http://localhost:9011/retrieve}"
ensure_dir() {
    local dir=$(dirname "$1")
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
    fi
}

NUM_SAMPLES="${NUM_SAMPLES:-1000}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_PATH="${OUTPUT_PATH:-${HERE}/result/results_${NUM_SAMPLES}_${TIMESTAMP}.jsonl}"
REPORT_PATH="${REPORT_PATH:-${HERE}/result/report_${NUM_SAMPLES}_${TIMESTAMP}.png}"

ensure_dir "$OUTPUT_PATH"
ensure_dir "$REPORT_PATH"
TEST_DATA="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/raw_sets/2wikimultihopqa/dev.jsonl"

cd "${ROOT}"
python -m AgenticDynamicRecallRag.method_validation_check.main \
  --llm-endpoint "${LLM_ENDPOINT}" \
  --llm-model "${LLM_MODEL}" \
  --retriever-endpoint "${RETRIEVER_ENDPOINT}" \
  --num-samples "${NUM_SAMPLES}" \
  --output-path "${OUTPUT_PATH}" \
  --report-path "${REPORT_PATH}" \
  --dataset-path "${TEST_DATA}"
