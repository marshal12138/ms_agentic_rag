#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
JUDGE_SCRIPT="${SCRIPT_DIR}/llm_judge_dpskv4f.sh"
CONFIG_PATH="${LLM_JUDGE_CONFIG:-${PROJECT_ROOT}/CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml}"
WAIT_TIMEOUT="${WAIT_TIMEOUT:-1200}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  "${JUDGE_SCRIPT}" start --config "${CONFIG_PATH}" --dry-run
  exit 0
fi

"${JUDGE_SCRIPT}" start --config "${CONFIG_PATH}" --wait --timeout "${WAIT_TIMEOUT}"
"${JUDGE_SCRIPT}" status --config "${CONFIG_PATH}"
