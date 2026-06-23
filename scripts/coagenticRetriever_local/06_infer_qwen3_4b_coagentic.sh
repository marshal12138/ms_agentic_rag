#!/usr/bin/env bash
set -euo pipefail

# CoAgenticRetriever eval wrapper with optional LLM-as-judge reranking.
#
# Usage example:
#   reranker=llm_as_judge RUN_MODE=full STRATEGY_NAME=my_judge_eval \
#     AGENT_MODEL=/path/to/global_step_x \
#     bash scripts/coagenticRetriever_local/06_infer_qwen3_4b_coagentic.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${ROOT}/CoAgenticRetriever}"

: "${TRAIN_DATA:=${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet}"
: "${VAL_DATA:=${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet}"
: "${DATA_PATH:=${VAL_DATA}}"
: "${GROUP_NAME:=coAgenticRetriever}"
: "${RUN_MODE:=full}"

RERANKER="${RERANKER:-${reranker:-dense_e5}}"
case "${RERANKER}" in
  dense|e5|dense-e5)
    RERANKER="dense_e5"
    ;;
  llm-as-judge|llm_judge|judge)
    RERANKER="llm_as_judge"
    ;;
  dense_e5|llm_as_judge) ;;
  *)
    echo "ERROR: unsupported reranker=${RERANKER}; use dense_e5 or llm_as_judge" >&2
    exit 2
    ;;
esac

: "${AGENT_GPU_IDS:=4,5}"
: "${AGENT_TP_SIZE:=2}"
: "${RANK_GPU_ID:=2}"
: "${RECALL_GPU_ID:=3}"
: "${LLM_JUDGE_GPU_IDS:=6,7}"
: "${LLM_JUDGE_TENSOR_PARALLEL_SIZE:=2}"
: "${LLM_JUDGE_SERVICE_CONFIG:=${PROJECT_ROOT}/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml}"
: "${LLM_JUDGE_ENDPOINT:=http://127.0.0.1:8067/v1/chat/completions}"
: "${LLM_JUDGE_MODEL:=DeepSeek-V4-Flash}"
: "${LLM_JUDGE_PROMPT_PATH:=${PROJECT_ROOT}/async_labeling/prompts/llm_judge_rank50_v1.md}"
: "${LLM_JUDGE_WAIT_SECONDS:=1800}"
: "${AUTO_START_LLM_JUDGE:=1}"
: "${AUTO_STOP_LLM_JUDGE:=0}"
: "${INJECT_TOOL_SCHEMA:=false}"

LLM_JUDGE_PID=""

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

llm_judge_models_url() {
  printf '%s\n' "${LLM_JUDGE_ENDPOINT%/v1/chat/completions}/v1/models"
}

check_llm_judge_service() {
  local models_url
  models_url="$(llm_judge_models_url)"
  "${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}" -c \
    "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('${models_url}', timeout=5).status < 500 else 1)" \
    >/dev/null 2>&1
}

cleanup_llm_judge_service() {
  if [[ -n "${LLM_JUDGE_PID}" ]] && is_truthy "${AUTO_STOP_LLM_JUDGE}"; then
    if kill -0 "${LLM_JUDGE_PID}" 2>/dev/null; then
      kill -TERM "${LLM_JUDGE_PID}" 2>/dev/null || true
      wait "${LLM_JUDGE_PID}" 2>/dev/null || true
    fi
  fi
}
trap cleanup_llm_judge_service EXIT INT TERM

ensure_llm_judge_service() {
  if [[ "${RERANKER}" != "llm_as_judge" || "${RUN_MODE}" == "no-ranker" ]]; then
    return 0
  fi
  if [[ ! -f "${LLM_JUDGE_SERVICE_CONFIG}" ]]; then
    echo "ERROR: LLM judge service config not found: ${LLM_JUDGE_SERVICE_CONFIG}" >&2
    exit 2
  fi
  if [[ ! -f "${LLM_JUDGE_PROMPT_PATH}" ]]; then
    echo "ERROR: LLM judge prompt not found: ${LLM_JUDGE_PROMPT_PATH}" >&2
    exit 2
  fi
  if check_llm_judge_service; then
    echo "LLM judge service already available: ${LLM_JUDGE_ENDPOINT}"
    return 0
  fi
  if ! is_truthy "${AUTO_START_LLM_JUDGE}"; then
    echo "ERROR: LLM judge service is unavailable and AUTO_START_LLM_JUDGE=${AUTO_START_LLM_JUDGE}" >&2
    echo "       endpoint=${LLM_JUDGE_ENDPOINT}" >&2
    exit 2
  fi
  if is_truthy "${DRY_RUN:-0}"; then
    LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS}" \
    LLM_JUDGE_TENSOR_PARALLEL_SIZE="${LLM_JUDGE_TENSOR_PARALLEL_SIZE}" \
      bash "${PROJECT_ROOT}/scripts/launch_llm_as_judge.sh" --config "${LLM_JUDGE_SERVICE_CONFIG}" --dry-run >/dev/null
    echo "DRY_RUN=1; LLM judge service config validated: ${LLM_JUDGE_SERVICE_CONFIG}"
    return 0
  fi

  echo "starting LLM judge service on GPUs ${LLM_JUDGE_GPU_IDS}; endpoint=${LLM_JUDGE_ENDPOINT}"
  LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS}" \
  LLM_JUDGE_TENSOR_PARALLEL_SIZE="${LLM_JUDGE_TENSOR_PARALLEL_SIZE}" \
    bash "${PROJECT_ROOT}/scripts/launch_llm_as_judge.sh" --config "${LLM_JUDGE_SERVICE_CONFIG}"

  local pid_file waited
  pid_file="${PROJECT_ROOT}/../log/llm_judge/vllm_gpu06_07_8067.pid"
  if [[ -f "${pid_file}" ]]; then
    LLM_JUDGE_PID="$(cat "${pid_file}")"
  fi
  waited=0
  while [[ "${waited}" -lt "${LLM_JUDGE_WAIT_SECONDS}" ]]; do
    if check_llm_judge_service; then
      echo "LLM judge service ready: ${LLM_JUDGE_ENDPOINT}"
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  echo "ERROR: timed out waiting for LLM judge service after ${LLM_JUDGE_WAIT_SECONDS}s" >&2
  exit 2
}

export TRAIN_DATA VAL_DATA DATA_PATH GROUP_NAME RUN_MODE
export RERANKER reranker="${RERANKER}"
export AGENT_GPU_IDS AGENT_TP_SIZE RANK_GPU_ID RECALL_GPU_ID
export LLM_JUDGE_ENDPOINT LLM_JUDGE_MODEL LLM_JUDGE_PROMPT_PATH
export LLM_JUDGE_GPU_IDS LLM_JUDGE_TENSOR_PARALLEL_SIZE
export INJECT_TOOL_SCHEMA

ensure_llm_judge_service

bash "${SCRIPT_DIR}/02_infer_qwen3_4b_ablation_val_only.sh" "$@"
