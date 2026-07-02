#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
cd "${ROOT}"

AGENT_CKPT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260622-220205-CAR_async_ranker_training_ds_flash_mix_signal_b3_v1_select_all/global_step_79"
DATA_PATH="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet"

AGENT_GPU_IDS="${AGENT_GPU_IDS:-3,4}"
AGENT_TP_SIZE="${AGENT_TP_SIZE:-2}"
RECALL_GPU_ID="${RECALL_GPU_ID:-5}"
LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS:-6,7}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}}"
WAIT_FOR_GPU_RELEASE="${WAIT_FOR_GPU_RELEASE:-1}"
WAIT_FOR_GPU_INTERVAL_SECONDS="${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}"
WAIT_FOR_GPU_TIMEOUT_SECONDS="${WAIT_FOR_GPU_TIMEOUT_SECONDS:-0}"
WAIT_FOR_GPU_LABEL="${WAIT_FOR_GPU_LABEL:-dpskv4f-as-rank eval GPU wait}"

source "${ROOT}/src/runtime/wait_for_gpus.sh"
wait_for_gpus_if_enabled

STRATEGY_NAME=llm_judge_as_rank_wt_trained_agt_llm \
EVAL_BUDGET_YAML="${EVAL_BUDGET_YAML:-scripts/coagenticRetriever_local/strategies_yaml/rollout_cosearch_aligned_budget.yaml}" \
INJECT_TOOL_SCHEMA="${INJECT_TOOL_SCHEMA:-false}" \
RUN_MODE=full \
reranker=llm_as_judge \
AGENT_MODEL="${AGENT_CKPT}" \
DATA_PATH="${DATA_PATH}" \
AGENT_GPU_IDS="${AGENT_GPU_IDS}" \
AGENT_TP_SIZE="${AGENT_TP_SIZE}" \
RECALL_GPU_ID="${RECALL_GPU_ID}" \
LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS}" \
LLM_JUDGE_TENSOR_PARALLEL_SIZE=2 \
LLM_JUDGE_MODEL=DeepSeek-V4-Flash \
LLM_JUDGE_ENDPOINT=http://127.0.0.1:8067/v1/chat/completions \
LLM_JUDGE_MAX_RETRIES=3 \
LLM_JUDGE_REQUEST_TIMEOUT=600 \
MAX_EVAL_NUM=-1 \
EVAL_BATCH_SIZE=16 \
KEEP_TRACE=partial \
bash scripts/coagenticRetriever_local/06_infer_qwen3_4b_coagentic.sh
