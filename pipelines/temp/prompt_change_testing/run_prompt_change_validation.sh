#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
WORK_DIR="${ROOT}/pipelines/temp/prompt_change_testing"
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"

TRAIN_SRC="${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet"
EVAL_SRC="${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet"
TRAIN_SUBSET="${WORK_DIR}/data/train_4.parquet"
EVAL_SUBSET="${WORK_DIR}/data/eval_1.parquet"

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-prompt_change_train}"
TRAIN_LOG_DIR="${WORK_DIR}/train_logs"
TRAIN_OUT_DIR="${WORK_DIR}/train_out"
TRAIN_IO="${WORK_DIR}/train_llm_io.jsonl"
TRAIN_LOG="${WORK_DIR}/${TRAIN_RUN_NAME}.stdout.log"
TRAIN_SEARCH_TIMING="${TRAIN_LOG_DIR}/${TRAIN_RUN_NAME}.search_timing.jsonl"

EVAL_RUN_NAME="${EVAL_RUN_NAME:-prompt_change_eval}"
EVAL_TRACE_DIR="${WORK_DIR}/eval_trace"
EVAL_RUNTIME_LOG_DIR="${WORK_DIR}/eval_runtime_logs"
EVAL_IO="${WORK_DIR}/eval_llm_io.jsonl"
EVAL_LOG="${WORK_DIR}/${EVAL_RUN_NAME}.stdout.log"
TRAJECTORY_JSON="${WORK_DIR}/agent_trajectory.json"

mkdir -p "${WORK_DIR}" "${TRAIN_LOG_DIR}" "${EVAL_TRACE_DIR}" "${EVAL_RUNTIME_LOG_DIR}"
rm -f "${TRAIN_IO}" "${EVAL_IO}" "${TRAIN_LOG}" "${EVAL_LOG}" "${TRAJECTORY_JSON}"
rm -rf "${TRAIN_OUT_DIR}" "${EVAL_TRACE_DIR}" "${EVAL_RUNTIME_LOG_DIR}"
mkdir -p "${EVAL_TRACE_DIR}" "${EVAL_RUNTIME_LOG_DIR}"

"${PY}" "${WORK_DIR}/prepare_prompt_change_validation_data.py" \
  --train-src "${TRAIN_SRC}" \
  --eval-src "${EVAL_SRC}" \
  --train-dst "${TRAIN_SUBSET}" \
  --eval-dst "${EVAL_SUBSET}" \
  --train-count 4 \
  --eval-count 1

(
  cd "${ROOT}"
  env \
    PY="${PY}" \
    EXP_NAME="${TRAIN_RUN_NAME}" \
    LOG_DIR="${TRAIN_LOG_DIR}" \
    OUT_DIR="${TRAIN_OUT_DIR}" \
    CHECKPOINT_ROOT="${WORK_DIR}/checkpoints" \
    TRAIN_DATA="${TRAIN_SUBSET}" \
    VAL_DATA="${EVAL_SUBSET}" \
    TRAIN_MAX_SAMPLES=4 \
    VAL_MAX_SAMPLES=1 \
    TRAIN_BATCH_SIZE=4 \
    ACTOR_BATCH_SIZE=4 \
    VAL_BATCH_SIZE=1 \
    TOTAL_STEPS=1 \
    N_ROLLOUTS=1 \
    AGENT_GPU_IDS="${TRAIN_AGENT_GPU_IDS:-0,1,2,3}" \
    RANK_GPU_ID="${TRAIN_RANK_GPU_ID:-4}" \
    RECALL_GPU_ID="${TRAIN_RECALL_GPU_ID:-4}" \
    RETRIEVER_DEVICE=cpu \
    RECALL_RETRIEVER_CONFIG_DEVICE=cpu \
    RECALL_RETRIEVER_DEVICE=cpu \
    RECALL_RETRIEVER_CONFIG_DEVICE=cpu \
    RETRIEVER_CONTRASTIVE_BATCH_SIZE=4 \
    RETRIEVER_STEPS_PER_GLOBAL_STEP=1 \
    MAX_ASSISTANT_TURNS=2 \
    MAX_USER_TURNS=1 \
    MAX_PROMPT_LENGTH=4096 \
    MAX_RESPONSE_LENGTH=160 \
    MAX_MODEL_LEN=8192 \
    MAX_TOOL_RESPONSE_LENGTH=2048 \
    TOP_N=5 \
    TOP_M=5 \
    TP_SIZE=1 \
    MAX_NUM_SEQS=1 \
    AGENT_WORKERS=1 \
    RAY_NUM_CPUS=8 \
    RAY_OBJECT_STORE_MEMORY=2147483648 \
    GPU_MEMORY_UTILIZATION=0.25 \
    RERANKER_TRAINABLE=false \
    COSEARCH_LLM_IO_JSONL="${TRAIN_IO}" \
    COSEARCH_LLM_IO_MAX_RECORDS=40 \
    ALLOW_RUN_REUSE=1 \
    ALLOW_DIR_REUSE=1 \
    bash "${ROOT}/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh"
) 2>&1 | tee "${TRAIN_LOG}"

TRAIN_CHECKPOINT="$(find "${TRAIN_OUT_DIR}" -maxdepth 1 -type d -name 'global_step_*' | sort | tail -n 1)"
if [[ -z "${TRAIN_CHECKPOINT}" ]]; then
  echo "ERROR: no global_step_* checkpoint found under ${TRAIN_OUT_DIR}" >&2
  exit 2
fi

(
  cd "${ROOT}"
  env \
    PY="${PY}" \
    RUN_NAME="${EVAL_RUN_NAME}" \
    RUN_MODE=co-training \
    CHECKPOINT_DIR="${TRAIN_CHECKPOINT}" \
    AGENT_MODEL="${TRAIN_CHECKPOINT}" \
    RERANKER_MODEL="/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B" \
    VAL_DATA="${EVAL_SUBSET}" \
    MAX_EVAL_STEPS=1 \
    VAL_BATCH_SIZE=1 \
    TOP_K=5 \
    RECALL_TOP_K=5 \
    AGENT_GPU_IDS="${EVAL_AGENT_GPU_IDS:-0}" \
    RERANKER_GPU_IDS="${EVAL_RERANKER_GPU_IDS:-1}" \
    AGENT_TP_SIZE=1 \
    RERANKER_TP_SIZE=1 \
    AGENT_PORT="${EVAL_AGENT_PORT:-8040}" \
    RERANKER_PORT="${EVAL_RERANKER_PORT:-8041}" \
    PROXY_PORT="${EVAL_PROXY_PORT:-8030}" \
    RETRIEVER_PORT_BASE="${EVAL_RETRIEVER_PORT_BASE:-8020}" \
    RETRIEVER_INSTANCES=1 \
    KEEP_TRACE=full \
    LLM_IO_JSONL="${EVAL_IO}" \
    COSEARCH_LLM_IO_MAX_RECORDS=40 \
    MAX_ASSISTANT_TURNS=2 \
    MAX_USER_TURNS=1 \
    MAX_PROMPT_LENGTH=4096 \
    MAX_RESPONSE_LENGTH=160 \
    MAX_MODEL_LEN=8192 \
    MAX_TOOL_RESPONSE_LENGTH=2048 \
    RERANKER_MAX_PROMPT_LENGTH=16384 \
    RERANKER_MAX_RESPONSE_LENGTH=160 \
    TEMPERATURE=0.0 \
    TOP_P=1.0 \
    RERANKER_TEMPERATURE=0.0 \
    GPU_MEMORY_UTILIZATION=0.25 \
    MAX_NUM_SEQS=1 \
    TRACE_DIR="${EVAL_TRACE_DIR}" \
    RUNTIME_LOG_DIR="${EVAL_RUNTIME_LOG_DIR}" \
    REPORT_PATH="${WORK_DIR}/${EVAL_RUN_NAME}.report.md" \
    bash "${ROOT}/scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh"
) 2>&1 | tee "${EVAL_LOG}"

"${PY}" "${WORK_DIR}/extract_prompt_change_trajectory.py" \
  --train-llm-io "${TRAIN_IO}" \
  --train-rollout-dir "${TRAIN_OUT_DIR}/rollout_data" \
  --train-search-timing "${TRAIN_SEARCH_TIMING}" \
  --eval-llm-io "${EVAL_IO}" \
  --eval-traces "${EVAL_TRACE_DIR}/traces.jsonl" \
  --out "${TRAJECTORY_JSON}"

echo "train_log=${TRAIN_LOG}"
echo "eval_log=${EVAL_LOG}"
echo "train_checkpoint=${TRAIN_CHECKPOINT}"
echo "trajectory_json=${TRAJECTORY_JSON}"
