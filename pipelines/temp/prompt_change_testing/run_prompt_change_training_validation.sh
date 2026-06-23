#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
WORK_DIR="${ROOT}/pipelines/temp/prompt_change_testing"
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"

TRAIN_SRC="${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet"
EVAL_SRC="${ROOT}/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet"
TRAIN_SUBSET="${WORK_DIR}/data/train_4.parquet"
EVAL_SUBSET="${WORK_DIR}/data/eval_1.parquet"

TRAIN_RUN_NAME="${TRAIN_RUN_NAME:-prompt_change_train_real}"
TRAIN_LOG_DIR="${WORK_DIR}/train_logs"
TRAIN_OUT_DIR="${WORK_DIR}/train_out"
TRAIN_ROLLOUT_DIR="${TRAIN_LOG_DIR}/rollout_data"
TRAIN_VALIDATION_DIR="${TRAIN_LOG_DIR}/validation_data"
TRAIN_IO="${WORK_DIR}/train_llm_io.jsonl"
TRAIN_LOG="${WORK_DIR}/${TRAIN_RUN_NAME}.stdout.log"
TRAIN_INNER_LOG="${TRAIN_LOG_DIR}/${TRAIN_RUN_NAME}.train.log"
TRAIN_SEARCH_TIMING="${TRAIN_LOG_DIR}/${TRAIN_RUN_NAME}.search_timing.jsonl"
TRAIN_METRICS="${TRAIN_LOG_DIR}/${TRAIN_RUN_NAME}.metrics.jsonl"
TRAIN_NVIDIA_SMI="${TRAIN_LOG_DIR}/${TRAIN_RUN_NAME}.nvidia_smi.csv"
TRAIN_REPORT_PREFIX="${TRAIN_LOG_DIR}/${TRAIN_RUN_NAME}.timing_report"
TRAJECTORY_JSON="${WORK_DIR}/agent_training_trajectory.json"

mkdir -p "${WORK_DIR}" "${TRAIN_LOG_DIR}"
rm -f "${TRAIN_IO}" "${TRAIN_LOG}" "${TRAIN_INNER_LOG}" "${TRAIN_SEARCH_TIMING}" "${TRAIN_METRICS}" "${TRAIN_NVIDIA_SMI}" "${TRAJECTORY_JSON}"
rm -rf "${TRAIN_OUT_DIR}" "${TRAIN_ROLLOUT_DIR}" "${TRAIN_VALIDATION_DIR}"

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
    ROLLOUT_DATA_DIR="${TRAIN_ROLLOUT_DIR}" \
    VALIDATION_DATA_DIR="${TRAIN_VALIDATION_DIR}" \
    CHECKPOINT_ROOT="${WORK_DIR}/checkpoints" \
    TRAIN_LOG="${TRAIN_INNER_LOG}" \
    METRICS_JSONL="${TRAIN_METRICS}" \
    SEARCH_TIMING_JSONL="${TRAIN_SEARCH_TIMING}" \
    NVIDIA_SMI_CSV="${TRAIN_NVIDIA_SMI}" \
    REPORT_PREFIX="${TRAIN_REPORT_PREFIX}" \
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
    RECALL_GPU_ID="${TRAIN_RECALL_GPU_ID:-5}" \
    MAX_ASSISTANT_TURNS=2 \
    MAX_USER_TURNS=1 \
    MAX_PROMPT_LENGTH=4096 \
    MAX_RESPONSE_LENGTH=256 \
    MAX_MODEL_LEN=8192 \
    MAX_TOOL_RESPONSE_LENGTH=2048 \
    TOP_N=5 \
    TOP_M=5 \
    TP_SIZE=1 \
    MAX_NUM_SEQS=4 \
    AGENT_WORKERS=1 \
    RAY_NUM_CPUS=8 \
    RAY_OBJECT_STORE_MEMORY=2147483648 \
    GPU_MEMORY_UTILIZATION=0.25 \
    TEMPERATURE=0.0 \
    TOP_P=1.0 \
    RERANKER_TRAINABLE=false \
    DISABLE_RERANKER_ROLLOUT=true \
    USE_LLM_RERANKER=false \
    COSEARCH_LLM_IO_JSONL="${TRAIN_IO}" \
    COSEARCH_LLM_IO_MAX_RECORDS=40 \
    COAGENTIC_EXTRA_ARGS="actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.actor.fsdp_config.use_torch_compile=False actor_rollout_ref.ref.use_torch_compile=False actor_rollout_ref.ref.fsdp_config.use_torch_compile=False reranker_actor_rollout_ref.actor.use_torch_compile=False reranker_actor_rollout_ref.actor.fsdp_config.use_torch_compile=False reranker_actor_rollout_ref.ref.use_torch_compile=False reranker_actor_rollout_ref.ref.fsdp_config.use_torch_compile=False" \
    ALLOW_RUN_REUSE=1 \
    ALLOW_DIR_REUSE=1 \
    bash "${ROOT}/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh"
) 2>&1 | tee "${TRAIN_LOG}"

if [[ ! -d "${TRAIN_ROLLOUT_DIR}" ]]; then
  echo "ERROR: rollout_data not found under ${TRAIN_ROLLOUT_DIR}" >&2
  exit 2
fi

"${PY}" "${WORK_DIR}/extract_training_prompt_change_trajectory.py" \
  --train-subset "${TRAIN_SUBSET}" \
  --train-llm-io "${TRAIN_IO}" \
  --train-search-timing "${TRAIN_SEARCH_TIMING}" \
  --train-rollout-dir "${TRAIN_ROLLOUT_DIR}" \
  --out "${TRAJECTORY_JSON}"

echo "train_log=${TRAIN_LOG}"
echo "train_rollout_dir=${TRAIN_ROLLOUT_DIR}"
echo "trajectory_json=${TRAJECTORY_JSON}"
