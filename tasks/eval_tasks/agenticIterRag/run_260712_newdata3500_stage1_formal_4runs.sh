#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
EVAL_ENTRY="${ROOT}/tasks/eval_tasks/agenticIterRag/eval_agent_search.sh"
DATA_PATH="${ROOT}/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet"
RUN_SPEC="${ROOT}/tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260712_3500_stage1_formal.json"
AGGREGATE_DIR="${ROOT}/reports/eval/agenticIterRag/260712-newdata3500-stage1-formal-aggregate"
START_MODEL_INDEX="${START_MODEL_INDEX:-0}"
AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1,2,3,4,5}"
RECALL_GPU_IDS="${RECALL_GPU_IDS:-6,7}"
AGENT_MAX_NUM_SEQS="${AGENT_MAX_NUM_SEQS:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-384}"

SEARCH_R1_512_MODEL="${ROOT}/checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8"
SEARCH_R1_5100_MODEL="${ROOT}/checkpoints/AIR/260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79"
SPAD_512_STAGE1_MODEL="${ROOT}/checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8"
SPAD_5100_STAGE1_MODEL="${ROOT}/checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/AgenticIterRag:${ROOT}/AgenticIterRag/verl:${PYTHONPATH:-}"

run_once() {
  local model_path="$1"
  local task_name="$2"

  echo "[$(date '+%F %T %Z')] starting ${task_name}"
  bash "${EVAL_ENTRY}" \
    --agent-model "${model_path}" \
    --data-path "${DATA_PATH}" \
    --max-samples 3500 \
    --task-name "${task_name}" \
    --repeat-id 1 \
    --agent-gpu-ids "${AGENT_GPU_IDS}" \
    --recall-gpu-ids "${RECALL_GPU_IDS}" \
    --agent-max-num-seqs "${AGENT_MAX_NUM_SEQS}" \
    --infer-batch-size "${INFER_BATCH_SIZE}"
  echo "[$(date '+%F %T %Z')] completed ${task_name}"
}

if (( START_MODEL_INDEX <= 0 )); then
  run_once "${SEARCH_R1_512_MODEL}" "260712-newdata3500-fastio-search-r1-512-run1"
fi
if (( START_MODEL_INDEX <= 1 )); then
  run_once "${SEARCH_R1_5100_MODEL}" "260712-newdata3500-fastio-search-r1-5100-run1"
fi
if (( START_MODEL_INDEX <= 2 )); then
  run_once "${SPAD_512_STAGE1_MODEL}" "260712-newdata3500-fastio-spad-512-stage1-run1"
fi
if (( START_MODEL_INDEX <= 3 )); then
  run_once "${SPAD_5100_STAGE1_MODEL}" "260712-newdata3500-fastio-spad-5100-stage1-run1"
fi

"${ROOT}/.venvs/ms_agt_rag_overlay/bin/python" \
  "${ROOT}/scripts/cosearch_local/aggregate_newdata_model_eval.py" \
  --run-spec "${RUN_SPEC}" \
  --data "${DATA_PATH}" \
  --output-dir "${AGGREGATE_DIR}" \
  --bootstrap-samples 10000 \
  --seed 42

echo "[$(date '+%F %T %Z')] all 3500-sample evaluations and aggregation completed"
