#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
EVAL_ENTRY="${ROOT}/tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh"
DATA_PATH="${ROOT}/data/global_train_eval_data/350e/co_search_ablation.eval.parquet"
RUN_SPEC="${ROOT}/tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260711_512_formal.json"
AGGREGATE_DIR="${ROOT}/reports/eval/agenticIterRag/260711-newdata512-formal-aggregate"
START_MODEL_INDEX="${START_MODEL_INDEX:-0}"

BASE_MODEL="/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-1.7B"
SEARCH_R1_MODEL="${ROOT}/checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8"
SPAD_STAGE1_MODEL="${ROOT}/checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8"
SPAD_STAGE3_MODEL="${ROOT}/checkpoints/AIR/260711-115144-826023-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stage3_resume/stages/train_agent/spad_rag/answer_distillation/grpo/grpo_checkpoint_verl/actor_model_hf/global_step_3"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/AgenticIterRag:${ROOT}/AgenticIterRag/verl:${PYTHONPATH:-}"

run_repeats() {
  local model_path="$1"
  local task_prefix="$2"
  local repeat_id task_name

  for repeat_id in 1 2 3; do
    task_name="${task_prefix}-run${repeat_id}"
    echo "[$(date '+%F %T %Z')] starting ${task_name}"
    bash "${EVAL_ENTRY}" \
      --agent-model "${model_path}" \
      --data-path "${DATA_PATH}" \
      --max-samples 350 \
      --task-name "${task_name}" \
      --repeat-id "${repeat_id}"
    echo "[$(date '+%F %T %Z')] completed ${task_name}"
  done
}

if (( START_MODEL_INDEX <= 0 )); then
  run_repeats "${BASE_MODEL}" "260711-newdata512-base-retry1"
fi
if (( START_MODEL_INDEX <= 1 )); then
  run_repeats "${SEARCH_R1_MODEL}" "260711-newdata512-search-r1"
fi
if (( START_MODEL_INDEX <= 2 )); then
  run_repeats "${SPAD_STAGE1_MODEL}" "260711-newdata512-spad-stage1"
fi
if (( START_MODEL_INDEX <= 3 )); then
  run_repeats "${SPAD_STAGE3_MODEL}" "260711-newdata512-spad-stage3-retry1"
fi

"${ROOT}/.venvs/ms_agt_rag_overlay/bin/python" \
  "${ROOT}/scripts/cosearch_local/aggregate_newdata_model_eval.py" \
  --run-spec "${RUN_SPEC}" \
  --data "${DATA_PATH}" \
  --output-dir "${AGGREGATE_DIR}" \
  --bootstrap-samples 10000 \
  --seed 42

echo "[$(date '+%F %T %Z')] all formal evaluations and aggregation completed"
