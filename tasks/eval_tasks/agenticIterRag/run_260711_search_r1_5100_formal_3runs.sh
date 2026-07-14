#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
EVAL_ENTRY="${ROOT}/tasks/eval_tasks/agenticIterRag/eval_spad_agent_search_350.sh"
DATA_PATH="${ROOT}/data/global_train_eval_data/350e/co_search_ablation.eval.parquet"
RUN_SPEC="${ROOT}/tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260711_search_r1_5100_formal.json"
AGGREGATE_DIR="${ROOT}/reports/eval/agenticIterRag/260711-newdata5100-search-r1-formal-aggregate"
MODEL_PATH="${ROOT}/checkpoints/AIR/260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79"
TASK_PREFIX="260711-newdata5100-search-r1-retry1"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/AgenticIterRag:${ROOT}/AgenticIterRag/verl:${PYTHONPATH:-}"

for repeat_id in 1 2 3; do
  task_name="${TASK_PREFIX}-run${repeat_id}"
  echo "[$(date '+%F %T %Z')] starting ${task_name}"
  bash "${EVAL_ENTRY}" \
    --agent-model "${MODEL_PATH}" \
    --data-path "${DATA_PATH}" \
    --max-samples 350 \
    --task-name "${task_name}" \
    --repeat-id "${repeat_id}"
  echo "[$(date '+%F %T %Z')] completed ${task_name}"
done

"${ROOT}/.venvs/ms_agt_rag_overlay/bin/python" \
  "${ROOT}/scripts/cosearch_local/aggregate_newdata_model_eval.py" \
  --run-spec "${RUN_SPEC}" \
  --data "${DATA_PATH}" \
  --output-dir "${AGGREGATE_DIR}" \
  --bootstrap-samples 10000 \
  --seed 42

echo "[$(date '+%F %T %Z')] Search-R1-5100 evaluations and aggregation completed"
