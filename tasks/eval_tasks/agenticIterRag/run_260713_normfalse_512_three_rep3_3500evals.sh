#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
EVAL_ENTRY="${ROOT}/tasks/eval_tasks/agenticIterRag/eval_agent_search.sh"
DATA_PATH="${ROOT}/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet"
RUN_SPEC="${ROOT}/tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260713_normfalse_512_three_repeats.json"
AGGREGATE_DIR="${ROOT}/reports/eval/agenticIterRag/260713-newdata3500-normfalse-512-three-repeats-aggregate"
START_MODEL_INDEX="${START_MODEL_INDEX:-0}"
AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1,2,3,4,5}"
RECALL_GPU_IDS="${RECALL_GPU_IDS:-6,7}"
AGENT_MAX_NUM_SEQS="${AGENT_MAX_NUM_SEQS:-64}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-384}"
FLUSH_EVERY_N="${FLUSH_EVERY_N:-500}"

MODELS=(
  "${ROOT}/checkpoints/AIR/260713-185433-916978-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512_normfalse_rep3/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8"
  "${ROOT}/checkpoints/AIR/260713-192527-981329-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_normfalse_rep3/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8"
  "${ROOT}/checkpoints/AIR/260713-201539-129092-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_v2_normfalse_rep3/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8"
)

TASK_NAMES=(
  "260713-newdata3500-search-r1-512-normfalse-rep3-run1"
  "260713-newdata3500-spad-512-stable-normfalse-rep3-run1"
  "260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep3-run1"
)

cd "${ROOT}"
export PYTHONPATH="${ROOT}/AgenticIterRag:${ROOT}/AgenticIterRag/verl:${PYTHONPATH:-}"

if [[ ! "${START_MODEL_INDEX}" =~ ^[0-2]$ ]]; then
  echo "ERROR: START_MODEL_INDEX must be in [0, 2]; got ${START_MODEL_INDEX}" >&2
  exit 2
fi
if [[ ! -f "${DATA_PATH}" ]]; then
  echo "ERROR: evaluation data not found: ${DATA_PATH}" >&2
  exit 2
fi
for model_path in "${MODELS[@]}"; do
  if [[ ! -d "${model_path}" ]]; then
    echo "ERROR: checkpoint not found: ${model_path}" >&2
    exit 2
  fi
done

run_once() {
  local index="$1"
  local model_path="${MODELS[$index]}"
  local task_name="${TASK_NAMES[$index]}"

  echo "[$(date '+%F %T %Z')] START index=${index} task=${task_name}"
  bash "${EVAL_ENTRY}" \
    --agent-model "${model_path}" \
    --data-path "${DATA_PATH}" \
    --max-samples 3500 \
    --task-name "${task_name}" \
    --repeat-id 1 \
    --agent-gpu-ids "${AGENT_GPU_IDS}" \
    --recall-gpu-ids "${RECALL_GPU_IDS}" \
    --agent-max-num-seqs "${AGENT_MAX_NUM_SEQS}" \
    --infer-batch-size "${INFER_BATCH_SIZE}" \
    --flush-every-n "${FLUSH_EVERY_N}"
  echo "[$(date '+%F %T %Z')] DONE  index=${index} task=${task_name}"
}

for index in "${!MODELS[@]}"; do
  if (( index >= START_MODEL_INDEX )); then
    run_once "${index}"
  fi
done

"${ROOT}/.venvs/ms_agt_rag_overlay/bin/python" \
  "${ROOT}/scripts/cosearch_local/aggregate_newdata_model_eval.py" \
  --run-spec "${RUN_SPEC}" \
  --data "${DATA_PATH}" \
  --output-dir "${AGGREGATE_DIR}" \
  --bootstrap-samples 10000 \
  --seed 42

echo "[$(date '+%F %T %Z')] ALL EVALUATIONS AND AGGREGATION COMPLETED"
