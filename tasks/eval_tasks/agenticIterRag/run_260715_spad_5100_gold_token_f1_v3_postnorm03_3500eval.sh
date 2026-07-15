#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"
EVAL_ENTRY="${ROOT}/tasks/eval_tasks/agenticIterRag/eval_agent_search.sh"
MODEL_PATH="${MODEL_PATH:-${ROOT}/checkpoints/AIR/260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm03_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_verl/global_step_79/hf_safetensors/actor}"
DATA_PATH="${ROOT}/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet"
REPEAT_ID="${REPEAT_ID:-1}"
TASK_NAME="${TASK_NAME:-$(date +%y%m%d-%H%M%S)-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run${REPEAT_ID}}"

cd "${ROOT}"

echo "[$(date '+%F %T %Z')] evaluating ${MODEL_PATH}"
echo "[$(date '+%F %T %Z')] task ${TASK_NAME}"

exec bash "${EVAL_ENTRY}" \
  --agent-model "${MODEL_PATH}" \
  --data-path "${DATA_PATH}" \
  --max-samples 3500 \
  --task-name "${TASK_NAME}" \
  --repeat-id "${REPEAT_ID}" \
  --agent-gpu-ids "0,1,2,3,4,5" \
  --recall-gpu-ids "6,7" \
  --agent-max-num-seqs 64 \
  --infer-batch-size 384 \
  --flush-every-n 500
