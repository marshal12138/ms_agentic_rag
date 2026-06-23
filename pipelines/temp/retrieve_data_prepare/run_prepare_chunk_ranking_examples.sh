#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PIPELINE_DIR="${ROOT}/pipelines/temp/retrieve_data_prepare"
SCRIPTS_DIR="${ROOT}/scripts/coagenticRetriever_local"
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"

TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/co_search/local_flashrag/co_search_ablation.train.parquet}"
RUN_NAME="${RUN_NAME:-retrieve_data_prepare_260608}"
WORK_DIR="${WORK_DIR:-${PIPELINE_DIR}/runs/${RUN_NAME}}"
SUBSET_DATA="${SUBSET_DATA:-${WORK_DIR}/train_subset_for_rollout.parquet}"
OUT_DIR="${OUT_DIR:-${WORK_DIR}/infer_out}"
DATASET_DIR="${DATASET_DIR:-${ROOT}/data/llm_judge/chunk_ranking/examples}"
CUSTOM_TOOL_CONFIG="${CUSTOM_TOOL_CONFIG:-${WORK_DIR}/co_search_tool_config_retrieval_only.yaml}"
PROXY_PORT="${PROXY_PORT:-8130}"

RESUME_FROM_PATH="${RESUME_FROM_PATH:-/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79}"
MODEL_PATH="${MODEL_PATH:-/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B}"

mkdir -p "${WORK_DIR}" "${DATASET_DIR}"

cat > "${CUSTOM_TOOL_CONFIG}" <<EOF
tools:
  - class_name: verl.tools.co_search_tool.CoSearchTool
    config:
      type: native
      retrieval_service_url: "http://127.0.0.1:${PROXY_PORT}/retrieve"
      timeout: 30
      max_retries: 1
      retry_delay: 0.5
      retry_backoff: 1.0
      default_top_n: 50
      default_top_m: 5
      hit_cutoffs: [1, 3, 5]
      tool_score_metric: "hit"
      trivial_answers: ["yes", "no", "true", "false"]
      format_penalty: -0.2
      max_concurrent_per_worker: ${TOOL_MAX_CONCURRENT_PER_WORKER:-5}
      use_reranker: false
      save_top_n_documents: true
EOF

"${PY}" "${PIPELINE_DIR}/make_train_subset.py" \
  --input "${TRAIN_DATA}" \
  --output "${SUBSET_DATA}" \
  --max-samples "${SUBSET_MAX_SAMPLES:-240}" \
  --start "${SUBSET_START:-0}"

export PY
export RUN_NAME
export MODEL_PATH
export RESUME_FROM_PATH
export TRAIN_DATA
export VAL_DATA="${SUBSET_DATA}"
export OUT_DIR
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-180}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-4}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
export N_ROLLOUTS="${N_ROLLOUTS:-1}"
export TOP_N=50
export TOP_M=5
export RERANKER_TRAINABLE=false
export PROXY_PORT
export RETRIEVER_INSTANCES="${RETRIEVER_INSTANCES:-5}"
export RETRIEVER_MIN_INSTANCES="${RETRIEVER_MIN_INSTANCES:-2}"
export AGENT_WORKERS="${AGENT_WORKERS:-5}"
export TOOL_MAX_CONCURRENT_PER_WORKER="${TOOL_MAX_CONCURRENT_PER_WORKER:-5}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-5}"
export ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-4}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-48}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
export MAX_ACTOR_CKPT_TO_KEEP=1
export TRAINER_LOGGER="${TRAINER_LOGGER:-['console','file']}"
export VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUT_DIR}/validation_data}"

bash "${SCRIPTS_DIR}/02_infer_qwen3_4b_ablation_val_only.sh" \
  +trainer.disable_reranker_rollout=true \
  "actor_rollout_ref.actor.checkpoint.load_contents=['model']" \
  custom_reward_function.path="${PIPELINE_DIR}/search_qa_f1_with_trace.py" \
  custom_reward_function.name=search_qa_f1_penalty_with_trace_compute_score \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${CUSTOM_TOOL_CONFIG}"

"${PY}" "${PIPELINE_DIR}/extract_chunk_ranking_examples.py" \
  --validation-dir "${VALIDATION_DATA_DIR}" \
  --output-dir "${DATASET_DIR}" \
  --target "${TARGET_EXAMPLES:-100}" \
  --shard-size 10
