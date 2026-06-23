#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU CoSearch smoke training derived from CoSearch/scripts/train_co_search_grpo.sh.
# It uses the official CoSearch VERL/Ray trainer and local Qwen3-0.6B assets.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="${ROOT}/CoSearch"
CHECKPOINT_VERL_ROOT="${CHECKPOINT_VERL_ROOT:-${PROJECT_ROOT}/verl}"
CHECKPOINT_CONVERT_ROLES="${CHECKPOINT_CONVERT_ROLES:-actor reranker_actor_rollout}"
PY="${PY:-/data04/envs/ms/ms_cosearch_official/bin/python}"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
source "${ROOT}/src/hydra_overrides/hydra_overrides.sh"
source "${ROOT}/src/checkpoints/checkpoint_conversion.sh"
GROUP_NAME="${GROUP_NAME:-cosearch}"
resolve_cosearch_group_identity "${GROUP_NAME}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
N_GPUS_PER_NODE="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
NNODES="${NNODES:-1}"
PORT="${PORT:-8010}"
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${PORT}/retrieve}"

MODEL_PATH="${MODEL_PATH:-/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-0.6B}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/data/co_search/local_flashrag/co_search_rl_smoke.train.parquet}"
VAL_DATA="${VAL_DATA:-${ROOT}/data/co_search/local_flashrag/co_search_7bench_smoke.eval.parquet}"
OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints/qwen3_4b_probe/${GROUP_SLUG}/${EXP_NAME:-official_verl_qwen3_0_6b_smoke}}"
EXP_NAME="${EXP_NAME:-official_verl_qwen3_0_6b_smoke}"
setup_cosearch_logging_defaults "${ROOT}" "default"

N_ROLLOUTS="${N_ROLLOUTS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-256}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-3072}"
MAX_TURNS="${MAX_TURNS:-2}"
MAX_USER_TURNS="${MAX_USER_TURNS:-${MAX_TURNS}}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-${MAX_TURNS}}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-1024}"
TOP_N="${TOP_N:-10}"
TOP_M="${TOP_M:-5}"
TOOL_MAX_CONCURRENT_PER_WORKER="${TOOL_MAX_CONCURRENT_PER_WORKER:-2}"
TP_SIZE="${TP_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-2147483648}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUT_DIR}/validation_data}"

ACTOR_LR="${ACTOR_LR:-1e-6}"
ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-2}"
ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
FORMAT_PENALTY="${FORMAT_PENALTY:--0.2}"
UID_GROUP_THRESHOLD="${UID_GROUP_THRESHOLD:-0.8}"
AGENT_THRESHOLD="${AGENT_THRESHOLD:-0.8}"
COND_THRESHOLD="${COND_THRESHOLD:-0.8}"
FILTER_NO_ANSWER_IN_DOCS="${FILTER_NO_ANSWER_IN_DOCS:-false}"
RERANKER_TRAINABLE="${RERANKER_TRAINABLE:-false}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
RETRIEVAL_PREFLIGHT="${RETRIEVAL_PREFLIGHT:-1}"
RETRIEVAL_PREFLIGHT_QUERY="${RETRIEVAL_PREFLIGHT_QUERY:-who got the first nobel prize in physics?}"
RETRIEVAL_PREFLIGHT_EXPECT="${RETRIEVAL_PREFLIGHT_EXPECT:-Röntgen}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','file']}"
HYDRA_OVERRIDE_YAMLS="${HYDRA_OVERRIDE_YAMLS:-}"
COSEARCH_STRATEGY_YAML="${COSEARCH_STRATEGY_YAML:-}"
COSEARCH_EXTRA_ARGS="${COSEARCH_EXTRA_ARGS:-}"

if [[ "${N_GPUS_PER_NODE}" -lt 2 ]]; then
  echo "ERROR: official CoSearch dual-agent resource split needs at least 2 visible GPUs; got GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi

for path in "${PROJECT_ROOT}/main_co_search_ppo.py" "${PROJECT_ROOT}/verl" "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"

TOOL_CONFIG="${TOOL_CONFIG:-${LOG_DIR}/${EXP_NAME}.co_search_tool_config.yaml}"
cat > "${TOOL_CONFIG}" <<EOF
tools:
  - class_name: verl.tools.co_search_tool.CoSearchTool
    config:
      type: native
      retrieval_service_url: "${RETRIEVAL_SERVICE_URL}"
      timeout: 30
      max_retries: 1
      retry_delay: 0.5
      retry_backoff: 1.0
      default_top_n: ${TOP_N}
      default_top_m: ${TOP_M}
      hit_cutoffs: [1, 3, 5]
      tool_score_metric: "hit"
      trivial_answers: ["yes", "no", "true", "false"]
      format_penalty: ${FORMAT_PENALTY}
      max_concurrent_per_worker: ${TOOL_MAX_CONCURRENT_PER_WORKER}
EOF

if [[ "${RETRIEVAL_PREFLIGHT}" == "1" || "${RETRIEVAL_PREFLIGHT}" == "true" || "${RETRIEVAL_PREFLIGHT}" == "yes" ]]; then
  "${PY}" "${ROOT}/scripts/cosearch_local/check_cosearch_tool_retrieval.py" \
    --project-root "${PROJECT_ROOT}" \
    --url "${RETRIEVAL_SERVICE_URL}" \
    --query "${RETRIEVAL_PREFLIGHT_QUERY}" \
    --top-n "${TOP_N}" \
    --top-m "${TOP_M}" \
    --expect-contains "${RETRIEVAL_PREFLIGHT_EXPECT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}"
if [[ "${PYTORCH_CUDA_ALLOC_CONF:-}" == *"expandable_segments:True"* && "${ALLOW_VLLM_EXPANDABLE_SEGMENTS:-0}" != "1" ]]; then
  echo "WARNING: ignoring PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF} because vLLM CuMem memory pool rejects expandable_segments:True" >&2
  unset PYTORCH_CUDA_ALLOC_CONF
elif [[ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF
fi
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export TOKENIZERS_PARALLELISM=false
export VLLM_DISABLE_FLASHINFER=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export WANDB_MODE=disabled

trainer_total_step_args=()
case "${TOTAL_STEPS}" in
  ""|auto|AUTO|null|NULL|none|None|-1)
    trainer_total_step_args+=(trainer.total_training_steps=null)
    ;;
  *)
    trainer_total_step_args+=(trainer.total_training_steps="${TOTAL_STEPS}")
    ;;
esac

cd "${PROJECT_ROOT}"

hydra_collect_yaml_override_files hydra_yaml_files \
  "${HYDRA_OVERRIDE_YAMLS}" \
  "${COSEARCH_STRATEGY_YAML}"
hydra_yaml_overrides_to_array hydra_yaml_args "${PY}" "${hydra_yaml_files[@]}"
if [[ "${#hydra_yaml_files[@]}" -gt 0 ]]; then
  echo "Hydra YAML override files: ${hydra_yaml_files[*]}"
fi

read -r -a cosearch_extra_args <<< "${COSEARCH_EXTRA_ARGS}"

set +e
"${PY}" main_co_search_ppo.py \
  algorithm.use_kl_in_reward=False \
  algorithm.adv_estimator=grpo \
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}" \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init.object_store_memory="${RAY_OBJECT_STORE_MEMORY}" \
  data.train_files="['${TRAIN_DATA}']" \
  data.val_files="['${VAL_DATA}']" \
  data.train_max_samples="${TRAIN_MAX_SAMPLES:-4}" \
  data.val_max_samples="${VAL_MAX_SAMPLES:-4}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.truncation=left \
  data.trust_remote_code=True \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK:-8}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-16}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.n="${N_ROLLOUTS}" \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.25}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_USER_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_ASSISTANT_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="${MAX_TOOL_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
  actor_rollout_ref.rollout.multi_turn.format=search_r1 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
  actor_rollout_ref.rollout.agent.num_workers="${AGENT_WORKERS}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=co_search_agent \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${PROJECT_ROOT}/config/co_search_agent_loop_config.yaml" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  critic.enable=False \
  reward_model.enable=False \
  reward_model.reward_manager=multiturn \
  reward_model.use_reward_loop=True \
  custom_reward_function.path="${PROJECT_ROOT}/verl/verl/utils/reward_score/search_qa_f1_with_format_penalty.py" \
  custom_reward_function.name=search_qa_f1_penalty_compute_score \
  +custom_reward_function.reward_kwargs.format_penalty="${FORMAT_PENALTY}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  trainer.total_epochs=1 \
  "${trainer_total_step_args[@]}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.resume_mode="${RESUME_MODE}" \
  +trainer.num_examine=0 \
  +trainer.val_num_examine=1 \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=co_search_official_verl_smoke \
  trainer.default_local_dir="${OUT_DIR}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  reranker_actor_rollout_ref.model.path="${MODEL_PATH}" \
  reranker_actor_rollout_ref.model.trust_remote_code=True \
  +reranker_actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  reranker_actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  reranker_actor_rollout_ref.model.lora_rank="${LORA_RANK:-8}" \
  reranker_actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-16}" \
  reranker_actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  reranker_actor_rollout_ref.rollout.n="${N_ROLLOUTS}" \
  reranker_actor_rollout_ref.rollout.temperature="${TEMPERATURE:-1.0}" \
  reranker_actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.25}" \
  reranker_actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  reranker_actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
  reranker_actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  reranker_actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  reranker_actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
  reranker_actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  reranker_actor_rollout_ref.rollout.mode=async \
  reranker_actor_rollout_ref.rollout.name=vllm \
  reranker_actor_rollout_ref.rollout.enforce_eager=True \
  reranker_actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  reranker_actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reranker_actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  reranker_actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_BATCH_SIZE}" \
  reranker_actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE_PER_GPU}" \
  reranker_actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  reranker_actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  reranker_actor_rollout_ref.actor.use_kl_loss=True \
  reranker_actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  reranker_actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  reranker_actor_rollout_ref.actor.fsdp_config.param_offload=True \
  reranker_actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  reranker_actor_rollout_ref.model.enable_gradient_checkpointing=True \
  reranker_actor_rollout_ref.trainable="${RERANKER_TRAINABLE}" \
  reranker_uid_group_function.path="${PROJECT_ROOT}/verl/verl/experimental/agent_loop/uid_group_functions.py" \
  reranker_uid_group_function.name=group_by_muid_ans_in_doc_subq_rougeL1 \
  +reranker_uid_group_function.uid_group_kwargs.threshold="${UID_GROUP_THRESHOLD}" \
  reranker_score_assign_function.path="${PROJECT_ROOT}/verl/verl/experimental/agent_loop/score_assign_functions.py" \
  reranker_score_assign_function.name=sum_tool_agent_score_with_cond_threshold \
  +reranker_score_assign_function.score_assign_kwargs.agent_threshold="${AGENT_THRESHOLD}" \
  +reranker_score_assign_function.score_assign_kwargs.cond_threshold="${COND_THRESHOLD}" \
  trainer.reranker_sampling_val_start_step=10000 \
  trainer.reranker_filter_no_answer_in_docs="${FILTER_NO_ANSWER_IN_DOCS}" \
  "${hydra_yaml_args[@]}" \
  "${cosearch_extra_args[@]}" \
  "$@"
TRAIN_STATUS="$?"
set -e

if [[ "${TRAIN_STATUS}" == "0" ]]; then
  CHECKPOINT_VERL_ROOT="${CHECKPOINT_VERL_ROOT}" \
  CHECKPOINT_CONVERT_ROLES="${CHECKPOINT_CONVERT_ROLES}" \
    run_verl_fsdp_checkpoint_conversion "${ROOT}" "${OUT_DIR}"
else
  echo "checkpoint conversion skipped: training exited with status ${TRAIN_STATUS}" >&2
fi

exit "${TRAIN_STATUS}"
