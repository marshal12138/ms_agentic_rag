#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU CoAgenticRetriever training/evaluation entry.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${COAGENTIC_PROJECT_ROOT:-${ROOT}/CoAgenticRetriever}"
if [[ -d "${PROJECT_ROOT}" ]]; then
  PROJECT_ROOT="$(cd "${PROJECT_ROOT}" && pwd)"
fi
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
source "${ROOT}/src/logs/report_system/logging_reports.sh"
source "${ROOT}/src/hydra_overrides/hydra_overrides.sh"
source "${SCRIPT_DIR}/00_project_paths.sh"
setup_agent_iteration_paths "${ROOT}"
GROUP_NAME="${GROUP_NAME:-coAgenticRetriever}"
resolve_coagentic_group_identity "${GROUP_NAME}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
VISIBLE_GPU_COUNT="$(awk -F',' '{print NF}' <<< "${GPU_IDS}")"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${VISIBLE_GPU_COUNT}}"
NNODES="${NNODES:-1}"
PORT="${PORT:-8010}"
RETRIEVAL_SERVICE_URL="${RETRIEVAL_SERVICE_URL:-http://127.0.0.1:${PORT}/retrieve}"

MODEL_PATH="${MODEL_PATH:-${EXTERNAL_MODEL_ROOT}/llm/Qwen3-0.6B}"
TRAIN_DATA="${TRAIN_DATA:-${LOCAL_FLASHRAG_ROOT}/co_search_ablation.train.parquet}"
VAL_DATA="${VAL_DATA:-${LOCAL_FLASHRAG_ROOT}/co_search_ablation.eval.parquet}"
OUT_DIR="${OUT_DIR:-${ROOT}/checkpoints/qwen3_4b_probe/${GROUP_SLUG}/${EXP_NAME:-official_verl_qwen3_0_6b_smoke}}"
EXP_NAME="${EXP_NAME:-official_verl_qwen3_0_6b_smoke}"
setup_coagentic_logging_defaults "${ROOT}" "default"

N_ROLLOUTS="${N_ROLLOUTS:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-1}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_TURNS="${MAX_TURNS:-2}"
MAX_USER_TURNS="${MAX_USER_TURNS:-${MAX_TURNS}}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-${MAX_TURNS}}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-2048}"
TOP_N="${TOP_N:-10}"
TOP_M="${TOP_M:-5}"
TOOL_MAX_CONCURRENT_PER_WORKER="${TOOL_MAX_CONCURRENT_PER_WORKER:-2}"
TP_SIZE="${TP_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
AGENT_WORKERS="${AGENT_WORKERS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-16}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-2147483648}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${OUT_DIR}/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${OUT_DIR}/validation_data}"
DUMP_ROLLOUT_EVERY_STEP_NUM="${DUMP_ROLLOUT_EVERY_STEP_NUM:-10}"
DUMP_ROLLOUT_NUM_EVERYTIME="${DUMP_ROLLOUT_NUM_EVERYTIME:-1}"
MAX_ROLLOUT_DUMP_NUM="${MAX_ROLLOUT_DUMP_NUM:--1}"
ROLLOUT_TRACE_MODE="${ROLLOUT_TRACE_MODE:-full}"
case "${ROLLOUT_TRACE_MODE}" in
  full|partial)
    ;;
  *)
    echo "ERROR: unsupported ROLLOUT_TRACE_MODE=${ROLLOUT_TRACE_MODE}; use full or partial" >&2
    exit 2
    ;;
esac

ACTOR_LR="${ACTOR_LR:-1e-6}"
ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-16}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
FORMAT_PENALTY="${FORMAT_PENALTY:--0.2}"
UID_GROUP_THRESHOLD="${UID_GROUP_THRESHOLD:-0.8}"
AGENT_THRESHOLD="${AGENT_THRESHOLD:-0.8}"
COND_THRESHOLD="${COND_THRESHOLD:-0.8}"
FILTER_NO_ANSWER_IN_DOCS="${FILTER_NO_ANSWER_IN_DOCS:-false}"
RERANKER_TRAINABLE="${RERANKER_TRAINABLE:-false}"
DISABLE_RERANKER_ROLLOUT="${DISABLE_RERANKER_ROLLOUT:-true}"
USE_LLM_RERANKER="${USE_LLM_RERANKER:-false}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-true}"
RETRIEVAL_PREFLIGHT="${RETRIEVAL_PREFLIGHT:-1}"
RETRIEVAL_PREFLIGHT_QUERY="${RETRIEVAL_PREFLIGHT_QUERY:-who got the first nobel prize in physics?}"
RETRIEVAL_PREFLIGHT_EXPECT="${RETRIEVAL_PREFLIGHT_EXPECT:-Röntgen}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','file']}"
COAGENTIC_MAIN="${COAGENTIC_MAIN:-${PROJECT_ROOT}/main_coagentic_retriever.py}"
COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-}"
SAVE_TOP_N_DOCUMENTS="${SAVE_TOP_N_DOCUMENTS:-false}"
COAGENTIC_RANKER_ENABLED="${COAGENTIC_RANKER_ENABLED:-}"
COAGENTIC_TOOL_CLASS_NAME="${COAGENTIC_TOOL_CLASS_NAME:-verl.tools.coagentic_retriever_tool.CoAgenticRetrieverTool}"
COAGENTIC_AGENT_LOOP_NAME="${COAGENTIC_AGENT_LOOP_NAME:-coagentic_retriever_agent}"
COAGENTIC_AGENT_LOOP_CONFIG="${COAGENTIC_AGENT_LOOP_CONFIG:-${PROJECT_ROOT}/config/coagentic_retriever_agent_loop_config.yaml}"
TOOL_CONFIG="${PROJECT_ROOT}/config/coagentic_retriever_tool_config.yaml"

load_static_tool_config() {
  local parsed
  parsed="$("${PY}" -c '
import shlex
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
except ModuleNotFoundError:
    from omegaconf import OmegaConf
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}

tool = (data.get("tools") or [{}])[0]
config = tool.get("config") or {}

def emit(name, value):
    if isinstance(value, bool):
        value = str(value).lower()
    elif value is None:
        value = ""
    else:
        value = str(value)
    print(f"{name}={shlex.quote(value)}")

emit("STATIC_TOOL_CLASS_NAME", tool.get("class_name", ""))
emit("STATIC_RETRIEVAL_SERVICE_URL", config.get("retrieval_service_url", ""))
emit("STATIC_DEFAULT_TOP_N", config.get("default_top_n", ""))
emit("STATIC_DEFAULT_TOP_M", config.get("default_top_m", ""))
emit("STATIC_FORMAT_PENALTY", config.get("format_penalty", ""))
emit("STATIC_MAX_CONCURRENT_PER_WORKER", config.get("max_concurrent_per_worker", ""))
emit("STATIC_RANKER_ENABLED", config.get("ranker_enabled", ""))
' "${TOOL_CONFIG}")"
  eval "${parsed}"

  COAGENTIC_TOOL_CLASS_NAME="${STATIC_TOOL_CLASS_NAME}"
  RETRIEVAL_SERVICE_URL="${STATIC_RETRIEVAL_SERVICE_URL}"
  TOP_N="${STATIC_DEFAULT_TOP_N}"
  RECALL_TOP_K="${STATIC_DEFAULT_TOP_N}"
  TOP_M="${STATIC_DEFAULT_TOP_M}"
  FORMAT_PENALTY="${STATIC_FORMAT_PENALTY}"
  COAGENTIC_RANKER_ENABLED="${STATIC_RANKER_ENABLED}"
  if [[ -z "${COAGENTIC_RANKER_ENABLED}" ]]; then
    echo "ERROR: tool config must explicitly set ranker_enabled." >&2
    exit 2
  fi
  if [[ -n "${STATIC_MAX_CONCURRENT_PER_WORKER}" ]]; then
    TOOL_MAX_CONCURRENT_PER_WORKER="${STATIC_MAX_CONCURRENT_PER_WORKER}"
  fi
}

load_static_tool_config

if [[ "${COAGENTIC_RANKER_ENABLED}" != "false" && "${N_GPUS_PER_NODE}" -lt 2 ]]; then
  echo "ERROR: CoAgenticRetriever resource split needs at least 2 visible GPUs; got GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi
if [[ "${N_GPUS_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "ERROR: N_GPUS_PER_NODE=${N_GPUS_PER_NODE} exceeds visible GPU count ${VISIBLE_GPU_COUNT} from GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi

required_paths=("${COAGENTIC_MAIN}" "${PROJECT_ROOT}/verl" "${MODEL_PATH}" "${TRAIN_DATA}" "${VAL_DATA}" "${COAGENTIC_AGENT_LOOP_CONFIG}" "${TOOL_CONFIG}")
for path in "${required_paths[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "ERROR: required path not found: ${path}" >&2
    exit 1
  fi
done

mkdir -p "${LOG_DIR}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1; official trainer configuration resolved"
  echo "PROJECT_ROOT=${PROJECT_ROOT}"
  echo "MODEL_PATH=${MODEL_PATH}"
  echo "LORA_RANK=${LORA_RANK}"
  echo "LORA_ALPHA=${LORA_ALPHA}"
  echo "TRAIN_DATA=${TRAIN_DATA}"
  echo "VAL_DATA=${VAL_DATA}"
  echo "OUT_DIR=${OUT_DIR}"
  echo "TOOL_CONFIG=${TOOL_CONFIG}"
  echo "RETRIEVAL_SERVICE_URL=${RETRIEVAL_SERVICE_URL}"
  echo "COAGENTIC_RANKER_ENABLED=${COAGENTIC_RANKER_ENABLED}"
  echo "ROLLOUT_TRACE_MODE=${ROLLOUT_TRACE_MODE}"
  echo "ASYNC_LABELING_YAML=${ASYNC_LABELING_YAML:-}"
  exit 0
fi

if [[ "${RETRIEVAL_PREFLIGHT}" == "1" || "${RETRIEVAL_PREFLIGHT}" == "true" || "${RETRIEVAL_PREFLIGHT}" == "yes" ]]; then
  "${PY}" "${SCRIPT_DIR}/00_check_coagentic_tool_retrieval.py" \
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
  "${HYDRA_OVERRIDE_YAMLS:-}" \
  "${RANKER_STRATEGY_YAML:-}" \
  "${ASYNC_LABELING_YAML:-}"
hydra_yaml_overrides_to_array hydra_yaml_args "${PY}" "${hydra_yaml_files[@]}"
if [[ "${#hydra_yaml_files[@]}" -gt 0 ]]; then
  echo "Hydra YAML override files: ${hydra_yaml_files[*]}"
fi

read -r -a coagentic_default_args <<< "${COAGENTIC_DEFAULT_EXTRA_ARGS:-}"
read -r -a coagentic_extra_args <<< "${COAGENTIC_EXTRA_ARGS}"

exec "${PY}" "${COAGENTIC_MAIN}" \
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
  data.dataloader_num_workers=0 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
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
  actor_rollout_ref.rollout.agent.default_agent_loop="${COAGENTIC_AGENT_LOOP_NAME}" \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${COAGENTIC_AGENT_LOOP_CONFIG}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  critic.enable=False \
  reward_model.enable=False \
  reward_model.reward_manager=multiturn \
  reward_model.use_reward_loop=True \
  custom_reward_function.path="${PROJECT_ROOT}/rewards/search_qa_f1_with_format_penalty.py" \
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
  +trainer.disable_reranker_rollout="${DISABLE_RERANKER_ROLLOUT}" \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.logger="${TRAINER_LOGGER}" \
  trainer.project_name=coagentic_retriever_verl_smoke \
  trainer.default_local_dir="${OUT_DIR}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
  trainer.dump_rollout_every_step_num="${DUMP_ROLLOUT_EVERY_STEP_NUM}" \
  trainer.dump_rollout_num_everytime="${DUMP_ROLLOUT_NUM_EVERYTIME}" \
  trainer.max_rollout_dump_num="${MAX_ROLLOUT_DUMP_NUM}" \
  trainer.rollout_trace_mode="${ROLLOUT_TRACE_MODE}" \
  trainer.validation_data_dir="${VALIDATION_DATA_DIR}" \
  "${coagentic_default_args[@]}" \
  "${hydra_yaml_args[@]}" \
  "${coagentic_extra_args[@]}" \
  "$@"
