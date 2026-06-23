#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

if [ ! -f "${PROJECT_ROOT}/main_coagentic_retriever.py" ]; then
    echo "ERROR: cannot find main_coagentic_retriever.py under ${PROJECT_ROOT}."
    exit 1
fi

if [ ! -f "${PROJECT_ROOT}/config/coagentic_retriever_agent_loop_config.yaml" ]; then
    echo "ERROR: agent loop config not found under ${PROJECT_ROOT}/config."
    exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl:${PYTHONPATH:-}"
export VLLM_DISABLE_FLASHINFER="${VLLM_DISABLE_FLASHINFER:-1}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B}"
TRAIN_DATA="${TRAIN_DATA:-['${PROJECT_ROOT}/data/coAgenticRetriever/albation_1/train.parquet']}"
VAL_DATA="${VAL_DATA:-['${PROJECT_ROOT}/data/coAgenticRetriever/albation_1/val.parquet']}"
PROJECT_NAME="${PROJECT_NAME:-coagentic_retriever}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/coagentic_retriever}"

AGENT_LOOP_CONFIG="${AGENT_LOOP_CONFIG:-${PROJECT_ROOT}/config/coagentic_retriever_agent_loop_config.yaml}"
TOOL_CONFIG="${PROJECT_ROOT}/config/coagentic_retriever_tool_config.yaml"

load_static_tool_config() {
    local parsed
    parsed="$(python -c '
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

config = ((data.get("tools") or [{}])[0].get("config") or {})
ranker = config.get("ranker") or {}

def emit(name, value):
    if value is None:
        value = ""
    print(f"{name}={shlex.quote(str(value))}")

emit("STATIC_RETRIEVAL_SERVICE_URL", config.get("retrieval_service_url", ""))
emit("STATIC_RANKER_MODEL_PATH", ranker.get("model_path", ""))
emit("STATIC_RANKER_DEVICE", ranker.get("device", ""))
emit("STATIC_RANKER_TOP_K", ranker.get("top_k", ""))
' "${TOOL_CONFIG}")"
    eval "${parsed}"

    RETRIEVAL_SERVICE_URL="${STATIC_RETRIEVAL_SERVICE_URL}"
    RANKER_MODEL_PATH="${STATIC_RANKER_MODEL_PATH}"
    RANKER_DEVICE="${STATIC_RANKER_DEVICE}"
    RANKER_TOP_K="${STATIC_RANKER_TOP_K}"
}

load_static_tool_config

NNODES="${NNODES:-${SLURM_NNODES:-1}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
TP_SIZE="${TP_SIZE:-1}"
N_ROLLOUTS="${N_ROLLOUTS:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-128}"
ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
ACTOR_LR_WARMUP_STEPS_RATIO="${ACTOR_LR_WARMUP_STEPS_RATIO:-0.04}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.001}"
SAVE_FREQ="${SAVE_FREQ:-10}"
TEST_FREQ="${TEST_FREQ:-20}"
FORMAT_PENALTY="${FORMAT_PENALTY:--0.2}"

RANKER_STEPS_PER_GLOBAL_STEP="${RANKER_STEPS_PER_GLOBAL_STEP:-2}"
RANKER_BATCH_SIZE="${RANKER_BATCH_SIZE:-32}"
RANKER_LR="${RANKER_LR:-2.0e-5}"

REWARD_FN_PATH="${REWARD_FN_PATH:-${PROJECT_ROOT}/rewards/search_qa_f1_with_format_penalty.py}"
TRAIN_REWARD_FN="${TRAIN_REWARD_FN:-search_qa_f1_penalty_compute_score}"
NUM_EXAMINE="${NUM_EXAMINE:-0}"
VAL_NUM_EXAMINE="${VAL_NUM_EXAMINE:-1}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-coagentic_retriever_grpo_N${N_ROLLOUTS}_T${TEMPERATURE}_ranker_steps${RANKER_STEPS_PER_GLOBAL_STEP}_${TIMESTAMP}}"

cd "${PROJECT_ROOT}"

python main_coagentic_retriever.py \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=20480 \
    data.max_response_length=4096 \
    data.truncation=error \
    actor_rollout_ref.model.path="${CHECKPOINT_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
    actor_rollout_ref.rollout.n="${N_ROLLOUTS}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=24576 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.prompt_length=20480 \
    actor_rollout_ref.rollout.response_length=4096 \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=6 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096 \
    actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=left \
    actor_rollout_ref.rollout.multi_turn.format=search_r1 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG}" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.agent.default_agent_loop=coagentic_retriever_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ACTOR_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ACTOR_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="${ACTOR_LR_WARMUP_STEPS_RATIO}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    critic.enable=False \
    reward_model.enable=False \
    reward_model.reward_manager=multiturn \
    reward_model.use_reward_loop=True \
    custom_reward_function.path="${REWARD_FN_PATH}" \
    custom_reward_function.name="${TRAIN_REWARD_FN}" \
    +custom_reward_function.reward_kwargs.format_penalty="${FORMAT_PENALTY}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.max_actor_ckpt_to_keep=1 \
    +trainer.num_examine="${NUM_EXAMINE}" \
    +trainer.val_num_examine="${VAL_NUM_EXAMINE}" \
    trainer.val_before_train=False \
    trainer.logger='[console]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.rollout_data_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME}/rollout_data" \
    trainer.validation_data_dir="${CHECKPOINT_DIR}/${PROJECT_NAME}/${EXP_NAME}/validation_data" \
    trainer.ranker_trainable=True \
    trainer.ranker_update_mode=contrastive \
    trainer.ranker_steps_per_global_step="${RANKER_STEPS_PER_GLOBAL_STEP}" \
    recall_retriever.service_url="${RETRIEVAL_SERVICE_URL}" \
    ranker.model_path="${RANKER_MODEL_PATH}" \
    ranker.device="${RANKER_DEVICE}" \
    ranker.top_k="${RANKER_TOP_K}" \
    ranker_training.batch_size="${RANKER_BATCH_SIZE}" \
    ranker_training.optim.lr="${RANKER_LR}"
