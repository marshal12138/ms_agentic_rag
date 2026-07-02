#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

cd "${ROOT}"

# 实验名；默认用于日志目录、checkpoint 目录和 Hydra experiment_name，可在外部提前设置 EXP_NAME 覆盖。
export EXP_NAME="${EXP_NAME:-CAR_async_ranker_training_ds_flash_mix_signal_b3_v1_select_all}"

# 实验分组名；用于组织日志和 checkpoint 的上级目录。
export GROUP_NAME="${GROUP_NAME:-coAgenticRetriever}"

# Agent LLM 训练/rollout 使用的 NPU/GPU ID 列表。
export AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1,2,3}"

# dense ranker 训练或推理使用的 NPU/GPU ID。
export RANK_GPU_ID="${RANK_GPU_ID:-4}"

# recall dense retriever HTTP 服务使用的 NPU/GPU ID。
export RECALL_GPU_ID="${RECALL_GPU_ID:-5}"

# LLM-as-judge 服务使用的 NPU/GPU ID 列表；这里配给 GLM 4.7 judge 容器。
export LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS:-6,7}"

# 是否启用异步 ranker 训练链路；1 表示使用 overlay 中的 async_ranker_training 配置。
export ENABLE_ASYNC_RANKER_TRAINING="${ENABLE_ASYNC_RANKER_TRAINING:-1}"

# recall 检索服务不可用时是否由训练 launcher 自动启动。
export AUTO_START_RECALL_SERVICE="${AUTO_START_RECALL_SERVICE:-1}"

# 训练脚本退出时是否自动停止本次拉起的 recall 检索服务。
export AUTO_STOP_RECALL_SERVICE="${AUTO_STOP_RECALL_SERVICE:-1}"

# 等待 recall 检索服务启动完成的最长秒数。
export RECALL_SERVICE_WAIT_SECONDS="${RECALL_SERVICE_WAIT_SECONDS:-240}"

# LLM judge 服务不可用时是否由训练 launcher 自动启动；本脚本前面也会显式启动 GLM 容器。
export AUTO_START_LLM_JUDGE="${AUTO_START_LLM_JUDGE:-1}"

# 训练脚本退出时是否自动停止本次拉起的 LLM judge 服务。
export AUTO_STOP_LLM_JUDGE="${AUTO_STOP_LLM_JUDGE:-1}"

# 训练前是否检查 LLM judge endpoint 可用。
export LLM_JUDGE_PREFLIGHT="${LLM_JUDGE_PREFLIGHT:-1}"

# 等待 LLM judge 服务启动完成的最长秒数。
export LLM_JUDGE_WAIT_SECONDS="${LLM_JUDGE_WAIT_SECONDS:-600}"

# 启动前需要等待释放的设备集合；默认覆盖 agent、ranker、recall 和 judge 的全部设备。
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}}"

# 是否启用设备占用等待；1 表示执行 wait_for_gpus_if_enabled。
export WAIT_FOR_GPU_RELEASE="${WAIT_FOR_GPU_RELEASE:-1}"

# 设备占用轮询间隔秒数。
export WAIT_FOR_GPU_INTERVAL_SECONDS="${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}"

# 等待设备释放时日志中显示的任务标签。
export WAIT_FOR_GPU_LABEL="${WAIT_FOR_GPU_LABEL:-mix-signal experiment GPU wait}"

source "${ROOT}/src/runtime/wait_for_gpus.sh"
wait_for_gpus_if_enabled

# 使用glm4.7作为judge服务，启动judge服务容器；临时fix代码；
CoAgenticRetriever/scripts/launch_glm47_flash_judge_container.sh --restart
CoAgenticRetriever/scripts/launch_glm47_flash_judge_container.sh --status

bash "${ROOT}/scripts/coagenticRetriever_v2/01_train_launcher.sh" \
  --trainer_main_hydra_config=coagentic_retriever_trainer \
  --DATA_CONFIG=co_search_ablation \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=cosearch_async_qwen3_4b \
  --RANKER_BASE_CONFIG=ranker_contrastive \
  --ASYNC_RANKER_TRAINING_BASE_CONFIG=async_ranker_training \
  --OVERLAY_YAML=scripts/coagenticRetriever_v2/strategies_yaml/async_ranker_training_glm_4_7_flash_rank50_select_all.yaml \
  --OVERLAY_YAML=tasks/train_tasks/coAgenticRetriever/configs/mix_signal_b3_overlay.yaml \
  --LLM_JUDGE_SERVICE_CONFIG=CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_glm_4_7_flash_gpu06_07.yaml \
  --actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
