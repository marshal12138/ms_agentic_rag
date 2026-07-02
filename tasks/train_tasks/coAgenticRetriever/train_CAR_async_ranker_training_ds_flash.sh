#!/usr/bin/env bash
set -euo pipefail

# 项目根目录：后续所有配置文件和训练入口都基于该路径定位。
ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

# 实验名：会进入 run-name/log/checkpoint 名称中；外部可通过 EXP_NAME=... 覆盖。
export EXP_NAME="${EXP_NAME:-CAR_async_ranker_training_ds_flash_v1}"

# agent rollout vLLM 的显存占用比例；这里不控制 LLM judge 服务的显存参数。
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"

# agent rollout vLLM 同时处理的最大序列数；越大吞吐越高，但显存压力也更高。
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

# agent 侧并发 worker 数；影响 rollout/tool call 采样吞吐。
export AGENT_WORKERS="${AGENT_WORKERS:-4}"

# 每个 agent worker 内部同时发起的工具调用数；总工具并发约为 AGENT_WORKERS * TOOL_MAX_CONCURRENT_PER_WORKER。
export TOOL_MAX_CONCURRENT_PER_WORKER="${TOOL_MAX_CONCURRENT_PER_WORKER:-4}"

# logprob/ref logprob 计算的每 GPU micro batch；主要影响显存和 logprob 计算吞吐。
export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-8}"

# actor 更新的每 GPU micro batch；主要影响 actor 训练显存占用。
export ACTOR_MICRO_BATCH_SIZE_PER_GPU="${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-4}"

# 检索工具保存的 dense-ranker 排序后候选数。async judge 依赖 rank 后 top50，不能改成 recall top50。
export TOP_N="${TOP_N:-50}"

# agent LLM 实际可见的 topM 文档数；默认只把 rank 后 top5 给 agent 决策，和 LLM judge 对 50 个 chunk 打分是两件事。
export TOP_M="${TOP_M:-5}"

# agent LLM rollout/update 使用的 GPU 组。
export AGENT_GPU_IDS="${AGENT_GPU_IDS:-0,1,2,3}"

# dense ranker 训练/服务使用的 GPU。
export RANK_GPU_ID="${RANK_GPU_ID:-4}"

# recall retriever 服务使用的 GPU。
export RECALL_GPU_ID="${RECALL_GPU_ID:-5}"

# 是否启用异步样本标注框架；0 表示回到不使用 async_ranker_training 的训练路径。
export ENABLE_ASYNC_RANKER_TRAINING="${ENABLE_ASYNC_RANKER_TRAINING:-1}"

# async_ranker_training 主配置：包含限流、buffer、sample_builder、judge client 等策略参数。
export ASYNC_RANKER_TRAINING_YAML="${ASYNC_RANKER_TRAINING_YAML:-${ROOT}/scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml}"

# judge endpoint 不可用时是否自动启动 LLM judge vLLM 服务。
export AUTO_START_LLM_JUDGE="${AUTO_START_LLM_JUDGE:-1}"

# 训练结束或脚本退出时是否自动停止本脚本启动的 LLM judge 服务。
export AUTO_STOP_LLM_JUDGE="${AUTO_STOP_LLM_JUDGE:-1}"

# LLM judge vLLM 启动配置：模型路径、GPU 6/7、端口、--max-model-len 等核心服务参数写在这里。
export LLM_JUDGE_SERVICE_CONFIG="${LLM_JUDGE_SERVICE_CONFIG:-${ROOT}/CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml}"

# LLM judge 的 OpenAI-compatible chat completions endpoint。
export LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-http://127.0.0.1:8067/v1/chat/completions}"

# 训练前是否做 judge endpoint 可用性检查；开启后 endpoint 不可用会先尝试自动启动或直接报错。
export LLM_JUDGE_PREFLIGHT="${LLM_JUDGE_PREFLIGHT:-1}"

# 传给底层 CoAgenticRetriever 训练入口的 Hydra 覆盖参数。
# - max_num_batched_tokens/multi_turn.max_parallel_calls 控制 rollout 吞吐和并发工具调用。
# - fsdp_config.*_offload=False 保持当前 GPU 训练配置，不启用参数/优化器 offload。
# - enable_thinking=False 关闭 Qwen3 thinking 模式，保持现有训练提示格式。
export COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-actor_rollout_ref.rollout.max_num_batched_tokens=32768 actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2 actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.ref.fsdp_config.param_offload=False ++data.apply_chat_template_kwargs.enable_thinking=False}"

# 常用外部覆盖参数：
# - TOTAL_STEPS=10 可用于 10 step smoke；不在本脚本内写默认值，避免改变正式训练步数。
# - RUN_STAMP=... 可固定 run-name 时间戳/前缀，便于复现实验日志目录。
bash "${ROOT}/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh" "$@"
