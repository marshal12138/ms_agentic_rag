#!/usr/bin/env bash
set -euo pipefail

# 项目根目录：后续所有配置文件和训练入口都基于该路径定位。
ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

# 实验名：会进入 run-name/log/checkpoint 名称中；外部可通过 EXP_NAME=... 覆盖。
export EXP_NAME="${EXP_NAME:-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all}"

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

# 是否启用异步样本标注框架；0 表示回到不使用 async_labeling 的训练路径。
export ENABLE_ASYNC_LABELING="${ENABLE_ASYNC_LABELING:-1}"

# async_labeling 主配置：包含限流、buffer、sample_builder、judge client 等策略参数。
export ASYNC_LABELING_YAML="${ASYNC_LABELING_YAML:-${ROOT}/scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml}"

# 训练 rollout budget 默认配置。具体 max_prompt/max_response/max_model_len/turn/tool-response budget
# 由 YAML 管理，task 脚本只选择默认配置；外部 HYDRA_OVERRIDE_YAMLS 排在它后面，可覆盖这些默认值。
export TRAIN_BUDGET_YAML="${TRAIN_BUDGET_YAML:-${ROOT}/scripts/coagenticRetriever_local/strategies_yaml/rollout_cosearch_aligned_budget.yaml}"
if [[ -n "${TRAIN_BUDGET_YAML}" && ! -f "${TRAIN_BUDGET_YAML}" ]]; then
  echo "ERROR: TRAIN_BUDGET_YAML not found: ${TRAIN_BUDGET_YAML}" >&2
  exit 2
fi
USER_HYDRA_OVERRIDE_YAMLS="${HYDRA_OVERRIDE_YAMLS:-}"
if [[ -n "${TRAIN_BUDGET_YAML}" ]]; then
  export HYDRA_OVERRIDE_YAMLS="${TRAIN_BUDGET_YAML}${USER_HYDRA_OVERRIDE_YAMLS:+ ${USER_HYDRA_OVERRIDE_YAMLS}}"
else
  export HYDRA_OVERRIDE_YAMLS="${USER_HYDRA_OVERRIDE_YAMLS}"
fi

# judge endpoint 不可用时是否自动启动 LLM judge vLLM 服务。
export AUTO_START_LLM_JUDGE="${AUTO_START_LLM_JUDGE:-1}"

# 训练结束或脚本退出时是否自动停止本脚本启动的 LLM judge 服务。
export AUTO_STOP_LLM_JUDGE="${AUTO_STOP_LLM_JUDGE:-1}"

# LLM judge vLLM 启动配置：模型路径、GPU、端口、--max-model-len 等核心服务参数写在这里。
export LLM_JUDGE_SERVICE_CONFIG="${LLM_JUDGE_SERVICE_CONFIG:-${ROOT}/CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml}"
export LLM_JUDGE_GPU_IDS="${LLM_JUDGE_GPU_IDS:-6,7}"

# LLM judge 的 OpenAI-compatible chat completions endpoint。
export LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-http://127.0.0.1:8067/v1/chat/completions}"

# 训练前是否做 judge endpoint 可用性检查；开启后 endpoint 不可用会先尝试自动启动或直接报错。
export LLM_JUDGE_PREFLIGHT="${LLM_JUDGE_PREFLIGHT:-1}"

# 是否向 Qwen chat template 注入 tools schema。CoSearch-aligned 训练默认不注入。
export INJECT_TOOL_SCHEMA="${INJECT_TOOL_SCHEMA:-false}"

# 本实验的变量：攒够 3 条 completed judge signals 后，一起进入 signal_builder/sample_builder。
# sample_builder 仍然只输出 num_groups_per_step 组对比学习样本，默认 32 组。
export SAMPLE_BUILDER_REQUEST_BATCH="${SAMPLE_BUILDER_REQUEST_BATCH:-3}"

# 当前实验还在占用 GPU 时，可以先启动本脚本；它会等目标 GPU 全部释放后再进入训练。
# 默认等待 agent/ranker/recall/judge 全部 GPU，避免和前一个完整 async-labeling 实验抢资源。
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-${AGENT_GPU_IDS},${RANK_GPU_ID},${RECALL_GPU_ID},${LLM_JUDGE_GPU_IDS}}"
export WAIT_FOR_GPU_RELEASE="${WAIT_FOR_GPU_RELEASE:-1}"
export WAIT_FOR_GPU_INTERVAL_SECONDS="${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}"
export WAIT_FOR_GPU_LABEL="${WAIT_FOR_GPU_LABEL:-mix-signal experiment GPU wait}"

source "${ROOT}/src/runtime/wait_for_gpus.sh"

# 传给底层 CoAgenticRetriever 训练入口的 Hydra 覆盖参数。
# - multi_turn.max_parallel_calls 控制并发工具调用。
# - fsdp_config.*_offload=False 保持当前 GPU 训练配置，不启用参数/优化器 offload。
# - inject_tool_schema 是本 task 的显式行为开关，默认 false，用于和 CoSearch-aligned eval 对齐。
# - sample_builder_request_batch 是本实验变量，始终追加在末尾，保证覆盖 YAML/default。
DEFAULT_COAGENTIC_EXTRA_ARGS="actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2 actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False actor_rollout_ref.ref.fsdp_config.param_offload=False"
export COAGENTIC_EXTRA_ARGS="${COAGENTIC_EXTRA_ARGS:-${DEFAULT_COAGENTIC_EXTRA_ARGS}} ++coagentic_retriever.agent.inject_tool_schema=${INJECT_TOOL_SCHEMA} ranker_training.async_labeling.sample_builder_request_batch=${SAMPLE_BUILDER_REQUEST_BATCH}"

# 常用外部覆盖参数：
# - TOTAL_STEPS=10 可用于 10 step smoke；不在本脚本内写默认值，避免改变正式训练步数。
# - RUN_STAMP=... 可固定 run-name 时间戳/前缀，便于复现实验日志目录。
wait_for_gpus_if_enabled

bash "${ROOT}/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh" "$@"
