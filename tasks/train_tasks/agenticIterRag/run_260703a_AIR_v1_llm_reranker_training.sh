#!/usr/bin/env bash
set -euo pipefail

# 本任务是 AgenticIterRag v1 的 LLM reranker branch GRPO 训练入口。
# shell 只选择配置组和 overlay；模型路径、数据路径、stage 范围、topN/topM、batch size 和 reward 策略都写在 YAML 中。
# 这样做的原因是：pipeline compiler 会把所有业务参数写入 final config，后续排查训练结果时可以完整复现。

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

cd "${ROOT}"

bash "${ROOT}/scripts/agenticIterRag_v1/01_pipeline_launcher.sh" \
  --main-run-config agentic_iter_rag_main \
  --DATA_CONFIG=co_search_ablation \
  --PIPELINE_CONFIG=offline_two_stage \
  --RESOURCE_CONFIG=local_8gpu_0_7 \
  --INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm \
  --INFER_BUDGET_CONFIG=air_aligned_budget \
  --RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=air_async_qwen3_4b \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/llm_reranker_training_overlay.yaml \
  "$@"
