#!/usr/bin/env bash
set -euo pipefail

# SPAD-RAG 第一版 smoke 入口。
# 这里只跑 train_agent.impl=spad_rag 的 smoke 后端，用于验证配置、manifest 和三阶段数据契约。

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
  --AGENT_TRAINING_CONFIG=spad_rag_base \
  --MODEL_CONFIG=qwen3_1_7b \
  --ROLLOUT_CONFIG=air_async_qwen3_1_7b \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_smoke_overlay.yaml \
  "$@"
