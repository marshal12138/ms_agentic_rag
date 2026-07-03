#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

cd "${ROOT}"

LAUNCHER_ARGS=(
  # v1 顶层 manifest，提供默认配置组、运行目录和审计策略。
  --main-run-config agentic_iter_rag_main

  # 训练/trace/infer 数据配置；infer-matrix-only 默认使用历史 ablation infer 数据。
  --DATA_CONFIG=co_search_ablation

  # 单 pipeline DAG 配置；infer-matrix-only overlay 会只启用 infer_matrix。
  --PIPELINE_CONFIG=offline_two_stage

  # 本地资源配置。
  --RESOURCE_CONFIG=local_8gpu_0_7

  # 推理 runtime 配置。
  --INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm

  # AIR 推理预算。
  --INFER_BUDGET_CONFIG=air_aligned_budget

  # LLM reranker 训练配置；infer-matrix-only 用于读取训练后 reranker 产物字段。
  --RERANKER_TRAINING_CONFIG=llm_reranker_base

  # Agent LLM 基座模型配置。
  --MODEL_CONFIG=qwen3_4b

  # 多轮 search-tool rollout 配置。
  --ROLLOUT_CONFIG=air_async_qwen3_4b

  # 本 infer-only task 的实验级 overlay。
  --OVERLAY_YAML=tasks/infer_tasks/agenticIterRag/configs/infer_matrix_only_overlay.yaml
)

bash "${ROOT}/scripts/agenticIterRag_v1/01_pipeline_launcher.sh" "${LAUNCHER_ARGS[@]}" "$@"
