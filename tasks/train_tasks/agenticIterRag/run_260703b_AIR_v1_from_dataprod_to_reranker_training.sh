#!/usr/bin/env bash
set -euo pipefail

# 本任务是 AgenticIterRag v1 end-point hard subset 的 LLM reranker stage2 正式训练入口。
# overlay 在 bash CLI 中显式列出，避免环境变量覆盖导致运行记录和脚本默认值不一致。

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
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_base_overlay.yaml \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_answer_evidence_w02_n8_1p5epoch_overlay.yaml \
  --data.trace_max_samples=-1 \
  "$@"
