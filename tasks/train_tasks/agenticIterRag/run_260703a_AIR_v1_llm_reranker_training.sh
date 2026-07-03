#!/usr/bin/env bash
set -euo pipefail

# 本任务是 AgenticIterRag v1 的 data produce 入口。
# 它只负责声明“这次实验选择哪些配置文件”，真正的业务参数必须写在 YAML 中。
#
# 配置读取顺序是：
#   1. main-run-config 先加载顶层 manifest。
#   2. DATA_CONFIG / PIPELINE_CONFIG / RESOURCE_CONFIG 等参数显式选择各个中间层 YAML。
#   3. OVERLAY_YAML 再覆盖本 task 特有的实验差异。
#   4. "$@" 允许临时追加 --key=value CLI override，用于 dry-run、断点续跑等少量运行时覆盖。
#
# main-run-config:
#   对应 AgenticIterRag/config/main_run/agentic_iter_rag_main.yaml。
#   负责项目名、默认实验名、运行目录根路径、artifact/report/log 根路径和审计策略。
#
# DATA_CONFIG:
#   对应 AgenticIterRag/config/data/co_search_ablation.yaml。
#   负责训练集、推理集、trace 生成数据、batch size、prompt/response 长度等数据侧参数。
#   当前复用历史 ablation parquet 数据文件；这只是已有磁盘数据目录，不是运行链路依赖。
#
# PIPELINE_CONFIG:
#   对应 AgenticIterRag/config/pipeline/offline_two_stage.yaml。
#   负责 pipeline 内部 stage 顺序和断点控制字段。
#   data produce overlay 会将实际执行范围限制为 generate_traces -> build_reranker_dataset。
#
# RESOURCE_CONFIG:
#   对应 AgenticIterRag/config/resource/local_8gpu_0_7.yaml。
#   负责本机硬件描述和 stage-level resource placement。
#   runner 会按 selected stages 生成 stage_resource_plan，而不是读取全局 agent/recall 绑卡。
#
# INFER_RUNTIME_CONFIG:
#   对应 AgenticIterRag/config/infer_runtime/agentic_iter_rag_vllm.yaml。
#   负责推理时的 run_mode、reranker 类型、retrieval top-N/top-M、LLM reranker 请求参数和推理产物路径字段。
#
# INFER_BUDGET_CONFIG:
#   对应 AgenticIterRag/config/infer_budget/air_aligned_budget.yaml。
#   负责推理样本数、最大 turn 数、prompt/response/tool token budget、采样参数和 vLLM 资源预算。
#
# RERANKER_TRAINING_CONFIG:
#   对应 AgenticIterRag/config/reranker_training/llm_reranker_base.yaml。
#   负责 LLM reranker 的基座模型、数据集 manifest 和训练超参。
#   注意：input_dataset/train_dataset 的生产策略不放在这里，而是放在 pipeline stage 子配置中。
#
# MODEL_CONFIG:
#   对应 AgenticIterRag/config/model/qwen3_4b.yaml。
#   负责 search-tool agent 的基座模型路径、trust_remote_code、gradient checkpointing 和 dtype。
#
# ROLLOUT_CONFIG:
#   对应 AgenticIterRag/config/rollout/air_async_qwen3_4b.yaml。
#   负责 agent 训练/推理时的多轮工具调用开关、最大 turn 数、并发工具调用数和采样参数。
#
# OVERLAY_YAML:
#   对应 tasks/train_tasks/agenticIterRag/configs/dataproduce_overlay.yaml。
#   负责本 data produce task 的实验级覆盖，例如实验名、已有 agent checkpoint、trace 生成数据
#   build_input_dataset/build_train_dataset 开关和数据生产策略。

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

cd "${ROOT}"

bash "${ROOT}/scripts/agenticIterRag_v1/01_pipeline_launcher.sh" \
  --main-run-config agentic_iter_rag_main \
  --DATA_CONFIG=co_search_ablation \
  --PIPELINE_CONFIG=offline_two_stage \
  --RESOURCE_CONFIG=local_8gpu_0_7 \
  --INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm \
  --INFER_BUDGET_CONFIG=air_aligned_budget \
  --RERANKER_TRAINING_CONFIG=llm_reranker_base \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=air_async_qwen3_4b \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/dataproduce_overlay.yaml \
  --data.trace_max_samples=-1 \
  --infer_budget.infer_batch_size=96 \
  --infer_budget.vllm.gpu_memory_utilization=0.8 \
  "$@"
