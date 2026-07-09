# AgenticIterRag v1 运行手册

## Data Produce Dry-run

本任务只执行 `generate_traces -> build_reranker_dataset`，用于从已有 agent checkpoint 生产 trajectory、input_dataset 和 GRPO train_dataset。

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh --dry-run
```

小样本真实生产可以临时覆盖样本数：

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh \
  --data.trace_max_samples=10
```

推理阶段默认每完成 10 条样本增量落盘一次，可通过 YAML 或 CLI dotlist 调整：

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh \
  --data.trace_max_samples=100 \
  --infer_runtime.artifacts.flush_every_n=10
```

增量落盘会重写当前已完成前缀的 `metrics.jsonl` 和 `traces.jsonl`，因此文件中不会出现重复样本；任务异常中断时，可以直接检查本次 `air_infer_trace` 目录下已经写出的前缀数据。

真实产物会写入长期数据目录：

```text
data/AgenticIterRag/trajectory/<trajectory_version>/
data/AgenticIterRag/llm_reranker_train_set/<input_dataset_version>/
```

## 全流程 Dry-run

```bash
bash tasks/train_tasks/agenticIterRag/run_00_AIR_v1_offline_example_task.sh --dry-run
```

## Infer-only Dry-run

```bash
bash tasks/infer_tasks/agenticIterRag/infer_AIR_v1_matrix.sh --dry-run
```

## 断点控制

只执行到 reranker 数据集构造：

```bash
bash tasks/train_tasks/agenticIterRag/run_00_AIR_v1_offline_example_task.sh \
  --dry-run \
  --pipeline.stop_after_stage=build_reranker_dataset
```

从 LLM reranker 训练阶段继续：

```bash
bash tasks/train_tasks/agenticIterRag/run_00_AIR_v1_offline_example_task.sh \
  --dry-run \
  --pipeline.resume_from_stage=train_llm_reranker
```

## 资源计划审计

AIR resource YAML 使用 stage-level placement。dry-run 后检查：

```text
log/agenticIterRag/<run>/outputs/execution_plan.yaml
```

其中 `stage_resource_plan` 会列出本次 selected stages 实际使用的 GPU、端口和服务实例。每个 stage manifest
也会在 `outputs.resource_plan` 中记录对应资源计划。

## 输出位置

dry-run 和真实运行都会写入：

- `log/agenticIterRag/<run>/runtime_logs/`
- `log/agenticIterRag/<run>/outputs/`
- `reports/agenticIterRag/`

`runtime_logs/` 保存 pipeline 级日志和最终配置快照；`outputs/` 保存 pipeline manifest、execution plan、stage manifest、checkpoint 和 stage 产物。

pipeline 顶层 manifest 和 execution plan 位于本次 run 的 `outputs/` 目录下。每个内部 stage 的 manifest 位于：

```text
outputs/stages/<stage>/manifest.json
```

`build_reranker_dataset` 还有两个内部子阶段 manifest：

```text
outputs/stages/build_reranker_dataset/build_input_dataset/manifest.json
outputs/stages/build_reranker_dataset/build_train_dataset/manifest.json
```
