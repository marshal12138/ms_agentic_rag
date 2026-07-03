# AgenticIterRag v1 架构

AgenticIterRag v1 采用单 pipeline 入口。外部调度系统只需要调度一次完整实验任务，pipeline 内部再按 YAML 定义的 stage 顺序执行。

## 单入口

正式 launcher 是：

```text
scripts/agenticIterRag_v1/01_pipeline_launcher.sh
```

正式完整实验 task 是：

```text
tasks/train_tasks/agenticIterRag/run_00_AIR_v1_offline_example_task.sh
```

该 task 虽然会触发完整离线两阶段 pipeline，但它仍是一个单独训练任务入口，因此放在 `tasks/train_tasks/`。`tasks/experiments/` 只用于更高层的多个 train/infer task 串行编排。

可选 infer-only task 是：

```text
tasks/infer_tasks/agenticIterRag/infer_AIR_v1_matrix.sh
```

旧的 train agent、generate traces、build dataset、train reranker、infer matrix stage 级 shell 入口不再作为正式入口存在。

## Pipeline DAG

v1 默认 pipeline 是：

```text
train_agent
generate_traces
build_reranker_dataset
train_llm_reranker
infer_matrix
```

这些 stage 是 Python pipeline runner 的内部节点，不是 task 层调度单元。

## Stage Resource Plan

AIR 的资源计划按本次 selected stages 生成，而不是按全局角色静态绑卡。resource YAML 描述硬件和
`stage_resources`，runner 根据 `resume_from_stage`、`stop_after_stage` 和 `skip_stages` 生成
`execution_plan.yaml` 中的 `stage_resource_plan`。

这保证同一张 GPU 可以在不同顺序 stage 间复用，同时 dry-run 能审计每个 stage 会启动哪些服务、占用哪些
GPU 和端口。

## 参考边界

CoAgenticRetriever v2 作为以下部分的参考：

- launcher 和 compiler 的分层方式
- 配置编译与审计文件落盘方式
- no-ranker agent 训练形态
- full trace 的字段预期
- infer matrix 的报告口径

AgenticIterRag v1 自身负责：

- 单 pipeline 入口
- 独立 AIR infer launcher、infer engine 和 search tool class
- trace 到 reranker dataset 的数据契约
- LLM reranker 数据和推理 adapter
- pipeline manifest 和 execution plan

AIR 运行时不调用 `scripts/coagenticRetriever_v2`。历史数据文件和 checkpoint 路径中可能仍包含
`coAgenticRetriever` 或 `CAR` 字符串，这只是已有磁盘目录命名，不表示 AIR 调用 CAR 代码链路。
