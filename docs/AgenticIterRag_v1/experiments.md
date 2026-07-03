# AgenticIterRag v1 实验设计

## Task 边界

完整实验 task：

```text
tasks/train_tasks/agenticIterRag/run_00_AIR_v1_offline_example_task.sh
```

该 task 表达一次完整 AgenticIterRag v1 离线两阶段实验。

由于该入口本身仍是一个具体训练 pipeline，而不是多个独立 train/infer task 的外层串行编排，因此放在 `tasks/train_tasks/`。

infer-only task：

```text
tasks/infer_tasks/agenticIterRag/infer_AIR_v1_matrix.sh
```

该 task 只用于已有 agent/reranker 产物的独立推理矩阵。

## 推理矩阵

初始推理矩阵必须包含四组对照：

1. `origin_agent`
2. `trained_agent`
3. `trained_agent_original_llm_reranker`
4. `trained_agent_trained_llm_reranker`

所有对照组必须使用相同的：

- 推理数据路径
- retrieval top-N
- visible top-M
- 生成 budget
- 最大轮数
- 指标名称

每份报告必须包含：

- agent 模型路径
- reranker 类型
- reranker 模型或 endpoint
- final config 路径
- metrics JSONL 路径
- trace 路径
