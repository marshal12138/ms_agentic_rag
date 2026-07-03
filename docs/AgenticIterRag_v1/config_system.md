# AgenticIterRag v1 配置体系

所有业务配置都必须有 YAML 字段作为基础来源。如果某个值在框架准备阶段尚未确定，也必须在 YAML 中显式保留字段，并将值设为 `null`。

## 分层职责

task 只表达一次实验配置，例如选择 main_run、各配置组 YAML 和 task overlay。

训练 task 必须显式写出中间层配置组选型，便于阅读和调度审计：

```bash
--DATA_CONFIG=co_search_ablation
--PIPELINE_CONFIG=offline_two_stage
--RESOURCE_CONFIG=local_8gpu_0_7
--INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm
--INFER_BUDGET_CONFIG=air_aligned_budget
--RERANKER_TRAINING_CONFIG=llm_reranker_base
--MODEL_CONFIG=qwen3_4b
--ROLLOUT_CONFIG=air_async_qwen3_4b
```

这些参数只选择 YAML 配置组，不承载具体业务参数。具体路径、batch size、top-N/top-M、服务端口和训练超参仍必须写在对应 YAML 或 overlay 中。

launcher 只负责调用 compiler、source runtime env、启动 pipeline runner。

pipeline YAML 负责描述内部 stage、断点控制、输入输出和资源需求。

Python runner 负责解释 pipeline YAML 并按 stage 执行。

resource YAML 必须描述硬件和 stage-level placement，而不是全局角色绑卡。AIR pipeline 可以通过
`resume_from_stage`、`stop_after_stage` 和 `skip_stages` 执行不同 stage 子集，因此资源计划必须按本次
selected stages 动态解析。

推荐结构：

```yaml
hardware:
  accelerator: auto
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]

stage_resources:
  generate_traces:
    services:
      agent_vllm:
        gpu_ids: [0, 1, 2, 3]
        tensor_parallel_size: 4
        port: 8140
      recall:
        gpu_ids: [5]
        port: 8130
        retrieval_service_url: http://127.0.0.1:8130/retrieve

  build_reranker_dataset:
    local_cpu:
      workers: 1
```

旧式全局字段不再支持：

```yaml
agent:
recall:
judge:
original_llm_reranker:
trained_llm_reranker:
```

## Shell 限制

Shell 脚本不得独立定义模型路径、数据路径、batch size、top-N/top-M、服务端口、服务开关、训练超参或 reranker 设置。

允许保留的少量环境变量包括：

- `EXP_NAME`
- `GROUP_NAME`
- `PY`
- accelerator 可见设备兼容变量，例如 `CUDA_VISIBLE_DEVICES` 和 `ASCEND_VISIBLE_DEVICES`
- compiler 生成的 runtime 变量

除非为了调试显式设置 `AGENTIC_ITER_RAG_ALLOW_SHELL_CONFIG=1`，否则 compiler 会拒绝 shell-only 业务配置环境变量。

## 优先级

配置优先级为：

```text
main_run 默认配置组 < task 显式配置组选型 < task overlay < CLI dotlist
```

每次 pipeline 运行都会落盘：

- `pipeline.final_config.yaml`
- `pipeline.final_config.json`
- `pipeline.env`
- `pipeline.args.txt`
- `pipeline.manifest.json`
- `execution_plan.yaml`
- `stages/<stage>/manifest.json`

`execution_plan.yaml` 必须包含本次 selected stages 的 `stage_resource_plan`，用于 dry-run 审计每个 stage
实际占用的 GPU、端口和服务实例。
