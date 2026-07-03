# AIR Agentic RAG with LLM Reranker 服务组装详细设计

更新日期：2026-07-03

## 1. 目标

训练完 LLM reranker 后，需要把它接回 agentic RAG 服务链路。

服务链路是：

```text
agent llm
  -> search tool
  -> retriever top50
  -> llm reranker reorder
  -> top5 observation
  -> agent llm
```

这篇文档设计 service bundle。它不是直接启动服务，而是产出一组配置，让外部服务启动脚本可以读取并组合完整服务。

## 2. 非目标

第一版不做：

- 不实现完整服务启动器。
- 不管理线上多副本部署。
- 不做服务健康检查系统。
- 不做 reranker 失败自动 fallback。
- 不把 bundle 写入长期数据目录。

第一版只把配置和 manifest 写到本次 run artifact 目录。

## 3. 输出目录

训练完成后输出：

```text
outputs/agenticIterRag/<group>/<run>/service_bundle/
  service_config.yaml
  tool_config.yaml
  manifest.json
```

文件说明：

- `service_config.yaml`：服务级组合配置，描述 agent、retriever、reranker。
- `tool_config.yaml`：search tool 配置，可被启动脚本或 tool registry 消费。
- `manifest.json`：bundle 来源、模型路径、配置 hash、生成时间。

## 4. service_config.yaml

示例：

```yaml
# AIR 带 LLM reranker 的 agentic RAG 服务配置。
# 该文件由 build_service_bundle stage 生成，供外部服务启动脚本读取。

# service bundle schema 版本。
schema_version: air_service_bundle_v1

# 服务类型；启动脚本用它判断如何组合 agent、retriever 和 reranker。
service_type: agentic_rag_with_llm_reranker

agent:
  # search agent 模型路径。通常来自已训练 agent checkpoint。
  model_path: /path/to/trained/search_agent

  # agent tokenizer 路径，默认和 Qwen3-4B 基座模型一致。
  tokenizer_path: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B

retriever:
  # retriever HTTP endpoint。部署侧可以按实际端口覆盖。
  endpoint: http://127.0.0.1:8130/retrieve

  # retriever 返回候选数量。
  top_n: 50

llm_reranker:
  # 训练完成后的 LLM reranker 模型路径。
  model_path: /path/to/trained/llm_reranker

  # reranker 基座模型路径，用于 tokenizer 或服务初始化兜底。
  base_model: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B

  # reranker prompt 模板版本。
  prompt_template_version: air_rerank_tags_v1_full50

  # reranker 输出解析器版本。
  output_parser: air_rerank_tags_full50

  # 第一版要求 reranker 必须可用，失败时不静默回退。
  required: true

observation:
  # agent 最终可见文档数。无论 reranker 排多少，agent 只看 top5。
  visible_top_m: 5

  # tool observation 格式版本。
  tool_response_format_version: air_search_tool_response_v1
```

注释要求：

- 生成的 YAML 模板必须包含中文注释。
- 注释要说明字段来源：训练产物、运行环境，还是部署侧可覆盖。

## 5. tool_config.yaml

示例：

```yaml
# AIR search tool 配置：retriever 后接 LLM reranker。
# 外部服务启动脚本可以读取该文件，并结合端口、GPU 等部署配置启动完整链路。

tools:
  - class_name: verl.tools.agentic_iter_rag_retriever_tool.AgenticIterRagRetrieverTool
    config:
      # 工具类型标记。
      type: native

      # retriever 服务地址；部署侧可覆盖。
      retrieval_service_url: http://127.0.0.1:8130/retrieve

      # retriever 返回 top50 候选。
      recall_final_top_n: 50

      # agent observation 只暴露 top5。
      searchTool_final_top_m: 5

      # 启用 reranker。
      ranker_enabled: true

      ranker:
        # 第一版使用 LLM reranker 服务后端。
        backend: llm_reranker_service

        # reranker 必须可用，失败直接报错。
        required: true

        # 训练完成后的 reranker 模型路径。
        model_path: /path/to/trained/llm_reranker

        # prompt 模板版本。
        prompt_template_version: air_rerank_tags_v1_full50

        # 输出解析器版本。
        output_parser: air_rerank_tags_full50
```

注意：现有 `AgenticIterRagRetrieverTool` 当前支持 disabled/ray_actor dense ranker。接入 LLM reranker 服务时，需要后续实现新的 backend 分支或服务 adapter。service bundle 先把配置契约写清楚。

## 6. Manifest

`manifest.json` 示例：

```json
{
  "type": "air_llm_reranker_service_bundle",
  "schema_version": "air_service_bundle_v1",
  "created_at": "2026-07-03T00:00:00Z",
  "service_config": ".../service_config.yaml",
  "tool_config": ".../tool_config.yaml",
  "source_train_stage_manifest": ".../stages/train_llm_reranker/manifest.json",
  "reranker_model": "/path/to/trained/llm_reranker",
  "agent_model": "/path/to/trained/search_agent",
  "retriever_top_n": 50,
  "visible_top_m": 5,
  "prompt_template_version": "air_rerank_tags_v1_full50",
  "output_parser": "air_rerank_tags_full50",
  "required": true,
  "config_hash": "..."
}
```

## 7. 生成模块

建议实现：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/service_bundle.py
```

核心函数：

```text
build_service_bundle(config: dict, train_manifest: dict, output_dir: Path) -> dict
validate_service_bundle(bundle_dir: Path) -> None
```

`build_service_bundle` 负责：

- 读取训练后的 reranker model path。
- 读取 frozen/search agent model path。
- 读取 retriever endpoint。
- 写 `service_config.yaml`。
- 写 `tool_config.yaml`。
- 写 `manifest.json`。

代码注释要求：

- 生成每个配置文件前写中文注释，说明文件用途。
- 字段来源映射处写中文注释，说明哪些来自训练产物，哪些来自 runtime config。
- fail-fast 策略处写中文注释，说明为什么不静默 fallback。

## 8. Fail-fast 策略

第一版 bundle 默认：

```yaml
llm_reranker:
  required: true
```

含义：

- reranker 服务不可用，search tool 报错。
- reranker 输出格式错，search tool 报错。
- 不回退 retriever top5。

原因：

训练和评估阶段最怕“看起来启用了 reranker，但实际没有生效”。fail-fast 能让问题尽早暴露。

后续线上部署如果需要可用性优先，可以新增：

```yaml
required: false
fallback: retriever_top5
```

但第一版不实现。

## 9. 启动脚本消费方式

外部服务启动脚本读取：

```text
service_bundle/service_config.yaml
```

然后做：

1. 启动 agent vLLM。
2. 启动 retriever 服务。
3. 启动 LLM reranker 服务。
4. 把 `tool_config.yaml` 注入 agentic RAG runtime。

bundle 不负责实际启动进程，只提供配置事实源。

## 10. 校验规则

生成后立即校验：

- `service_config.yaml` 存在。
- `tool_config.yaml` 存在。
- `manifest.json` 存在。
- reranker model path 非空。
- agent model path 非空。
- `retriever.top_n == 50`。
- `observation.visible_top_m == 5`。
- `visible_top_m <= top_n`。
- `llm_reranker.required == true`。
- parser 和 prompt version 非空。

校验失败时，build_service_bundle stage 失败。

## 11. Runner 集成

新增 pipeline stage：

```text
build_service_bundle
```

输入：

- `train_llm_reranker.outputs.reranker_model`
- `infer_runtime.models.trained_agent_model`
- `infer_runtime.models.recall_model_path`
- retriever endpoint resource plan

输出：

- `service_bundle_dir`
- `service_config.yaml`
- `tool_config.yaml`
- `manifest.json`

dry-run 时只写预期路径和配置摘要。

## 12. 测试计划

### 12.1 正向生成测试

输入 fake train manifest。

期望：

- 生成三个文件。
- YAML 可解析。
- manifest 指向真实文件。

### 12.2 字段校验测试

删除 reranker model path。

期望：

- 校验失败。

把 `visible_top_m=60`。

期望：

- 校验失败。

### 12.3 Tool Config 初始化测试

用 tool registry 尝试读取 `tool_config.yaml`。

期望：

- 配置结构能被解析。

如果代码尚未实现 `llm_reranker_service` backend，则测试只校验配置结构，不实例化真实后端。

### 12.4 注释验收

人工检查：

- 生成的 YAML 模板有中文注释。
- 注释说明字段来源。
- fail-fast 策略有中文说明。
