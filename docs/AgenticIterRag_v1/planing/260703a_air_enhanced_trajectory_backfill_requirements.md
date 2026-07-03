# AIR 增强轨迹补产需求文档

更新日期：2026-07-03

## 1. 背景

这篇文档只讨论一件事：为了训练 AIR 的 LLM reranker，我们到底还缺什么轨迹数据，以及需要怎么补产。

现在 AIR 已经可以做 data produce：

- 用已训练好的 search agent 对训练集 query 做 rollout。
- 保存原始 raw trace。
- 抽取 canonical trajectory。
- 再从 trajectory 生成 LLM reranker 的 input dataset 和 train dataset。

这些数据已经足够训练一个“静态 reranker prompt 数据集”。也就是说，给 reranker 一个 `question + sub_query + top50 docs`，让它学会输出 `<reason>` 和 `<rerank>`。

但我们现在要做的不是静态排序训练，而是更接近真实 agentic RAG 的训练：

1. 选中原 agent 轨迹里的某一步 search。
2. 让 reranker 改这一步 top50 docs 的排序。
3. 只把排序后的 top5 作为新的 observation 给回 search agent。
4. 让 search agent 从这个点继续 rollout。
5. 用最终 answer 的 reward 反过来训练 reranker。

这个训练方式要求我们能精确回到原轨迹中的某个 search 点位。当前 AIR trajectory 不够，因为它只记录了 `sub_query`、`recall_topn_docs`、`visible_docs`、`final_answer` 和 `metrics` 这些结果字段，没有保存“这一轮 search 前后 agent 的完整上下文状态”。

所以，第一步必须补产一种增强轨迹。它不是替代现有 trajectory，而是在现有 trajectory 基础上增加 continuation 所需的上下文快照。

## 2. 当前数据缺口

当前 canonical trajectory 的粒度是“一次 search call 一条记录”。它能回答：

- 这个 search call 属于哪个样本。
- agent 当时生成了什么 `sub_query`。
- retriever 返回了哪些 topN docs。
- 原链路暴露给 agent 的 topM docs 是什么。
- 最后原轨迹答案是什么，reward 是多少。

但它不能回答几个更关键的问题：

- 这个 search call 发生前，agent 的完整 messages 是什么。
- 当前 assistant tool_call 已经写入上下文了吗。
- tool observation 是以什么格式插入的。
- 如果把 observation 替换成 reranker top5，应该接在哪个位置。
- 替换以后继续 rollout 时，agent 应该看到的历史和原 data produce 是否完全一致。

如果没有这些信息，我们只能近似重构上下文。近似重构会带来很大的训练噪声：reranker 改的是轨迹 A 的某个 query/doc，但 agent continuation 看到的上下文可能不是当时真实上下文，甚至可能 role 顺序、tool response 格式、chat template 都不一致。

这会直接违反 reranker 训练里的两个关键约束：

- query 轨迹要对齐。
- 上下文格式要一致。

所以增强轨迹不是锦上添花，它是严格实现该训练方法的前置条件。

## 3. 增强轨迹的目标

增强轨迹要支持一个很具体的操作：

给定一条样本和一个 search step，我们能稳定构造：

```text
原始上下文到当前 tool_call 为止
+ 新的 tool observation
+ frozen search agent 后续继续 rollout
```

也就是说，它必须让训练代码可以做下面这个动作：

```text
messages_before_tool_response
  + render_tool_message(reranker_top5_docs)
  -> continue_rollout_with_frozen_agent()
  -> new_answer
  -> reward(new_answer)
```

补产增强轨迹后，reranker 训练不再需要猜测历史上下文，也不需要从字符串里硬拆 prompt。它只需要读结构化字段。

## 4. 产物位置

建议把增强轨迹作为 trajectory 数据集内部的可选增强文件，而不是另开一类完全独立数据集。

推荐目录：

```text
data/AgenticIterRag/
  trajectory/
    <trajectory_version>/
      trajectory.jsonl
      raw_traces.jsonl
      metrics.jsonl
      enhanced_trajectory.jsonl
      enhanced_example.json
      enhanced_summary.json
      manifest.json
      final_config.yaml
```

字段说明：

- `enhanced_trajectory.jsonl`：增强轨迹主文件，一行是一条完整原始样本的 rollout。
- `enhanced_example.json`：从增强轨迹里抽取的一条样例，方便人工检查。
- `enhanced_summary.json`：增强轨迹统计，例如样本数、总 search step 数、最大 step 数、空 search 样本数。
- `manifest.json`：增加增强轨迹相关索引字段。

不建议把增强轨迹拆成每个 search step 一行的主文件。原因是继续 rollout 的上下文天然属于一条完整 trajectory，按完整 trajectory 存可以保留更清晰的关系。后续要做 per-step dataset 时，再从 `steps[]` 里展开即可。

## 5. 增强轨迹顶层格式

`enhanced_trajectory.jsonl` 中每一行是一条完整 rollout，推荐格式如下：

```json
{
  "schema_version": "air_enhanced_trajectory_v1",
  "trajectory_id": "sample-000123",
  "sample_id": "original-sample-id",
  "data_source": "nq",
  "source_index": 123,
  "question": "what are three branches of government in the united states?",
  "gold_answers": ["legislative, executive, and judicial"],
  "agent_model": "/path/to/search-agent-checkpoint",
  "agent_model_role": "trained_agent",
  "context_format_version": "air_agent_messages_v1",
  "tool_response_format_version": "air_search_tool_response_v1",
  "chat_template_source": "qwen3_chat_template",
  "tokenizer_name_or_path": "/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B",
  "baseline_final_answer": "legislative, executive, and judicial",
  "baseline_reward": 1.0,
  "baseline_metrics": {
    "em": 1.0,
    "f1": 1.0,
    "tool_calls": 2,
    "status": "answered"
  },
  "initial_prompt_messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "final_messages": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "<reason>...</reason>\n<tool_call>...</tool_call>"
    },
    {
      "role": "tool",
      "content": "[1] ...\n[2] ..."
    },
    {
      "role": "assistant",
      "content": "<reason>...</reason>\n<answer>...</answer>"
    }
  ],
  "steps": [],
  "raw_trace_ref": {
    "path": "data/AgenticIterRag/trajectory/<version>/raw_traces.jsonl",
    "line_index": 123
  },
  "created_at": "2026-07-03T00:00:00Z"
}
```

顶层字段解释：

- `schema_version`：增强轨迹 schema 版本。第一版固定为 `air_enhanced_trajectory_v1`。
- `trajectory_id`：增强轨迹内部 ID。后续 reranker 样本必须用它做主键之一。
- `sample_id`：源数据样本 ID，优先继承原 parquet 或 raw trace 中的 uid/id/index。
- `data_source`：数据集来源，例如 `nq`。
- `source_index`：源数据行号或本次 infer 行号，方便回查。
- `question`：原始用户问题。
- `gold_answers`：标准答案列表。
- `agent_model`：生成该轨迹的 search agent checkpoint。
- `agent_model_role`：通常是 `trained_agent`，用于区分 origin/trained agent。
- `context_format_version`：messages 结构版本。
- `tool_response_format_version`：tool observation 渲染格式版本。
- `chat_template_source`：agent 推理时使用的 chat template 来源。
- `tokenizer_name_or_path`：agent tokenizer 路径。
- `baseline_final_answer`：原 no-ranker rollout 最终答案。
- `baseline_reward`：原 no-ranker rollout 的最终 reward。
- `baseline_metrics`：原轨迹完整 metrics。
- `initial_prompt_messages`：源 parquet 中原始 prompt messages。
- `final_messages`：原轨迹完整 messages，包含所有 assistant/tool 轮次。
- `steps`：该轨迹中每次 search 的结构化快照。
- `raw_trace_ref`：回指 raw trace 的位置。
- `created_at`：写入时间。

## 6. Search Step 格式

`steps[]` 是增强轨迹最关键的部分。每个元素对应一次 search tool call。

推荐格式：

```json
{
  "step_index": 0,
  "turn_index": 0,
  "sub_query": "three branches of government in the united states",
  "tool_call": {
    "name": "search",
    "arguments": {
      "query": "three branches of government in the united states"
    }
  },
  "assistant_tool_call_message": {
    "role": "assistant",
    "content": "<reason>...</reason>\n<tool_call>{...}</tool_call>"
  },
  "messages_before_tool_response": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "<reason>...</reason>\n<tool_call>{...}</tool_call>"
    }
  ],
  "assistant_turns_so_far": 1,
  "user_turns_so_far": 0,
  "recall_topn_docs": [
    {
      "doc_id": "2819066",
      "id": "2819066",
      "rank": 1,
      "recall_rank": 1,
      "title": "Federal government of the United States",
      "text": "...",
      "contents": "...",
      "score": 0.8955078125,
      "recall_score": 0.8955078125
    }
  ],
  "original_ranked_docs": [
    {
      "doc_id": "2819066",
      "rank": 1,
      "text": "..."
    }
  ],
  "original_visible_docs": [
    {
      "doc_id": "2819066",
      "rank": 1,
      "text": "..."
    }
  ],
  "original_tool_message": {
    "role": "tool",
    "content": "[1] Federal government of the United States\n..."
  },
  "messages_after_original_tool_response": [
    {
      "role": "user",
      "content": "..."
    },
    {
      "role": "assistant",
      "content": "<reason>...</reason>\n<tool_call>{...}</tool_call>"
    },
    {
      "role": "tool",
      "content": "[1] Federal government of the United States\n..."
    }
  ],
  "doc_id_order": [
    "2819066",
    "6259445"
  ],
  "original_visible_doc_ids": [
    "2819066",
    "6259445"
  ],
  "step_metrics": {
    "num_recall_docs": 50,
    "num_agent_visible_docs": 5,
    "retrieve_s": 0.123,
    "ranker_s": 0.0
  }
}
```

字段解释：

- `step_index`：该 trajectory 内第几次 search，从 0 开始。
- `turn_index`：对应 assistant turn，从 0 开始。第一版可以等于 `step_index`，但建议保留独立字段。
- `sub_query`：agent 生成的 search query。它必须等于 tool call JSON 里的 `arguments.query`。
- `tool_call`：结构化工具调用。
- `assistant_tool_call_message`：当前 assistant 生成 tool_call 的完整消息。
- `messages_before_tool_response`：最重要字段。它停在当前 assistant tool_call 之后，tool observation 之前。
- `assistant_turns_so_far`：到当前 tool_call 为止已经有多少 assistant turn。
- `user_turns_so_far`：到当前 tool_call 为止已经插入多少 tool/user observation。
- `recall_topn_docs`：retriever 返回的 topN，默认 top50。
- `original_ranked_docs`：原链路排序后的 docs。no-ranker 时应等于 recall 顺序。
- `original_visible_docs`：原链路交给 agent 的 topM，默认 top5。
- `original_tool_message`：原链路真实插入的 tool message。
- `messages_after_original_tool_response`：原链路插入原 top5 observation 后的上下文。
- `doc_id_order`：`recall_topn_docs` 的 doc_id 顺序。
- `original_visible_doc_ids`：原 top5 doc_id。
- `step_metrics`：这一步的检索和格式统计。

## 7. 关于 `messages_before_tool_response`

这个字段要特别强调。它不是“search 前的上下文”，也不是“完整最终上下文”，而是：

```text
当前 assistant 已经生成了 <tool_call>
但 tool observation 还没有插入
```

举个例子，原始轨迹长这样：

```text
user: Question
assistant: <reason>...</reason><tool_call>{"query": "q1"}</tool_call>
tool: [1] docA ...
assistant: <reason>...</reason><tool_call>{"query": "q2"}</tool_call>
tool: [1] docB ...
assistant: <reason>...</reason><answer>...</answer>
```

如果我们训练时选择第 2 步 search，也就是 `q2`，那么 `messages_before_tool_response` 应该是：

```text
user: Question
assistant: tool_call q1
tool: original observation for q1
assistant: tool_call q2
```

然后训练时会拼：

```text
messages_before_tool_response
+ new tool observation from reranker(q2 top50)
```

再让 frozen agent 继续生成。

这样才能保证 reranker 只改变目标 step 的 observation，不会顺手改掉前面历史。

## 8. 文档字段规范

为了兼容现有代码，doc 建议同时保留几种字段：

```json
{
  "doc_id": "2819066",
  "id": "2819066",
  "title": "Federal government of the United States",
  "text": "...",
  "contents": "...",
  "rank": 1,
  "recall_rank": 1,
  "score": 0.8955078125,
  "recall_score": 0.8955078125
}
```

要求：

- `doc_id` 必须存在，是后续所有对齐校验的主键。
- `text` 必须存在，供 AIR reranker prompt 使用。
- `contents` 建议保留，兼容旧 search tool formatter。
- `rank` 和 `recall_rank` 必须能表达原 retriever 顺序。
- 允许保留额外字段，但不允许缺失 `doc_id` 或 `text/contents`。

如果原始 retriever 只返回 `id` 和 `contents`，增强轨迹抽取时必须补出：

- `doc_id = str(id)`
- `text = contents`

## 9. Manifest 增强

`trajectory/<version>/manifest.json` 需要增加：

```json
{
  "enhanced_trajectory_jsonl": "data/AgenticIterRag/trajectory/<version>/enhanced_trajectory.jsonl",
  "enhanced_example_json": "data/AgenticIterRag/trajectory/<version>/enhanced_example.json",
  "enhanced_summary_json": "data/AgenticIterRag/trajectory/<version>/enhanced_summary.json",
  "enhanced_schema_version": "air_enhanced_trajectory_v1",
  "enhanced_record_count": 1000,
  "enhanced_search_step_count": 2345,
  "context_format_version": "air_agent_messages_v1",
  "tool_response_format_version": "air_search_tool_response_v1"
}
```

`enhanced_summary.json` 建议包含：

```json
{
  "dataset_type": "enhanced_trajectory",
  "schema_version": "air_enhanced_trajectory_v1",
  "record_count": 1000,
  "search_step_count": 2345,
  "records_without_search": 12,
  "max_steps_per_record": 6,
  "avg_steps_per_record": 2.345,
  "top_n": 50,
  "top_m": 5,
  "context_format_version": "air_agent_messages_v1",
  "tool_response_format_version": "air_search_tool_response_v1"
}
```

## 10. 补产方式

第一版推荐直接改 AIR infer backend，让它在 data produce 时同步写增强字段。

原因很简单：infer backend 在 rollout 当下最清楚 messages 是什么、当前 assistant 输出是什么、tool response 是怎么拼进去的。这个时候保存上下文最稳。

具体要在 AIR infer 的多轮循环里，在每次 tool response 插入前后记录：

1. assistant tool_call message。
2. tool_call 结构化 JSON。
3. `messages_before_tool_response`。
4. retriever top50 docs。
5. 原链路 final top5 docs。
6. 原 tool response message。
7. `messages_after_original_tool_response`。

不推荐先从现有 `raw_traces.jsonl` 反推增强轨迹。现有 raw trace 里虽然有 prompt、sub_queries 和 doc 数组，但没有完整 messages 快照，反推只能做到近似。

## 11. 校验规则

补产增强轨迹时必须做强校验。

每个 step 必须满足：

- `sub_query == tool_call.arguments.query`。
- `messages_before_tool_response[-1].role == "assistant"`。
- `messages_before_tool_response[-1].content` 中必须包含 `<tool_call>`。
- `original_tool_message.role == "tool"`。
- `len(recall_topn_docs) > 0`。
- `len(original_visible_docs) <= top_m`。
- `original_visible_doc_ids` 必须是 `doc_id_order` 的前 topM，no-ranker 模式下尤其如此。
- `messages_after_original_tool_response == messages_before_tool_response + [original_tool_message]` 在结构上成立。

每条 trajectory 必须满足：

- `len(steps) == baseline_metrics.tool_calls`，除非原推理状态明确异常。
- `baseline_reward` 必须存在。
- `gold_answers` 必须非空。
- `context_format_version` 必须存在。
- `tool_response_format_version` 必须存在。

如果校验失败，默认 fail-fast，不要把坏样本静默写进训练数据。

## 12. 和 Reranker Dataset 的关系

增强轨迹补产后，后续 reranker branch dataset 应该从 `enhanced_trajectory.jsonl` 构造，而不是从旧 `trajectory.jsonl` 构造。

两者关系如下：

```text
enhanced_trajectory.jsonl
  -> build_reranker_branch_dataset
  -> reranker GRPO train dataset
```

branch dataset 中每条样本必须绑定：

- `trajectory_id`
- `step_index`
- `sub_query`
- `candidate_doc_ids`
- `messages_before_tool_response`

这样训练时才能严格知道：reranker 改的是哪条轨迹的哪一步。

## 13. 验收标准

小样本补产验收：

```bash
bash tasks/train_tasks/agenticIterRag/run_260702a_AIR_v1_dataproduce.sh \
  --data.trace_max_samples=10
```

通过条件：

- `enhanced_trajectory.jsonl` 存在。
- `enhanced_example.json` 存在。
- `enhanced_summary.json` 存在。
- 每条有 search 的样本至少有一个 `steps[]`。
- 每个 step 都有 `messages_before_tool_response`、`recall_topn_docs`、`original_visible_docs`。
- 随机抽 3 条样本，用 `messages_before_tool_response + original_tool_message` 重新渲染，格式和原始 tool observation 一致。

负向验收：

- 删除某个 step 的 `messages_before_tool_response`，branch dataset 构造必须失败。
- 修改某个 `sub_query`，branch dataset 构造必须失败。
- 修改某个 doc_id 顺序，branch dataset 构造必须失败。

## 14. 第一版不做的事情

第一版增强轨迹补产不做这些事：

- 不训练 reranker。
- 不设计多步 reranker 同时生效的 reward。
- 不把历史旧 `raw_traces.jsonl` 强行转换成严格增强轨迹。
- 不支持模糊匹配 step。
- 不允许跨 trajectory 复用上下文。

第一版只把继续 rollout 所需的上下文快照保存正确。只要这个数据基础是干净的，后面的 reranker GRPO 才有意义。
