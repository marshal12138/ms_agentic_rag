# AIR Reranker Continuation Rollout 详细设计

更新日期：2026-07-03

## 1. 目标

continuation rollout 是这套训练方法最核心的动作。

它要做的是：

```text
拿到 reranker 新排序
取 top5 docs
把 top5 渲染成新的 tool observation
接到原 agent 历史上下文里
让 frozen search agent 从这里继续 rollout
最后得到新 answer
```

这一步必须非常严格。因为 reranker reward 就来自这个新 answer。如果上下文拼错，reward 就不可信。

## 2. 非目标

第一版 continuation 不做：

- 不训练 search agent。
- 不在后续 search 里继续调用 reranker。
- 不从头重新跑完整轨迹。
- 不尝试修复格式错误的 reranker 输出。
- 不支持 all-steps reranker action 的 credit assignment。

第一版只支持一个 reranker action 改一个 search step。

## 3. 输入和输出

输入：

- `messages_before_tool_response`
- reranker 排序后的 top5 docs
- frozen search agent 模型
- retriever-only search tool 配置
- gold answers
- continuation budget

输出：

```json
{
  "final_answer": "new answer",
  "assistant_text": "<reason>...</reason><answer>...</answer>",
  "metrics": {
    "status": "answered",
    "tool_calls_after_branch": 1,
    "agent_turns_after_branch": 2
  },
  "continuation_trace": {
    "messages": [],
    "new_tool_message": {},
    "visible_doc_ids": []
  }
}
```

## 4. 上下文拼接规则

核心拼接：

```text
branch_messages = messages_before_tool_response + [new_tool_message]
```

`messages_before_tool_response` 必须停在当前 assistant tool_call 之后。

例如原轨迹：

```text
user: Question
assistant: tool_call q1
tool: old observation q1
assistant: tool_call q2
tool: old observation q2
assistant: answer
```

如果替换 q2，则：

```text
messages_before_tool_response =
  user: Question
  assistant: tool_call q1
  tool: old observation q1
  assistant: tool_call q2

new_tool_message =
  tool: reranker top5 observation for q2
```

拼好后从这里继续：

```text
user
assistant q1
tool old q1
assistant q2
tool new q2
assistant ...
```

这保证只改变目标 step，不改变前面的历史。

代码注释要求：

- 拼接处必须写中文注释，说明这里为什么不能用 `final_messages` 或 `messages_after_original_tool_response`。
- role 顺序校验处必须写中文注释，说明 tool message 必须接在 assistant tool_call 后面。

## 5. Tool Observation 渲染

new tool message 只包含 top5 docs。

格式必须复用 AIR search tool 当前 observation 格式。

推荐接口：

```text
render_air_tool_observation(
    docs: list[dict],
    max_tool_response_length: int,
    format_version: str,
) -> dict
```

返回：

```json
{
  "role": "tool",
  "content": "[1] title\ntext\n[2] title\ntext..."
}
```

要求：

- 输入 docs 最多 5 篇。
- index 从 1 开始重新编号。
- 不保留原 candidate index。
- 超长时按 AIR 当前 tool response 截断策略处理。

注意：reranker 输出的是 50 排序，但 agent 看到的是重新编号后的 top5 observation。agent 不需要知道它们原来是候选 `[27]` 或 `[3]`。

## 6. 后续 Search Tool 规则

continuation 后续如果 agent 再调用 search：

```text
只能调用 retriever-only search tool
```

也就是说：

- 不调用正在训练的 reranker。
- 不调用训练后的 reranker。
- 不调用任何 dense ranker。

这样 reward 才能归因到“当前这一步被替换的 observation”。

配置必须写死：

```yaml
reranker_training:
  continuation:
    search_tool_mode: retriever_only
```

如果配置不是 `retriever_only`，训练入口直接失败。

代码注释要求：

- 创建 search tool config 的地方必须写中文注释，说明后续 search 禁用 reranker 是为了保证单步 counterfactual 归因。

## 7. Frozen Agent 调用

continuation 使用 frozen search agent。

要求：

- agent 模型不更新。
- 使用与 data produce 相同的 chat template。
- 使用与 data produce 相同的 `<reason>/<tool_call>/<answer>` 格式约束。
- sampling 默认 temperature=0。
- max turns 从 `reranker_training.continuation` 读。

接口建议：

```text
class ContinuationRunner:
    async def run(
        self,
        messages_before_tool_response: list[dict],
        visible_docs: list[dict],
        gold_answers: list[str],
        extra_info: dict,
    ) -> ContinuationResult
```

`ContinuationResult`：

```text
final_answer: str
assistant_text: str
status: str
messages: list[dict]
metrics: dict
trace: dict
```

## 8. 终止条件

continuation 终止条件和 AIR infer 保持一致：

- agent 输出 `<answer>`。
- 达到 `max_assistant_turns`。
- 达到 `max_user_turns`。
- prompt 超过 `max_prompt_length`。
- response 超过 `max_response_length`。
- 多 tool_call 或非法格式导致无法继续。

状态值建议：

- `answered`
- `max_turns`
- `max_user_turns`
- `prompt_too_long`
- `multiple_tool_calls`
- `no_valid_answer`
- `tool_error`
- `agent_error`

这些状态要写入 reward extra info。

## 9. 上下文一致性校验

运行 continuation 前先校验：

- `messages_before_tool_response` 非空。
- 最后一条 message 是 assistant。
- 最后一条 assistant content 包含 `<tool_call>`。
- new tool message role 是 `tool`。
- `context_format_version == air_agent_messages_v1`。
- `tool_response_format_version == air_search_tool_response_v1`。

校验失败直接抛错，不给模型 penalty。

原因：这是数据或实现错误，不是 reranker 输出错误。

## 10. 错误处理

模型行为错误：

- reranker 格式错误：reward 模块处理，continuation 不运行。

数据错误：

- messages 缺失。
- role 顺序不对。
- top5 doc 缺 text。

处理：直接失败。

基础设施错误：

- frozen agent 服务连接失败。
- retriever 服务连接失败。
- timeout。

处理：第一版直接失败，不转成模型 penalty。

原因是这些错误和 reranker 排序无关，转成 penalty 会污染训练。

## 11. 日志和 Trace

每次 continuation 至少记录：

- `trajectory_id`
- `step_index`
- `visible_doc_ids`
- `final_answer`
- `status`
- `tool_calls_after_branch`
- `agent_turns_after_branch`
- `continuation_s`
- `retriever_call_count`
- `reranker_call_count_after_branch`

其中 `reranker_call_count_after_branch` 必须是 0。

如果不是 0，说明后续 search 错误接入 reranker，训练必须失败。

## 12. 实现计划

建议新增：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/continuation_rollout.py
```

核心函数：

```text
validate_branch_context(extra_info) -> None
render_new_tool_message(docs, config) -> dict
build_branch_messages(messages_before_tool_response, new_tool_message) -> list[dict]
run_frozen_agent_continuation(messages, config) -> ContinuationResult
```

代码注释要求：

- `validate_branch_context` 写清楚每条校验保护什么。
- `build_branch_messages` 写清楚只替换当前 tool observation。
- `run_frozen_agent_continuation` 写清楚 frozen agent 不更新参数。
- 后续 search tool 初始化处写清楚 retriever-only 约束。

## 13. 测试计划

### 13.1 拼接正向测试

输入：

- 一段合法 `messages_before_tool_response`。
- 5 篇 doc。

期望：

- 输出 messages 长度增加 1。
- 最后一条是 tool。
- 前序 messages 未被修改。

### 13.2 拼接负向测试

输入最后一条不是 assistant。

期望：

- 报错。

输入最后 assistant 不含 `<tool_call>`。

期望：

- 报错。

### 13.3 Top5 Observation 测试

输入 50 排序 docs。

期望：

- observation 只含前 5 篇。
- 第 6 篇内容不出现。

### 13.4 后续 Retriever-only 测试

mock agent 在 continuation 后续再次 search。

期望：

- retriever 被调用。
- reranker 没有被调用。
- `reranker_call_count_after_branch=0`。

### 13.5 Infra Error 测试

mock frozen agent 连接失败。

期望：

- 抛 infra error。
- 不写成 `format_penalty`。

### 13.6 注释验收

人工检查：

- 上下文拼接处有中文注释。
- role 顺序校验有中文注释。
- 禁用后续 reranker 的地方有中文注释。
