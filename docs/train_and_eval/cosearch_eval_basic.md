# CoSearch_derevitives 框架

## 评估

`scripts/cosearch_local/11_evaluate_cosearch_base.sh` 是 CoSearch 模型的基础评估入口。

该评估器有意设计为仅使用 vLLM：

- 启动 dense retriever 实例和一个轮询式 retrieval proxy。
- 为主 agent 启动一个兼容 OpenAI API 的 vLLM server。
- 为 reranker agent 启动一个兼容 OpenAI API 的 vLLM server。
- 在 VERL 之外执行 rollout、检索、重排、EM/F1 评分、延迟聚合、trace 保存和 markdown 报告生成。
- 不得使用 VERL 作为模型加载或评估运行时。

默认资源和数据：

- Agent 模型：`/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- Reranker 模型：`/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- 评估数据：`data/co_search/local_flashrag/co_search_ablation.eval.parquet`
- Dense retriever 实例数：`4`
- Dense retriever 默认模式：`gpu`
- Dense retriever 默认 GPU：`5`
- Agent GPU：`0,1`
- Reranker GPU：`2,3`
- Agent tensor parallel size：`2`
- Reranker tensor parallel size：`2`
- 评估 batch size：`32`
- `MAX_EVAL_NUM=-1`，表示评估完整数据集。

启动行为：

- 所有已配置的 dense retriever 实例会并行启动，然后统一进行健康检查。
- 如果配置端口上已有健康的 retriever 实例，则会复用这些实例。
- retriever、proxy、agent vLLM 和 reranker vLLM 的运行时日志会保存在评估结果目录中。

Prompt 和 rollout 限制与 `09` 训练路径保持一致：

- `TOP_N=50`
- `TOP_M=5`
- `MAX_ASSISTANT_TURNS=6`
- `MAX_USER_TURNS=6`
- `MAX_PROMPT_LENGTH=11264`
- `MAX_RESPONSE_LENGTH=1024`
- `MAX_MODEL_LEN=12288`
- `MAX_TOOL_RESPONSE_LENGTH=4096`

用法：

```bash
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

常用覆盖项：

```bash
STRATEGY_NAME=my_policy \
AGENT_MODEL=/path/to/model_or_checkpoint_step \
RERANKER_MODEL=/path/to/model_or_checkpoint_step \
MAX_EVAL_NUM=100 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

模型路径解析：

- 如果 `AGENT_MODEL` 或 `RERANKER_MODEL` 直接指向可加载的 HuggingFace 模型目录，则使用该目录。
- 如果它指向 VERL checkpoint step 目录，评估器会在其下查找可加载的 HF safetensors。
- 对于 agent 角色，优先使用 `hf_safetensors/actor`。
- 对于 reranker 角色，优先使用 `hf_safetensors/reranker_actor_rollout`。

示例：

```text
global_step_79/
  hf_safetensors/
    actor/
    reranker_actor_rollout/
```

两个角色都传入 `global_step_79` 时，会自动解析到上述两个角色专属的 HF 目录。

输出：

- 报告：`reports/eval/<group>/<yymmdd>-<hhmm>-<strategy-slug>.report.md`
- 评估结果目录：`log/eval_res/<group>/<yymmdd>-<hhmm>-<strategy-slug>/`
- 评估结果文件：
  - `traces.jsonl`
  - `metrics.jsonl`
  - `summary.json`
  - `run_config.json`
- 运行时日志：
  - `runtime_logs/retriever_<port>.log`
  - `runtime_logs/retrieval_proxy.log`
  - `runtime_logs/agent_vllm_<port>.log`
  - `runtime_logs/reranker_vllm_<port>.log`

`KEEP_TRACE=partial` 会保存每个 query 的 prompt、sub-query、重排后的 top-5 chunks、最终答案和 ground-truth answers。

`KEEP_TRACE=full` 会额外保存每次 tool call 的检索 top-50 chunks。

## 训练/评估 Prompt 一致性

评估脚本必须保持与训练相同的 prompt 语义。对于 tool call 之后的 agent turn，这一点尤其重要，因为模型在训练时是在通过 Qwen chat-template 的 tool message 读取检索段落之后，才生成最终的 `<answer>...</answer>`。

当前 vLLM 评估器通过以下规则保持与训练路径一致：

- 初始 agent prompt 从数据集的 `prompt` 字段加载，并使用 `tokenizer.apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` 渲染。
- 第一次 agent 输出会作为模型生成的 assistant 文本追加进去。
- 检索输出使用训练 formatter 格式化，而不是使用单独的 eval-only formatter：
  - `verl.tools.utils.search.format_tool_response(...)` 用于 agent 可见的 top-5 passages。
  - `verl.tools.utils.search.format_tool_response_with_docid_map(...)` 用于 reranker prompts。
- 检索结果会作为 chat message 插入，其中 `role="tool"`，`content=<tool_response>...</tool_response>`，并由训练工具格式化。
- Tool message 使用 `tokenizer.apply_chat_template([tool_message], add_generation_prompt=True, tokenize=True)` 进行 tokenization。
- 将 tool-message tokens 追加到已有 prompt 时，评估器会像训练流程一样移除 tokenizer 生成的前置 system/chat prefix，然后保留新的 tool/user block 以及下一个 assistant generation prefix。

最终 answer-stage prompt 必须具有以下结构：

```text
assistant
<reason>...</reason>
<tool_call>
{
  "name": "search",
  "arguments": {
    "query": "..."
  }
}
</tool_call>
user
<tool_response>
[1] "Title"
passage text
[2] "Title"
passage text
...
</tool_response>
assistant
```

保留 Qwen special tokens 时，同一个边界会显示为：

```text
</tool_call><|im_start|>user
<tool_response>
...
</tool_response><|im_end|>
<|im_start|>assistant
```

不要手动把 `"\n<tool_response>...\n"` 这样的普通字符串拼接进 prompt。应使用 `role="tool"` message 和 tokenizer chat template。手动字符串拼接可能会悄悄移除 user/tool 边界，造成训练/评估 prompt 漂移。

当前评估器还支持用于一致性检查的 LLM IO tracing：

```bash
LLM_IO_JSONL=/path/to/eval_llm_io.jsonl \
COSEARCH_LLM_IO_MAX_RECORDS=20 \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

JSONL 记录包含 `role`、`assistant_turn`、解码后的 `prompt_text`、解码后的 `output_text`、token counts 和 sampling parameters。进行训练/评估调试时，请检查 `assistant_turn=2` 的 agent 记录；这是插入 tool response 之后的最终答案生成 prompt。

来自 Qwen3-4B 一致性检查的已知对比结果：

- 训练 rollout 日志使用 `skip_special_tokens=True` 解码 special tokens，因此 tool 边界可能显示为 `</tool_call>user\n<tool_response>...`。
- Eval LLM IO 在解码后的 prompt 中保留 Qwen special tokens，因此同一个边界会显示为 `</tool_call><|im_start|>user\n<tool_response>...`。
- 这两种视图是等价的 chat-template 渲染。关键不变量是：tool output 是一个 tool/user message，后面跟随新的 assistant generation prefix，而不是追加在上一个 assistant turn 内部的原始文本。
