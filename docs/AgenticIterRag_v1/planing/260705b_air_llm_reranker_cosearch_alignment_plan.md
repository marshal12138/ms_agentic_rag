# AIR LLM Reranker 与 CoSearch Reranker 行为对齐改造计划

## 1. 背景

当前 AIR LLM reranker 训练链路里，reranker prompt 使用的是 `AIR_RERANK_FULL50_PROMPT_WITH_INITIAL_QUERY`。

它的行为是：

- 输入 retriever 召回的 top50 文档。
- 要求 reranker 输出完整 50 个文档的排序。
- parser 校验 50 个 index 是否完整、去重、合法。
- agent observation 最后只取 reranker 排序后的 top5。

这个设计有一个很现实的问题：模型为了输出完整 50 个 index，需要生成很长的 `<rerank>` 内容，再加上 `<reason>`，很容易把 `max_response_length=1024` 打满。当前真实训练里已经看到过这种现象：response length clip ratio 长期接近 1，format reward 经常拿不到有效信号，训练效率非常差。

CoSearch 中的 reranker 行为不是这样。CoSearch reranker 的核心行为是：

- 输入 N 篇候选文档，实际场景里通常是 top50。
- 输出 EXACTLY M 篇最相关文档，默认 M=5。
- 输出格式同样是 `<reason>...</reason>` 加 `<rerank>...</rerank>`。
- `<rerank>` 里只需要给出 top5 index，不需要完整 50 排序。

所以这次改造的核心目标就是：AIR LLM reranker 的训练行为和 CoSearch reranker 完全对齐。不要再训练 full50 全排序，而是训练 top50 输入、top5 输出。

## 2. 总体目标

这次改造完成后，AIR LLM reranker 的默认训练行为应该是：

- 输入：原始 question、中间 query、retriever 召回的 top50 文档。
- 输出：`<reason>...</reason>` 和 `<rerank>...</rerank>`。
- `<rerank>` 内容：只输出 5 个不同的 index。
- index 范围：仍然是 `[1, 50]`，因为候选文档还是 50 篇。
- parser 校验：只校验 exactly 5 个 index，而不是 50 个。
- stage1 reward：只看 reranker 输出格式和长度。
- stage2 reward：把 reranker 输出的 top5 文档作为新的 tool observation，接 frozen agent 继续 rollout。
- 默认仍然关闭 stage2，先把 stage1 format training 跑通、跑快、跑稳定。

说得更口语化一点：之前是让 reranker 把 50 篇文章全排一遍，但 agent 实际只看前 5 篇，这个任务太重、太慢、也没必要。现在改成跟 CoSearch 一样，直接让 reranker 从 50 篇里挑出最该给 agent 看的 5 篇。

## 3. 非目标

这次不做下面这些事情：

- 不保留 full50 训练行为的兼容开关。
- 不做 full50 parser 的向后兼容。
- 不继续支持 `prompt_template_version=air_rerank_tags_v1_full50` 作为训练默认路径。
- 不改数据生产 stage 的 agent rollout 逻辑。
- 不改 `generate_traces`、`build_reranker_dataset` 这些数据生产相关 stage 的核心语义。
- 不在 stage2 默认启用 agentic rollout reward。
- 不把 CoSearch 项目代码作为运行时依赖直接 import 到 AIR 训练链路。

这点要特别明确：这版是覆盖旧 LLM reranker 训练流程，不是给旧方案加一个兼容分支。旧 full50 方案本身已经证明训练效率不合适，所以不应该继续扩大复杂度。

## 4. 对齐后的行为定义

### 4.1 输入保持 top50

retriever 仍然召回 top50 文档，branch dataset 里仍然保存 50 篇候选文档。

这部分不变，因为 top50 是 reranker 可选择的候选池。如果只输入 top5，那 reranker 没有发挥空间；如果输入太多，prompt 又会过长。所以当前继续使用 top50。

配置上继续保留：

```yaml
branch_dataset:
  # retriever 给 reranker 的候选文档数量。这里保持 50，表示 reranker 从 50 篇里挑 top5。
  candidate_top_n: 50
```

### 4.2 输出改成 top5

reranker 不再输出 50 个 index，而是输出 exactly 5 个 index。

目标格式示例：

```text
<reason>
The query asks about ...
I select passages that directly mention ...
</reason>
<rerank>
[8] > [3] > [1] > [17] > [5]
</rerank>
```

这里有几个要求：

- 必须有 `<reason>`。
- 必须有 `</reason>`。
- 必须有 `<rerank>`。
- 必须有 `</rerank>`。
- `<rerank>` 里必须正好 5 个 index。
- 5 个 index 必须不同。
- 每个 index 必须在 `[1, 50]` 范围内。
- 不接受 `<think>` 作为思考内容标签。

配置上明确：

```yaml
branch_dataset:
  # reranker 最终要输出的文档数量，也是 agent 当前 observation 实际能看到的文档数量。
  visible_top_m: 5
```

### 4.3 Prompt 文本与 CoSearch 完全一致

AIR 里原本已有一个 topM prompt：`AIR_RERANK_PROMPT_WITH_INITIAL_QUERY`。它和 CoSearch 的 prompt 基本同源，但是当前检查发现不是完全 byte-level 一致，有一些标点差异，例如：

- `5-8` 和 `5–8` 的差异。
- ASCII apostrophe 和 curly apostrophe 的差异。

这次要做到行为完全对齐，所以 AIR 的 topM prompt 文本需要改成和 CoSearch 的 `RERANK_PROMPT_WITH_INITIAL_QUERY` 完全一致。

实现上不建议 AIR 运行时直接 import CoSearch 的 prompt。原因很简单：AIR 应该有自己的训练模块边界，不能让训练入口依赖 CoSearch 的内部包路径。更合理的做法是：

- 在 AIR 自己的 `llm_reranker/format.py` 中保留 prompt 常量。
- 把 prompt 内容改成和 CoSearch 当前 prompt 完全一致。
- 增加测试，比较 AIR prompt 和 CoSearch prompt 的文本是否一致。

这样既能行为对齐，又不会把两个工程运行时绑死。

### 4.4 文档格式与 CoSearch 对齐

CoSearch 的文档渲染逻辑大致是：

```text
[1] Title: xxx
contents...

[2] Title: yyy
contents...
```

字段读取优先级是：

- `contents`
- `text`
- `passage`
- 空字符串

超过最大长度时，会截断并追加 `...`。

AIR 当前 formatter 和 CoSearch 还有细节差异，所以这次也要对齐：

- 字段优先级改成 `contents -> text -> passage`。
- 截断时追加 `...`。
- 有 title 时使用 `[i] Title: {title}\n{contents}`。
- 没有 title 时使用 `[i] {contents}`。
- 文档编号仍然从 1 开始。

这部分会直接影响 prompt token 分布和模型看到的输入结构，所以必须一起改，而不是只改 prompt 文本。

## 5. 配置改造计划

### 5.1 主配置文件

核心配置仍然挂在：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/AgenticIterRag/config/main_run/agentic_iter_rag_main.yaml
```

reranker 训练专用配置仍然是：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/AgenticIterRag/config/reranker_training/llm_reranker_grpo_branch.yaml
```

`llm_reranker_grpo_branch.yaml` 是 LLM reranker 训练的核心配置文件，但它必须挂到 AIR 主 pipeline 配置里，因为 reranker training 现在是 AIR 全流程 pipeline 的一个 stage，不是游离在外面的单独任务。

### 5.2 branch dataset 配置

需要把 prompt 版本从 full50 改成 topM 对齐版本。

建议配置：

```yaml
branch_dataset:
  # retriever 召回给 reranker 的候选文档数量。这里保持 50，表示 reranker 从 50 篇里选择 top5。
  candidate_top_n: 50

  # reranker 需要输出的文档数量，也是后续 agent observation 实际使用的文档数量。
  visible_top_m: 5

  # prompt 模板版本。这里明确使用与 CoSearch reranker 对齐的 topM prompt，不再使用 full50 全排序 prompt。
  prompt_template_version: cosearch_rerank_topm_v1
```

### 5.3 reward 配置

stage1 使用 `reranker_format_reward`。

stage1 的 parser 参数应该从配置里明确表达：

```yaml
reranker_training:
  stage1_format:
    reward:
      # stage1 只训练格式，不接 agentic rollout。
      name: reranker_format_reward

      # reranker 必须输出 exactly 5 个 index。
      expected_count: 5

      # index 的最大合法范围来自候选文档数量，也就是 [1, 50]。
      max_index: 50

      # 格式正确且输出长度不超过 512 时给满分。
      max_full_score_response_length: 512

      # 格式正确但输出长度超过 512 时给半分，避免模型靠长篇啰嗦 reason 拿满分。
      long_response_score: 0.5

      # 格式正确且长度合规时给满分。
      valid_score: 1.0

      # 格式错误时给负分。格式错代表动作不可执行，所以不能给 0。
      invalid_score: -0.5
```

stage2 使用 `agentic_rag_rollout_reward`，但默认关闭。

```yaml
reranker_training:
  stage2_agentic:
    # stage2 暂时默认关闭。等 stage1 训练稳定后，再打开 agentic rollout reward。
    enabled: false

    reward:
      # stage2 的大 reward：先解析 reranker top5，再接 frozen agent 继续 rollout，最后用 answer reward 打分。
      name: agentic_rag_rollout_reward

      # reranker 输出数量仍然是 5。
      expected_count: 5

      # index 合法范围仍然来自 50 篇候选文档。
      max_index: 50

      # 子策略默认复用 answer_reward，即直接用新 answer 和 gold answer 算分。
      answer_score_strategy: answer_reward
```

### 5.4 中文注释要求

所有新增和修改的 YAML 配置项都必须补充中文注释。

注释风格参考现有 AIR 配置文件，不要写成只有字段名没有解释的注释。每个关键字段至少说明三件事：

- 这个字段控制什么。
- 默认值为什么这么设。
- 改它会影响什么。

尤其下面这些字段必须有中文注释：

- `candidate_top_n`
- `visible_top_m`
- `prompt_template_version`
- `expected_count`
- `max_index`
- `max_full_score_response_length`
- `valid_score`
- `long_response_score`
- `invalid_score`
- `stage1_format.enabled`
- `stage2_agentic.enabled`

## 6. 代码改造计划

### 6.1 Prompt 和 formatter

目标文件：

```text
AgenticIterRag/agentic_iter_rag/llm_reranker/format.py
```

要做的事情：

- 把 AIR topM prompt 文本改成与 CoSearch `RERANK_PROMPT_WITH_INITIAL_QUERY` 完全一致。
- 停止在 branch dataset 中使用 `render_air_rerank_full50_prompt(...)`。
- 默认使用 `render_air_rerank_tags_prompt(...)`。
- 调整 `format_air_passages(...)`，让文档渲染格式和 CoSearch 一致。
- 保留 `<reason>` 和 `<rerank>`，不引入 `<think>`。

代码实现时必须补充中文注释，尤其要解释：

- 为什么输入仍然是 top50。
- 为什么输出只要求 top5。
- 为什么不再使用 full50 全排序。
- 为什么 formatter 要和 CoSearch 保持一致。

### 6.2 Branch dataset builder

目标文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/branch_dataset.py
```

要做的事情：

- import 从 `render_air_rerank_full50_prompt` 改成 topM prompt renderer。
- 构造 prompt 时传入 `top_m=visible_top_m`。
- JSONL schema 中保留 50 篇 candidate docs。
- `extra_info` 中明确记录：
  - `candidate_top_n=50`
  - `visible_top_m=5`
  - `prompt_template_version=cosearch_rerank_topm_v1`
  - `candidate_index_to_doc_id`
  - `selected_point_policy`
  - `branch_step_index`
- README 或 manifest 里不再写 full50，而是写 top50 input / top5 output。

代码实现时必须补充中文注释，尤其要解释：

- branch sample 的 prompt 为什么只要求输出 top5。
- `candidate_index_to_doc_id` 为什么仍然要覆盖 50 篇。
- `visible_top_m` 和 agent observation top5 的关系。

### 6.3 Parser

目标文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/parser.py
```

当前 parser 的核心假设是 `expected_n=50`。这次要改成：

```python
parse_rerank_response(
    text: str,
    expected_count: int = 5,
    max_index: int = 50,
)
```

parser 要校验：

- `<reason>` 标签存在。
- `</reason>` 标签存在。
- `<rerank>` 标签存在。
- `</rerank>` 标签存在。
- `<reason>` 在 `<rerank>` 前面。
- reason 内容非空。
- rerank 内容非空。
- rerank 中 exactly 5 个 index。
- index 不重复。
- index 范围在 `[1, 50]`。
- 不允许出现非法字符。

格式错误要返回明确 error code，例如：

- `missing_reason_tag`
- `missing_rerank_tag`
- `empty_reason`
- `empty_rerank`
- `wrong_index_count`
- `duplicate_index`
- `index_out_of_range`
- `invalid_rerank_text`

代码实现时必须补充中文注释，尤其要解释：

- `expected_count` 和 `max_index` 为什么拆开。
- 为什么输出 5 个 index 但 index 最大值是 50。
- 为什么 `<think>` 不被接受。

### 6.4 Stage1 format reward

目标文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/rewards/reranker_format_reward.py
```

stage1 reward 规则改成：

- 格式错误：`-0.5`
- 格式正确且 response length <= 512：`1.0`
- 格式正确但 response length > 512：`0.5`

这里的 response length 应优先使用 token length。如果当前 reward 函数只能拿到文本长度，需要在实现里明确字段含义，并尽量接入 tokenizer 或 VERL 传入的 response length。

reward extra info 中建议写入：

- `format_valid`
- `format_error_code`
- `expected_count`
- `max_index`
- `ranked_indices`
- `response_length`
- `max_full_score_response_length`
- `length_penalty_applied`
- `reward_value`

代码实现时必须补充中文注释，尤其要解释：

- 为什么格式错给负分，而不是给 0。
- 为什么长度超过 512 只给 0.5。
- 为什么 stage1 不调用 continuation rollout。

### 6.5 Stage2 agentic rollout reward

目标文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/rewards/agentic_rag_rollout_reward.py
AgenticIterRag/agentic_iter_rag/reranker_training/continuation_reward.py
```

stage2 默认关闭，但代码结构要按 top5 行为改好。

stage2 的完整链路是：

```text
reranker 输出 top5 index
-> parser 校验 expected_count=5, max_index=50
-> index 映射到 doc_id/doc 内容
-> 这 5 篇 doc 渲染成新的 tool observation
-> messages_before_tool_response + new_tool_message
-> frozen agent 继续 rollout
-> 后续 search 仍然 retriever-only
-> agent 输出 final answer
-> answer_reward 或 delta_answer_reward 子策略打分
```

注意这里不再是：

```text
reranker 输出 50
-> 取前 5
```

而是直接：

```text
reranker 输出 5
-> 这 5 个就是 observation
```

代码实现时必须补充中文注释，尤其要解释：

- stage2 为什么默认关闭。
- reranker 输出 top5 后为什么可以直接作为 observation。
- continuation rollout 中为什么后续 search 仍然只用 retriever。
- 为什么不在后续 search 中继续调用 reranker。

### 6.6 Trainer entry

目标文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/trainer_entry.py
```

要做的事情：

- VERL command 中不再传 `expected_n=50`。
- stage1 传：
  - `expected_count=visible_top_m`
  - `max_index=candidate_top_n`
- stage2 传：
  - `expected_count=visible_top_m`
  - `max_index=candidate_top_n`
- dry-run 输出里要能清楚看到 top50 input / top5 output。
- `trainer.rollout_data_dir` 和 `trainer.validation_data_dir` 的动态构造保持不变。
- stage1/stage2 日志目录继续隔离。

代码实现时必须补充中文注释，尤其要解释：

- 为什么 reward parser 参数从一个 `expected_n` 拆成两个字段。
- 为什么 stage1 和 stage2 使用同一套 parser 参数。

### 6.7 Service bundle

目标文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/service_bundle.py
```

服务 bundle 需要表达新的运行时行为：

- retriever 召回 top50。
- LLM reranker 从 top50 中输出 top5。
- search tool 给 agent 的 observation 是 reranker top5。
- prompt version 是 `cosearch_rerank_topm_v1`。
- parser expected count 是 5。
- parser max index 是 50。

生成的 YAML 模板必须补充中文注释，说明每个字段来自哪里：

- 训练产物。
- 运行环境。
- 部署侧覆盖。

## 7. 日志和报告改造

训练日志结构当前已经支持 stage1/stage2 分目录保存：

```text
runtime_logs/train_llm_reranker/stage1_format/rollout_data
runtime_logs/train_llm_reranker/stage1_format/validation_data
runtime_logs/train_llm_reranker/stage2_agentic/rollout_data
runtime_logs/train_llm_reranker/stage2_agentic/validation_data
```

这次改造后，需要在 rollout JSONL 中能直接看出：

- prompt 里是 `Rank EXACTLY 5 passages`。
- prompt 里是 `Passages (50 total)`。
- output 里 `<rerank>` 只有 5 个 index。
- score 的 extra info 里 `expected_count=5`、`max_index=50`。

如果 stage1 仍然出现大量 response 打满 1024，要优先看 rollout output 内容，判断是不是模型在 `<reason>` 里啰嗦、没学会收敛，还是 prompt/template/chat thinking 没关干净。

## 8. 测试计划

### 8.1 单元测试

parser 测试：

- 合法输出 5 个 index，返回 valid。
- 输出 4 个 index，报 `wrong_index_count`。
- 输出 6 个 index，报 `wrong_index_count`。
- 输出重复 index，报 `duplicate_index`。
- 输出 `[51]`，报 `index_out_of_range`。
- 只有 `<think>` 没有 `<reason>`，报 `missing_reason_tag`。
- `<rerank>` 中包含非 index 文本，报 `invalid_rerank_text`。

formatter 测试：

- 有 title 时格式和 CoSearch 一致。
- 无 title 时格式和 CoSearch 一致。
- 字段读取优先级是 `contents -> text -> passage`。
- 超长截断时追加 `...`。

prompt 对齐测试：

- AIR prompt 常量和 CoSearch prompt 常量文本完全一致。
- branch dataset 生成的 prompt 包含 `Rank EXACTLY 5 passages`。
- branch dataset 生成的 prompt 包含 `Passages (50 total)`。
- branch dataset 生成的 prompt 不包含 full50 全排序要求。

reward 测试：

- 格式正确且长度 <=512，reward 为 `1.0`。
- 格式正确但长度 >512，reward 为 `0.5`。
- 格式错误，reward 为 `-0.5`。
- extra info 中包含 `expected_count=5` 和 `max_index=50`。

### 8.2 Dry-run 测试

运行：

```bash
bash tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh --dry-run
```

检查：

- selected stages 不变。
- stage1 reward 是 `reranker_format_reward`。
- stage1 reward 参数是 `expected_count=5`、`max_index=50`。
- 不再出现 `expected_n=50`。
- branch dataset prompt version 是 `cosearch_rerank_topm_v1`。
- stage2 默认关闭。
- `trainer.rollout_data_dir` 仍然动态落到当前 run 的 runtime_logs 下。

### 8.3 小样本真实训练测试

先跑小样本：

```bash
AIR_RERANKER_SMOKE=1 bash tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh
```

或者使用等价的小样本覆盖参数。

检查：

- stage1 能真实在 NPU 上跑通。
- rollout output 中 `<rerank>` 开始出现 5 个 index。
- response length clip ratio 明显低于 full50 版本。
- reward 不再长期卡在 `-0.5`。
- training report 和 latest 曲线能正常刷新。
- checkpoint 仍然按配置保存。

### 8.4 回归测试

必须确认这次改造不影响数据生产相关 stage：

- `generate_traces` 不应该感知 LLM reranker prompt 改造。
- `build_reranker_dataset` 不应该被 parser 参数变化破坏。
- dataproduce 脚本不应该因为 reranker training 配置变化而失败。
- `run_260702a_AIR_v1_dataproduce.sh --dry-run` 应该仍能通过。

重点边界：

- 只能改 LLM reranker training stage 和它直接依赖的 branch dataset / reward / service bundle。
- 不要改 search agent 的 prompt。
- 不要改 trace generation 的上下文格式。
- 不要改 retriever-only search tool 的默认行为。

## 9. 验收标准

这次改造可以认为完成，需要满足下面这些条件：

- AIR branch dataset 生成的是 CoSearch 对齐 topM prompt。
- prompt 文本与 CoSearch reranker prompt 完全一致。
- 文档格式与 CoSearch reranker 输入格式一致。
- reranker 输出要求从 full50 改成 exactly top5。
- parser 使用 `expected_count=5` 和 `max_index=50`。
- stage1 reward 使用新的三档规则：`1.0 / 0.5 / -0.5`。
- stage2 reward 代码路径按 top5 输出设计好，但默认关闭。
- dry-run 不再出现 `expected_n=50`。
- stage1 真实 NPU 训练能跑通。
- rollout 日志能看到真实 input/output/score。
- 数据生产顶层 stage 不受影响。

## 10. 执行顺序

建议按这个顺序实施：

1. 改 prompt 和 formatter，让 AIR 文本与 CoSearch 对齐。
2. 改 branch dataset，让训练样本从 full50 输出切到 top5 输出。
3. 改 parser，把 `expected_n` 拆成 `expected_count` 和 `max_index`。
4. 改 stage1 format reward，接入 top5 parser 和三档长度规则。
5. 改 stage2 agentic rollout reward，让它消费 top5 输出。
6. 改 trainer entry，把 VERL reward 参数传成 `expected_count=5`、`max_index=50`。
7. 改配置文件，所有新增和修改字段补中文注释。
8. 改 service bundle，让部署配置也表达 top50 input / top5 output。
9. 跑 parser、formatter、reward 单测。
10. 跑 pipeline dry-run。
11. 跑小样本真实 NPU 训练。
12. 如果小样本训练速度和 reward 表现正常，再启动正式训练。

## 11. 风险点

### 11.1 Prompt 完全一致不等于训练完全一致

Prompt 和 formatter 对齐后，AIR 训练仍然可能和 CoSearch 有一些环境差异，比如：

- base model 不同。
- chat template 不同。
- sampling 参数不同。
- reward 结构不同。
- AIR 是 GRPO 训练，CoSearch 可能是运行时 rerank 或其他训练方式。

所以这里说的“完全对齐”，指的是 reranker 的输入输出协议、prompt 文本、文档格式、parser 规则对齐，不代表训练框架完全一样。

### 11.2 top5 输出会降低生成长度，但不保证马上学会格式

从 full50 改成 top5，模型输出长度压力会明显下降。但如果 base model 对 `<reason>/<rerank>` 仍然不熟，stage1 仍然可能需要一定训练步数才能稳定格式。

所以 rollout_data_dir 必须打开，不能再只看 aggregate metrics。否则看不到模型到底在输出什么。

### 11.3 不要把 CoSearch 包变成 AIR 运行时依赖

测试里可以比较 AIR prompt 和 CoSearch prompt，但训练链路不要 runtime import CoSearch。否则后面部署 service bundle 时会增加不必要的路径依赖。

## 12. 结论

这次改造的核心不是小修 prompt，而是把 AIR LLM reranker 的训练任务从“50 篇全排序”改成“50 篇里选 top5”，并且把这个行为和 CoSearch reranker 对齐。

改完后，训练目标会更短、更直接、更贴近真实 agent observation，也更容易让 `max_response_length=1024` 不再被无意义地打满。

后续如果要做更复杂的 reranker，比如输出 top10、动态 M、或者多步 rollout 每步都 rerank，应该在这个 topM 对齐版本稳定之后再扩展。
