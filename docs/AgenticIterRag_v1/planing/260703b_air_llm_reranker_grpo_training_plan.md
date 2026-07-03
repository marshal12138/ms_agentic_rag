# AIR LLM Reranker GRPO 训练实现计划

更新日期：2026-07-03

## 1. 这件事到底要做什么

这篇文档讨论 AIR v1 里 LLM reranker 的训练实现方案。

先用比较直白的话说清楚目标：

我们不是要训练一个“看 query 和 doc，然后静态输出排序标签”的 reranker。我们要训练的是一个真正放在 agentic search 链路里的 reranker。

它的训练信号来自后续 agent 的最终答案。

训练时，大概是这样：

1. 已经有一个训练好的 search agent。
2. 已经用这个 search agent 对训练集 query 产出过 rollout 轨迹。
3. 每条轨迹里有若干次 search，每次 search 有 agent 生成的 `sub_query` 和 retriever 返回的 top50 docs。
4. 我们从一条轨迹里选一个 search step。
5. 让 LLM reranker 对这一步的 50 篇 doc 重新排序。
6. 只取 reranker 排序后的 top5，作为新的 tool observation。
7. 把这个 observation 插回原 agent 历史上下文。
8. 冻结 search agent，让它从这里继续 rollout 到最终答案。
9. 用原 search agent 的 reward function 给这个新答案打分。
10. 这个分数就是 reranker 这次排序的 reward。
11. 用 GRPO 训练 reranker。

所以，这里的 reranker reward 不是独立的排序指标，而是“这个排序对 agent 最终答题有没有帮助”。

这点很重要。它决定了我们不能只看 doc relevance，也不能只看 Hit@5。最终要看 agent 拿到这些 doc 后有没有答得更好。

## 2. 前置条件

第一版正式训练前，必须先满足一个条件：

必须有增强轨迹。

增强轨迹需求见：

```text
docs/AgenticIterRag_v1/planing/260703a_air_enhanced_trajectory_backfill_requirements.md
```

原因是当前 AIR canonical trajectory 不足以严格继续 rollout。它记录了 search step 的结果，但没有记录某个 search step 发生时，agent 当时的完整 messages 状态。

如果没有 `messages_before_tool_response`，训练时就只能重构上下文。重构上下文会带来两个问题：

- 可能把轨迹 A 的 reranker 结果塞给轨迹 B。
- 可能 continuation 的上下文格式和 data produce 时不一致。

这两个问题都会污染 reward。

所以第一版训练入口要做强校验：如果 dataset manifest 指向的不是增强轨迹构造出来的 branch dataset，就直接失败。

## 3. 第一版实现范围

第一版做这些事情：

- 训练 LLM reranker。
- search agent 冻结。
- reranker 每次只改变一条完整轨迹中的一个 search step。
- reranker 输出完整 50 个候选的排序。
- agent 只看排序后的 top5。
- continuation 过程里的后续 search tool 仍然只有 retriever。
- reward 默认复用原 search agent reward。
- 支持可选 baseline delta reward。
- 训练完成后输出 service bundle，供后续 agentic RAG 服务启动脚本读取。

第一版不做这些事情：

- 不重新训练 search agent。
- 不实现 agent 和 reranker 交替训练。
- 不实现每一步 search 都接入 reranker 的训练模式。
- 不引入 LLM-as-judge reward。
- 不把原 no-ranker trace 顺序当成强监督标签。
- 不在 reranker 服务失败时静默回退 retriever top5。

这些后续可以做，但第一版先把单步 counterfactual rollout 训练闭环做稳。

## 4. 两个新子模块

需要新增两个大的子模块。

第一个是训练模块：

```text
llm_reranker_training
```

它负责：

- 从增强轨迹构造 branch dataset。
- 启动 reranker GRPO 训练。
- 在训练中调用 frozen agent 做 continuation rollout。
- 计算 reranker reward。
- 保存 reranker checkpoint 和训练 manifest。

第二个是服务组装模块：

```text
agentic_rag_with_llm_reranker
```

它负责：

- 训练完成后生成带 LLM reranker 的 search tool 配置。
- 生成 service bundle。
- 让外部服务启动脚本可以读取该 bundle，把 agent、retriever、reranker 组合成完整 agentic RAG 服务。

这两个模块逻辑上分开。训练模块关心“怎么学”，服务组装模块关心“怎么用”。

## 5. 训练入口

预设训练入口是：

```text
tasks/train_tasks/agenticIterRag/run_260703a_AIR_v1_llm_reranker_training.sh
```

这个 shell 入口保持 AIR 当前风格：

- 只选择配置组。
- 只指定 overlay。
- 不在 shell 里写业务参数。
- 允许少量 CLI dotlist 覆盖，用于 dry-run、smoke、断点恢复。

推荐入口结构：

```bash
bash scripts/agenticIterRag_v1/01_pipeline_launcher.sh \
  --main-run-config agentic_iter_rag_main \
  --DATA_CONFIG=co_search_ablation \
  --PIPELINE_CONFIG=offline_two_stage \
  --RESOURCE_CONFIG=local_8gpu_0_7 \
  --INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm \
  --INFER_BUDGET_CONFIG=air_aligned_budget \
  --RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=air_async_qwen3_4b \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/llm_reranker_training_overlay.yaml \
  "$@"
```

注意，这里 `RERANKER_TRAINING_CONFIG` 建议新增一个专用配置，而不是继续使用当前比较基础的 `llm_reranker_base.yaml`。

建议新增：

```text
AgenticIterRag/config/reranker_training/llm_reranker_grpo_branch.yaml
```

## 6. Pipeline Stage 调整

当前 AIR pipeline 已有：

```text
train_agent
generate_traces
build_reranker_dataset
train_llm_reranker
infer_matrix
```

对这次任务来说，我们不需要从头训练 agent，也不需要重新 infer matrix。训练入口应该实际执行：

```text
build_reranker_branch_dataset
train_llm_reranker
build_llm_reranker_service_bundle
```

有两种落地方式：

第一种是把 `build_reranker_branch_dataset` 做成 `build_reranker_dataset` 的新子阶段。

第二种是把它做成独立 pipeline stage。

推荐第二种。

原因是 branch dataset 已经不是普通 reranker prompt dataset。它绑定了 continuation context、baseline reward、step policy 和 GRPO reward 逻辑。它的语义更接近训练前准备数据，而不是原来的静态 `input_dataset -> train_dataset`。

推荐新增 pipeline stage：

```text
build_reranker_branch_dataset
train_llm_reranker
build_service_bundle
```

在完整 offline two-stage 里，未来可以插到：

```text
train_agent
generate_traces
build_reranker_dataset
build_reranker_branch_dataset
train_llm_reranker
build_service_bundle
infer_matrix
```

但这次 `run_260703a` 默认从 `build_reranker_branch_dataset` 开始。

## 7. Branch Dataset 设计

branch dataset 是 reranker GRPO 真正消费的数据。

它从增强轨迹构造。每条样本对应：

```text
某条 trajectory 的某一个 search step
```

推荐目录：

```text
data/AgenticIterRag/
  llm_reranker_branch_train_set/
    <branch_dataset_version>/
      dataset.jsonl
      dataset.parquet
      example.json
      manifest.json
      source_enhanced_trajectory.manifest.json
      final_config.yaml
```

每条样本推荐格式：

```json
{
  "sample_id": "traj-123:step:0",
  "data_source": "agentic_iter_rag.llm_reranker.branch_grpo",
  "ability": "llm_reranker",
  "prompt": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "reward_model": {
    "style": "rule",
    "ground_truth": {
      "target": ["gold answer"]
    }
  },
  "prompt_template_version": "air_rerank_tags_v1_full50",
  "formatter": "verl_chat",
  "target_text": null,
  "extra_info": {
    "trajectory_id": "traj-123",
    "sample_id": "source-sample-id",
    "step_index": 0,
    "step_policy": "type0",
    "question": "original question",
    "sub_query": "agent generated sub query",
    "candidate_doc_ids": ["doc1", "doc2"],
    "candidate_index_to_doc_id": {
      "1": "doc1",
      "2": "doc2"
    },
    "candidate_docs": [],
    "messages_before_tool_response": [],
    "original_visible_doc_ids": ["doc1", "doc2"],
    "baseline_final_answer": "old answer",
    "baseline_reward": 0.5,
    "baseline_metrics": {},
    "context_format_version": "air_agent_messages_v1",
    "tool_response_format_version": "air_search_tool_response_v1",
    "reward_strategy": "answer_reward",
    "source_enhanced_trajectory_ref": {
      "path": ".../enhanced_trajectory.jsonl",
      "line_index": 123
    }
  }
}
```

重点字段：

- `sample_id`：必须包含 trajectory 和 step，方便排查。
- `prompt`：给 reranker 的输入。
- `reward_model.ground_truth.target`：gold answers。
- `extra_info.messages_before_tool_response`：continuation 的起点。
- `extra_info.candidate_docs`：50 篇候选文档。
- `extra_info.candidate_index_to_doc_id`：reranker 输出 `[i]` 后映射真实 doc_id。
- `extra_info.baseline_reward`：delta reward 会用。
- `extra_info.step_policy`：记录该样本是 type1/type-1/type0 哪种策略选出来的。

## 8. Reranker Prompt 和输出格式

第一版 prompt 复用 AIR 当前 reranker prompt 风格，但要调整输出约束。

之前 prompt 里说：

```text
Rank EXACTLY M passages
```

现在训练需要完整 50 排序，所以要改成：

```text
Rank ALL 50 passages
```

输出格式仍然是：

```text
<reason> ... </reason>
<rerank> ... </rerank>
```

`<rerank>` 里必须包含 50 个 distinct index：

```text
<rerank>[27] > [3] > [1] > ... > [44]</rerank>
```

为什么要求 50 个，而不是只输出 top5？

原因有两个：

1. 训练时只有 top5 会给 agent，但完整排序更容易稳定约束模型行为。
2. 后续如果要做 ranking metric、topK ablation 或服务端调试，完整排序更有用。

不过 agent observation 永远只取前 5。

格式错误直接给 `-0.5`，不继续 rollout。

格式错误包括：

- 缺 `<reason>`。
- 缺 `<rerank>`。
- tag 顺序不对。
- `<rerank>` 里不是候选 index。
- index 不在 `[1, 50]`。
- index 重复。
- index 数量不是 50。

## 9. Step Policy

第一版支持三种单步替换策略。

### type1

只替换第一步 search。

适合看 reranker 对开局证据选择的影响。很多任务第一步搜错，后面就会一路偏。

### type-1

只替换最后一步 search。

适合看 reranker 对最终补证据的影响。这个策略通常 continuation 更短，训练成本更低。

### type0

默认策略。

每条轨迹用固定 seed 随机选择一个 search step。

要求：

- 同一个 `seed + trajectory_id` 必须选出同一个 step。
- 换 seed 可以改变分布。
- 不能跨 trajectory 选 step。
- 如果某条 trajectory 没有 search step，直接跳过。

配置建议：

```yaml
reranker_training:
  branch_dataset:
    step_policy: type0
    random_seed: 20260703
    allow_no_search: false
```

### multi-step mode

完整轨迹的每一步 search 都接入 reranker，这个模式先不实现。

原因是 reward credit assignment 会复杂很多：

- 一条最终 answer reward 要分给多个 reranker action。
- 多个 reranker 输出之间可能互相影响。
- GRPO grouping 也要重新设计。

第一版只在文档和配置里保留：

```yaml
reranker_training:
  branch_dataset:
    step_policy: all_steps
    supported: false
```

如果用户配置 `all_steps`，训练入口直接报错，提示该模式留待后续实现。

## 10. Continuation Rollout

训练时 search agent 是 frozen 的。

对于一个有效 reranker 输出，continuation 逻辑是：

1. 解析 reranker 输出，得到完整 doc 排序。
2. 取前 5 个 doc。
3. 用 AIR search tool 当前 formatter 渲染成 tool message。
4. 拼接：

```text
messages_before_tool_response + [new_tool_message]
```

5. 用 frozen agent 从这个 messages 继续生成。
6. 如果 agent 后续继续 search，search tool 仍然只有 retriever。
7. 直到 agent 输出 `<answer>` 或达到最大 turn/budget。

注意：后续 search 不允许再调用训练中的 reranker。

这点要写死。因为第一版训练的是“只改变一步 search result”的 counterfactual。如果后续 search 也接入 reranker，那 reward 就不再只归因于当前 reranker action。

## 11. Reward 设计

第一版至少支持两组 reward 策略。

### answer_reward

默认策略。

公式：

```text
reranker_reward = reward_fn(new_final_answer, gold_answers)
```

这里 `reward_fn` 复用 search agent 原来的 reward function。比如当前已有的 F1 + format penalty 系列。

如果 agent continuation 格式错误，则由原 reward function 自己处理格式惩罚。

### delta_answer_reward

可选策略。

公式：

```text
reranker_reward = reward_fn(new_final_answer, gold_answers) - baseline_reward
```

其中 `baseline_reward` 来自增强轨迹顶层。

这个策略的含义是：reranker 不是因为题目简单就拿高分，而是要比原 no-ranker 轨迹更好才拿正收益。

但默认不启用。原因是 delta reward 方差可能更大，早期训练更容易不稳定。

### format penalty

reranker 自己的输出格式错误时：

```text
reranker_reward = -0.5
```

并且不触发 agent continuation。

这个规则要放在最前面。

也就是说：

```text
if reranker_format_invalid:
    return -0.5
else:
    continue_agent_rollout_and_score()
```

配置建议：

```yaml
reranker_training:
  reward:
    strategy: answer_reward
    format_penalty: -0.5
    answer_reward_function:
      path: AgenticIterRag/rewards/search_qa_f1_with_format_penalty.py
      name: search_qa_f1_penalty_compute_score
    require_baseline_reward_for_delta: true
```

## 12. GRPO 分组

GRPO 的关键是同一个 prompt 下采多个 reranker 输出，然后比较 reward。

推荐 UID：

```text
trajectory_id:step_index
```

如果后续要加难度分桶，可以扩展为：

```text
trajectory_id:step_index:baseline_bucket
```

第一版先简单一点：

- 同一个 trajectory 的同一个 step 是一个 GRPO group。
- reranker 对同一个 prompt 采样多个排序。
- 每个排序分别触发 continuation rollout。
- 得到多个 reward。
- 用这些 reward 做组内相对优势。

需要注意：

- 如果某个 group 全部 reward 一样，可以按现有 VERL 逻辑过滤或保留 fallback。
- 如果有效样本数量不足，要在 metrics 里清楚记录。

## 13. 训练后端

采用 AIR 专用 GRPO 路线。

也就是说，复用 VERL 的底层 actor/ref/rollout/optimizer 能力，但不要直接照搬旧 `search_r1_reranker_reward_agent_loop_worker` 的数据假设。

原因：

- 旧 worker 更像在线完整 rollout 中采 reranker 输出。
- 这次需求明确是基于已产出的轨迹，替换某个历史 search step。
- AIR 需要自己的数据契约、manifest、stage、service bundle。

可以参考旧代码里的几个点：

- reranker 输出处理。
- UID grouping。
- group filtering。
- reward extra info 写法。
- batch 对齐到 GPU 数。

但训练主语义要写在 AIR 自己模块里。

建议新增代码区域：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/
  branch_dataset.py
  continuation_rollout.py
  reward.py
  parser.py
  trainer_entry.py
  service_bundle.py
```

职责：

- `branch_dataset.py`：从增强轨迹构造 branch dataset。
- `continuation_rollout.py`：拼新 observation 并调用 frozen agent 继续 rollout。
- `reward.py`：格式 reward、answer reward、delta reward。
- `parser.py`：解析 `<reason>` 和 `<rerank>`。
- `trainer_entry.py`：把 AIR config 翻译成 VERL GRPO 训练任务。
- `service_bundle.py`：训练结束后生成服务 bundle。

## 14. 配置设计

新增或扩展 `reranker_training` 配置：

```yaml
reranker_training:
  name: llm_reranker_grpo_branch
  base_model: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B

  input:
    enhanced_trajectory_manifest: null
    branch_dataset_manifest: null

  branch_dataset:
    enabled: true
    version: null
    overwrite: false
    step_policy: type0
    random_seed: 20260703
    candidate_top_n: 50
    visible_top_m: 5
    prompt_template_version: air_rerank_tags_v1_full50
    formatter: verl_chat
    max_doc_chars: 2000

  continuation:
    agent_model: null
    use_frozen_agent: true
    search_tool_mode: retriever_only
    max_assistant_turns: 6
    max_user_turns: 6
    max_prompt_length: 11264
    max_response_length: 1024
    max_tool_response_length: 4096
    temperature: 0.0
    top_p: 1.0

  reward:
    strategy: answer_reward
    format_penalty: -0.5
    answer_reward_function:
      path: AgenticIterRag/rewards/search_qa_f1_with_format_penalty.py
      name: search_qa_f1_penalty_compute_score

  trainer:
    method: grpo
    total_epochs: 1
    train_batch_size: 16
    micro_batch_size_per_gpu: 1
    n_samples_per_prompt: 4
    learning_rate: 2.0e-5
    max_prompt_length: 12000
    max_response_length: 2048
    save_freq: 100
```

资源配置要补充：

```yaml
resource:
  stage_resources:
    build_reranker_branch_dataset:
      local_cpu:
        workers: 1

    train_llm_reranker:
      services:
        reranker_actor:
          gpu_ids: [0, 1, 2, 3]
          tensor_parallel_size: 4
        frozen_agent_vllm:
          gpu_ids: [4, 5, 6, 7]
          tensor_parallel_size: 4
        recall:
          gpu_ids: [7]
          port: 8130
          retrieval_service_url: http://127.0.0.1:8130/retrieve
```

实际 GPU 怎么分，可以后续按机器调整，但配置语义要是 stage-level placement。

## 15. Service Bundle

训练完成后输出：

```text
outputs/agenticIterRag/<group>/<run>/service_bundle/
  service_config.yaml
  tool_config.yaml
  manifest.json
```

`service_config.yaml` 表达服务级组合：

```yaml
service_type: agentic_rag_with_llm_reranker
schema_version: air_service_bundle_v1

agent:
  model_path: /path/to/trained/search_agent
  tokenizer_path: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B

retriever:
  endpoint: http://127.0.0.1:8130/retrieve
  top_n: 50

llm_reranker:
  model_path: /path/to/trained/llm_reranker
  base_model: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B
  prompt_template_version: air_rerank_tags_v1_full50
  output_parser: air_rerank_tags_full50
  required: true

observation:
  visible_top_m: 5
  tool_response_format_version: air_search_tool_response_v1
```

`tool_config.yaml` 表达 search tool 如何接入 reranker：

```yaml
tools:
  - class_name: verl.tools.agentic_iter_rag_retriever_tool.AgenticIterRagRetrieverTool
    config:
      type: native
      retrieval_service_url: http://127.0.0.1:8130/retrieve
      recall_final_top_n: 50
      searchTool_final_top_m: 5
      ranker_enabled: true
      ranker:
        backend: llm_reranker_service
        required: true
        model_path: /path/to/trained/llm_reranker
        prompt_template_version: air_rerank_tags_v1_full50
        output_parser: air_rerank_tags_full50
```

第一版建议 fail-fast：

- reranker 服务不可用，search tool 报错。
- reranker 输出格式错，search tool 报错。
- 不静默回退 retriever top5。

原因是训练和评估时必须知道 reranker 真的生效了。静默回退会让指标解释变得很混乱。

## 16. Test Plan

### 16.1 增强轨迹前置测试

目标：确认训练数据真的能支持严格 continuation。

测试点：

- `enhanced_trajectory.jsonl` 存在。
- 每条有 search 的 trajectory 都有 `steps[]`。
- 每个 step 都有 `messages_before_tool_response`。
- 每个 step 都有 `recall_topn_docs`，且默认 50 篇。
- `sub_query == tool_call.arguments.query`。
- `messages_before_tool_response[-1]` 是 assistant tool_call。
- `messages_after_original_tool_response` 等于 `messages_before_tool_response + original_tool_message`。

负向测试：

- 删掉 `messages_before_tool_response`，branch dataset 构造失败。
- 改掉 `sub_query`，branch dataset 构造失败。
- 改乱 top50 doc_id 顺序，branch dataset 构造失败。

### 16.2 要点1：Agent observation 是 top5

要测两条链路。

no-ranker 链路：

- retriever 返回 top50。
- tool response 只包含 retriever top5。

with-reranker 链路：

- reranker 输出完整 50 排序。
- tool response 只包含 reranker top5。

断言：

- continuation prompt 中不出现第 6 到第 50 篇 doc 的文本。
- `num_agent_visible_docs == 5`。
- `visible_doc_ids == reranker_ranked_doc_ids[:5]`。

### 16.3 要点2：Query 轨迹对齐

构造 branch dataset 时强校验：

- `trajectory_id` 存在。
- `step_index` 存在。
- `sub_query` 和增强轨迹 step 完全一致。
- `candidate_doc_ids` 和增强轨迹 step 的 `doc_id_order` 完全一致。
- `messages_before_tool_response` 来自同一个 trajectory 的同一个 step。

负向测试：

- 把样本 A 的 `messages_before_tool_response` 换成样本 B 的，必须失败。
- 把 step0 的 top50 docs 换成 step1 的，必须失败。
- 把 type0 随机选出的 step 改成不存在的 step，必须失败。

### 16.4 要点3：上下文格式一致性

测试方法：

1. 从增强轨迹取一个 step。
2. 用 `messages_before_tool_response + original_tool_message` 重建上下文。
3. 用同一个 tokenizer/chat template 渲染。
4. 和 data produce 当时的上下文格式做结构对比。

断言：

- role 顺序一致。
- assistant tool_call 内容未被改写。
- tool message formatter 版本一致。
- chat template source 写入 manifest。

这个测试的目的不是要求 token 完全相同到每个空格，而是确保 role 结构、tool response 插入位置和 formatter 一致。

### 16.5 要点4：Reranker 生效点位

type1：

- 多步轨迹只选 `step_index=0`。

type-1：

- 多步轨迹只选最后一个 step。

type0：

- 同一个 seed 下选择结果稳定。
- 换 seed 后允许选择变化。
- 不允许跨 trajectory 选 step。

all_steps：

- 第一版配置后训练入口直接失败。
- 错误信息说明该模式留待后续实现。

### 16.6 要点5：Continuation 搜索工具

测试目标：只在被替换的那一步使用 reranker，后续 search 仍然只有 retriever。

用 mock 服务做：

- mock reranker 记录 call count。
- mock retriever 记录 call count。
- continuation agent 后续再发 search。

断言：

- 初始替换 step 使用 reranker 输出。
- continuation 后续 search 不调用 reranker。
- reranker call count 等于训练样本的 reranker generation 次数。
- 后续 search tool mode 是 `retriever_only`。

### 16.7 要点6：Reward function 配置

格式错误：

- reranker 输出缺 `<rerank>`。
- reward 直接是 `-0.5`。
- frozen agent continuation 没有被调用。

answer_reward：

- mock agent 返回固定 answer。
- reward 等于原 answer reward function 输出。

delta_answer_reward：

- mock baseline_reward = 0.4。
- mock new reward = 0.7。
- reranker reward = 0.3。

baseline 缺失：

- `delta_answer_reward` 直接失败。
- `answer_reward` 不受影响。

### 16.8 GRPO batch 测试

同一个 prompt 采样多个 reranker 输出。

断言：

- 同一个 `trajectory_id:step_index` 进入同一个 UID group。
- 每个输出都有自己的 continuation reward。
- reward 写到 reranker response 的最后一个有效 token。
- group 内 reward 全相同的样本按配置过滤或 fallback。

### 16.9 Service Bundle 测试

训练结束后检查：

- `service_bundle/service_config.yaml` 存在。
- `service_bundle/tool_config.yaml` 存在。
- `service_bundle/manifest.json` 存在。
- `service_config.yaml` 能解析。
- `tool_config.yaml` 能被 tool registry 初始化。
- reranker model path 指向训练产物。
- topN=50，topM=5。
- reranker `required=true`。

负向测试：

- 删除 reranker model path，bundle 校验失败。
- 把 topM 改成大于 topN，bundle 校验失败。
- parser 配置缺失，bundle 校验失败。

## 17. 里程碑

### Milestone 1：增强轨迹补产

完成：

- AIR infer backend 写出增强轨迹。
- manifest 和 summary 增强。
- 小样本补产通过。

验收：

- 10 条样本 smoke。
- 增强轨迹结构校验通过。

### Milestone 2：Branch Dataset

完成：

- 从增强轨迹构造 branch dataset。
- 支持 type1/type-1/type0。
- prompt 要求完整 50 排序。

验收：

- branch dataset example 可人工阅读。
- 负向对齐测试通过。

### Milestone 3：Reward Runner

完成：

- reranker 输出 parser。
- 格式 penalty。
- continuation rollout。
- answer reward 和 delta reward。

验收：

- mock 单测覆盖格式错、answer reward、delta reward。

### Milestone 4：GRPO 训练入口

完成：

- `run_260703a_AIR_v1_llm_reranker_training.sh`。
- `main_train_llm_reranker.py` 接入真实训练。
- stage manifest 写清楚输入、输出、checkpoint。

验收：

- dry-run 成功。
- 小样本 smoke 能跑完整 batch。

### Milestone 5：Service Bundle

完成：

- 训练产物转服务配置。
- 写入 artifact `service_bundle`。

验收：

- bundle 配置可解析。
- tool config 可初始化。

## 18. 默认决策

第一版默认：

- 使用 Qwen3-4B 作为 reranker base model。
- search agent 冻结。
- step policy 使用 `type0`。
- reranker 输出完整 50 排序。
- agent observation 只取 top5。
- continuation 后续 search 只用 retriever。
- reward 使用 `answer_reward`。
- reranker 格式错误 reward 是 `-0.5`。
- service bundle 只写 artifact 目录。
- reranker 服务失败 fail-fast。

这些默认值不是随便选的。它们的目标是先把训练闭环做干净，保证 reward 能归因到“这一条 reranker 排序”上。
