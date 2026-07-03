# AIR Reranker Branch Dataset 详细设计

更新日期：2026-07-03

## 1. 目标

branch dataset 是 LLM reranker GRPO 训练真正吃的数据。

它和现有的 `llm_reranker_train_set` 不一样。现有数据更像静态 prompt 数据：给 reranker 一个 query 和 top50 docs，让它输出排序。

branch dataset 多了一层关键能力：它要告诉训练器，如果 reranker 改了这一步排序，应该从原 agent 轨迹的哪个上下文继续 rollout。

所以每条 branch sample 本质上是：

```text
某条增强轨迹中的某一个 search step
```

它必须同时包含：

- reranker prompt。
- 当前 step 的 top50 docs。
- 当前 step 对应的 `messages_before_tool_response`。
- 原轨迹 baseline reward。
- gold answer。
- step 选择策略。

## 2. 非目标

第一版 branch dataset 不做这些事：

- 不直接训练模型。
- 不调用 frozen agent。
- 不计算 continuation reward。
- 不做 all-steps reranker 训练。
- 不从普通 canonical trajectory 近似重构上下文。

如果没有增强轨迹，branch dataset 构造必须失败。

## 3. 输入

输入是增强轨迹 manifest：

```text
data/AgenticIterRag/trajectory/<trajectory_version>/manifest.json
```

manifest 里必须包含：

```json
{
  "enhanced_trajectory_jsonl": ".../enhanced_trajectory.jsonl",
  "enhanced_schema_version": "air_enhanced_trajectory_v1",
  "context_format_version": "air_agent_messages_v1",
  "tool_response_format_version": "air_search_tool_response_v1"
}
```

如果 `enhanced_trajectory_jsonl` 不存在，直接报错。

如果 schema 版本不是 `air_enhanced_trajectory_v1`，直接报错。

## 4. 输出目录

建议输出到长期数据目录：

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

文件说明：

- `dataset.jsonl`：branch dataset 主文件。
- `dataset.parquet`：可选镜像，依赖可用时写入。
- `example.json`：样例记录。
- `manifest.json`：数据版本、来源、策略、样本数、字段版本。
- `source_enhanced_trajectory.manifest.json`：来源增强轨迹 manifest 快照。
- `final_config.yaml`：构造本数据时使用的最终配置。

版本名建议：

```text
<trajectory_version>__branch_<step_policy>_top50_top5_<prompt_template_version>
```

例如：

```text
260703f_AIR_v1_traj_xxx__branch_type0_top50_top5_air_rerank_tags_v1_full50
```

## 5. 样本 Schema

每条样本是 VERL 可消费的 GRPO 数据，同时包含 AIR continuation 需要的 `extra_info`。

推荐格式：

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
    "source_index": 123,
    "step_index": 0,
    "turn_index": 0,
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

字段说明：

- `sample_id`：必须稳定，推荐 `trajectory_id:step:<step_index>`。
- `data_source`：固定为 `agentic_iter_rag.llm_reranker.branch_grpo`。
- `ability`：固定为 `llm_reranker`。
- `prompt`：reranker prompt，formatter 为 VERL chat message。
- `reward_model.ground_truth.target`：gold answer，continuation reward 会用。
- `target_text`：第一版为空，因为没有静态监督答案。
- `extra_info`：AIR 训练闭环需要的全部上下文。

## 6. `extra_info` 关键字段

### 6.1 对齐字段

这些字段用于防止错配：

- `trajectory_id`
- `step_index`
- `sub_query`
- `candidate_doc_ids`
- `source_enhanced_trajectory_ref`

构造器必须保证：

```text
extra_info.sub_query == enhanced.steps[step_index].sub_query
extra_info.candidate_doc_ids == enhanced.steps[step_index].doc_id_order
```

如果不一致，直接失败。

### 6.2 Continuation 字段

这些字段用于后续 frozen agent 继续 rollout：

- `messages_before_tool_response`
- `context_format_version`
- `tool_response_format_version`
- `candidate_docs`
- `candidate_index_to_doc_id`

训练时会解析 reranker 输出的 index，然后用 `candidate_index_to_doc_id` 找回真实 doc，再从 `candidate_docs` 里取 top5 拼 observation。

### 6.3 Reward 字段

这些字段用于 reward：

- `reward_model.ground_truth.target`
- `baseline_final_answer`
- `baseline_reward`
- `baseline_metrics`
- `reward_strategy`

默认策略 `answer_reward` 不需要 baseline。`delta_answer_reward` 必须要求 `baseline_reward` 存在。

## 7. Step Policy

### type1

只选第一步 search。

实现：

```text
selected_step = steps[0]
```

如果 `steps` 为空，按 `allow_no_search` 决定跳过或报错。

### type-1

只选最后一步 search。

实现：

```text
selected_step = steps[-1]
```

### type0

固定 seed 随机选一步。

实现要求：

```text
selected_index = stable_hash(random_seed, trajectory_id) % len(steps)
```

不要用 Python 进程全局 random 状态直接 `random.choice`，否则并发和数据分片时不稳定。

### all_steps

第一版不支持。

如果配置为 `all_steps`，构造器直接失败，提示：

```text
all_steps is reserved for future multi-step reranker training and is not supported in AIR v1
```

## 8. Builder 模块设计

建议新增：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/branch_dataset.py
```

核心函数：

```text
load_enhanced_trajectory_manifest(path) -> dict
iter_enhanced_trajectories(manifest) -> Iterator[dict]
select_step(trajectory, policy, seed, allow_no_search) -> dict | None
validate_selected_step(trajectory, step, config) -> None
render_branch_prompt(trajectory, step, config) -> list[dict]
build_branch_sample(trajectory, step, config, source_ref) -> dict
build_branch_dataset(config_path, output_manifest_path) -> dict
```

代码实现时要补中文注释：

- `select_step` 要解释 type0 为什么用稳定 hash。
- `validate_selected_step` 要解释为什么 sub_query 和 doc_id 顺序必须强校验。
- `build_branch_sample` 要解释 `extra_info` 哪些字段给 continuation 用，哪些字段给 reward 用。
- 写 manifest 的地方要说明该数据集和增强轨迹的追溯关系。

## 9. Manifest 设计

`manifest.json` 推荐格式：

```json
{
  "dataset_type": "llm_reranker_branch_train_set",
  "schema_version": "air_reranker_branch_dataset_v1",
  "version": "<branch_dataset_version>",
  "version_dir": "...",
  "created_at": "2026-07-03T00:00:00Z",
  "source_enhanced_trajectory_manifest": "...",
  "source_enhanced_trajectory_jsonl": "...",
  "step_policy": "type0",
  "random_seed": 20260703,
  "candidate_top_n": 50,
  "visible_top_m": 5,
  "prompt_template_version": "air_rerank_tags_v1_full50",
  "formatter": "verl_chat",
  "dataset_jsonl": ".../dataset.jsonl",
  "dataset_parquet": ".../dataset.parquet",
  "example_json": ".../example.json",
  "sample_count": 1000,
  "skipped_no_search_count": 0,
  "final_config_yaml": ".../final_config.yaml",
  "config_hash": "..."
}
```

## 10. 错误处理

构造器默认 fail-fast。

必须报错的情况：

- 增强轨迹 manifest 缺失。
- `enhanced_trajectory_jsonl` 不存在。
- `steps` 为空且 `allow_no_search=false`。
- `messages_before_tool_response` 缺失。
- `recall_topn_docs` 少于 `candidate_top_n`。
- doc_id 缺失或重复。
- `sub_query` 和 tool call 不一致。
- `visible_top_m > candidate_top_n`。
- `step_policy=all_steps`。

允许跳过的情况：

- `steps` 为空且 `allow_no_search=true`。
- 某条轨迹原始状态是非正常完成，但配置显式允许跳过异常样本。

跳过样本必须写入 summary，不允许静默吞掉。

## 11. 测试计划

### 11.1 正向测试

用 2 条增强轨迹构造 branch dataset：

- 一条单步 search。
- 一条多步 search。

分别测试 type1、type-1、type0。

期望：

- `dataset.jsonl` 非空。
- `example.json` 存在。
- `manifest.json` 字段完整。
- `extra_info.messages_before_tool_response` 存在。

### 11.2 对齐负向测试

人为修改：

- `sub_query`
- `candidate_doc_ids`
- `step_index`
- `messages_before_tool_response`

期望构造器失败，错误信息包含具体字段路径。

### 11.3 type0 稳定性测试

同一份数据，同一个 seed，运行两次。

期望：

- 选中的 `trajectory_id:step_index` 完全一致。

换 seed 后：

- 允许有样本选择不同 step。

### 11.4 dry-run 测试

pipeline dry-run 时不写长期数据集，只写 stage manifest。

manifest 中应包含：

- 预期输出目录。
- step policy。
- candidate_top_n。
- visible_top_m。

### 11.5 注释验收

人工检查新增 builder 代码：

- 复杂字段转换有中文注释。
- 强校验逻辑有中文注释。
- manifest 写入逻辑有中文注释。
