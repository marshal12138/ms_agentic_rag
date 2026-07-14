# Search-R1 结构化多 Gold 奖励代码实现报告

> **已废止（2026-07-11）**：本文围绕旧 Search-R1 与所谓“结构化 Search-R1”的数据/reward
> 差异展开，不符合后续明确的实验口径，不得作为当前代码验收或模型比较依据。当前实现范围与
> 验收以 `260711-09a_新数据Search-R1与SPAD代码实现报告.md` 为准。

生成时间：2026-07-11 08:10 CST

## 1. 完成结论

已按 `260710-21a_Search-R1多Gold答案语义与EM奖励审查报告.md` 的实施顺序完成代码和数据改造：

1. 对旧 Search-R1 实际训练 512 个 prompt 中的 29 个多 Gold 样本完成全量分类。
2. 对冻结 350 评估集中的 150 个多 Gold 样本完成全量分类。
3. 新增 AND-of-OR 结构化答案契约、结构化 EM、Answer Group F1/Recall。
4. 保留 Legacy EM/F1，历史指标仍可复现。
5. 新增独立 `search_r1_structured` reward，不改变旧 reward 名称和默认行为。
6. 训练和评估均使用独立数据、配置、任务名和 checkpoint。
7. 增加别名、集合、复合槽位、污染、歧义排除和顺序不变性测试。

## 2. 数据冻结与分类

### 2.1 数据版本处理

原审查报告基于 2026-07-10 的 5100/350 数据，但默认数据路径随后被另一项清洗任务覆盖。为避免把不同
数据版本混入本实验，本实现没有覆盖当前默认数据，而是从替换备份恢复原审查数据，并结合旧训练
rollout 中实际出现的 512 个问题生成隔离实验集。

源文件：

- Train：`data/global_train_eval_data/replaced_backups/20260710-231218/data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet`
- Eval：`data/global_train_eval_data/replaced_backups/20260710-231218/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- 旧 rollout：`log/agenticIterRag/260710-113003-543853-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data`

生成结果：

| 文件 | 行数 | SHA256 |
| --- | ---: | --- |
| `search_r1_structured.train.parquet` | 512 | `631b6023750bc3d8333bb4deda92ea7eccbe61307ea890ed1366a2e5e6c8237d` |
| `search_r1_structured.eval.parquet` | 350 | `ce01777aabcbfae4e48343b09fc76bb6f043f500177a8f51df039beea47453db` |
| `multi_gold_classification.jsonl` | 179 | `1662a48949a10df6882c367bc255bdcedb3ca7e21631185e0e622bcdaaffcb39` |
| `multi_gold_classification.tsv` | 179 | `b5687317fc4f7921cabea11bd31b5d3fbe7a175844047b7084b5f18f24c9f63f` |

统一目录：`data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/`。

### 2.2 全量分类结果

训练 29 条和评估 150 条多 Gold 样本共 179 条，分类如下：

| 类型 | 数量 | 处理 |
| --- | ---: | --- |
| `alias_or` | 108 | 一个 OR group，任一别名完整匹配 |
| `required_set` | 12 | 多个必答 group，全部覆盖才是结构化 EM=1 |
| `multi_slot` | 4 | 每个必答槽位为独立 group |
| `contaminated` | 44 | 删除中间跳或错误候选，只保留确认后的最终答案 group |
| `ambiguous` | 11 | 无法可靠消歧时标为结构化 reward 不可用 |

共 166 条可用于结构化 reward，13 条不可用。单答案样本保持单 group，不改变 Legacy 指标。

分类和覆盖的主要依据为：PopQA `obj_id` 实体别名、MuSiQue decomposition 的最终跳答案，以及对 NQ、
TriviaQA、2Wiki 多答案样本的逐条语义审查。所有人工结果均写入 JSONL/TSV，可独立复核。

## 3. 新答案契约

`reward_model.ground_truth` 在原 `target` 外新增：

```json
{
  "target": ["legacy answer 1", "legacy answer 2"],
  "required_answer_groups": [
    ["alias A1", "alias A2"],
    ["required answer B"]
  ],
  "answer_semantics": "required_set",
  "structured_reward_eligible": true
}
```

外层 group 是 AND，内层 alias 是 OR。规则为：

- 单 group：Structured EM 与旧 OR-reference EM 完全一致。
- 多 group：预测中必须包含每个 group 的至少一个连续归一化 alias，group 顺序无关。
- Structured EM 是严格 0/1 奖励。
- Answer Group Recall 衡量必答 group 覆盖率。
- Answer Group F1 同时约束 group 覆盖和预测 token precision，完整且简短的答案优于单成员或冗长答案。
- `structured_reward_eligible=false` 的样本不进入结构化指标分母，也不产生结构化正奖励。

归一化继续沿用 Search-R1/AIR 历史规则：小写、删除 ASCII 标点和英文冠词、合并空白。

## 4. 代码实现

### 4.1 共享指标模块

新增：

- `AgenticIterRag/agentic_iter_rag/metrics/__init__.py`
- `AgenticIterRag/agentic_iter_rag/metrics/answer_metrics.py`

该模块统一提供 Legacy EM/F1、结构化字段解析、AND-of-OR EM、Group F1/Recall，训练和评估调用同一份
实现，避免 reward 与报表口径漂移。

### 4.2 训练 reward

修改：

`AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py`

新增 reward 类型 `search_r1_structured`。返回值包含：

- `score` / `search_r1_answer_em`：结构化严格 EM。
- `legacy_em`、`legacy_f1`：兼容诊断。
- `structured_em`、`answer_group_f1`、`answer_group_recall`。
- `matched_group_count`、`required_group_count`、`answer_semantics`、`structured_reward_eligible`。

同时修复了 ground truth 字段为 NumPy array 时用布尔 `or` 选字段可能触发歧义异常的问题。旧
`search_r1_original` 类型仍存在，行为不变。

### 4.3 端到端评估

修改：

`scripts/agenticIterRag_v1/assets/infer_backend/infer_air_vllm.py`

每个样本同时写 Legacy 和 Structured 指标；汇总表增加 Structured N、Structured EM、Group F1、
Group Recall。结构化分母只包含 eligible 样本，trace 保存答案语义和 required groups。

### 4.4 数据与复评工具

新增：

- `scripts/cosearch_local/build_structured_answer_data.py`
- `scripts/cosearch_local/replay_structured_search_r1_reward.py`
- `scripts/cosearch_local/rescore_structured_eval.py`

`rescore_structured_eval.py` 会校验每轮 350 个 index 完整唯一、数据源一致、问题文本与新 parquet
逐条一致，然后才允许对不同模型输出做同一新数据口径的离线复评。

### 4.5 独立训练入口

新增：

- `tasks/train_tasks/agenticIterRag/configs/search_r1_structured_qwen3_1_7b_512_overlay.yaml`
- `tasks/train_tasks/agenticIterRag/run_260711a_AIR_search_r1_structured_qwen3_1_7b_512.sh`

配置显式指定隔离数据和 `search_r1_structured`，不会覆盖或续训旧 Search-R1 checkpoint。

## 5. 旧 rollout 奖励重放

对旧模型 512 prompt x 8 = 4096 条冻结 rollout 重放新 reward：

| 指标 | 数量 |
| --- | ---: |
| 旧 reward 正例 | 639 |
| 结构化 reward 正例 | 640 |
| 旧正例但结构化为 0 | 5 |
| group 内 reward 非常量的问题 | 137 / 512 |
| group 全 0 的问题 | 354 / 512 |
| group 全 1 的问题 | 21 / 512 |

5 个旧假阳性包括只回答 `Ghana` 的集合题、只回答单个演员的多成员题；同时 6 个包含完整 21 人
cast 的回答从旧 EM=0 修正为结构化 EM=1。正例总量几乎不变，说明改造主要纠正 reward 语义，
并未人为增加奖励密度。

重放汇总：`data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/legacy_rollout_replay/summary.json`。

## 6. 测试与验证

新增 `AgenticIterRag/tests/test_answer_metrics.py`，包含 8 个定向用例：

- alias OR 等价性。
- required set 单项不再得满分。
- required set 完整答案和顺序变化。
- multi-slot 全槽位覆盖。
- contaminated 中间答案不再得分。
- ambiguous 样本排除。
- reward 端到端集成。
- Legacy 兼容行为。

最终使用仓库的 `unittest discover` 运行 45 个测试，结果 45/45 通过；Python 语法编译和
`git diff --check` 也通过。数据构建、4096 rollout 重放、跨模型轨迹复评以及新增三轮结构化模型
复评均成功完成。

## 7. 已知边界

1. `required_answer_groups` 仍是规则匹配，不是语义模型判断；未列出的同义表达可能漏判。
2. 多 group 通过连续 token span 匹配，适合当前短答案任务，但不替代生成式 judge。
3. 13 条无法可靠消歧的样本被排除，而非强行赋予不可信答案。
4. 训练完成后修正了一个仅影响 `structured_reward_eligible` 日志诊断字段的条件；该条件不参与
   实际训练 `score` 计算，因此 checkpoint 数值奖励不受影响，当前代码已修正诊断含义。
