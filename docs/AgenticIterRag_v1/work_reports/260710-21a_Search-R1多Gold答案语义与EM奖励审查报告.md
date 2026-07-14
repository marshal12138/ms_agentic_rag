# Search-R1 多 Gold 答案语义与 EM 奖励审查报告

生成时间：2026-07-10 21:15 CST

## 1. 结论

当前数据中的 `gold_answers: list[str]` 同时承载了至少五种不同语义：

1. 同一答案的别名、缩写或格式变体。
2. 多个都必须回答的集合成员。
3. 一个复合问题中不同必答槽位的答案。
4. 问题缺少时间、版本或对象限定时的多个可能答案。
5. 最终答案与中间跳答案混入同一列表的数据污染。

Search-R1 上游及本地实现均无条件把该列表解释为 **OR reference 集合**：模型答案经归一化后，只要
与任意一个 gold 完全相同，EM reward 就是 1。当前端到端评估的 EM 也采用 `any`，F1 则是与每个
gold 分别计算后取 `max`。

该语义对别名列表是正确的，但对集合型、复合槽位型和污染数据不正确。它可能奖励只回答一个成员的
不完整答案，也可能拒绝包含全部正确成员的完整答案。因此，当前实现没有偏离 Search-R1 官方代码；
真正的问题是 **扁平 `list[str]` 数据契约与部分 QA 的答案结构不兼容**。

## 2. 审查范围与统计

本报告检查了以下数据和实现：

- 完整训练集：`data/AgenticIterRag/source/co_search_ablation.train.parquet`。
- 当前评测集：`data/AgenticIterRag/source/co_search_ablation.infer.parquet`。
- 最新 Search-R1 训练 8 个 step 的实际 rollout。
- AIR 端到端评估脚本。
- 本地 Search-R1 original reward 适配。
- 仓库内 vendored Search-R1 源码。
- GitHub 官方 `PeterGriffinJin/Search-R1` 当前 `main`，检查 commit
  `598e61bd1d36895726d28a8d06b3a15bed19f5d3`。

总体统计：

| 数据范围 | 总问题数 | 多 gold 问题数 | 占比 |
| --- | ---: | ---: | ---: |
| 完整训练集 | 5100 | 399 | 7.8% |
| 最新 Search-R1 实际训练 prompt | 512 | 29 | 5.7% |
| 当前评测集 | 350 | 150 | 42.9% |

最新训练每个 prompt 采样 8 条 rollout，因此 29 个多-gold prompt 对应 232 个 rollout 位置。它们均
使用当前 OR 型 EM reward。

350 条评测集按数据源统计：

| 数据集 | 总数 | 多 gold | 占比 |
| --- | ---: | ---: | ---: |
| 2wikimultihopqa | 50 | 34 | 68% |
| bamboogle | 50 | 0 | 0% |
| hotpotqa | 50 | 0 | 0% |
| musique | 50 | 15 | 30% |
| nq | 50 | 21 | 42% |
| popqa | 50 | 35 | 70% |
| triviaqa | 50 | 45 | 90% |

PopQA、TriviaQA 和 2Wiki 中有大量合理的实体别名，因此多-gold 比例高不等同于数据错误。风险来自
同一字段无法区分别名 OR、集合 AND、复合槽位和中间答案。

## 3. 多 Gold 的实际类型

### 3.1 合理的别名 OR

以下列表中的元素语义等价，模型输出任意一种即可：

```text
Question: who is recognized as the founder of islam?
Gold: ["the Islamic prophet Muhammad", "Muhammad"]

Question: the measured amount of alcohol in a drink is called?
Gold: ["alcohol by volume", "ABV"]

Question: In what city was Jerrold Katz born?
Gold: ["Washington, D.C.", "Washington DC", "D.C.", ...]
```

对这类数据使用 `any(normalized_prediction == normalized_gold)` 是合理的。

### 3.2 必须完整回答的集合 AND

```text
Question: what are three branches of government in the united states?
Gold: ["legislative", "executive", "judicial"]
```

该问题明确要求三个成员。当前 reward 却把三个字符串当成三个可替代答案，产生：

```text
prediction = "legislative"                       -> Search-R1 EM = 1
prediction = "legislative, executive, judicial" -> Search-R1 EM = 0
```

当前评估 F1 也逐个对单项 gold 计算再取最大值，因此完整答案的 F1 为 `0.5`，只输出
`legislative` 的 F1 反而为 `1.0`。已有历史 trajectory 正好记录了完整三项答案的 `EM=0, F1=0.5`。

最新 Search-R1 训练中的同类样本还包括：

```text
Question: three countries in africa that the greenwich meridian passes through?
Gold: ["Ghana", "Algeria", "Mali"]

Question: where does the superior vena cava return blood from?
Gold: ["upper limbs", "eyes", "neck"]
```

### 3.3 复合问题的多个必答槽位

```text
Question: who is the father of genetics and what did he study?
Gold: ["the common edible pea", "Mendel", "pea plants",
       "variation in plants", "Gregor Mendel"]
```

该问题至少有两个槽位：人物和研究对象。当前只输出 `Mendel` 即可获得 EM=1，没有检查第二个槽位。

另一个评测样本：

```text
Question: where does the thames river begin and end?
Gold: ["Lighthouse Cove", "Near Tavistock"]
```

问题要求起点和终点，但当前输出任意一个就算完全正确。

### 3.4 多版本、时间变化或问题本身欠限定

```text
Question: when did the twenty one pilots hiatus start?
Gold: ["November 2016", "July 2017"]

Question: when did the yugioh card game come out?
Gold: ["in 1999 in Japan", "March 2002 in North America"]
```

这类数据可能是在容忍不同地区、事件定义或来源，也可能是原问题缺少限定。OR 型评估能够容忍多个
解释，但会掩盖问题本身不唯一，不能等同于高质量别名。

### 3.5 最终答案与中间跳答案污染

MuSiQue 中存在不能解释为别名的列表：

```text
Question: How many students attend where Andre Dreiding is employed?
Gold: ["nearly 25,000", "University of Zurich"]
```

`University of Zurich` 是“任职机构”这一中间跳实体，不是“多少学生”的最终答案。当前输出它仍会
获得 EM reward 1。

另一个样本：

```text
Question: What political party was the socialist candidate part of who ran for president in 1912?
Gold: ["Socialist Party of America", "Democrat", "Democratic Party"]
```

这些候选并不语义等价。扁平 OR 会给错误关系或中间候选提供假阳性奖励。

## 4. GitHub 官方 Search-R1 实现

官方数据处理脚本从 `RUC-NLPIR/FlashRAG_datasets` 读取数据，并直接执行：

```python
solution = {
    "target": example["golden_answers"],
}
```

它没有区分别名、集合成员、槽位或中间答案，也没有重组多答案结构：

- [官方训练数据处理脚本](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/scripts/data_process/qa_search_train_merge.py)
- [官方评测数据处理脚本](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/scripts/data_process/qa_search_test_merge.py)

官方 `em_check` 将 prediction 和每个 `golden_answer` 做相同归一化，只要某一项完全相等就返回 1：

- [官方 `qa_em.py`](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/verl/utils/reward_score/qa_em.py)

所以，本项目当前的“任一 gold 命中即得 1”继承了 Search-R1 原始语义，并非本地适配独自引入。

## 5. 本地数据处理与 Search-R1 Reward

本地数据准备函数位于：

`scripts/cosearch_local/prepare_cosearch_data.py`

`normalize_answers()` 读取 `golden_answers/answers/answer` 后统一转成 `list[str]`，随后直接写入：

```python
"reward_model": {
    "style": "rule",
    "ground_truth": {"target": answers},
}
```

这一步同样没有保留答案之间的逻辑关系。

当前 Search-R1 original reward 位于：

`AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py`

核心逻辑是：

```python
def _answer_em(prediction, gold_answers):
    normalized_prediction = normalize_answer(prediction)
    for gold in gold_answers:
        if normalized_prediction == normalize_answer(gold):
            return 1.0
    return 0.0
```

reward 还具有以下行为：

- 抽取最后一个完整闭合的 `<answer>...</answer>`。
- 任一 gold 精确匹配时 reward 为 1。
- 不匹配、没有答案或标签未闭合时 reward 为 0。
- 不要求输出所有 gold。
- 不判断 evidence 是否支持答案。
- 不给部分正确答案连续分数。

现有 Search-R1 reward 单元测试只覆盖单 gold `Paris`，没有多-gold alias、集合、复合槽位或污染案例。

## 6. 当前端到端评估如何计算

评估入口位于：

`scripts/agenticIterRag_v1/assets/infer_backend/infer_air_vllm.py`

EM：

```python
def exact_match(prediction, answers):
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(answer) for answer in answers))
```

Token F1：

```python
def token_f1(prediction, answers):
    return max(one_f1(answer) for answer in answers)
```

归一化包括：

- 转小写。
- 删除 ASCII 标点。
- 删除英文冠词 `a/an/the`。
- 合并连续空白。

它不会做语义同义判断、集合比较、槽位覆盖或时态消歧。额外解释文字也会导致严格 EM 失败。

当前 350 样本评估报告中的 EM/F1 全部采用该 OR-reference 口径。所有模型在同一数据和实现上比较，
所以相对比较仍有一定意义；但集合型、复合型和污染问题上的绝对分数不应解释为真实问答正确率。

SPAD 中使用的 token F1 也对 gold 列表取最大值：

`AgenticIterRag/agentic_iter_rag/agent_training/spad/reward.py`

Teacher PE 的 S/I/A 指标则是对人工 S/I/A 标签评分，不直接使用该 EM；只有 gold-aware prompt 会把
gold 放入输入。

## 7. 对训练与评估的影响

### 7.1 合理别名

OR 型 EM 能容忍简称、全称、大小写和标点变体。这是其主要合理用途。

### 7.2 集合型问题产生错误优化方向

模型输出一个成员即可得到满分，输出完整集合反而可能得 0。GRPO 会把不完整回答视为优轨迹，把
完整回答视为劣轨迹，直接形成错误 credit assignment。

### 7.3 复合问题只覆盖一个槽位也可得满分

人物、时间、地点、研究对象等多个必答槽位被压平后，reward 无法检查完整性。

### 7.4 中间答案污染导致假阳性

模型可能只完成第一跳或输出无关中间实体，却因为该实体被写入 `target` 而得到 reward 1。这会削弱
Search-R1 学习继续搜索和完成多跳链路的动力。

### 7.5 当前指标同时可能被抬高和压低

- 输出任意单项即可匹配，会抬高不完整答案的 EM。
- 输出完整集合无法匹配任何单项，会压低真正完整答案的 EM。
- F1 的 `max` 能给完整列表部分分数，但仍常常奖励单项高于完整集合。

因此不能简单判断现有总 EM 是净高估还是净低估；方向取决于样本中别名、集合和污染数据的比例。

## 8. 建议的数据契约

不能把所有多-gold 列表改成“必须全部命中”，因为大量列表确实只是别名。建议显式表示
**AND of OR groups**：

```json
{
  "required_answer_groups": [
    ["legislative"],
    ["executive"],
    ["judicial"]
  ]
}
```

外层 group 必须全部满足；同一 group 内的字符串是可替代别名。普通别名问题表示为：

```json
{
  "required_answer_groups": [
    ["Muhammad", "the Islamic prophet Muhammad"]
  ]
}
```

复合槽位可进一步保留槽位名：

```json
{
  "answer_slots": {
    "person": ["Gregor Mendel", "Mendel"],
    "studied_subject": ["pea plants", "the common edible pea"]
  }
}
```

数据污染样本不能靠 reward 规则修复，应回到源记录确认最终答案并删除中间跳候选。

## 9. 建议的实施顺序

1. 对最新实际训练中的 29 个多-gold prompt 全量人工分类。
2. 对 350 评测集中的 150 个多-gold 样本全量分类。
3. 标签至少包括：`alias_or`、`required_set`、`multi_slot`、`ambiguous`、`contaminated`。
4. 保留当前 Search-R1 EM 为 `legacy_em`，保证历史报告可复现。
5. 新增 `structured_em`、`set_f1` 或 `slot_f1`，与 legacy 指标并行报告。
6. 在新 reward 上重新训练独立实验，不与旧 reward 训练 checkpoint 直接混合比较。
7. 增加多-gold 单元测试，至少覆盖别名、集合、复合槽位、中间答案污染和完整答案顺序变化。

在完成分类前，不建议直接把所有多 gold 改为 AND，也不建议继续把任一列表元素命中解释为所有 QA
类型上的“完全正确”。
