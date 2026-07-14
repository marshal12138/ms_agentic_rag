# Search-R1 动作指标与 SPAD 下一轮奖励决策

生成时间：2026-07-10 16 时

命名说明：`260710-16a` 表示 2026 年 7 月 10 日 16 点生成的第 1 篇报告；同理，已有的
`260710-15a` 表示当天 15 点生成的第 1 篇报告。

## 结论

最新 Search-R1 的训练与确定性评估表明，不能把“平均搜索次数更高”直接视为搜索策略
改善，也不应把“第 6 步平均搜索次数不低于 1.5”继续作为 SPAD reward 实验的主要
通过门槛。

Search-R1 在训练 rollout 中逐渐收缩到一次搜索，但最终 checkpoint 在温度为 0 的
确定性评估中形成了明显的两极分化：一部分样本一次搜索后正常回答，另一部分样本重复
相同 query 直到耗尽五次搜索预算。三次评估中，多搜轨迹有 82.2% 包含重复 query；
搜满五次的 393 条轨迹 EM 为 0。由此可见，当前首要问题不是“搜索次数不足”，而是：

1. 模型没有稳定学会何时停止。
2. 遇到证据不足时，模型经常重复旧 query，而不是围绕缺失事实进行 reformulation。
3. 训练时的随机 rollout action 分布不能可靠预测温度为 0 时的最终策略。
4. Search-R1 的总体指标提升主要来自单跳数据，不能证明多跳检索策略已经改善。

因此，下一轮首先应建立“结果对齐的简单 reward + 训练评测一致的重复 query 防护”
基线；在该基线下确认动作稳定后，再考虑引入更复杂的 evidence-gain shaping。

## 数据来源

最新已完成 Search-R1 训练 run：

`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/agenticIterRag/260710-113003-543853-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal`

最终 checkpoint：

`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260710-113003-543853-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`

三次确定性评估汇总：

`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/docs/AgenticIterRag_v1/work_report/260710-12f_Search-R1最新第8步三次重评报告.md`

作为对照的 SPAD `bad_stop=-0.20` run：

`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/agenticIterRag/260710-151318-734570-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_formal`

该 SPAD run 在第 6/8 步确认早停趋势后按决策停止，未进入 Stage2、Stage3，也未启动
350 条评估。

## Search-R1 训练动作指标

每个 step 包含 64 个 prompt，每个 prompt 采样 8 条轨迹，共 512 条 rollout。
“非重复多搜”指 `search_count >= 2` 且 `duplicate_query_count = 0`。

| Step | 平均搜索次数 | 多搜轨迹 | 非重复多搜 | 含重复 query 的轨迹 | 合法闭合 answer | 非法 answer | EM reward 均值 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.8379 | 211 | 104 | 107 | 349 | 163 | 0.1113 |
| 2 | 2.0117 | 242 | 125 | 117 | 345 | 167 | 0.0723 |
| 3 | 1.7754 | 183 | 72 | 111 | 337 | 175 | 0.1348 |
| 4 | 1.7109 | 166 | 72 | 94 | 305 | 207 | 0.0977 |
| 5 | 1.7070 | 143 | 47 | 96 | 324 | 188 | 0.1895 |
| 6 | 1.2793 | 74 | 32 | 42 | 391 | 121 | 0.1914 |
| 7 | 1.2090 | 59 | 19 | 40 | 368 | 144 | 0.2363 |
| 8 | 1.2539 | 86 | 52 | 34 | 368 | 144 | 0.2148 |

第 8 步的搜索次数分布为：

| 搜索次数 | 轨迹数 |
| ---: | ---: |
| 0 | 3 |
| 1 | 423 |
| 2 | 59 |
| 3 | 14 |
| 4 | 6 |
| 5 | 7 |

第 8 步中，一次搜索轨迹的 EM reward 均值为 0.2104，二次搜索轨迹为 0.3220。
这说明少数有效二搜确实能够从最终答案 EM 获得正向信号。但 64 个 prompt group 中有
37 组 reward 全为 0，只有 23 组存在 reward 方差，训练信号仍然较稀疏。

在同时包含单搜和多搜 rollout 的 29 个 group 中，多搜最优 reward 高于单搜的有 5 组，
单搜高于多搜的有 4 组，另外 20 组持平。简单 outcome reward 没有系统性惩罚多搜，
但也不会奖励没有改善最终答案的额外搜索。

## Search-R1 确定性评估动作指标

三次评估各包含相同的 350 条数据，共计 1050 条轨迹。温度为 0，最多执行五次搜索。

| 搜索次数 | 三次合计轨迹数 | EM | F1 |
| ---: | ---: | ---: | ---: |
| 1 | 432 | 0.2431 | 0.3548 |
| 2 | 185 | 0.1784 | 0.2491 |
| 3 | 32 | 0.1562 | 0.2479 |
| 4 | 8 | 0.0000 | 0.1917 |
| 5 | 393 | 0.0000 | 0.0017 |

整体动作统计：

| 指标 | 数值 |
| --- | ---: |
| 平均 tool calls | 2.7571 |
| 多搜轨迹 | 618 / 1050 |
| 非重复多搜轨迹 | 110 / 618，17.8% |
| 含重复 query 的多搜轨迹 | 508 / 618，82.2% |
| 搜满五次 | 393 / 1050，37.4% |
| 五搜后 `max_turns` | 390 / 393 |

393 条五搜轨迹的唯一 query 数量分布为：

| 唯一 query 数 | 轨迹数 |
| ---: | ---: |
| 1 | 204 |
| 2 | 149 |
| 3 | 25 |
| 4 | 13 |
| 5 | 2 |

即 89.8% 的五搜轨迹最多只产生两个唯一 query。典型失败形式是将
`author of Revolution`、`Chances Are composer` 或 `director of Slim` 原样重复五次。

数据 prompt 已经明确要求：证据不足时识别缺失事实并发出新的 query，且不得重复或
改写旧 query。最新 Search-R1 仍出现上述行为，说明仅靠 prompt 无法约束确定性策略。

## 指标提升来自哪里

Search-R1 `global_step_8` 三次均值相对 base 提升 EM 0.0324、F1 0.0203，但不同数据集
的方向并不一致：

| 数据集 | Base EM / F1 | Search-R1 EM / F1 | 结论 |
| --- | ---: | ---: | --- |
| 2WikiMultiHopQA | 0.0733 / 0.1412 | 0.0333 / 0.0903 | 明显下降 |
| Bamboogle | 0.1133 / 0.2045 | 0.1000 / 0.1563 | 下降 |
| HotpotQA | 0.0333 / 0.1042 | 0.0600 / 0.1037 | EM 上升，F1 持平 |
| MuSiQue | 0.0933 / 0.1164 | 0.0467 / 0.0991 | 下降 |
| NQ | 0.0733 / 0.1750 | 0.1867 / 0.3049 | 明显上升 |
| PopQA | 0.1267 / 0.1952 | 0.1867 / 0.2165 | 上升 |
| TriviaQA | 0.2133 / 0.3180 | 0.3400 / 0.4258 | 明显上升 |

训练 parquet 的 512 条抽样只包含 NQ、HotpotQA、2WikiMultiHopQA 和 MuSiQue；评估还
包含 Bamboogle、PopQA 和 TriviaQA。因此总体提升主要反映单跳问答和答案生成能力，
不能解释为多跳 query reformulation 已经学会。

## 对原实验门槛的修正

原计划使用以下训练 rollout 门槛：

- 第 6 步平均搜索次数不低于 1.5。
- 非重复多搜轨迹不少于 80/512。
- 多搜的组内 reward 胜率不低于单搜。

最新 Search-R1 说明，这些指标只适合作为诊断量，不能单独作为通过条件。训练第 8 步
平均搜索仅 1.2539，但确定性评估平均搜索达到 2.7571，并出现大量重复五搜。训练与
评估 action 分布之间存在明显偏移。

新的原则是：不设最低平均搜索次数。目标应是“证据充分时及时停止；证据不足时执行有
信息增益的新搜索；无法取得增益时避免重复调用”，而不是机械提高 tool calls。

## 下一轮首个 SPAD reward 实验

第一组实验采用简单、结果对齐的 reward，暂时去掉 noisy status 的强负分支：

```text
合法停止且 teacher 正常输出：
  reward = F1(teacher_answer, gold)
         - 0.10 * duplicate_attempt_count

teacher 判断 insufficient_evidence / ambiguous_evidence：
  reward = 0
         - 0.10 * duplicate_attempt_count
  status 仅作为诊断字段，不触发 bad-stop penalty

格式非法 / 没有搜索证据 / 未合法结束：
  reward = -0.5

teacher 格式错误：
  reward = -0.1

search_cost = 0
bad_stop penalty = 0
```

该设计与 Search-R1 的 outcome reward 保持同一方向：只有检索证据最终改善 teacher
answer F1 时，额外搜索才获得正向收益。它不奖励搜索次数本身，也不会因为 teacher 的
`ambiguous` 噪声无条件施加强负分。

## 重复 query 防护

训练与评测 agent loop 应使用完全相同的精确重复 query 防护：

1. 对 query 做小写、首尾空白和连续空白归一化。
2. 命中历史 query 时不调用检索服务，避免重复消耗 tool budget。
3. 向模型返回简短的重复拒绝 observation，要求使用新实体或关系。
4. 记录 `duplicate_attempt_count`，并在所有合法 reward 分支中扣除每次 0.10。
5. 连续两次重复后以 `duplicate_loop` 结束，防止温度为 0 时无限重试。
6. 分别记录 attempted query 和实际 executed query，避免 guard 将模型行为问题隐藏掉。

首版只拦截精确归一化重复，不直接拦截语义近似 query，以降低误杀正常 reformulation
的风险。

## 为什么暂不直接加入 evidence gain

扩展人工审查显示，现有 teacher 状态总体一致率为 78.5%，其中：

| Teacher 状态 | 人工一致率 |
| --- | ---: |
| `supported_answer` | 80.0% |
| `insufficient_evidence` | 84.0% |
| `ambiguous_evidence` | 59.5% |

在该准确率下继续让同一 teacher 判断更细的 `none / partial / sufficient` 边际增益，很
可能把当前状态噪声放大为更密集的错误 shaping signal。尤其 `ambiguous` 约四成误判，
不适合直接作为强负 reward 或 query gain 的可靠监督。

因此 evidence gain 放在第二组实验。开始前应先改进 teacher 对关系谓词、passage 引用、
简单组合推理和真实歧义的判定，并对新的边际增益标签单独做人工审查。

## 训练与评估方案

第一轮仍固定：

- 同一份 512 条训练数据。
- `data_seed=42`。
- 每个 prompt 采样 8 条轨迹。
- 先只执行 Stage1，到第 6 步保存 checkpoint。
- 不立即进入 Stage2 和 Stage3。

由于训练 rollout 不能代表温度为 0 的最终动作策略，第 6 步 checkpoint 必须执行同配置
的确定性 350 条评估。服务启动耗时占比较高，缩小到 70 条节省有限，因此直接保留完整
350 条更有统计意义。

## 新的通过门槛

训练 rollout 指标只用于诊断，不再要求平均搜索次数达到固定下限。主要通过条件来自
确定性评估：

| 指标 | 通过门槛 |
| --- | --- |
| 首轮搜索执行率 | 100% |
| `max_turns` 比例 | 不超过 20% |
| `duplicate_loop` 比例 | 不超过 5% |
| 实际执行的重复 query | 0；重复 attempt 单独报告 |
| 有效 answer 比例 | 不低于同配置 base |
| 四个多跳数据集平均 EM/F1 | 不低于同配置 base |
| 总体 EM/F1 | 不低于同配置 base，并报告相对 Search-R1 的差值 |

同时必须报告：

- 每个搜索次数桶的 EM、F1 和状态分布。
- 非重复多搜与重复 attempt 轨迹的比例。
- 每条轨迹的唯一 query 数和执行 query 数。
- 单跳与多跳数据集分开统计的 action 和准确率。
- 训练 rollout 与温度为 0 的评估 action 分布差异。

只有简单 outcome reward 在上述门槛下动作稳定、且多跳指标仍缺少提升时，才进入第二个
evidence-gain reward 实验。
