# GLM-4.7 Teacher Prompt Engineering 消融历史

更新时间：2026-07-10 18:23 CST

## 约束与验收

- 固定人工表：237 条，按规范化 Original question 分组为 dev 178、holdout 59。
- 主要指标：`I precision` 与 `I recall`；`S/A` 互相混淆暂时容忍。
- `0.98/0.98` 是理想上限；未达到时选择 holdout 效果、格式稳定性、时耗的 Pareto 最优方案。
- few-shot 禁止包含或改写 237 条中的问题、答案、evidence、数值、实体或人工判断。
- 所有合法 prompt 均为 no-thinking，除 C1 单独的 thinking 成本试验。
- 有效消融和独立复跑禁止读取 runner response cache；必须满足 `run.json.cache_hits=0`。
- 每个结果的完整指标见 `RESULTS_INDEX.md`，逐样本 messages/response/error 位于相应 `results/` 目录。

## 数据与基线

1. 冻结 `manual_judgments_237.tsv`、`benchmark_237.jsonl` 与 manifest；同题不跨 split。
2. A0 当前 v2 baseline：全量 237 条；dev I=`0.738/0.950`，holdout I=`0.870/0.800`。
3. K1 历史 v1 layout 严格复跑：历史 teacher 标签复现率仅 75.9%，说明 layout 不是唯一漂移来源。
4. K2 正在使用原 run 拓扑（单 TP=2 replica、并发 16、独立缓存）复跑，以测批调度影响。

## A: 单 Prompt Instruction-only

- A1 `candidate_count`：dev `0.833/0.875`，是单 prompt dev 最均衡方案；holdout `0.905/0.760`。
- A2 `candidate_count_i_guard`：dev `0.814/0.875`，增加缺桥规则未超过 A1。
- A3 compact：dev `0.655/0.950`，指令压缩导致过度判 I。
- A4 binary-I gate：dev `0.830/0.488`，过度放宽 candidate，漏掉大量 I。
- A5 balanced：dev `0.785/0.913`，改善 recall 但 precision 下降。

结论：system wording 可以移动 precision/recall operating point，但没有同时消除两侧错误。

## B: 增加 Gold Answer

- B1 gold hypothesis：dev `0.759/0.788`。
- B2 gold strict audit：dev `0.787/0.600`。
- B3 gold binary support：dev `0.769/0.875`，holdout `0.767/0.920`。

结论：gold 造成锚定；严格要求 gold 被支持又会把“证据支持其他答案”的 non-I 判成 I。Gold-aware
不优于无 gold baseline，不能作为当前生产选择。

## C: Thinking 与 Few-shot

- C1 thinking：dev `0.873/0.863`，但 parse=0.792、平均 1095 completion tokens、墙钟 135.3s。
  当前 chat template 把 thinking 写入 content，37 条在 XML 前截断，因此暂不继续。
- C2b 完全虚构 few-shot：dev `0.811/0.750`，没有收益。
- 早期 C2/E1 曾使用从现有数据派生或直接取自 observation 的示例，已移至
  `results/_invalid_data_leakage/`，从所有比较中永久排除。

## D: 串行 Critic / Debate

- D1 多草稿 arbiter：dev `0.861/0.775`。
- D2 A1 单草稿 critic：dev `0.807/0.888`。
- D3 检察/辩护/中性三方 arbiter：dev `0.855/0.738`。
- D4 虚构争议 few-shot arbiter：dev `0.841/0.725`。

结论：GLM 阅读自己的自然语言草稿后倾向把候选重新合理化；多阶段自然语言裁决增加时耗但未提高泛化。

## F/G: 自一致性与结构化 Worker

- A1 低温三 seed 加 temperature=0 的 4 路自一致性，最佳阈值 2/4：dev `0.841/0.925`；并行墙钟
  约 70s，格式错误增多，未跑 holdout。
- G1 entity/predicate/bridge certificate：dev `0.831/0.613`。
- G2 missing-bridge auditor：dev `0.700/0.875`。

结论：随机采样略改善 operating point，但成本明显；强制证书没有让 Flash 模型更可靠。

## H/J: One-token 概率 Gate

- vLLM `structured_outputs.choice=[I,N]` 可将每条输出压到 1 token。
- H1 全量 237 条墙钟 5.63s，但 dev `0.806/0.725`、holdout `0.792/0.760`。
- H2-H5 分别测试 exact relation、missing fact、candidate existence 与 gold support，均未超过 A1/A0。
- J1-J3 把 A1/A3 草稿送入 one-token final gate，仍未改善。
- 5 路 log-odds 线性校准 holdout 约 `0.778/0.840`，判别信息不足，未纳入候选。

结论：one-token gate 是最佳速度方案，但当前准确率不够；可作为未来低成本 cascade 的组件，不能单独替换 teacher。

## V: 多 Prompt 投票

- dev 冻结组合：A0/A1/A2/A3/B3 中至少 4 路判 I，dev `0.913/0.913`。
- holdout 仅 `0.905/0.760`，与 A1 相同级别；平均串行请求延迟 `28.85s/sample`，约
  `385.7` completion tokens。

结论：投票提高 dev，但没有 holdout 泛化收益，成本不成立。

## 当前选择状态

- 尚无方案达到 `0.98/0.98`。
- M2/P1/S0 去掉 sub_query、保留完整 passage、并在尾部重申 Original question，历史 6 次独立运行的
  holdout 均值约 I=`0.855/0.853`。最新 cache-free 三重复为 `0.856/0.867`，I F1 `0.861`；仍是当前
  稳定性最佳的单调用策略。
- A1 precision 更高（0.905），但 recall 降到 0.760；若业务更重视不误判 I，可作为另一 Pareto 点，
  但不满足“precision/recall 同时尽量高”的主目标。

## 每十次固定复盘

### Batch 1：#1-#10

| # | 方案 | 关键结果/结论 |
| ---: | --- | --- |
| 1 | A0 current-v2 baseline | dev `0.738/0.950`；holdout `0.870/0.800` |
| 2 | A1 candidate count | dev `0.833/0.875`，单 prompt dev 最均衡 |
| 3 | A2 I guard | dev `0.814/0.875`，规则变长无收益 |
| 4 | A3 compact guard | dev `0.655/0.950`，明显过度判 I |
| 5 | A4 binary gate | dev `0.830/0.488`，明显漏判 I |
| 6 | A5 balanced | dev `0.785/0.913`，precision/recall 仍交换 |
| 7 | B1 gold hypothesis | dev `0.759/0.788`，gold 锚定 |
| 8 | B2 gold strict audit | dev `0.787/0.600`，过度相信 gold |
| 9 | B3 gold binary support | dev `0.769/0.875`，未超过无 gold |
| 10 | C1 thinking | dev `0.873/0.863`，但 parse 0.792、135.3s、1095 tokens |

反思：单纯增删规则只能移动 operating point；gold 和 thinking 均有明显副作用。下一批应测试
few-shot、critic、投票和多阶段，而不是继续同义改写单 prompt。

### Batch 2：#11-#20

| # | 方案 | 关键结果/结论 |
| ---: | --- | --- |
| 11 | C2b 纯虚构 few-shot | dev `0.811/0.750`，few-shot 无收益 |
| 12 | D1 多草稿 arbiter | dev `0.861/0.775` |
| 13 | D2 单草稿 critic | dev `0.807/0.888` |
| 14 | D3 检察/辩护 arbiter | dev `0.855/0.738` |
| 15 | D4 虚构 debate few-shot | dev `0.841/0.725` |
| 16 | V1 五路 4/5 投票 | dev `0.913/0.913`；holdout `0.905/0.760`，约 387 tokens |
| 17 | F 四路低温自一致性 | dev 最佳 `0.841/0.925`，并行约 70s |
| 18 | G1 entailment certificate | dev `0.831/0.613` |
| 19 | G2 missing bridge auditor | dev `0.700/0.875` |
| 20 | H1 one-token direct gate | holdout `0.792/0.760`；全量仅 5.63s |

反思：自然语言串行裁决会继承并合理化前级错误；投票在 dev 上过拟合且成本过高；one-token gate
速度优秀但判别力不足。下一批应测试概率 gate 变体、草稿级联以及历史布局/服务拓扑。

### Batch 3：#21-#30

| # | 方案 | 关键结果/结论 |
| ---: | --- | --- |
| 21 | H2 exact-relation gate | holdout `0.708/0.680` |
| 22 | H3 missing-fact gate | holdout `0.590/0.920` |
| 23 | H4 candidate-existential gate | holdout `0.741/0.800` |
| 24 | H5 gold-support gate | holdout `0.629/0.880` |
| 25 | J1 A1 draft + direct gate | holdout `0.760/0.760` |
| 26 | J2 A1/A3 + exact gate | holdout `0.643/0.720` |
| 27 | J3 A1/A3 + gap gate | holdout `0.667/0.880` |
| 28 | K1 historical v1 / 4 replicas | holdout `0.808/0.840`；历史标签复现率 75.9% |
| 29 | K2 historical v1 / 1 replica | holdout `0.870/0.800`；109.9s，无准确率收益 |
| 30 | L1 v2 evidence-only（去 sub_query） | holdout `0.840/0.840`；27.84s，当前最佳均衡候选 |

反思：服务拓扑会改变个别边界输出，但单 replica 的 4 倍墙钟不成立。最有价值的新信息是
`sub_query` 会干扰 teacher 对 Original question 的判断；去掉它比堆指令、多模型投票或 gold 更有效，
而且不增加延迟。Batch 4 将围绕 evidence 呈现与低成本 cascade 做正交消融。

### Batch 4：#31-#40

| # | 方案 | 关键结果/结论 |
| ---: | --- | --- |
| 31 | L2 A1 + evidence-only | holdout `0.870/0.800`，12 条格式错误，不如 baseline instruction |
| 32 | L3 去 passage title | holdout `0.792/0.760`，title 是有用实体信号 |
| 33 | L4 evidence-only top-3 | holdout `0.733/0.880`，删后两篇提高 recall、严重降低 precision |
| 34 | L5 去 round 层级 | holdout `0.760/0.760`，扁平化丢失结构信息 |
| 35 | L6 evidence-only 单 replica | holdout `0.808/0.840`，L1 的 `0.840/0.840` 未稳定复现 |
| 36 | M1 current-v2 单 replica | holdout `0.778/0.840`，单服务基线也有调度漂移 |
| 37 | M2 去 sub_query + 尾部重申问题 | holdout `0.875/0.840`、F1 `0.857`，本批最佳单次结果 |
| 38 | M3 显式 passage delimiters | holdout `0.870/0.800`，无额外收益 |
| 39 | M4 concise balanced + evidence-only | holdout `0.778/0.840`，输出更短但 5 条格式错误 |
| 40 | N1 保留 sub_query + 尾部重申，四 replica | holdout 从 `0.769/0.800` 到 `0.880/0.880`，方差过大 |

反思：尾部重申 Original question 是当前最有希望的布局改动；只删 sub_query 的收益不稳定，title 与
round 层级不应删除。更重要的是，同权重、temperature=0 的四个 replica 在相同 59 条 holdout 上仍有
明显差异，单次最高分不能直接作为结论。Batch 5 首先对 M2 做四 replica 复跑并报告均值/范围，然后
再决定是否注册生产 prompt；离线执行继续尽量并行，但每个候选策略保持单 teacher 调用。

### Batch 5：#41-#50

| # | 方案 | 关键结果/结论 |
| ---: | --- | --- |
| 41 | P1 M2 四 replica 复跑 | holdout 范围 `0.808/0.840` 到 `0.920/0.920`；均值约 `0.858/0.840` |
| 42 | Q1 top-3 + question tail | holdout `0.786/0.880`，仍牺牲 precision |
| 43 | Q2 去 title + question tail | 单次 `0.875/0.840`，需稳定性复跑 |
| 44 | Q3 delimiters + question tail | 单次 `0.875/0.840`，token 更多，无明显增益 |
| 45 | Q4 short focused question tail | `0.840/0.840`，约 53 tokens，但有格式风险 |
| 46 | R1 Q2 四 replica 复跑 | 均值约 `0.830/0.824`，去 title 不稳定且不如保留 title |
| 47 | R2 Q4 四 replica 复跑 | 均值约 `0.898/0.792`，短输出换来 recall 下降和格式错误 |
| 48 | S1 question 只在尾部出现 | holdout `0.808/0.840`，问题应在首尾都保留 |
| 49 | S2 XML-tagged question | holdout `0.800/0.800`，标签无益 |
| 50 | S3 尾部候选计数提醒 | holdout `0.905/0.760`，提高 precision、降低 recall |

反思：M2 的有效组合是“首部问题 + 完整 title/passage + 不显示 sub_query + 尾部重复问题”。删 title、
删首部问题、加结构标签或加判定规则都会降低稳定均值。累计 6 次 M2 同策略运行后，holdout 平均
precision/recall 约 `0.855/0.853`；单次 `0.920/0.920` 不应被单独采用。详细范围见
`REPLICA_STABILITY.md`。除非后续有新的正交变量，生产候选应收敛到 M2，而不是继续堆叠指令。

## Top 5 cache-free 三重复验证

这 15 次是已有五种策略的重复验证，不计入新的消融方案编号。每个策略在三个不同 replica 上重新跑
全量 237 条，纳入统计的每次运行均满足 `cache_hits=0`。表中为 59 条 holdout 的三次均值：

| 排名 | 策略 | I precision | I recall | I F1 | Parse | 237 条墙钟 | 平均请求耗时 | Completion tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | question-tail evidence-only | 0.8556 | 0.8667 | 0.8606 | 1.0000 | 117.88s | 7.71s | 83.1 |
| 2 | question-tail with sub_query | 0.8426 | 0.8533 | 0.8478 | 1.0000 | 119.59s | 7.84s | 82.3 |
| 3 | title-free question-tail | 0.8494 | 0.8267 | 0.8378 | 1.0000 | 118.60s | 7.75s | 82.3 |
| 4 | short focused question-tail | 0.8832 | 0.7867 | 0.8307 | 0.9831 | 104.05s | 6.82s | 53.6 |
| 5 | evidence-only without tail | 0.8309 | 0.7867 | 0.8081 | 1.0000 | 105.87s | 6.99s | 82.8 |

第一次启动第二轮时，四个目录意外命中默认共享 response cache（`cache_hits=237`）。这些结果已移到
`results/_invalid_cached_replay/`，永久排除。runner 随后增加 `--disable-cache`；本次有效重复均没有
复用历史响应。

反思：新的严格三重复再次确认，隐藏 sub_query 并在证据末尾重申 Original question 的组合是准确率
最佳方案。short-focused 是 Top 5 中纯速度最快的方案，平均请求快约 11.5%、输出少约 35.5%，但
I recall 低 0.08 且存在 1.69% 格式失败；综合准确率、稳定性和耗时后仍选择 question-tail
evidence-only。
