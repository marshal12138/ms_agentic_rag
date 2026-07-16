# SPAD 新数据 Teacher Prompt 与组合策略消融对比报告

生成时间：2026-07-15T23:56:39+08:00（北京时间）

> 状态：基于新 5100-step 训练 rollout 的 512 条分层样本，人工 S/I/A 标注、384 dev
> prompt 消融、128 holdout 评估、组合策略三重复和耗时审计均已完成。当前领先方案是
> `hard_gate_r5_literal_canonical_v2`。该方案目前只在 Teacher PE harness 中实现，尚未接入
> 正式训练 runtime。四个 PE vLLM 服务继续保持运行。

## 1. 结论

本轮目标同时关注两项同等重要的能力：

1. Teacher 对 `insufficient_evidence`（I）的判别能力。
2. Teacher 在人工标注为 S 的样本上，生成答案对数据集 gold answer 的 token-F1 覆盖。

在实际 `teacher_called=true` 的 221 条 dev 样本上，三次 cache-free fresh 推理均值如下：

| 策略 | I Precision | I Recall | I F1 | Gold F1 覆盖 | 人工答案 F1 覆盖 | 等权指标 | 推理预算 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 训练 Production prompt | 0.8970 | 0.8881 | **0.8924** | 0.3180 | **0.6133** | 0.6052 | 1.0000x |
| 单 prompt R5 | 0.8287 | **0.9051** | 0.8651 | 0.6399 | 0.4408 | 0.7525 | 1.0000x |
| **Hard-gate v2** | 0.8970 | 0.8881 | **0.8924** | **0.6825** | 0.4833 | **0.7874** | 1.3558x |

核心结论：

- 训练 Production prompt 的 I 判别最好，但不看 gold，Gold F1 覆盖只有 `0.3180`。
- R5 用一次 gold-aware 调用把 Gold F1 提高到 `0.6399`，但 I F1 降到 `0.8651`。
- Hard-gate v2 把 I 二分类边界完全交给 Production prompt，并只在非 I 路径调用 R5，因此
  I P/R/F1 与 Production prompt 逐样本完全相同，同时把 Gold F1 提高到 `0.6825`。
- Hard-gate v2 的等权指标为 `0.7874`，比 Production prompt 高 `0.1822`，比单 prompt R5
  高 `0.0349`，且平均推理预算 `1.3558x`，低于用户允许的 `2x`。
- 允许 gold-aware Stage B 推翻 Stage-A I 的 Dual-all v2 在 holdout 上出现 I recall 崩塌，
  已明确淘汰。Gold literal 出现在 passage 中不能替代完整关系与 bridge 的证据判断。

## 2. 数据、抽样与人工标注

### 2.1 来源训练

样本来自正式训练 run：

```text
260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm03_stage1
```

该 run 共有 79 个 step、5,056 个 question group、每组 8 条 rollout，共 40,448 条轨迹。

### 2.2 分层抽样

抽样遵循以下冻结规则：

- 每个 question group 只抽一条轨迹。
- 按 step 每 20 步分层，形成 L1 `1-20`、L2 `21-40`、L3 `41-60`、L4 `61-79`。
- 每层 128 条，共 512 条；优先抽取该 group 中 `teacher_called=true` 的代表轨迹。
- 最终包含 293 条 teacher-called 轨迹和 219 条非调用 control。

### 2.3 人工 S/I/A 标注

人工判断只使用 Original question 与累计可见 Search evidence：

- S：证据支持一个完整、唯一的答案。
- I：缺少必要事实、谓词关系或多跳 bridge。
- A：证据支持多个同等满足问题约束、且互不兼容的完整答案。

Gold answer、Actor answer 和历史 Teacher 输出只用于审计，不作为人工标签证据。

512 条标签分布为 S/I/A `241/241/30`。冻结 benchmark 使用 384 dev 和 128 holdout：

| Split | 总数 | S | I | A | 每个 step 层 |
|---|---:|---:|---:|---:|---:|
| dev | 384 | 181 | 181 | 22 | 96 |
| holdout | 128 | 60 | 60 | 8 | 32 |

## 3. 训练实际 Teacher Prompt 的身份确认

### 3.1 配置与代码

源训练的最终配置明确记录：

```yaml
agent_training:
  teacher_answerer:
    prompt_version: spad_teacher_evidence_status_answer_v2
```

对应位置：

- 训练配置：`log/agenticIterRag/260715-005906-987696-.../runtime_logs/pipeline/pipeline.final_config.yaml`
- 生产 prompt registry：`AgenticIterRag/agentic_iter_rag/agent_training/spad/prompts.py`
- PE registry：`pipelines/formal/agenticIterRag/teacher_PE/prompt_variants.py`

PE 中用 `baseline_current_v2` 表示该生产 prompt。已经用同一冻结 case 做程序化逐字比较：

```text
production system message == PE system message: true
production user message   == PE user message:   true
```

因此本文的 fresh Production prompt 指标是训练实际 prompt 的同构复现，不是近似替代方案。

### 3.2 历史输出与 fresh 复现

原训练保存的 293 条 teacher-called 历史输出中有一条缺少可解析 status。历史结果为：

| 范围 | N | I Precision | I Recall | I F1 | Gold F1 覆盖 | 等权指标 |
|---|---:|---:|---:|---:|---:|---:|
| 全部历史 teacher-called | 293 | 0.9121 | 0.8830 | 0.8973 | 0.3086 | 0.6030 |
| dev 历史 teacher-called | 221 | 0.8971 | 0.8905 | 0.8938 | 0.2831 | 0.5884 |
| dev fresh 三次均值 | 221 | 0.8970 | 0.8881 | 0.8924 | 0.3180 | 0.6052 |

历史与 fresh 的 I F1 只差 `0.0014`，说明复现实验与训练中的实际判别行为一致。Gold F1 的
差异反映 GLM-4.7-Flash 在 temperature=0 下仍存在生成级非完全确定性，因此候选选择使用三次
fresh 均值而不是单次历史输出。

## 4. 指标定义与主切片

### 4.1 I 二分类

将人工 I 视为正类，将 S/A 合并为非 I：

```text
I Precision = 正确预测 I / 全部预测 I
I Recall    = 正确预测 I / 全部人工 I
I F1        = I Precision 与 I Recall 的调和平均
```

本轮主要关心 I，而不是强行区分 S/A。S/A confusion 仍单独记录，但不进入第一指标。

### 4.2 答案覆盖

只在人工标签为 S 的样本上计算：

- 策略返回 S 时，计算 Teacher answer 对所有 reference gold 的最大 token-F1。
- 策略返回 I/A 或格式解析失败时，该样本贡献 0。
- `Gold F1 覆盖` 是人工 S 样本上的平均值。
- `人工答案 F1 覆盖` 使用人工 evidence-grounded answer，作为 gold 偏移诊断。

### 4.3 等权指标

```text
Equal objective = 0.5 * I_F1 + 0.5 * Gold_F1_coverage_on_manual_S
```

所有策略选择首先看实际 `teacher_called=true` 切片。全 dev 包含较容易、训练中未调用 Teacher
的 control，因此只作为第二切片。

## 5. 策略一：训练 Production Prompt

### 5.1 输入与输出

Production prompt 不读取 gold，输入包括：

- Original question。
- 每轮 sub-query。
- 每轮最多 5 个检索文档的 title 和 passage。
- 截至当前轨迹的全部累计 evidence。

单次调用输出：

```text
<reason>...</reason>
<status>supported_answer|insufficient_evidence|ambiguous_evidence</status>
<answer>...</answer>
```

system prompt 要求只使用 evidence；一个完整答案用 S，缺少必要事实用 I，多个互斥完整答案用 A。

### 5.2 优点

- 不受 gold 诱导，I 判别最稳定，dev I F1 为 `0.8924`。
- dev 人工答案 F1 为 `0.6133`，三种候选中最高。
- 一条样本只调用一次 Teacher，逻辑与当前训练完全一致。

### 5.3 局限

- 不知道 reference gold，答案可能是 evidence 支持的别名、描述或另一正确候选。
- teacher-called dev 的 Gold F1 只有 `0.3180`，在新等权目标下成为主要瓶颈。
- 等权指标只有 `0.6052`，不能充分提供 Gold Token-F1 reward 信号。

### 5.4 适用位置

当运行时没有 gold 时，它仍是默认方案。在 Hard-gate v2 中，它被保留为不可推翻的 I gate。

## 6. 策略二：单 Prompt R5

### 6.1 Prompt 与布局

R5 registry 名为 `gold_support_evidence_only_v3`，输入包含：

- Original question。
- Reference gold answers。
- 完整 title/passage evidence。

它隐藏 sub-query，不在尾部重复 question。system prompt 明确规定 gold 是待验证 hypothesis，
不是 evidence；必须检查 exact entity、predicate、scope 和全部 bridge。

### 6.2 单调用流程

```text
question + gold + evidence -> reason + S/I/A + answer
```

如果 evidence 支持 gold 或等价别名，返回最短答案；如果 gold 不受支持但另一个答案受支持，可以
返回非 gold 答案；证据缺失则返回 I。

### 6.3 优点

- 一次调用即可使用 gold，推理预算仍为 `1.0x`。
- dev Gold F1 从 Production 的 `0.3180` 提高到 `0.6399`。
- 等权指标从 `0.6052` 提高到 `0.7525`。
- 三次 dev 等权指标范围仅 `[0.7516, 0.7535]`，稳定性较高。

### 6.4 局限

- I Precision 从 Production 的 `0.8970` 降到 `0.8287`。
- I F1 从 `0.8924` 降到 `0.8651`。
- 人工答案 F1 从 `0.6133` 降到 `0.4408`，说明 gold-aware prompt 会偏向数据集 gold 措辞，
  不总是保持人工认为最直接的 evidence answer。

### 6.5 适用位置

当 gold 可用、且严格只允许一次 Teacher 调用时，R5 是当前最佳稳定单 prompt 方案。

## 7. 策略三：Hard-Gate v2

### 7.1 结构

正式名称：`hard_gate_r5_literal_canonical_v2`。

```text
                         +--> I: 直接返回 Stage-A I，不调用 Stage B
question + evidence --> Stage A Production prompt
                         +--> S/A: 调用 Stage-B R5（question + gold + evidence）
                                      |
                                      +--> 保持 Stage-A I/non-I 边界
                                      +--> supported answer 择优
                                      +--> evidence-literal gold 规范化
```

### 7.2 I 边界规则

Hard-gate 的含义是 I 二分类完全由 Stage A 决定：

1. Stage A 返回 I 时，Stage B 不调用，最终一定是 I。
2. Stage A 返回 S/A、Stage B 返回 I 或解析失败时，回退到 Stage A，最终仍是非 I。
3. Stage B 只能处理非 I 内部的 S/A 和答案内容，不能改变 I/non-I 边界。

因此：

```text
Final is I <=> Stage A is I
```

三次 dev 中，Hard-gate v2 与对应 Production Stage A 的 I confusion matrix 逐样本完全相同，
I P/R/F1 均为 `0.8970/0.8881/0.8924`。这不是四舍五入后的相近，而是合并规则保证的相等。

### 7.3 非 I 答案合并

只有 Stage A 为 S/A 时才进入答案阶段：

1. 两个阶段中只有一个给出 S，保留唯一 supported answer。
2. 两个阶段都给出 S，选择对 reference gold token-F1 更高的答案。
3. 如果 reference gold 的规范化字面值确实出现在 Search evidence 中，且替换能提高 Gold F1，
   才将答案规范化为该 gold。
4. Gold literal 不在 evidence 时禁止注入。
5. 人工 label 和人工答案不会进入运行时策略。

这使 gold 只参与非 I 答案选择，不参与证据充分性判断。

### 7.4 效果

- 保持 Production I F1 `0.8924`。
- Gold F1 从 Production 的 `0.3180` 提高到 `0.6825`。
- Gold F1 比单 prompt R5 的 `0.6399` 再提高 `0.0426`。
- 等权指标达到 `0.7874`，是当前主切片最高值。
- 人工答案 F1 为 `0.4833`，高于 R5 的 `0.4408`，但低于 Production 的 `0.6133`。
- 三次 parse rate 均为 `1.0`。

### 7.5 成本

dev 三次实测：

| 成本项 | 均值或范围 |
|---|---:|
| 全 dev Stage-B 调用率 | 51.6%-52.1% |
| teacher-called Stage-B 调用率 | 37.6%-40.3%，均值约 38.6% |
| 全 dev 平均 elapsed ratio | 1.3558x |
| teacher-called 平均两阶段累计推理时间 | 9.28 秒 |
| 用户允许上限 | 2.0x |

PE 使用 4 个无持续排队副本测量 token 生成时间。正式训练只能使用单副本，因此 `1.3558x`
表示平均单样本生成工作量，而不是对正式训练 wall time 的直接保证。接入后仍需用小规模单副本训练
pilot 测量队列、rollout 与 Teacher 的流水重叠。

### 7.6 当前状态

该策略已经在 PE harness 中实现、三次复验并通过预算审计，但尚未修改正式训练 Teacher runtime。
生产集成应是后续独立任务，不能把 PE runner 直接视为训练代码已完成。

## 8. 淘汰对照：Dual-All v2

Dual-all v2 对所有样本调用 R5，并允许 Stage B 在以下条件下推翻 Stage-A I：

```text
Stage B status == S and token_F1(Stage-B answer, gold) >= 0.8
```

dev teacher-called 三次均值：

| I F1 | Gold F1 | 等权指标 | 平均耗时 |
|---:|---:|---:|---:|
| 0.8812 | 0.7220 | 0.8016 | 1.7904x |

该方案 dev 等权指标表面上高于 Hard-gate，但 holdout 暴露严重问题：

| I Precision | I Recall | I F1 | Gold F1 | 等权指标 |
|---:|---:|---:|---:|---:|
| 0.9473 | **0.6993** | **0.8045** | 0.9222 | 0.8634 |

许多人工 I 样本的 passage 中出现了 gold 字面值，但缺少问题要求的完整谓词关系或多跳 bridge。
Gold token-F1 阈值只能证明答案字符串相似，不能证明 evidence sufficiency。因此 Dual-all v2 被淘汰，
并形成后续组合策略的硬约束：Stage B 不得推翻 Production I gate。

## 9. 全 Dev 与实际 Teacher-Called 的差异

全 384 dev 包含 163 条训练中未实际调用 Teacher 的 control。三次均值如下：

| 策略 | 全 dev I F1 | 全 dev Gold F1 | 全 dev 人工答案 F1 | 全 dev 等权指标 |
|---|---:|---:|---:|---:|
| Production | 0.8605 | 0.6267 | 0.7241 | 0.7436 |
| R5 | 0.8424 | **0.8153** | **0.7214** | 0.8288 |
| Hard-gate v2 | **0.8605** | 0.8013 | 0.7083 | **0.8309** |

Control 通常证据更充分、答案更容易，所以全 dev 的 Gold F1 明显高于 teacher-called 切片。正式选择
以 teacher-called 为第一切片，不能用 control 的高分掩盖困难样本表现。

## 10. Holdout 结果与统计边界

### 10.1 三次诊断均值

128 holdout 中有 72 条 teacher-called 样本：

| 策略 | I Precision | I Recall | I F1 | Gold F1 | 人工答案 F1 | 等权指标 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production fresh | 0.9478 | 0.8235 | **0.8812** | 0.4141 | **0.7615** | 0.6476 | 1.0000x |
| R5 frozen | 0.9321 | 0.8039 | 0.8632 | 0.7667 | 0.6156 | 0.8149 | 1.0000x |
| Hard-gate v2 | 0.9478 | 0.8235 | **0.8812** | **0.9000** | 0.5267 | **0.8906** | 1.2307x |

Hard-gate 再次逐样本保持 Production 的 I 边界，并显著提高 Gold F1。

### 10.2 必须保留的限制

- R5 是在 dev 冻结后第一次打开 holdout，属于原始冻结评估。
- 组合策略随后使用同一 128 holdout 诊断并淘汰 Dual-all。
- Hard-gate 的 holdout 复验发生在 holdout 已影响组合策略决策之后，因此只能称为 reused-holdout
  diagnostic，不能称为新的未触碰最终估计。
- Hard-gate holdout 人工答案 F1 `0.5267` 低于 R5 的 `0.6156`，说明 evidence-literal gold
  规范化会靠近数据集 gold，但可能远离人工 evidence answer 的措辞。
- 不应继续使用 3500e 调 Teacher prompt。3500e 应保留为 Actor policy 的最终泛化评估。

## 11. 策略选择建议

| 约束 | 建议策略 | 原因 |
|---|---|---|
| 运行时没有 reference gold | Production prompt | 不需要 gold，I 判别稳定，单调用 |
| gold 可用但严格限制一次调用 | 单 prompt R5 | 当前最佳稳定单调用等权指标 |
| gold 可用、允许最多 2x、I 与 Gold 同等重要 | **Hard-gate v2** | 保持 Production I 边界，Gold F1 与等权指标最高 |

当前建议把 Hard-gate v2 作为下一轮生产集成候选，但在正式大训练前必须完成两件事：

1. 将两阶段 gate、fallback、答案择优和 literal guard 接入正式 Teacher runtime，并补齐单元测试。
2. 从新的训练 rollout 重新抽取未触碰验证样本，确认 I F1 与 Gold F1 的联合提升，不再使用当前
   128 holdout 调规则或门限。

## 12. 产出与复现位置

Teacher PE 根目录：

```text
pipelines/formal/agenticIterRag/teacher_PE
```

关键产出：

| 文件或目录 | 作用 |
|---|---|
| `NEW_DATA_STRATEGY_COMPARISON.md` | PE 目录中的逐策略详细审计文档 |
| `NEW_DATA_PE_WORKLOG.md` | 抽样、标注、单 prompt 与组合消融时间线 |
| `NEW_DATA_RESULTS_INDEX.md` | 41 个持久化结果目录的自动指标索引 |
| `NEW_DATA_STABILITY.md` | 三次重复均值、范围、人工答案与预算汇总 |
| `manual_judgments_newdata_512.tsv` | 512 条人工 S/I/A 判断 |
| `benchmark_newdata_512_ablation.jsonl` | 冻结的 384 dev / 128 holdout benchmark |
| `composite_prompt_variants.py` | 组合策略 registry |
| `run_composite_ablation.py` | 两阶段推理、合并、评分与预算落盘 |
| `derive_composite_policy.py` | 对独立持久化阶段输出做确定性策略复算 |
| `results_newdata/` | 单 prompt、组合、重复与 holdout 的完整输出 |

最终一致性审计覆盖 41 个完整结果目录和 12,672 条预测；预测数不匹配、cache hit、请求错误和
超过 `2x` 预算的组合 run 均为 0。CPU 回归测试 19/19 通过。

## 13. 最终判断

在本轮新数据分布和用户定义的等权目标下，Hard-gate v2 是目前唯一同时满足以下条件的方案：

- I F1 不低于训练 Production prompt。
- Gold F1 高于单 prompt R5。
- 平均推理预算低于 2x。
- gold 不参与 I gate，且 gold literal 不在 evidence 时禁止答案注入。

因此，当前工程选择应从“是否继续修改单个 prompt”转为“是否把 Hard-gate v2 集成到单副本正式
训练 runtime，并用新的未触碰样本验证”。在完成该验证前，应称其为领先生产候选，而不是已经
获得无偏最终证明的默认方案。

## 14. 消融过程

本节补充时间：2026-07-16T09:41:24+08:00（北京时间）。

本轮消融不是一次性比较几个 prompt，而是分成六个阶段逐步缩小问题：

```text
冻结新数据与指标
        |
        v
无 gold 的旧成功布局复现
        |
        v
gold 输入与布局因子消融
        |
        v
重复运行，确认 R5/R9 的稳定性并冻结 R5
        |
        v
在 2x 预算内组合 I gate 与答案生成
        |
        v
holdout 诊断，淘汰会破坏 I 的组合，再保留 Hard-gate
```

### 14.1 本节自定义术语表

以下名称是本次 PE 工作中的局部命名，用来标识实验对象或处理步骤，不是模型或算法的标准名称。

#### `Production prompt`

指训练配置中实际使用的 `spad_teacher_evidence_status_answer_v2`。它不读取 gold，一次调用同时
完成证据充分性判断、S/I/A 状态判断和答案生成。在 PE registry 中的名称是 `baseline_current_v2`。

#### `Teacher-called`

指原始 Actor rollout 中 `teacher_called=true` 的轨迹。它代表训练时真的会产生 Teacher reward
或 Teacher fallback 的运行样本，是本轮 prompt 选择的第一指标切片。没有调用 Teacher 的样本称为
`non-called control`，只用于观察容易样本和分层漂移，不能替代主切片。

#### `Round`

`Round 1` 到 `Round 10` 是单 prompt dev 消融的时间顺序编号，不代表训练 step，也不代表模型版本。
每个 Round 绑定一个 prompt registry variant、一个结果目录和一次 cache-free 推理。

#### `Gold-aware`

指 prompt 的输入中包含 reference gold answer。Gold-aware 不等于允许模型相信 gold；R5 等策略
明确要求把 gold 当作待验证 hypothesis，只能在 evidence 支持时使用。

#### `Gold hypothesis`

指 gold 在 prompt 中的角色：它是需要被证据验证的候选答案，不是新的检索证据，也不能自动补齐
缺失的实体关系、谓词或多跳 bridge。

#### `Evidence-only`

本轮名称中的 `evidence_only` 不是“完全不包含 gold”。在 R5 的具体命名里，它表示 user layout
只保留 Original question、gold 和 title/passage evidence，并隐藏 sub-query 与 question tail；
R5 仍然是 gold-aware。

#### `Sub-query`

Actor 为检索器生成的查询字符串。它记录搜索历史，不一定等于用户真正的问题。隐藏 sub-query
是为了检验 Teacher 是否会把检索中间产物误当成最终问题。

#### `Question tail`

把 Original question 在 evidence 块之后再次重复一次的 user layout。它是布局变量，不是额外的
问题内容。本轮结果显示，在 gold-aware prompt 中重复 question tail 会伤害 teacher-called 指标。

#### `Answer alignment`

Round 2 的局部命名，表示在不改变 Round 1 user layout 的情况下，增加“答案类型、最短 passage span、
不要输出解释句”的生成指导。它只改变 system instruction，不改变输入证据。

#### `Gold layout 2x2`

一次布局因子实验，不是 2x2 模型。两个因子分别是“是否保留 sub-query”和“是否保留 question tail”，
四个角点是 Round 3、4、5、6。它的目的不是找四个独立最佳 prompt，而是隔离两个 user-layout 因子。

#### `R5`

Round 5 的简称，对应 `gold_support_evidence_only_v3`。它是四个 gold-layout 角点中 teacher-called
等权指标最好的单 prompt，并被后续组合策略用作 Stage B。

#### `R9`

Round 9 的简称，对应 `gold_binary_support_evidence_only_v3`。它在 R5 layout 上加入历史 gold-relation
binary support gate。R9 的单次 all-dev 分数较高，但三次重复的 teacher-called 等权指标低于 R5。

#### `Decoupled status/answer`

Round 7 的局部命名。prompt 试图让模型用一个 gold-hidden 的逻辑判断 S/I/A，再用另一段规则生成
gold-aligned answer。这里的“解耦”是 prompt instruction 上的角色解耦，不是两个真实模型或两个
独立服务。实际结果是 Stage 1 过度放宽，人工 I 被大量改成 S。

#### `I-guard`

Round 8 的局部命名，表示在 gold-aware prompt 中加入更严格的 I/missing-bridge 审计指令。它并不
等同于 Hard-gate，因为它仍由同一个 gold-aware prompt 自己决定 I，不能从结构上锁住 I 边界。

#### `Binary support`

Round 9 的局部命名，表示先问“gold 关系是否被 evidence 支持”的二元问题，再映射到 S/I/A。它是
prompt 内部的判断顺序，不是独立的分类器，也没有独立的概率 gate。

#### `Compact balanced`

Round 10 的局部命名，表示删除一部分 system prompt 文字，保留候选计数、gold 非证据和非 gold
答案规则。它用于测试 prompt 压缩的成本收益；实际造成 XML parse failure 和 I recall 回归。

#### `Stage A` 与 `Stage B`

组合策略中的两个连续模型调用阶段。Stage A 是 Production prompt，负责 evidence sufficiency 和
I/non-I 边界；Stage B 是 R5 或其他 gold-aware answer prompt，负责非 I 范围内的答案生成。它们不是
Actor 的训练 stage，也不是 SPAD 的 Stage1/Stage2 训练阶段。

#### `Verifier`、`Extractor` 与 `Selector`

这是对三类 Stage-B 职责的局部简称。`Verifier` 重新检查 gold 与完整问题关系，并同时给出 S/A
和答案；`Extractor` 接受 Stage-A non-I 已经绑定的前提，只负责抽取短答案；`Selector` 在 Stage-A
draft、reference gold 和其他 evidence 候选之间做选择。它们都使用同一个 GLM-4.7-Flash，不是三个
不同模型。

#### `C1` 到 `C6`

`C` 是 Composite ablation 的编号前缀。C1-C6 只表示组合策略实验的时间与推理顺序，和单 prompt
的 Round 1-10 分开，也不对应训练 step。

#### `I gate` 与 `I boundary`

本轮组合策略中的结构化约束。`I gate` 是决定“是否证据不足”的判断入口；`I boundary` 是最终 I 与
非 I 的分类边界。Hard-gate 中二者都归 Stage A 所有，Stage B 无权修改。

#### `Hard-gate`

指“前一阶段的 gate 结果具有绑定效力”的组合方式。Hard-gate v2 的绑定规则是：Stage-A I 直接
结束；Stage-A 非 I 才进入 Stage B；Stage B 无论返回 I、S 还是解析失败，都不能把 Stage-A 非 I
改成 I。

#### `Dual-all`

指 Stage B 对所有样本调用的组合方式。它与 Hard-gate 的主要区别是：即使 Stage A 判断 I，也会
继续调用 Stage B，并可能允许 Stage B 反转 I。Dual-all v2 在 holdout 上证实这种反转会破坏 I recall。

#### `Gold-F1 override threshold`

Dual-all v1/v2 使用的反转条件：Stage B 返回 S，且 Stage-B answer 对 reference gold 的 token-F1
至少为 `0.8`，才允许推翻 Stage-A I。这个阈值是本轮实验参数，不是训练 reward 的阈值；holdout
表明字符串相似度阈值不能保证完整关系被证据支持。

#### `Supported-answer chooser`

Hard-gate 的答案合并规则：如果 Stage A 和 Stage B 都返回 S，就选择对 gold token-F1 更高的答案；
如果只有一方返回 S，就保留唯一的 S。它只选择答案，不改变 I/non-I 标签边界。

#### `Fallback`

指下游 Stage B 返回 I、格式解析失败或没有可用答案时，保留上游 Stage-A non-I 输出。Hard-gate
使用 fallback 防止 Stage B 制造新的 I；它不是训练 reward 中的 Teacher fallback group。

#### `Evidence-literal gold canonicalization`

答案后处理规则的完整名称。若某个 reference gold 的规范化字面值确实出现在 title/passage evidence
中，并且替换当前答案能提高 Gold F1，才用该 gold 字符串替换答案。它不允许把 evidence 中不存在的
gold 注入答案，因此又称 `literal guard`。这一步可能让答案更接近数据集 gold、但远离人工答案措辞。

#### `Derived run` 与 `deterministic postprocess`

`Derived run` 是不重新请求模型、只对已经落盘的独立 Stage-A/Stage-B 输出应用合并规则的结果目录。
`deterministic postprocess` 是其中的确定性答案选择和 literal canonicalization。derived run 的
metadata 明确写入 `model_requests_this_derivation=0`，不能算作新的模型重复实验。

#### `Fresh` 与 `cache-free`

`Fresh` 表示该 run 重新向 vLLM 发起模型请求；`cache-free` 表示显式关闭 response cache 且
`cache_hits=0`。它们用于避免把历史输出复放误当成独立重复。temperature=0 仍不保证完整输出在
并发服务下逐字相同，因此需要三次重复。

#### `Reused-holdout diagnostic`

指 holdout 已经影响过策略选择后，再在同一 holdout 上运行其他候选得到的诊断结果。它可以暴露
失败模式，但不再是未触碰、无选择偏差的最终估计。本报告中的 Hard-gate holdout 属于这一类。

#### `Equal objective`

本轮用户指定的第一选择分数：`0.5 * I_F1 + 0.5 * Gold F1 coverage on manual-S`。它把 I 判别
和 gold answer 命中视为同等重要，但不能取代 manual-answer F1、parse rate 和 holdout 限制。

#### `Manual-answer F1`

使用人工标注者根据 evidence 写出的答案计算 token-F1。它不参与 runtime 决策，只用于发现“模型更
接近数据集 gold，但不一定更接近当前 evidence 的最佳表述”的偏移。

### 14.2 第一阶段：先固定数据和评价口径

1. 从 5,056 个 group、40,448 条 rollout 中每组抽一条轨迹。
2. 按 step 每 20 步分成四层，每层抽 128 条，得到 512 条样本。
3. 只依据 Original question 和 evidence 完成 S/I/A 人工标注，防止 gold 或历史 Teacher 反向
   污染标签。
4. 用种子 `260715` 固定为 384 dev、128 holdout；dev 用于 prompt 选择，holdout 只在选择冻结后打开。
5. 确定 teacher-called 为主切片，定义 I F1、Gold F1 覆盖、manual-answer F1 和等权 objective。

这一步的思路是先把“模型是否会判断证据不足”和“答案是否命中数据集 gold”拆成可独立观察的
指标，否则单看 answer EM/F1 无法知道模型是答错了，还是错误地把 I 判成了 S。

### 14.3 第二阶段：无 gold 旧成功路径复现

#### Round 1：`baseline_question_tail_evidence_only_v2`

思路是先复现此前 237 样本 PE 中胜出的 no-gold question-tail 布局，作为新数据的迁移基线。它包含
evidence-only 输入并在尾部重复 question，但第一次运行有 1 条 parse failure。

结果：teacher-called I F1 `0.8873`，Gold F1 `0.2951`，等权 `0.5912`。

#### Round 2：`question_tail_answer_alignment_v3`

只改 system instruction，增加答案类型识别、最短 passage span 和 canonical answer 指导，保持
Round 1 user layout 字节级不变。

结果：teacher-called I F1 `0.8897`，Gold F1 `0.2785`，等权 `0.5841`。虽然 all-dev objective
提高到 `0.7342`，但主切片下降，因此没有沿着 answer-alignment 方向继续加规则。

**阶段结论：** no-gold prompt 的 I 能力可迁移，但新目标的 Gold F1 瓶颈不能只靠要求“答案更短、
更 canonical”解决。

### 14.4 第三阶段：gold 输入与布局因子消融

这一阶段引入 reference gold，同时系统改变 sub-query 和 question tail 两个 user-layout 因子。

| Round | 策略与实际改变 | Teacher-called I F1 | Gold F1 | 等权 | 结论 |
|---|---|---:|---:|---:|---|
| R3 | `gold_support_question_tail_v3`：gold + 无 sub-query + 有 question tail | 0.8531 | 0.5546 | 0.7039 | Gold 有收益，但 I 下降明显 |
| R4 | `gold_support_check`：gold + 有 sub-query + 无 question tail | 0.8451 | 0.6303 | 0.7377 | 比 R3 好，但 sub-query 仍可能干扰 |
| R5 | `gold_support_evidence_only_v3`：gold + 无 sub-query + 无 question tail | 0.8660 | 0.6411 | **0.7535** | 第一轮单 prompt 主切片领先者 |
| R6 | `gold_support_subquery_question_tail_v3`：gold + 有 sub-query + 有 question tail | 0.8369 | 0.5566 | 0.6968 | 两个干扰因素同时存在，淘汰 |

这四个 Round 构成上文所说的 `Gold layout 2x2`：

- 隐藏 sub-query 有利于 Teacher 回到 Original question。
- 重复 question tail 在 gold-aware 设置下有害。
- 最佳布局是 question + gold + full evidence，且不放 sub-query、不重复 question tail。

Round 6 还离线分析过一种两调用拼接：用 no-gold prompt 的 status 决定 I/non-I，再取 gold-aware
prompt 的答案作为 non-I 输出。该拼接的 teacher-called 等权只有 `0.7155`，低于 R5，因此当时
没有继续扩大多调用预算。之后用户明确允许最多 `2x`，才进入后续结构化组合消融。

### 14.5 第四阶段：gold-aware system instruction 消融

布局固定为 R5，只改变 system instruction，测试不同的“如何使用 gold”思路：

| Round | 策略思路 | Teacher-called I F1 | Gold F1 | 等权 | 主要失败 |
|---|---|---:|---:|---:|---|
| R7 | `gold_decoupled_status_answer_v3`：prompt 内分离 status 判断与答案生成 | 0.7265 | 0.7327 | 0.7296 | 过度放宽，77 个人工 I 被改成 S；另有 parse failure |
| R8 | `gold_i_guard_evidence_only_v3`：加入更长、更严格的 missing-bridge 审计 | 0.7438 | 0.6636 | 0.7037 | 规则变长但仍过度接受，漏掉 70 个 manual-I |
| R9 | `gold_binary_support_evidence_only_v3`：先做 gold relation 二元支持判断 | 0.8600 | 0.5889 | 0.7244 | all-dev 曾领先，但 A recall 为 0，主切片低于 R5 |
| R10 | `gold_compact_balanced_v3`：压缩 R5 system prompt | 0.8339 | 0.5381 | 0.6860 | parse rate 降至 0.9688，出现 12 个格式失败 |

这一步的思路是判断性能问题来自“gold 的存在”还是“gold 使用说明的写法”。结果表明：

- status/answer 的显式 prompt 解耦会让模型把“能够猜到答案”误当成“证据足够”。
- 增加更多 missing-bridge 文字不能保证模型执行结构化检查。
- binary gate 在单 prompt 内仍不是结构性约束，不能保证 I 边界。
- 过度压缩 system prompt 会损害 XML 遵循和 I recall。

### 14.6 第五阶段：重复运行与单 prompt 冻结

对最有希望的 R5 和 R9 各做两次独立 cache-free fresh dev repeat：

| 候选 | 三次 teacher-called I F1 | Gold F1 | 等权 | 平均 wall time |
|---|---:|---:|---:|---:|
| R5 | 0.8651 | 0.6399 | **0.7525** | 35.89 秒 |
| R9 | 0.8594 | 0.5811 | 0.7202 | 38.68 秒 |

R9 的单次 all-dev objective 曾高于 R5，但重复后主切片明显落后。因此冻结 R5 作为后续组合中
的 Stage-B answer prompt，并打开原本封存的 128 holdout 做 R5 三重复：teacher-called I F1
`0.8632`、Gold F1 `0.7667`、等权 `0.8149`。

### 14.7 第六阶段：在允许 2x 预算后做组合消融

用户将 Teacher 推理预算上限放宽到单 prompt 的 2 倍。这里不再尝试把所有职责塞进一个 prompt，
而是把“证据充分性判断”和“gold-aware 答案生成”拆为两个有明确权限边界的阶段。

#### C1：Hard-gate + R5 verifier

Stage A 使用新训练实际 Production prompt。若 Stage A 判 I，直接结束；若判 S/A，调用 R5。
这首先验证“保留 I gate、只增加 answer stage”是否可行。

结果：teacher-called I F1 `0.8922`，Gold F1 `0.5770`，等权 `0.7346`，预算 `1.3524x`。
I F1 与 Stage A 完全相同，说明硬性 Stage-A I gate 能保住判别边界。

#### C2：Hard-gate + dedicated gold extractor

Stage B 不复用 R5，而是建立一个专门的 gold-aware extractor，要求在 non-I 前提下抽取最短、最
接近 gold 的 evidence span。目的在于测试“专门答案抽取器”是否优于通用 R5。

结果：I F1 仍为 `0.8922`，但 Gold F1 降到 `0.4587`，等权 `0.6755`，预算 `1.4146x`。
专门 extractor 的指令更长、更强调候选格式，反而弱于已验证的 R5，因此淘汰。

#### C3：Hard-gate + draft/gold selector

Stage B 被要求比较 Stage-A draft、reference gold 和 evidence 中的其他候选，选择一个完整支持的
答案。目的在于测试“显式候选选择”是否能解决模型输出描述句或非 canonical span 的问题。

结果：I F1 `0.8922`，Gold F1 `0.3852`，等权 `0.6387`，预算 `1.5019x`。让模型同时做候选
比较、关系校验和答案生成增加了认知负担，淘汰。

#### C4：Dual-all + Gold-F1 0.8 override

Stage B 对所有样本调用 R5；如果 Stage B 返回 S 且答案对 gold 的 token-F1 至少 `0.8`，允许
它推翻 Stage-A I。思路是恢复 Stage A 可能漏掉的 gold answer。

三次 dev 原始 dual-all 均值：I F1 `0.8812`，Gold F1 `0.6283`，等权 `0.7547`，预算
`1.7904x`。I 已经比 Production 下降，说明 override 不是无代价的。

#### C5：Dual-all + 确定性答案后处理

在 C4 的独立输出上增加两项不再请求模型的后处理：

1. 两个阶段都为 S 时选 gold token-F1 更高的答案。
2. evidence 中出现 reference gold literal 且能提高 Gold F1 时，替换为该 gold。

这得到 dev teacher-called I F1 `0.8812`、Gold F1 `0.7220`、等权 `0.8016`，预算仍为
`1.7904x`。它在 dev 上看起来最好，但三次 holdout 的 I recall 降至 `0.6993`、I F1 降至
`0.8045`，Gold F1 虽有 `0.9222`，却掩盖了严重的证据充分性错误。C5 淘汰。

#### C6：Hard-gate v2

吸收 C5 的答案择优和 evidence-literal 规范化，但撤销“Stage B 可以推翻 I”的权限：

- Stage-A I 不调用 Stage B。
- Stage-A S/A 才调用 R5。
- Stage-B I 或格式失败时回退 Stage A，不能产生新 I。
- Gold 只参与 non-I 答案选择，不能参与 I gate。

三次 dev teacher-called 均值：I F1 `0.8924`、Gold F1 `0.6825`、等权 `0.7874`、预算
`1.3558x`。它比 C5 的 Gold F1 略低，但保住了 Production I，成为当前组合候选。

### 14.8 为什么最终不是“让 gold 直接控制 I”

消融过程形成了一个明确的因果链：

```text
gold literal 命中
        !=
问题要求的完整 predicate 被 evidence 支持
        !=
I gate 可以被安全推翻
```

Dual-all 的 holdout 失败给出了直接证据：很多人工 I 样本包含与 gold 相同的字符串，但缺少问题
要求的关系 bridge。若让 gold token-F1 threshold 直接修改 I，Gold F1 会上升，I recall 会下降。
因此最终职责拆分为：

| 职责 | 唯一负责人 | gold 是否可用 |
|---|---|---|
| 判断是否缺证据 | Production Stage A | 否 |
| 判断 non-I 内的 S/A 和答案表述 | R5 Stage B | 是 |
| 答案选择 | Hard-gate deterministic merge | gold 只作择优参考 |
| literal canonicalization | evidence literal guard | 必须先在 evidence 中出现 |

### 14.9 本轮消融产生的工程决策

1. 单 prompt 继续堆 instruction 不是主要突破方向。R7/R8/R10 说明更长或更紧的文字不能稳定修复
   gold-aware prompt 的 I 漂移。
2. R5 的价值主要在答案生成，不应让它接管 I gate。
3. Hard-gate v2 是当前最符合“保住 I、提高 Gold F1、预算小于 2x”的结构。
4. `manual-answer F1` 必须继续保留。Hard-gate 的 holdout Gold F1 为 `0.9000`，但 manual-answer
   F1 为 `0.5267`，说明 gold 命中与 evidence 文字一致性不是同一个目标。
5. 当前 holdout 已参与组合策略判断，Hard-gate holdout 结果只能称为诊断，不能作为新的未触碰最终
   估计。下一步应从新的训练 rollout 抽样，不能反复使用 3500e 调 prompt。

### 14.10 复现与审计产物

消融过程的原始证据全部保存在：

```text
pipelines/formal/agenticIterRag/teacher_PE
```

关键位置：

- `260715_NEW_DATA_PE_WORKLOG.md`：按时间记录每个 Round、失败启动、稳定性重复和组合过程。
- `NEW_DATA_RESULTS_INDEX.md`：每个结果目录的 split、I 指标、Gold F1、manual-answer F1、预算和
  cache/error 状态。
- `NEW_DATA_STABILITY.md`：Production、R5、R9、Hard-gate、Dual-all 的三次均值与范围。
- `prompt_variants.py`：单 prompt registry 和 user-layout 因子。
- `composite_prompt_variants.py`：Stage-B 组合策略 registry。
- `run_ablation.py`：单 prompt 请求、XML parse 和评分。
- `run_composite_ablation.py`：Stage A/B 推理、gate、答案合并和预算审计。
- `derive_composite_policy.py`：对独立 Stage 输出做不新增模型请求的确定性复算。

本节的最终结果不是“某个 prompt 在一次 dev 上最高”，而是从单调用布局因子、gold 使用方式、
稳定性重复、组合权限边界和 holdout 失败模式逐步得到 Hard-gate v2 的过程。
