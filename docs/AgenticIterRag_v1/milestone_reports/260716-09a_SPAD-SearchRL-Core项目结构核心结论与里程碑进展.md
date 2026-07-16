# SPAD-SearchRL Core：项目结构、核心结论与里程碑进展

日期：2026-07-16，09am（北京时间）

> 状态：本文汇总截至 2026-07-16 09 时已有的正式训练、3500e 评估、Teacher PE、代码结构与
> 工程运行经验。当前已经完成实验验收的主方案是 `SPAD-SearchRL Core V3 / PostNorm 0.1`；
> `Boundary-Locked Teacher v2` 已进入正式训练代码与配置，但尚未完成 production smoke、5100
> 正式训练和 3500e 评估，因此仍是候选方案，不是新的已验证默认方案。

## 1. 里程碑结论

当前项目已经从“验证 SPAD 三阶段流水线能否跑通”，推进到“围绕一个真正有效的 Search Agent
RL 核心进行 reward、Teacher 和训练稳定性优化”。现阶段最重要的判断如下：

1. SPAD 的三阶段框架应继续保留，但当前有明确大规模效果证据的是 Stage1
   `search_policy_rl`。Stage2 `answer_refresh_data` 和 Stage3 `answer_distillation` 已经工程跑通，
   尚未证明在现有数据与训练方式下能带来稳定的增量收益。
2. 本文将当前有效的 Stage1 算法框架命名为 **`SPAD-SearchRL Core`**。这个名字描述它在 SPAD
   中的语义角色，不把整个项目永久降格为单阶段，也避免后续 Stage2/Stage3 恢复有效时重新改名。
3. 当前已验收配方是 **`SPAD-SearchRL Core V3 / PostNorm 0.1`**，代码 reward 名称为
   `spad_em_teacher_backoff_gold_token_f1_bonus_v3`。它在固定 3500e 上达到 EM `0.1994`、
   F1 `0.2787`、完整答案率 `0.8340`，是现有七组同协议结果中综合表现最好的工作点。
4. Teacher fallback 组不能与 Actor EM 命中组获得相同的标准化后梯度权重。将 Teacher 组的
   post-norm scale 从 `0.1` 提高到 `0.3`，没有获得显著的 EM/F1 改善，却使完整答案率下降
   `0.1240`，重复查询率提高 `0.4078`，因此 `0.3` 不应晋升为默认值。
5. 当前 reward 的下一个主要瓶颈不是再调一个连续 scale，而是 Teacher 同时承担“证据是否充分”
   和“答案是否贴近 gold”时的目标冲突。Teacher PE 表明，锁住证据充分性边界、只在非 I 路径
   做 gold-aware 答案对齐，是目前更有希望的方向。
6. `Boundary-Locked Teacher v2` 在 Teacher PE 主切片保持 Production 的 I F1 `0.8924`，同时将
   Gold F1 覆盖从 `0.3180` 提高到 `0.6825`，平均推理预算为 `1.3558x`。生产集成代码已经存在，
   但只有完成新的未触碰样本验证、单副本 smoke、正式训练和 3500e 评估后，才能判断它是否真的
   改善 Actor policy。
7. 当前结论来自单个正式训练 seed。3500e paired bootstrap 可以衡量同一批问题上的 checkpoint
   差异，但不能替代跨 seed 重训；因此不能把“V3 在本次 run 最好”扩大为“训练策略已证明跨 seed
   稳定最优”。

## 2. 本文术语

为避免把工程 stage 名、算法语义和具体配置版本混在一起，本文采用以下术语。

| 本文术语 | 含义 | 对应代码或配置 |
|---|---|---|
| **SPAD** | 保留的多阶段 Search Policy and Answer Distillation 总框架 | `train_agent.impl=spad_rag` |
| **SPAD-SearchRL Core** | 当前真正有效的搜索策略强化学习核心，即 SPAD Stage1 | `search_policy_rl` |
| **Actor-First Backoff Reward** | Actor EM 优先；整组 Actor EM 全零时才启用 Teacher 回退 | `spad_em_teacher_backoff*` reward 家族 |
| **Post-Norm Difficulty Weight** | 组内 GRPO 标准化后，对 Teacher fallback 困难组整体缩放 advantage | `teacher_group_postnorm_scale` |
| **Core V3 / P01** | 当前已验收主方案，V3 reward、post-norm scale `0.1` | `spad_em_teacher_backoff_gold_token_f1_bonus_v3` |
| **Boundary-Locked Teacher v2** | Stage A 锁住 I/non-I 边界，Stage B 只做非 I 答案对齐 | `spad_teacher_hard_gate_r5_literal_canonical_v2` |

`SPAD-SearchRL Core` 是框架名称，`Core V3 / P01` 是当前具体配方。未来 reward 进入 V4，或 Teacher
策略替换后，仍可沿用同一个框架名称。

## 3. 项目结构

### 3.1 两层 stage 必须区分

项目中存在两种不同层级的 stage，不能把它们混称为同一条三阶段 pipeline。

第一层是 AgenticIterRag（AIR）的外层实验 pipeline：

```text
train_agent
    -> generate_traces
    -> build_reranker_dataset
    -> train_llm_reranker
    -> infer_matrix
```

第二层是 `train_agent.impl=spad_rag` 内部保留的 SPAD 三阶段：

```text
SPAD Stage1  search_policy_rl
    -> Stage2 answer_refresh_data
    -> Stage3 answer_distillation
```

当前所有 Gold Token-F1 V2/V3 正式实验都通过 AIR 单入口进入 `train_agent`，但只选择 SPAD Stage1，
并显式关闭 Stage2/Stage3。这不等于删除多阶段框架，只表示当前默认实验面已经收敛到
`SPAD-SearchRL Core`。

### 3.2 AIR 外层职责

AIR 使用单 launcher：

```text
scripts/agenticIterRag_v1/01_pipeline_launcher.sh
```

外层 pipeline 负责：

- 合并 base config、resource config 和实验 overlay。
- 生成 final config、execution plan、stage resource plan 和 manifest。
- 根据 `resume_from_stage`、`stop_after_stage`、`skip_stages` 选择本次执行节点。
- 在顺序 stage 之间复用 NPU，并将上游 checkpoint、trace 和数据 manifest 传给下游。
- 统一组织日志、checkpoint、outputs 和评估产物。

AIR 参考过 CoAgenticRetriever v2 的 launcher/compiler 分层与 trace 契约，但当前运行时不调用 CAR
代码链路。旧目录或 checkpoint 路径中出现 `CAR`/`coAgenticRetriever` 只反映历史命名。

### 3.3 SPAD 内层职责

| SPAD 阶段 | 设计职责 | 当前证据状态 |
|---|---|---|
| Stage1 `search_policy_rl` | Actor 生成搜索轨迹与最终答案，使用 VERL GRPO 更新搜索策略 | **当前有效核心；已有 5100 训练和 3500e 验证** |
| Stage2 `answer_refresh_data` | 用 Stage1 checkpoint 重新 rollout，Teacher 标注并生成训练 pair | 工程已跑通；是数据生产阶段，不直接构成收益证明 |
| Stage3 `answer_distillation` | 使用 Stage2 数据进行 DPO/规划中的其他蒸馏训练 | 工程已跑通；现有 500 样本 DPO 未证明明确增益 |

2026-07-10 的 500 样本三阶段运行中，Stage1 完成 7 步 GRPO，Stage2 从 478 条有效轨迹生成
252 个 DPO pair，Stage3 完成 4 个 DPO 优化步。350 样本评估中，Stage3 最终模型相对基础模型
仅从 F1 `0.2512` 变为 `0.2519`，EM 同为 `0.1543`，且低于 Search-R1 original 的
EM/F1 `0.1714/0.2665`。因此 Stage2/Stage3 目前应视为“保留的研究能力”，不能写成已经有效的
主训练策略。

### 3.4 代码边界

| 路径 | 主要职责 |
|---|---|
| `AgenticIterRag/agentic_iter_rag/pipeline` | AIR pipeline runner、stage 调度和状态传递 |
| `AgenticIterRag/agentic_iter_rag/agent_training/spad` | SPAD 三阶段编排、Stage1 reward 路由、Teacher 策略与 manifest |
| `AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards` | stable、Gold Token-F1 V1/V2/V3 及候选 Hard-Gate reward |
| `AgenticIterRag/verl` | GRPO/PPO 训练、Ray worker、FSDP checkpoint 和 post-norm advantage 缩放 |
| `AgenticIterRag/agentic_iter_rag/trajectory` | 完整搜索轨迹的数据契约与处理 |
| `AgenticIterRag/agentic_iter_rag/reranker_dataset` | trace 到 reranker 数据集的转换 |
| `AgenticIterRag/agentic_iter_rag/reranker_training` | LLM reranker 训练链路 |
| `AgenticIterRag/agentic_iter_rag/infer_matrix` | 多配置推理矩阵与评估执行 |
| `AgenticIterRag/agentic_iter_rag/metrics` | EM、token-F1 和行为指标 |
| `tasks/train_tasks/agenticIterRag` | 正式训练 wrapper 与实验 overlay |
| `tasks/eval_tasks/agenticIterRag` | 固定协议评估入口与多模型编排 |
| `pipelines/formal/agenticIterRag/teacher_PE` | Teacher prompt/组合策略的独立 PE harness |

### 3.5 一次 Core 训练的数据流

```text
Question + Gold
      |
      v
Actor rollout x 8 ----> Recall/Search service ----> 累计 evidence ----> Actor answer
      |                                                        |
      |                                                        v
      +---- Actor EM group gate --------------------------> Teacher（仅 fallback 组）
                                                               |
                                                               v
             raw reward -> group normalize -> difficulty weight -> VERL policy update
```

训练中 Actor、Recall 和 Teacher 是不同服务角色。资源绑定来自 selected stage 的 resource plan，而
不是全局固定绑卡。正式产物至少应包括 final config、execution plan、stage manifest、逐 step rollout、
VERL checkpoint 和可直接评估的 HF safetensors。

## 4. SPAD-SearchRL Core 的当前算法

### 4.1 训练对象

当前正式设置以 Qwen3-1.7B Base 为 Actor，5100 条训练数据中按 79 steps、每 step 64 个问题组
运行；实际使用 `79 x 64 = 5056` 个 prompt slot。每题采样 8 条 rollout，共 40,448 条训练轨迹。
Actor rollout 使用随机采样，Teacher 使用 GLM-4.7-Flash 的确定性解码设置。

### 4.2 Actor-First Backoff Reward

对同一问题的 8 条 rollout 组成的组 `G`，先计算每条轨迹的 Actor EM：

```text
e_i = EM(actor_answer_i, gold_answers)  # 0 或 1
```

reward 有严格的组级优先级。

当组内至少一条 Actor answer 命中 EM 时：

```text
r_i = e_i
```

此时只使用 Actor 自己的可验证命中，不让 Teacher 改写该组排序。

当整组 Actor EM 全为 0 时，才进入 Teacher fallback：

```text
base_i = 0.1 * I(Teacher status in {supported_answer, ambiguous_evidence})
bonus_i = 0.1 * token_F1(Teacher answer, gold)
r_i = base_i + bonus_i
```

`bonus_i` 还要求 Actor 输出合法闭合的 `<answer>...</answer>`、Teacher 成功解析且状态合格。
Teacher 请求失败、格式失败、无合法 status 或 `insufficient_evidence` 不产生正向 backoff。这里的
Gold Token-F1 比较对象是 Teacher answer 与 reference gold，不是直接奖励错误 Actor answer 的
字符串相似度。

这一结构的核心动机是：

- 容易组已有 Actor EM，可使用强、无歧义的 0/1 监督。
- 困难组 8 条 rollout 可能全部 EM=0，如果完全置零，该问题对 GRPO 没有区分信号。
- Teacher 使用实际可见 evidence 提供稠密但可信度较低的排序信号，避免困难题全部掉出训练。

### 4.3 Post-Norm Difficulty Weight

V3 恢复 `norm_adv_by_std_in_grpo=true`。每个问题组先完成标准化：

```text
z_i = (r_i - group_mean) / (group_std + 1e-6)
```

然后按 reward 来源对整组统一缩放：

```text
A_i = 1.0 * z_i  # 组内存在 Actor EM 命中
A_i = 0.1 * z_i  # 整组 Actor EM 为 0，使用 Teacher fallback
```

本文将这个 `0.1` 称为 **Post-Norm Difficulty Weight**。它不是 raw reward 系数，而是困难组在
完成组内归一化后获得的整体优化预算。缩放必须满足：

- 同一 UID 的 8 条 rollout 使用相同 scale。
- scale 为正有限值。
- 只有 `norm_adv_by_std_in_grpo=true` 时允许配置 post-norm scale key。
- 保留缩放前后 advantage，支持逐条审计。

这个顺序解决了一个关键问题：如果只把 raw Teacher reward 设得很小，组内 std 标准化仍可能把
微小差异重新放大到与 Actor EM 组相同的梯度量级。V3 将“组内谁更好”和“这一组整体应占多少训练
预算”拆成两个独立问题。

### 4.4 当前正式配方

| 配置项 | 当前值 |
|---|---|
| Reward | `spad_em_teacher_backoff_gold_token_f1_bonus_v3` |
| Stable backoff | `0.1` |
| Gold Token-F1 bonus weight | `0.1` |
| Teacher group post-norm scale | **`0.1`** |
| GRPO std normalization | `true` |
| Rollout per prompt | `8` |
| Batch / steps | `64 / 79` |
| Data seed | `42` |
| `stream_group_max_inflight` | `2` |
| SPAD Stage2 / Stage3 | `disabled / disabled` |

正式 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/
spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_stage1_overlay.yaml
```

## 5. 关键进展

### 5.1 从三阶段跑通到 Stage1 聚焦

早期 500 样本实验完成了 SPAD Stage1/2/3 的端到端工程验证，但最终 DPO 模型没有获得明确的
精度增益。这个结果促使项目将主要实验预算转向 Stage1：先把搜索策略本身的 reward、终止格式、
Teacher fallback 和训练调度做正确，再讨论后续数据刷新与蒸馏。

这不是放弃多阶段框架，而是把“工程可执行”和“算法有效”分开验收。当前 Stage2/Stage3 的再次启用
条件应是：相对同一个 Stage1 checkpoint，有独立且可复现的增量，而不是只看 DPO loss 是否下降。

### 5.2 Stable reward：解决全零困难组

`spad_em_teacher_backoff` 建立了 Actor EM 优先、Teacher 只回退全零组的基本框架。它证明 Teacher
信号可以帮助 Stage1 超过单纯 Actor EM 的 Search-R1 基线，但也暴露了搜索变长、重复查询偏高的
行为副作用。

### 5.3 Gold Token-F1 V1/V2：改善 Teacher 答案对齐

Gold Token-F1 分支在 Teacher 状态奖励之上增加答案覆盖信号。V1 的资格规则与标准化组合不理想，
V2 修正 eligibility 并关闭 std normalization 后，完整答案率明显改善，但 EM 仍没有超过现有最佳。
这说明“加入更细 reward”本身不够，reward 在 GRPO 标准化后的实际梯度权重同样关键。

### 5.4 V3/P01：当前里程碑基线

V3 保留 V2 raw reward，恢复组内 std normalization，并在标准化后把 Teacher fallback 组乘 `0.1`。
正式 5100 训练完成 79 steps，自动保存并导出 HF checkpoint；固定 3500e 评估 3500/3500 成功。

相对 Search-R1 5100：

- EM `+0.0194`，95% CI `[0.0094, 0.0297]`。
- F1 `+0.0277`，95% CI `[0.0169, 0.0387]`。
- 完整答案率 `+0.1023`，95% CI `[0.0863, 0.1183]`。

相对 Gold Token-F1 V2：

- EM `+0.0163`，95% CI `[0.0077, 0.0249]`。
- F1 `+0.0114`，95% CI `[0.0022, 0.0208]`。

相对 stable 5100 的 EM/F1 点估计为 `+0.0071/+0.0086`，但置信区间跨 0，因此不能声称
V3/P01 已显著优于 stable 的精度；V3/P01 更确定的优势是完整答案率和搜索行为。

### 5.5 P03 消融：困难组权重不是越大越好

P03 只把 Teacher fallback 组的 post-norm scale 从 `0.1` 改为 `0.3`，其余 reward、数据、seed、
rollout 数与归一化配置保持一致。结果显示 raw reward 层面变化很小，但推理行为显著恶化。

这支持一个当前工作假设：Teacher fallback 组包含 Actor 尚未解决的困难问题，也包含 Teacher 噪声、
不充分证据和搜索策略失败。统一提高这些组的梯度预算，会同时放大有用信号和“继续搜索也许会有答案”
的错误倾向，最终表现为更长、更重复、更容易耗尽 turn budget 的搜索。

该解释与单 seed 结果一致，但还不是跨 seed 因果定律。当前决策只需要更保守的结论：`0.3` 没有
显示出可接受的收益风险比，默认值继续使用 `0.1`。

## 6. 固定 3500e 横向结果

所有行使用同一 3500e、no-ranker 搜索协议、temperature=0、Recall topN=50、Actor 可见 topM=5、
最多 6 个 assistant turns 和完整 trace。表中每行只有一个训练 checkpoint，不把确定性重复推理
当作独立训练样本。

| 模型 | 关键训练语义 | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Search-R1 512 | Actor EM；norm=true；512 条 | 0.1180 | 0.1965 | 0.6271 | 2.3489 | 0.3640 | 0.2569 |
| Search-R1 5100 | Actor EM；norm=true；5100 条 | 0.1800 | 0.2509 | 0.7317 | 1.7291 | 0.1786 | 0.1549 |
| SPAD stable 5100 | Actor-first Teacher backoff；norm=true | 0.1923 | 0.2700 | 0.7220 | 2.6557 | 0.5906 | 0.2443 |
| Gold Token-F1 V1 5100 | V1 eligibility；norm=true | 0.1837 | 0.2576 | 0.6334 | 3.0071 | 0.5763 | 0.3589 |
| Gold Token-F1 V2 5100 | V2 eligibility；norm=false | 0.1831 | 0.2673 | 0.7906 | 1.8889 | 0.2154 | 0.1863 |
| **Core V3 / P01 5100** | **V2 raw reward；norm=true；Teacher post-norm x0.1** | **0.1994** | **0.2787** | **0.8340** | **1.6969** | **0.1571** | **0.1369** |
| Core V3 / P03 5100 | V2 raw reward；norm=true；Teacher post-norm x0.3 | 0.1929 | 0.2734 | 0.7100 | 2.6883 | 0.5649 | 0.2714 |

P03 相对 P01 的 paired bootstrap：

| 指标 | P03 - P01 | 95% CI | 判断 |
|---|---:|---:|---|
| EM | -0.0066 | [-0.0171, 0.0043] | 区间跨 0 |
| F1 | -0.0053 | [-0.0167, 0.0062] | 区间跨 0 |
| 完整答案率 | **-0.1240** | **[-0.1403, -0.1080]** | 明确恶化 |
| 平均搜索数 | +0.9914 | - | 明显增加 |
| 重复查询率 | +0.4078 | - | 明显增加 |
| Max-turn 率 | +0.1345 | - | 明显增加 |

因此，项目的主指标不能只保留 EM/F1。完整答案率、重复查询率、平均搜索数和 Max-turn 率应继续
作为与精度并列的 policy quality gate；否则一个更容易反复搜索的模型可能在小幅精度波动下被错误
判断为等价方案。

## 7. Teacher 研究结论

### 7.1 Teacher 的两个目标存在张力

在 fallback 轨迹上，Teacher 实际承担两个目标：

1. 判断当前累计 Search evidence 是否足以支持一个完整答案，即 I/non-I 边界。
2. 在证据充分时生成尽量贴近数据集 reference gold 的短答案，以提供 Token-F1 区分信号。

Production prompt 不读取 gold，证据边界稳定但 Gold F1 覆盖偏低；gold-aware R5 能显著提高答案
覆盖，却会受 gold 诱导而降低 I 判别。这说明两个目标不宜继续完全塞进一个 prompt。

### 7.2 Teacher PE 数据与主切片

最新 PE 从 V3/P03 正式训练的 40,448 条轨迹中，按训练 step 分层抽取 512 个问题组，每组只取一条
代表轨迹。人工 S/I/A 标签分布为 `241/241/30`，冻结为 384 dev 和 128 holdout。

主指标只看 dev 中训练时真实 `teacher_called=true` 的 221 条困难样本。全 dev 还包含 163 条较容易
的 non-called control，不能用 control 的高分掩盖困难样本表现。

### 7.3 Boundary-Locked Teacher v2

本文将 PE 中的 Hard-Gate v2 语义命名为 **Boundary-Locked Teacher v2**。它不是简单串联两个
Teacher，而是把两类决策权限拆开：

```text
Stage A: Production prompt(question + evidence)
    |
    +-- I --------------------> 直接返回 I；不调用 Stage B
    |
    +-- S/A --> Stage B: R5(question + gold + evidence)
                     |
                     +--> 只能处理非 I 内部状态与答案
                     +--> 不得把 Stage-A non-I 改为 I
                     +--> supported answer 按 Gold Token-F1 择优
                     +--> gold 字面值必须真实存在于 evidence 才能规范化
```

其硬不变量是：

```text
Final is insufficient_evidence <=> Stage A is insufficient_evidence
```

因此 gold 只能参与答案对齐，不能参与证据是否充分的最终边界。

### 7.4 PE 结果

在 teacher-called dev 221 条样本上的三次 cache-free fresh 推理均值：

| 策略 | I Precision | I Recall | I F1 | Gold F1 覆盖 | 人工答案 F1 | 等权指标 | 推理预算 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Production | 0.8970 | 0.8881 | **0.8924** | 0.3180 | **0.6133** | 0.6052 | 1.0000x |
| 单 prompt R5 | 0.8287 | **0.9051** | 0.8651 | 0.6399 | 0.4408 | 0.7525 | 1.0000x |
| **Boundary-Locked Teacher v2** | 0.8970 | 0.8881 | **0.8924** | **0.6825** | 0.4833 | **0.7874** | 1.3558x |

128 holdout 中 teacher-called 的 72 条诊断结果同样有利：Boundary-Locked Teacher 保持
Production I F1 `0.8812`，Gold F1 达到 `0.9000`，等权指标 `0.8906`，预算 `1.2307x`。
但该 holdout 后续参与过组合策略淘汰，因此已经是 reused-holdout diagnostic，不能继续称为未触碰的
最终估计。

### 7.5 已淘汰方向

Dual-All v2 允许 gold-aware Stage B 在高 Gold Token-F1 时推翻 Stage-A I。它在 holdout 上虽然
Gold F1 达到 `0.9222`，但 I recall 降到 `0.6993`、I F1 降到 `0.8045`。根因是 passage 中出现
gold 字面值并不等于证据支持问题要求的实体、谓词、scope 或多跳 bridge。

由此形成一条应冻结的算法原则：

> Reference gold 可以约束答案表示，但不能替代 evidence sufficiency 判断，也不能推翻由
> gold-hidden evidence judge 给出的 I 边界。

## 8. 当前生产集成状态

截至本文时间点，Boundary-Locked Teacher v2 已从 PE 结论推进到生产代码集成，当前工作区已经包含：

- Teacher 策略注册与冻结语义：
  `AgenticIterRag/agentic_iter_rag/agent_training/spad/teacher_strategies.py`。
- 独立 reward：
  `search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2.py`。
- Stage1 reward 路由、策略 ID 传播与配置校验。
- Stage A I 锁定、Stage B 失败回退、supported answer 择优、evidence literal guard 和审计字段测试。
- 64 条 smoke overlay/wrapper 与 5100 正式 overlay/wrapper。

候选 reward 的精确名称为：

```text
spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2
```

候选配置继续使用 V3/P01 的 reward 权重和 Post-Norm Difficulty Weight：

```yaml
partial_reward: 0.1
gold_token_f1_bonus: 0.1
teacher_group_postnorm_scale: 0.1
norm_adv_by_std_in_grpo: true
```

但以下证据尚未在当前工作报告中出现，因此不能视为已经完成：

1. 当前 production 集成测试在目标训练环境中的完整通过记录。
2. 64 条、1-step 单副本真实 smoke 的完整 reward/scale/Teacher 审计。
3. Stage-B 调用率、队列等待和单副本训练 wall time 实测。
4. 新的未触碰 Teacher 验证样本。
5. 5100 正式训练、HF checkpoint 自动导出和 3500e Actor policy 评估。

这一区分很重要：PE 证明的是 Teacher 层联合指标更好；只有正式训练与 Actor 评估才能证明更好的
Teacher 信号会转化为更好的 search policy。

## 9. 工程成熟度与风险

### 9.1 已具备能力

- 单入口配置编译，支持 base/resource/overlay 分层。
- final config、execution plan、stage manifest 和 rollout 的可审计落盘。
- reward 类型精确路由，V3 与历史 stable/V2 文件保持独立。
- GRPO post-norm scale 的正有限校验、UID 组内一致性校验和缩放前后 advantage 审计。
- 3500e 固定协议、完整 trace、按题 paired bootstrap 和跨模型行为指标比较。
- Teacher PE 的人工 S/I/A benchmark、cache-free 重复、主切片/对照切片与推理预算审计。

### 9.2 PostNorm 0.3 暴露的生命周期问题

P03 训练实际完成了 79 steps，但 `verl_train.log` 在 step 22 后停止刷新。连续的 79 个 rollout 文件、
最终 checkpoint marker 和模型权重证明训练没有停在 step 22；当时的误判来自把单个日志 mtime 当作
任务真值，而没有优先检查 rollout、Ray worker 和 checkpoint 三类证据。

更深层的问题是监控会话与 pipeline 父级生命周期没有充分解耦。父级日志捕获/收尾控制流失效后，
Ray 子任务继续计算，但正常 finalizer 没有完整执行，导致：

- Teacher/Recall 服务没有按 pipeline 正常路径自动清理。
- VERL FSDP checkpoint 没有按预期自动转换为 HF safetensors。
- 后续需要手动清理服务并补做 HF 导出；第一次导出还因基础 Python 环境缺少 `tensordict` 失败，
  切换到仓库兼容环境后才成功。

这不是 reward 算法错误，但会影响训练状态判断、资源释放和评估启动时间。后续工程规则应固定为：

1. “停止持续观察”只停止轮询和日志跟踪，绝不向训练、pipeline 父进程或服务发送退出信号。
2. 只有收到明确的“停止训练/终止 pipeline/清理进程”指令时才允许改变任务生命周期。
3. 状态判断按 `process -> rollout continuity -> checkpoint marker -> manifest -> log` 的多证据顺序，
   不能只依据 `verl_train.log` 是否刷新。
4. 服务清理和 HF 转换应设计成幂等、可重入的 finalizer/reconciler，即使父级日志会话断开也可恢复。
5. HF exporter 必须固定到仓库兼容环境，并在正式训练前做依赖 preflight。

### 9.3 统计与数据风险

- 当前每个 5100 配方只有一个 seed，不能估计训练方差。
- Actor temperature=1，异步 Ray/vLLM 调度没有为每个 question/rollout/step 绑定独立 seed，训练不是
  位级复现。
- Teacher temperature=0 仍观察到生成级非完全确定性，所以 PE 使用三次 fresh 均值。
- 当前 128 holdout 已被组合策略选择过程复用，需要新样本才能恢复未触碰验证边界。
- 3500e 应继续只用于最终 Actor policy 泛化评估，不应用于反复调 Teacher prompt。

## 10. 下一阶段决策门槛

### 10.1 Boundary-Locked Teacher 生产验收

先完成独立单元与路由测试，再运行 64 条、1-step smoke。smoke 至少必须验证：

- 每个 UID 恰好 8 条 rollout，组内无混合 scale。
- Stage A 为 I 时 Stage B 调用数为 0，最终仍为 I。
- Stage A 为非 I 时，Stage B 失败能回退，不产生格式污染或异常正 reward。
- `teacher_i_boundary_preserved=true`，Teacher 总调用数、Stage-B called/used、selection reason 完整落盘。
- raw reward、Gold Token-F1 bonus、`advantage_pre_group_scale` 和
  `advantage_post_group_scale` 满足逐条计算关系。
- 无 NaN、无 rollout 丢失、自动 checkpoint 保存、服务清理和 HF 导出链路可执行。

smoke 通过后才启动 5100 正式训练。正式训练后继续使用同一 3500e 协议，与 Core V3/P01 做按题
比较，至少同时验收 EM、F1、完整答案率、平均搜索数、重复查询率和 Max-turn 率。

### 10.2 Core V3/P01 稳定性

在继续细调 scale 前，优先补至少两个独立 seed。跨 seed 需要报告 checkpoint 级均值/离散程度，
不能把 3500 个题目当作 3500 次独立训练。

如果继续做 Post-Norm Difficulty Weight 消融，可考虑 `0.15` 或 `0.2`，但只有在具备来源分组的
`mean(abs(advantage))`、policy-gradient norm 与行为指标后才值得启动。当前证据不支持再次测试
大于等于 `0.3` 的统一 scale。

### 10.3 SPAD Stage2/Stage3 的保留与复活条件

Stage2/Stage3 不删除，但默认关闭。再次投入正式资源前应满足：

1. 明确说明要修复 Stage1 的哪类残余错误，而不是泛化地“再蒸馏一次”。
2. 使用同一个 Stage1 checkpoint 构造严格对照。
3. Stage2 数据质量除 pair 数量外，还要审计证据充分性、chosen/rejected 差异和行为覆盖。
4. Stage3 最终模型必须在同一 3500e 上相对输入 Stage1 checkpoint 提供可复现增益。
5. 如果只降低训练 loss、但 EM/F1/完整答案率或搜索行为不改善，则不晋升为有效阶段。

## 11. 当前项目判断

截至 2026-07-16 09 时，项目不应再被描述为“一个已经有效的三阶段 SPAD 训练算法”，也不应被
描述为“已经放弃 SPAD、只剩一个普通 Stage1”。更准确的定义是：

> **SPAD 是被保留的多阶段研究框架；SPAD-SearchRL Core 是当前已经产生明确效果、正在持续迭代的
> 算法核心；Core V3/P01 是当前已验收配方；Boundary-Locked Teacher v2 是已经进入生产集成、
> 但尚待正式训练验证的下一候选。**

当前最可靠的算法认识是：Actor EM 应保持最高优先级；Teacher 负责填补全零困难组，但其标准化后
优化预算必须受控；gold 可以改善答案表示，却不能决定证据是否充分。当前最重要的工程认识是：训练
生命周期、日志观察、服务清理和 checkpoint 转换必须解耦，任何单一日志都不能成为任务状态的唯一
真值。

## 12. 主要依据

- `docs/AgenticIterRag_v1/architecture.md`
- `docs/AgenticIterRag_v1/work_reports/260710-12a_SPAD五百样本训练评估报告.md`
- `docs/AgenticIterRag_v1/work_reports/260714-14a_SPAD_GoldTokenF1V2_5100_NormFalse_Stage1训练与3500评估报告.md`
- `docs/AgenticIterRag_v1/work_reports/260715-00a_SPAD_GoldTokenF1V3_PostNormScale01_5100训练与3500评估报告.md`
- `docs/AgenticIterRag_v1/work_reports/260715-09a_SPAD_GoldTokenF1V3_PostNormScale03_5100训练与3500评估报告.md`
- `docs/AgenticIterRag_v1/work_reports/260715-23a_SPAD新数据Teacher组合策略消融与对比报告.md`
- `AgenticIterRag/agentic_iter_rag/agent_training/spad/teacher_strategies.py`
- `AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2.py`
- `tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1_overlay.yaml`
