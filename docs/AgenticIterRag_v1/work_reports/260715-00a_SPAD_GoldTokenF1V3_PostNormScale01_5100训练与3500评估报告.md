# SPAD Gold Token-F1 V3：Teacher 组 Post-Norm 0.1、5100 Stage1 训练与 3500 评估报告

日期：2026-07-15，12am（北京时间）

> 状态：V3 独立实现、单步 64 条真实训练验证、5100 Stage1 正式训练、HF checkpoint
> 导出、3500e 单次确定性评估和 10,000 次 paired bootstrap 均已完成。64 条验证训练的
> 日志与 checkpoint 已按要求删除。本实验只执行 Stage1，未执行 Stage2 或 Stage3。

## 1. 结论

本轮实现了 `spad_em_teacher_backoff_gold_token_f1_bonus_v3`。V3 保留 Gold Token-F1 V2
的原始 reward，恢复 `norm_adv_by_std_in_grpo=true`，但只对整组 Actor EM 全为 0、依赖
Teacher backoff 的组，在完成组内标准化后把整组 advantage 乘 `0.1`。

5100 正式训练完成 79 steps，最终 checkpoint 在固定 3500e 上取得：

| N | 成功 | 失败 | EM | F1 | 完整答案率 | 首轮搜索率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3500 | 3500 | 0 | **0.1994** | **0.2787** | **0.8340** | 0.9971 | **1.6969** | **0.1571** | **0.1369** |

与 Search-R1-512、Search-R1-5100 及三个历史 SPAD-5100 checkpoint 的同协议单次评估相比，
V3 的 EM、F1、完整答案率均为最高，平均搜索数、重复查询率和 Max-turn 率均为最低：

| 模型 | 关键训练语义 | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Search-R1 512 | Actor EM reward；norm=true；512 条 | 0.1180 | 0.1965 | 0.6271 | 2.3489 | 0.3640 | 0.2569 |
| Search-R1 5100 | Actor EM reward；norm=true；5100 条 | 0.1800 | 0.2509 | 0.7317 | 1.7291 | 0.1786 | 0.1549 |
| SPAD stable 5100 | stable reward；norm=true；inflight=2 | 0.1923 | 0.2700 | 0.7220 | 2.6557 | 0.5906 | 0.2443 |
| Gold Token-F1 V1 5100 | V1 eligibility；norm=true；inflight=2 | 0.1837 | 0.2576 | 0.6334 | 3.0071 | 0.5763 | 0.3589 |
| Gold Token-F1 V2 5100 | V2 eligibility；norm=false；inflight=2 | 0.1831 | 0.2673 | 0.7906 | 1.8889 | 0.2154 | 0.1863 |
| **Gold Token-F1 V3 5100** | **V2 eligibility；norm=true；Teacher 组 post-norm x0.1** | **0.1994** | **0.2787** | **0.8340** | **1.6969** | **0.1571** | **0.1369** |

按题 paired bootstrap 的结论是：

1. V3 相对 Search-R1-5100 的 EM `+0.0194`，95% CI `[0.0094, 0.0297]`；
   F1 `+0.0277`，95% CI `[0.0169, 0.0387]`；完整答案率 `+0.1023`，
   95% CI `[0.0863, 0.1183]`。三项区间均不跨 0。
2. V3 相对 Search-R1-512 的 EM/F1/完整答案率分别提高
   `+0.0814/+0.0821/+0.2069`，三项区间均不跨 0。
3. V3 相对 V2 的 EM `+0.0163`，95% CI `[0.0077, 0.0249]`；F1 `+0.0114`，
   95% CI `[0.0022, 0.0208]`，两项区间均不跨 0。
4. V3 相对 V1 的 EM/F1 为 `+0.0157/+0.0211`，区间均不跨 0。
5. V3 相对 stable 的 EM/F1 点估计为 `+0.0071/+0.0086`，但区间跨 0，不能确认
   精度显著优于 stable。
6. V3 相对三个 SPAD 对照的完整答案率提升区间均不跨 0；相对 stable/V1/V2 分别为
   `+0.1120/+0.2006/+0.0434`。

Qwen3-1.7B Base 没有在本轮 3500e 上运行。可用的历史 Base 结果来自另一份 350e 数据的三次
推理，均值为 EM `0.0810`、F1 `0.1567`、完整答案率 `0.5905`。它只能用于确认模型能力的大致
量级，不能与本次 V3 做同样本差值或 paired bootstrap；详细边界见第 8 节。

这些置信区间只刻画既定 checkpoint 在 3500 个问题上的逐题差异，不包含重新训练方差。
V3 当前只有一个训练 run，不能据此宣称该训练策略跨 seed 稳定优于所有对照。

## 2. V3 的训练语义

### 2.1 原始 reward 保持 V2 不变

每题采样 8 条 rollout，并按题分组：

1. 如果组内至少一条 Actor answer 命中 gold EM，每条轨迹只按自身 Actor EM 得 `1/0`。
2. 如果整组 Actor EM 全为 0，调用 GLM-4.7-Flash 判断检索证据。证据支持答案或存在歧义时，
   stable backoff 给 `0.1`，否则给 `0`。
3. 只有 Actor 输出合法闭合 `<answer>...</answer>`，Teacher 成功解析且证据状态为
   `supported_answer` 或 `ambiguous_evidence` 时，才追加 Gold Token-F1 bonus：

```text
teacher_gold_token_f1 = max(token_f1(teacher_answer, gold_alias_i))
bonus = 0.1 * teacher_gold_token_f1
raw_reward = stable_base_reward + bonus
```

V3 没有把上述 raw reward 本身乘 0.1。因此训练日志中的 `critic/score/mean`、
`final_reward/mean` 和 EM 等原始 reward 指标仍可直接与 V2 比较。

### 2.2 只缩放 Teacher fallback 组的标准化后 advantage

对每个 GRPO 问题组先按原配置计算：

```text
z_i = (r_i - group_mean) / (group_std + 1e-6)
```

随后按整组来源缩放：

```text
A_i = z_i        if group has any Actor EM hit
A_i = 0.1 * z_i  if group_all_em_zero=true (Teacher fallback group)
```

关键性质：

- 缩放发生在组内标准化之后，避免 `0.1` backoff 被 std 归一化重新放大到与 EM 组相同量级。
- 同一 UID 组的所有正、负 advantage 使用同一个 scale，保持组内相对排序和近零均值。
- Actor EM 组 scale 为 `1.0`，不削弱主要监督信号。
- Teacher 组内很小的原始 reward 差异仍会先被标准化；V3 限制的是该组整体梯度强度，
  不是恢复原始 reward 差值大小。

## 3. 实现与独立性

### 3.1 独立 Reward 模块

新增文件：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/
search_policy_teacher_reward_gold_match_bonus_v3.py
```

V3 先调用 stable reward，再复用 V2 bonus helper，最后只追加 advantage 来源和缩放审计字段：

```text
advantage_source
advantage_postnorm_scale
advantage_postnorm_scale_version=teacher_fallback_v1
```

stable `spad_em_teacher_backoff` 和现有 V2 reward 文件没有被改写。

### 3.2 GRPO 后归一化缩放

修改：

```text
AgenticIterRag/verl/verl/trainer/ppo/core_algos.py
AgenticIterRag/verl/verl/trainer/ppo/ray_trainer.py
```

新增行为：

- `compute_grpo_outcome_advantage` 接受可选 `group_postnorm_scales`。
- 校验每条 response 都有正有限 scale。
- 校验同一 UID 组 scale 完全一致，混组立即报错。
- 完成中心化/标准化后再乘 scale。
- 训练 rollout 额外落盘 `advantage_pre_group_scale` 和
  `advantage_post_group_scale`，支持逐条审计。
- 配置了 post-norm scale key 但 `norm_adv_by_std_in_grpo=false` 时拒绝启动，避免语义含混。

### 3.3 路由与配置

`search_policy_rl.py` 对 V3 精确选择独立模块和 batch 函数，并向 VERL 传入：

```yaml
algorithm.norm_adv_by_std_in_grpo: true
algorithm.group_postnorm_advantage_scale_key: advantage_postnorm_scale
algorithm.group_postnorm_advantage_scale_version: teacher_fallback_v1
```

正式 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/
spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_stage1_overlay.yaml
```

正式入口：

```text
tasks/train_tasks/agenticIterRag/
run_260714_AIR_spad_qwen3_1_7b_glm47_5100_gold_token_f1_v3_postnorm01_stage1.sh
```

## 4. 测试与 64 条真实训练验证

### 4.1 单元与数学验证

- reward/路由相关 `unittest`：26 项通过。
- CPU Torch 数学验证：Teacher 组标准化 advantage
  `[1.22473, 0, -1.22473, 0]` 缩放为
  `[0.122473, 0, -0.122473, 0]`；Actor EM 组保持不变。
- 同一 UID 混合 scale 会按预期抛错。
- 正式配置 dry-run 解析到 V3 独立模块、`norm=true`、scale key/version、79 steps 和 5100 样本。
- `git diff --check` 与正式 shell 语法检查通过。

当前环境没有安装 `pytest`，因此没有声称执行完整 pytest suite。

### 4.2 64 条、1-step 真实训练审计

验证 run 完成 `1/1` step，产生 64 个问题组、512 条 rollout。审计结果：

| 项目 | 结果 |
|---|---:|
| UID 组数 / rollout 数 | 64 / 512 |
| 每组大小 | 全部为 8 |
| Teacher fallback 组 | 47 组 / 376 条 |
| Actor EM 组 | 17 组 / 136 条 |
| Teacher 组正 / 负非零 advantage | 93 / 115 条 |
| Teacher 缩放前范围 | `[-2.474804, 2.474839]` |
| Teacher 缩放后范围 | `[-0.247480, 0.247484]` |
| 逐条 `post = pre * scale` 最大误差 | `0.0` |
| 最大组内 pre/post 均值绝对值 | `4.28e-7 / 6.71e-8` |
| 错标、混合 scale、错误组大小 | 0 |

这证明真实训练链路中的正、负 Teacher advantage 均在标准化后统一乘 0.1，Actor EM 组保持 1.0。

按用户要求，验证完成后已删除该 64 条 run 的全部日志、checkpoint 和启动包装目录；不保留可误用的
64 条模型产物。验证 overlay 保留为实现证据和可复现配置。

## 5. 5100 正式训练

### 5.1 关键配置

| 参数 | 值 |
|---|---|
| 初始模型 | Qwen3-1.7B Base |
| 训练数据 | `data/global_train_eval_data/5100t/co_search_ablation.train.parquet` |
| train max samples | 5100 |
| 实际 prompt slots | 79 x 64 = 5056 |
| 每题 rollout | 8 |
| 总 rollout | 40448 |
| steps / save freq | 79 / 79 |
| data shuffle / seed | true / 42 |
| Actor temperature / top_p | 1 / 1 |
| `norm_adv_by_std_in_grpo` | true |
| Teacher fallback post-norm scale | 0.1 |
| `stream_group_max_inflight` | 2 |
| Stage2 / Stage3 | 关闭 / 关闭 |

Teacher 参数：GLM-4.7-Flash，NPU 4-5，TP=2，BF16，temperature=0，top_p=1，
max_tokens=512，timeout=180 秒，thinking=false，batch workers=16。

已固定数据 seed 和 Teacher 确定性解码；Actor rollout 仍为 temperature=1，异步 Ray/vLLM 调度也
没有为每个 `question + rollout_index + step` 绑定独立 seed，因此训练不是位级可复现。

### 5.2 Run、耗时和 checkpoint

正式 run：

```text
260714-175600-957643-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_stage1
```

耗时：

| 阶段 | 耗时 |
|---|---:|
| 服务与模型加载 | 约 9 分钟 |
| 79-step 训练进度 | 5 小时 46 分 57 秒 |
| 含最终保存、HF 导出和流水线收尾 | 约 5 小时 58 分钟 |

最终 HF checkpoint：

```text
checkpoints/AIR/260714-175600-957643-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_stage1/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79
```

模型文件 `model.safetensors` 大小 `4,063,515,640` bytes，SHA256：

```text
5065eccb098797d75f52c7955a47ab49c45514c1ecc9948d2b2d17e6a80a0f8a
```

评估 manifest 计算的模型 fingerprint：

```text
36e79ae66691b96716f2f4c4ed0e0db81b1d41aacf7f0dedd2d43aaaff90b4f4
```

### 5.3 训练曲线

完整 79-step 曲线数据：

```text
reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-aggregate/training_curve.csv
```

曲线统计：

| 指标 | 全程均值 | 前 10 步均值 | 后 10 步均值 | 后 3 步均值 |
|---|---:|---:|---:|---:|
| Final reward | 0.3093 | 0.2159 | 0.3170 | 0.3354 |
| EM reward | 0.2805 | 0.1785 | 0.2898 | 0.3112 |
| Rollout F1 | 0.3521 | 0.2622 | 0.3630 | 0.3793 |
| Teacher Token-F1 bonus | 0.00682 | 0.01029 | 0.00594 | 0.00469 |
| Teacher fallback 率 | 0.5756 | 0.6625 | 0.5656 | 0.5521 |
| 平均 post-norm scale | 0.4820 | 0.4037 | 0.4909 | 0.5031 |
| 平均搜索数 | 1.3072 | 1.3879 | 1.4658 | 1.4733 |

按 10-step 分箱：

| Steps | Reward | EM | Rollout F1 | Teacher bonus | Fallback 率 | Mean scale | 搜索数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-10 | 0.2159 | 0.1785 | 0.2622 | 0.01029 | 0.6625 | 0.4037 | 1.3879 |
| 11-20 | 0.2843 | 0.2564 | 0.3294 | 0.00716 | 0.5969 | 0.4628 | 1.1785 |
| 21-30 | 0.3083 | 0.2826 | 0.3486 | 0.00477 | 0.5750 | 0.4825 | 1.1764 |
| 31-40 | 0.3469 | 0.3209 | 0.3878 | 0.00622 | 0.5344 | 0.5191 | 1.2182 |
| 41-50 | 0.3181 | 0.2852 | 0.3596 | 0.00807 | 0.5781 | 0.4797 | 1.2469 |
| 51-60 | 0.3363 | 0.3074 | 0.3759 | 0.00651 | 0.5531 | 0.5022 | 1.3596 |
| 61-70 | 0.3448 | 0.3197 | 0.3843 | 0.00547 | 0.5453 | 0.5092 | 1.4465 |
| 71-79 | 0.3215 | 0.2947 | 0.3709 | 0.00602 | 0.5573 | 0.4984 | 1.4588 |

训练 reward 单步波动较大：最低为 step 3 的 `0.1483`，最高为 step 68 的 `0.4442`。
第 69 步曾降至 `0.2560`，但第 65-69 步均值为 `0.3435`，高于历史 V2 同期的 `0.3323`；
它是 batch EM 命中波动，不是 post-norm scale 直接压低 raw reward。

整体趋势是 EM/F1 明显高于前 10 步，Teacher fallback 率和 bonus 均下降，说明后期更多问题组由
Actor 自身 EM 信号接管。后 10 步平均搜索数回升到 1.47，但评估时仍表现为比历史模型更短的搜索链。

## 6. 3500e 评估

### 6.1 协议与耗时

评估 task：

```text
260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1
```

评估数据：

```text
data/global_train_eval_data/3500e/co_search_ablation.eval.parquet
SHA256 bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283
```

| 项目 | 设置 |
|---|---|
| 模式 | no-ranker |
| Actor vLLM | NPU 0-5，6 个 DP replica，TP=1 |
| Recall | NPU 6-7，2 个 backend |
| Infer batch | 384 |
| 每 Actor `max_num_seqs` | 64 |
| Flush every N | 500 |
| temperature / top_p | 0 / 1 |
| Recall Top N / 模型可见 Top M | 50 / 5 |
| 最大 assistant turns | 6 |
| 单轮 response 上限 | 1024 tokens |
| Trace | full |

纯推理墙钟为 `516.8171s`，即 8 分 37 秒；含模型指纹、服务加载、预检、写盘和清理的总墙钟
约 15 分钟。3500 条全部成功，失败 0，且 `output_reuse=false`。

### 6.2 总体与状态

| 状态 | 数量 | 比例 |
|---|---:|---:|
| `answered` | 2917 | 0.8334 |
| `no_valid_answer` | 94 | 0.0269 |
| `max_turns` | 479 | 0.1369 |
| `multiple_tool_calls` | 8 | 0.0023 |
| `direct_answer_before_search` | 2 | 0.0006 |

聚合器的“完整答案率”按 `final_answer` 非空计算，为 2919/3500 = `0.8340`；它包含 2 条
`direct_answer_before_search`。严格 `answered` 状态比例为 `0.8334`。

搜索次数分布：

| 搜索次数 | 比例 | 约计条数 |
|---|---:|---:|
| 0 | 0.0029 | 10 |
| 1 | 0.7294 | 2553 |
| 2 | 0.1126 | 394 |
| 3 | 0.0154 | 54 |
| 4 | 0.0026 | 9 |
| 5+ | 0.1371 | 480 |

### 6.3 分数据源

| 数据源 | N | EM | F1 | 平均搜索数 |
|---|---:|---:|---:|---:|
| 2WikiMultiHopQA | 563 | 0.1545 | 0.1948 | 1.7691 |
| Bamboogle | 125 | 0.0800 | 0.1721 | 1.4240 |
| HotpotQA | 562 | 0.2189 | 0.3062 | 1.4324 |
| MuSiQue | 562 | 0.0356 | 0.0807 | 2.2740 |
| NQ | 562 | 0.3060 | 0.3870 | 1.3594 |
| PopQA | 563 | 0.3357 | 0.3722 | 1.9858 |
| TriviaQA | 563 | 0.1723 | 0.3547 | 1.4210 |

Macro-average EM/F1 为 `0.1861/0.2668`。MuSiQue 仍是最弱且搜索最多的数据源。

V3 相对 V2：7 个数据源中 6 个 EM 提高，只有 Bamboogle 下降；F1 在 2Wiki、HotpotQA、
PopQA、TriviaQA 提高，在 Bamboogle、MuSiQue、NQ 略降。因此总体提升不是所有数据源一致改善。

## 7. Paired Bootstrap

设置：3500 个问题按题配对，10,000 次 bootstrap，seed 42。核心结果：

| 对照 -> V3 | Delta EM [95% CI] | Delta F1 [95% CI] | Delta 完整答案率 [95% CI] |
|---|---:|---:|---:|
| Search-R1 512 -> V3 | +0.0814 `[0.0706, 0.0923]` | +0.0821 `[0.0712, 0.0933]` | +0.2069 `[0.1891, 0.2246]` |
| Search-R1 5100 -> V3 | +0.0194 `[0.0094, 0.0297]` | +0.0277 `[0.0169, 0.0387]` | +0.1023 `[0.0863, 0.1183]` |
| stable -> V3 | +0.0071 `[-0.0034, 0.0174]` | +0.0086 `[-0.0028, 0.0199]` | +0.1120 `[0.0963, 0.1280]` |
| Gold V1 -> V3 | +0.0157 `[0.0046, 0.0266]` | +0.0211 `[0.0091, 0.0329]` | +0.2006 `[0.1840, 0.2177]` |
| Gold V2 -> V3 | +0.0163 `[0.0077, 0.0249]` | +0.0114 `[0.0022, 0.0208]` | +0.0434 `[0.0300, 0.0569]` |

V3 相对 V2 的行为点估计同时改善：平均搜索 `-0.1920`、重复查询率 `-0.0583`、
Max-turn 率 `-0.0494`。当前聚合脚本没有为这三项输出 bootstrap CI，因此这里不声称其统计显著性。

## 8. Base、Search-R1 与历代 SPAD 的比较边界

### 8.1 同一 3500e 协议下的直接比较

以下模型均在同一份 3500e、no-ranker、Top N=50、Top M=5、temperature=0 协议下各评估一次，
因此可以做逐题 paired bootstrap：

| 模型 | 训练规模 | Reward / advantage 关键差别 | EM | F1 | 完整答案率 |
|---|---:|---|---:|---:|---:|
| Search-R1 | 512 | Actor EM；norm=true | 0.1180 | 0.1965 | 0.6271 |
| Search-R1 | 5100 | Actor EM；norm=true | 0.1800 | 0.2509 | 0.7317 |
| SPAD stable | 5100 | EM + Teacher backoff；norm=true | 0.1923 | 0.2700 | 0.7220 |
| Gold Token-F1 V1 | 5100 | 初版 bonus；norm=true | 0.1837 | 0.2576 | 0.6334 |
| Gold Token-F1 V2 | 5100 | 收紧 bonus eligibility；norm=false | 0.1831 | 0.2673 | 0.7906 |
| **Gold Token-F1 V3** | **5100** | **V2 eligibility；norm=true；Teacher 组 post-norm x0.1** | **0.1994** | **0.2787** | **0.8340** |

V3 相对每个同协议对照的描述性行为差值：

| V3 相对对照 | Delta EM | Delta F1 | Delta 完整答案率 | Delta 平均搜索数 | Delta 重复查询率 | Delta Max-turn 率 |
|---|---:|---:|---:|---:|---:|---:|
| Search-R1 512 | +0.0814 | +0.0821 | +0.2069 | -0.6520 | -0.2069 | -0.1200 |
| Search-R1 5100 | +0.0194 | +0.0277 | +0.1023 | -0.0322 | -0.0215 | -0.0180 |
| SPAD stable 5100 | +0.0071 | +0.0086 | +0.1120 | -0.9588 | -0.4335 | -0.1074 |
| Gold Token-F1 V1 5100 | +0.0157 | +0.0211 | +0.2006 | -1.3102 | -0.4192 | -0.2220 |
| Gold Token-F1 V2 5100 | +0.0163 | +0.0114 | +0.0434 | -0.1920 | -0.0583 | -0.0494 |

精度结论必须和第 7 节的区间一起看：V3 对 Search-R1-512、Search-R1-5100、V1、V2 的
EM/F1 提升区间均不跨 0；对 stable 的 EM/F1 区间跨 0。完整答案率方面，V3 对上述五个对照
的提升区间均不跨 0。行为差值没有 bootstrap CI，只作描述性比较。

这些同协议历史 checkpoint 为：

| 模型 | HF checkpoint |
|---|---|
| Search-R1 512 | `checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` |
| Search-R1 5100 | `checkpoints/AIR/260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` |
| SPAD stable 5100 | `checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` |
| Gold Token-F1 V1 5100 | `checkpoints/AIR/260713-022724-631051-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_bonus_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` |
| Gold Token-F1 V2 5100 | `checkpoints/AIR/260714-091019-055405-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v2_normfalse_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` |
| Gold Token-F1 V3 5100 | `checkpoints/AIR/260714-175600-957643-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` |

### 8.2 历史 Base 参照

Base 目前只有旧 350e 数据上的三次独立确定性评估：

| 模型 | 数据 / repeats | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-1.7B Base | 350e / 3 | 0.0810 +/- 0.0087 | 0.1567 +/- 0.0092 | 0.5905 +/- 0.0092 | 2.3943 +/- 0.0234 | 0.3676 +/- 0.0119 | 0.2248 +/- 0.0082 |

来源为 `data/global_train_eval_data/350e/co_search_ablation.eval.parquet`，SHA256
`ddd7297f5f77253392ccfca331639280bdef672e0c85210ad1267a711601b660`；聚合报告位于：

```text
reports/eval/agenticIterRag/260711-newdata5100-search-r1-formal-aggregate/report.md
```

若只看点估计，V3 减去这份 Base 历史均值为 EM `+0.1184`、F1 `+0.1220`、完整答案率
`+0.2435`、平均搜索数 `-0.6974`、重复查询率 `-0.2105`、Max-turn 率 `-0.0879`。
但 350e 和 3500e 是不同文件、不同样本规模，不能把这些差值解释为同协议提升，更不能据此给出
paired bootstrap 或显著性结论。若需要正式回答 V3 相对 Base 的提升，应补跑 Base 的同一 3500e
评估，而不是复用这份 350e 数字。

## 9. 解释与局限

当前结果支持“标准化后再弱化 Teacher fallback 组”比两种极端更合适：

- V1/stable 的 `norm=true` 会把全零 EM 组内的 0/0.1/0.2 差异标准化到较强梯度。
- V2 的 `norm=false` 保留原始小尺度，但同时取消了所有组的 std 归一化，训练语义改动较大。
- V3 保留 Actor EM 组的标准 GRPO 归一化，只把 Teacher 组整体压到 0.1，既保留 Teacher 排序，
  又限制其相对主 EM 信号的梯度强度。

但不能把 V3 对 V2 的单 run 提升严格归因于 post-norm scale，原因是 V2 为 `norm=false`，
V3 为 `norm=true + Teacher scale=0.1`，二者不是只改一个布尔参数。后续最有价值的受控消融是：

1. 固定 V2 eligibility、5100 数据、seed、inflight 和所有服务参数。
2. 比较 `norm=true + Teacher scale=1.0/0.3/0.1/0.03`。
3. 每个设置至少进行 3 次独立训练，而不是只重复同一 checkpoint 的确定性评估。
4. 继续同时报告准确率、完整答案率、搜索数、重复查询和 Max-turn，避免只优化单一 EM/F1。

## 10. 产物

训练日志根目录：

```text
log/agenticIterRag/260714-175600-957643-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_stage1
```

自动评估报告：

```text
reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1.report.md
```

完整 trace 与逐题 metrics：

```text
log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1/trace
```

历史聚合、bootstrap、完整训练曲线 CSV 和曲线统计：

```text
reports/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-aggregate/
  report.md
  summary.json
  training_curve.csv
  training_curve_summary.json
```

比较 run spec：

```text
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260715_gold_token_f1_v3.json
```

## 11. 最终判断

V3 本轮达到了预期的工程和实验目标：实现独立、真实单步审计正确、正式 5100 训练与 3500 评估
完整成功。单 checkpoint 上，V3 相对 Search-R1-512、Search-R1-5100、V2 和 V1 的 EM/F1 提升
通过按题 bootstrap，完整答案和搜索行为也明显改善；相对 stable 的精度点估计更高但区间跨 0，
不能宣称显著优于 stable。Base 因缺少同一 3500e 评估，仅能作历史量级参照。

因此 V3 适合作为下一轮多 seed 与 scale 消融的首选候选，而不是仅凭本次单 run 立即替代所有 stable
checkpoint。
