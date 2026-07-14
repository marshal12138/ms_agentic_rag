# GRPO 关闭组内 Std 归一化：三类 Reward 三重复训练与 3500 评估报告

> 日期：2026-07-13  
> 状态：九次 Stage1 训练、九次 3500 单次评估、paired bootstrap 聚合均已完成  
> 代码仓库：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives`  
> 本报告只讨论 Stage1；按照当前约定，后续 SPAD 实验默认不启动 Stage2/Stage3，除非另行说明。

## 1. 实验目的

本轮实验针对 GRPO 的组内 advantage 计算方式做消融：

```yaml
norm_adv_by_std_in_grpo: false
```

此前训练使用 `true`。在 `true` 下，组内 reward 先减均值、再除以组内标准差。对于只存在两个 reward 水平的组，整体乘以常数通常不会改变标准化后的相对 advantage。因此，Teacher backoff 的 `0.1` 与 EM reward 的 `1.0` 在不同组之间原本希望表达的绝对强弱，可能被组内标准化明显削弱。

本轮关闭除以标准差的步骤，保留组内中心化，使 reward 的绝对尺度进入 advantage：

```text
advantage_i = reward_i - mean(reward_group)
```

在每题 8 条 rollout 的设置下，若组内只有一条 reward=1，其余为 0，则 advantage 最大值为 `0.875`、最小值为 `-0.125`；若只有一条 reward=0.1，其余为 0，则最大值约为 `0.0875`。这使 Teacher backoff 信号在数值上保持约为 EM 信号的十分之一。

为同时观察该改动对不同 reward 的影响，本轮训练三类模型，每类重复三次：

1. Search-R1 原始 EM reward；
2. SPAD `spad_em_teacher_backoff`；
3. SPAD `spad_em_teacher_backoff_gold_token_f1_bonus` V2。

九个 checkpoint 均在同一份 512 训练数据上完成 Stage1，并在同一份 3500e 数据上做一次确定性评估。三次 repeat 使用相同显式配置和 `data_seed=42`，用于观察相同配置重复运行的稳定性；它们不是三个不同 seed 的实验。

## 2. 结论摘要

### 2.1 3500 评估结果

| 训练方法 | Repeat | EM | F1 | 完整答案率 | 首轮搜索率 | 平均搜索数 | 重复查询率 | 最大轮次率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Search-R1，norm=false | 1 | 0.1271 | 0.2108 | 0.6620 | 0.9783 | 2.2660 | 0.3194 | 0.2489 |
| Search-R1，norm=false | 2 | 0.1017 | 0.1771 | 0.5543 | 0.9994 | 2.7989 | 0.4994 | 0.3783 |
| Search-R1，norm=false | 3 | 0.1106 | 0.1896 | 0.5891 | 0.9991 | 2.5929 | 0.4343 | 0.3317 |
| Search-R1，norm=false | 三次均值 | 0.1131 | 0.1925 | 0.6018 | 0.9923 | 2.5526 | 0.4177 | 0.3196 |
| SPAD stable，norm=false | 1 | 0.1129 | 0.1904 | 0.6291 | 0.9689 | 2.2183 | 0.3120 | 0.2049 |
| SPAD stable，norm=false | 2 | 0.1297 | 0.2174 | 0.6911 | 0.9817 | 2.2757 | 0.3177 | 0.2303 |
| SPAD stable，norm=false | 3 | 0.1231 | 0.2086 | 0.6617 | 0.9703 | 2.1923 | 0.2940 | 0.2160 |
| SPAD stable，norm=false | 三次均值 | 0.1219 | 0.2055 | 0.6607 | 0.9736 | 2.2288 | 0.3079 | 0.2170 |
| SPAD Gold Token-F1 V2，norm=false | 1 | 0.1174 | 0.2032 | 0.6486 | 0.9834 | 2.3043 | 0.3349 | 0.2477 |
| SPAD Gold Token-F1 V2，norm=false | 2 | 0.1286 | 0.2110 | 0.6566 | 0.9800 | 2.3423 | 0.3494 | 0.2560 |
| SPAD Gold Token-F1 V2，norm=false | 3 | 0.1157 | 0.1944 | 0.6360 | 0.9697 | 2.3397 | 0.3686 | 0.2109 |
| SPAD Gold Token-F1 V2，norm=false | 三次均值 | 0.1206 | 0.2029 | 0.6470 | 0.9777 | 2.3288 | 0.3510 | 0.2382 |

### 2.2 三次重复的均值与离散程度

这里的 `±` 是三次训练结果的总体标准差，不是评估样本上的置信区间。三次训练使用同一个显式 `data_seed=42`，所以 `n=3` 反映的是并发训练链路在相同配置下的重复运行波动，不是跨 seed 方差。

| 方法 | EM 均值 ± SD | F1 均值 ± SD | 完整答案率均值 | F1 极差 |
|---|---:|---:|---:|---:|
| Search-R1，norm=false | 0.1131 ± 0.0105 | 0.1925 ± 0.0139 | 0.6018 | 0.0337 |
| SPAD stable，norm=false | 0.1219 ± 0.0069 | 0.2055 ± 0.0113 | 0.6607 | 0.0270 |
| SPAD Gold Token-F1 V2，norm=false | 0.1206 ± 0.0057 | 0.2029 ± 0.0068 | 0.6470 | 0.0166 |

Gold Token-F1 V2 的三次 F1 总体标准差仍最小，但第三次明显低于前两次，F1 极差由两次时的 `0.0078` 扩大至 `0.0166`。Stable 的三次平均 EM/F1 反而略高于 Gold V2；两者平均差只有 EM `0.0013`、F1 `0.0026`，仍小于各自的训练重复波动，不能宣称任一 reward 稳定胜出。

### 2.3 核心判断

1. **关闭 Std 归一化恢复了 reward 绝对尺度，但没有消除训练波动。** 九次训练日志中 advantage 极值均符合未除标准差后的原始尺度；配置确实生效。
2. **不能认定 `norm=false` 稳定提升 Search-R1。** Repeat 1 相对历史 norm=true 提高，Repeat 2 明显下降，Repeat 3 略低且 EM/F1 的单 checkpoint CI 均触及或跨过 0；三次均值低于历史值。
3. **SPAD stable 相对同为 inflight=2 的历史 norm=true 对照，三次均有提升。** 第三次也达到 EM `0.1231`、F1 `0.2086`，但三次均值仍低于历史 inflight=1 stable 最佳，不能把差异只归因于 norm 开关。
4. **第三次结果推翻了 Gold V2 均值略高的暂时排序。** 当前 Stable 三次均值比 Gold V2 高 EM `0.0013`、F1 `0.0026`；差距远小于训练波动，没有稳健优势证据。
5. **现阶段更强的结论仍是“训练随机性不可忽略”。** Search-R1 不涉及 Teacher 或 `0.1` backoff，但三次训练 F1 极差仍达 `0.0337`，说明异步 rollout、服务调度和采样顺序本身足以造成明显差异。

## 3. 三类 Reward 的实现语义

### 3.1 Search-R1 原始 Reward

Search-R1 直接按 Actor 最终答案对 gold answer 的 EM 计分：

```text
Actor EM 命中 -> 1
Actor EM 未命中 -> 0
```

它不调用 Teacher。因此，该组也可作为判断训练链路自身非确定性的对照。

### 3.2 SPAD stable：`spad_em_teacher_backoff`

每个问题采样 8 条 rollout，并按问题分组：

1. 如果组内至少一条 Actor 答案对 gold EM 命中，每条轨迹只按自身 EM 得 `1/0`；不调用 Teacher backoff。
2. 如果整组 Actor EM 均为 0，则由 GLM-4.7-Flash 基于检索证据判断该轨迹是否得到证据支持。
3. Teacher 判为 `supported` 或 `ambiguous` 时给予 `0.1`；证据不足、无证据、格式错误等情况为 `0`。

这个 reward 的设计目标不是让 Teacher 替 Actor 作答，而是在 gold EM 全零组中保留“检索轨迹有价值”的弱排序信号。

### 3.3 Gold Token-F1 Bonus V2

正式 reward 名称：

```text
spad_em_teacher_backoff_gold_token_f1_bonus
```

实现模块：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/
search_policy_teacher_reward_gold_match_bonus.py
```

V2 eligibility：

```text
actor_answer_closed_teacher_supported_v2
```

它独立组合 stable reward，不修改 `spad_em_teacher_backoff` 原实现。总 reward 为：

```text
stable_reward + extra_bonus

extra_bonus = 0.1 * max_token_f1(Teacher answer, each gold alias)
```

额外 bonus 只在以下条件全部满足时发放：

- 本题 8 条 rollout 的 Actor EM 全为 0；
- Teacher 已调用并成功解析，且没有格式错误；
- Teacher 状态为 `supported` 或 `ambiguous`；
- Actor 自身存在合法、闭合的最终答案；
- Teacher answer 与各 gold alias 计算词级 F1，取最大值。

重要区别：即使 Actor 没有合法闭合答案，只要检索证据被 Teacher 认可，原 stable `0.1` backoff 仍可发放；但 V2 的额外 token-F1 bonus 不再发放。这避免“Actor 未作答，仅 Teacher 答对，Actor 仍获得额外 bonus”的目标错位。

历史 Gold V1 的 bonus eligibility 更宽松。本轮同时改变了 `V1 -> V2` 与 `norm=true -> false`，所以历史 Gold V1 与本轮 Gold V2 的差异不能只归因于其中一个因素。

## 4. 代码与配置改动

### 4.1 将 norm 开关显式传给 VERL

训练计划构造代码：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/search_policy_rl.py
```

新增 Hydra override：

```text
algorithm.norm_adv_by_std_in_grpo=<trainer.norm_adv_by_std_in_grpo>
```

默认值设为 `false`，并在以下基础配置中显式记录：

```text
AgenticIterRag/config/agent_training/spad_rag_base.yaml
AgenticIterRag/config/agent_training/search_r1_original.yaml
```

这意味着当前代码基线默认关闭 GRPO 组内 Std 归一化。历史 checkpoint 的实际配置不因此改变。

### 4.2 九个训练 overlay

```text
tasks/train_tasks/agenticIterRag/configs/
search_r1_original_qwen3_1_7b_512_normfalse_rep1_overlay.yaml
search_r1_original_qwen3_1_7b_512_normfalse_rep2_overlay.yaml
search_r1_original_qwen3_1_7b_512_normfalse_rep3_overlay.yaml
spad_qwen3_1_7b_glm47_512_stable_normfalse_rep1_overlay.yaml
spad_qwen3_1_7b_glm47_512_stable_normfalse_rep2_overlay.yaml
spad_qwen3_1_7b_glm47_512_stable_normfalse_rep3_overlay.yaml
spad_qwen3_1_7b_glm47_512_gold_token_f1_v2_normfalse_rep1_overlay.yaml
spad_qwen3_1_7b_glm47_512_gold_token_f1_v2_normfalse_rep2_overlay.yaml
spad_qwen3_1_7b_glm47_512_gold_token_f1_v2_normfalse_rep3_overlay.yaml
```

统一编排入口：

```text
tasks/train_tasks/agenticIterRag/run_260713_normfalse_512_stage1_six_trainings.sh
tasks/train_tasks/agenticIterRag/run_260713_normfalse_512_stage1_three_rep3_trainings.sh
```

### 4.3 评估入口与聚合 spec

```text
tasks/eval_tasks/agenticIterRag/run_260713_normfalse_512_six_3500evals.sh
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260713_normfalse_512.json
tasks/eval_tasks/agenticIterRag/run_260713_normfalse_512_three_rep3_3500evals.sh
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260713_normfalse_512_three_repeats.json
```

第一份评估 spec 纳入 4 个历史 checkpoint 与前 6 个本轮 checkpoint；第二份纳入相同 4 个历史 checkpoint 与全部 9 个本轮 checkpoint，并补齐 Rep3 相关 paired comparison。

聚合器的 `runs` 语义是“同一 checkpoint 的多次独立推理”，要求模型指纹一致、`repeat_id` 依次为 1/2/3。本轮三个 repeat 是三个不同训练 checkpoint，各只评估一次，因此在 spec 中保持为三个独立模型，不能把它们伪装成一个模型的三次推理重复。训练方法的三次均值与总体 SD 由本报告跨 checkpoint 计算；paired bootstrap 则始终针对单 checkpoint、同一 3500 问题集合进行。

### 4.4 实现验证

训练启动前完成：

- plan/routing 测试：9 项通过；
- Gold V2 reward 测试：11 项通过；
- 三类 reward dry-run 均通过；
- 九次正式训练的日志均显示 `algorithm.norm_adv_by_std_in_grpo=False`；
- 各 step 的 advantage 最大/最小值为未标准化尺度，未出现仍按单位方差标准化的迹象。

## 5. 训练数据与控制变量

### 5.1 共同设置

| 项目 | 设置 |
|---|---|
| 基座模型 | Qwen3-1.7B |
| 训练数据量 | 512 questions |
| 每题 rollout 数 | 8 |
| 训练 batch | 64 questions |
| 总 step | 8 |
| 学习率 | `1e-6` |
| Actor rollout temperature | 1.0 |
| Actor rollout top_p | 1.0 |
| `norm_adv_by_std_in_grpo` | `false` |
| `stream_group_max_inflight` | 2 |
| `data_seed` | 42 |
| 训练阶段 | 仅 Stage1 |

SPAD stable 和 Gold V2 的 Teacher 共同设置：

| 项目 | 设置 |
|---|---|
| Teacher | GLM-4.7-Flash |
| temperature | 0 |
| top_p | 1 |
| max_tokens | 512 |
| reward backoff | 0.1 |

### 5.2 重复训练的随机性限制

三次 repeat 使用：

- 相同训练数据文件；
- 相同 `data_seed=42`；
- 相同基座模型；
- 相同超参数；
- 相同 `stream_group_max_inflight=2`；
- 相同硬件编排方式。

但三次仍生成不同 checkpoint。主要原因是训练链路包含异步 rollout、多个 Actor worker、Teacher/Recall 服务请求调度，以及采样请求到达顺序。显式 seed 不能完全固定这些并发顺序。

因此，本轮的两个 repeat 应表述为：

> 相同显式 seed 和配置下的重复运行稳定性检查。

不应表述为：

> 三个独立随机 seed 的统计实验。

若后续需要正式多 seed 结论，应显式使用至少 3 至 5 个不同训练 seed，并同步记录数据顺序、rollout seed 与服务调度相关参数。

## 6. 九次训练产物

所有训练都正常完成至 `global_step_8`。

### 6.1 Search-R1

#### Repeat 1

```text
checkpoints/AIR/
260713-103539-495712-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512_normfalse_rep1/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`7d6b598ca64bb741140629ed9178ef70ea275991b1e868edcf67f918e3a5dfeb`
- checkpoint fingerprint：`c1647d95921af68740c3d8b109f34a7ddea115bfa9d9cf52c7f2246b851a3ed2`
- 实际训练进度耗时：24 分 22 秒
- 流水线墙钟耗时：约 31 分钟

#### Repeat 2

```text
checkpoints/AIR/
260713-110639-534549-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512_normfalse_rep2/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`f7b0e46fea58eae8b69ea7be8c57ff377571baac33bba1193b60e8f1274fea06`
- checkpoint fingerprint：`7beed372e3d12d0f6fb87e3627d3c53550942e6c9bee441ea6cbfa1dc31edddb`
- 实际训练进度耗时：25 分 57 秒
- 流水线墙钟耗时：约 32 分 31 秒

#### Repeat 3

```text
checkpoints/AIR/
260713-185433-916978-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512_normfalse_rep3/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`4e8a0181aaf8584f601d4d8a4c7ccc5afe8dd25a0f2965f21e746f4b78b0934d`
- checkpoint fingerprint：`ee2236ad6cbfe04c6083e82872b2bdd0b73b370cf1e1f21c268eafe1e3e6bdec`
- 实际训练进度耗时：24 分 20 秒
- 流水线墙钟耗时：30 分 54 秒

### 6.2 SPAD stable

#### Repeat 1

```text
checkpoints/AIR/
260713-113910-342014-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_normfalse_rep1/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`6968fed1e9e7853fd7b3106a0cc1eeb457a1db99477ac0e0262b5b7249df7aaf`
- checkpoint fingerprint：`89a0414e634d623a7034b06370acb1e88c35270f44417ebb0a9fa639efd3c2ba`
- 实际训练进度耗时：约 40 分 12 秒
- 流水线墙钟耗时：约 50 分 04 秒

#### Repeat 2

```text
checkpoints/AIR/
260713-122915-051755-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_normfalse_rep2/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`1fbc84fcccd1f1634fc1c6516bf73e1f403f84014560c0e1e1788da6275af584`
- checkpoint fingerprint：`0201bc111c72e86a871a30e02fefd45ed865ba0663641008ba8de63f831e30f2`
- 实际训练进度耗时：约 41 分 48 秒
- 流水线墙钟耗时：约 52 分 12 秒

#### Repeat 3

```text
checkpoints/AIR/
260713-192527-981329-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_normfalse_rep3/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`1f1d6e96022d1c87d33cbc4e3a92957795e55ca029457177b5ee97b4bdd361bd`
- checkpoint fingerprint：`febdeb69663496e61008025c08e659018dffae854e56690c90bf748a8c3d5a03`
- 实际训练进度耗时：40 分 05 秒
- 流水线墙钟耗时：50 分 11 秒

### 6.3 SPAD Gold Token-F1 Bonus V2

#### Repeat 1

```text
checkpoints/AIR/
260713-132127-010666-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_v2_normfalse_rep1/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`e1cc096c1a5ad82bcfd141d09089fb2e983b270a9048ae67ef8205f7ed2f21c7`
- checkpoint fingerprint：`c8200818498737469551da21e9665471fa71573d7222ae2d59f61124ef2bfa6f`
- 实际训练进度耗时：约 39 分 16 秒
- 流水线墙钟耗时：约 49 分 29 秒

#### Repeat 2

```text
checkpoints/AIR/
260713-141055-402010-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_v2_normfalse_rep2/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`af1deb8a1290c715f5b83fd0ae0b1c75a0e0061bdcd179558fd35b16cfe745ed`
- checkpoint fingerprint：`633a0e057dee65762549c2fb1769a5e48af7fb91b743226e6b7c84400a2fdd1b`
- 实际训练进度耗时：约 38 分 17 秒
- 流水线墙钟耗时：约 48 分 01 秒

#### Repeat 3

```text
checkpoints/AIR/
260713-201539-129092-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_v2_normfalse_rep3/
stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
```

- `model.safetensors` SHA256：`ef1ecf32f5ccc2bc6799aced6f02350168dd9a23846ed3ec4ab8c8e8bf2a4a49`
- checkpoint fingerprint：`eea21584dabc4cf2a083f56af0ecc9e0decfa515783671e6b2e769ba7953ef63`
- 实际训练进度耗时：38 分 10 秒
- 流水线墙钟耗时：48 分 09 秒

## 7. 逐 Step 训练指标

### 7.1 Search-R1

`score` 即 Actor answer EM reward 均值。

| Step | Rep1 score | Rep1 step 秒 | Rep2 score | Rep2 step 秒 |
|---:|---:|---:|---:|---:|
| 1 | 0.0918 | 182.18 | 0.1270 | 182.79 |
| 2 | 0.1035 | 170.77 | 0.1113 | 184.11 |
| 3 | 0.1230 | 181.69 | 0.1172 | 208.49 |
| 4 | 0.1738 | 179.02 | 0.1797 | 206.98 |
| 5 | 0.1758 | 186.67 | 0.1699 | 193.39 |
| 6 | 0.2148 | 179.60 | 0.2148 | 186.63 |
| 7 | 0.2012 | 188.87 | 0.2129 | 191.90 |
| 8 | 0.2012 | 176.45 | 0.2090 | 185.95 |

- Rep1 最后 3 step 平均训练 reward：0.2057。
- Rep2 最后 3 step 平均训练 reward：0.2122。

Repeat 3：

| Step | score | step 秒 |
|---:|---:|---:|
| 1 | 0.1094 | 179.85 |
| 2 | 0.1309 | 174.25 |
| 3 | 0.1270 | 184.35 |
| 4 | 0.1777 | 184.20 |
| 5 | 0.1914 | 181.54 |
| 6 | 0.1934 | 179.32 |
| 7 | 0.2051 | 183.99 |
| 8 | 0.2227 | 175.18 |

- Rep3 最后 3 step 平均训练 reward：0.2070。
- 三次末段训练 reward 排序为 Rep2 > Rep3 > Rep1，3500 F1 排序却为 Rep1 > Rep3 > Rep2。这说明训练 batch reward 不能直接预测跨数据集泛化。

### 7.2 SPAD stable

| Step | Rep1 总 reward | Rep1 Actor EM | Rep1 秒 | Rep2 总 reward | Rep2 Actor EM | Rep2 秒 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1270 | 0.1016 | 331.41 | 0.1123 | 0.0801 | 306.12 |
| 2 | 0.0930 | 0.0684 | 297.44 | 0.1168 | 0.0918 | 321.53 |
| 3 | 0.1152 | 0.0898 | 282.77 | 0.1111 | 0.0781 | 301.78 |
| 4 | 0.1797 | 0.1523 | 290.93 | 0.1771 | 0.1543 | 327.11 |
| 5 | 0.1707 | 0.1348 | 295.80 | 0.1701 | 0.1406 | 309.31 |
| 6 | 0.2094 | 0.1855 | 296.52 | 0.2037 | 0.1816 | 313.40 |
| 7 | 0.1750 | 0.1523 | 286.31 | 0.1779 | 0.1543 | 313.42 |
| 8 | 0.1844 | 0.1563 | 313.09 | 0.2039 | 0.1777 | 322.53 |

- Rep1 最后 3 step 平均总 reward：0.1896。
- Rep2 最后 3 step 平均总 reward：0.1952。

Repeat 3：

| Step | 总 reward | Actor EM | step 秒 |
|---:|---:|---:|---:|
| 1 | 0.1363 | 0.1055 | 321.49 |
| 2 | 0.1137 | 0.0898 | 291.94 |
| 3 | 0.1129 | 0.0820 | 304.73 |
| 4 | 0.1820 | 0.1582 | 305.27 |
| 5 | 0.1541 | 0.1211 | 300.49 |
| 6 | 0.1986 | 0.1758 | 299.00 |
| 7 | 0.1988 | 0.1758 | 282.56 |
| 8 | 0.1869 | 0.1641 | 283.39 |

- Rep3 最后 3 step 平均总 reward：0.1948；Actor EM 均值：0.1719。
- stable 总 reward 高于 Actor EM 的部分来自 Teacher backoff。
- Rep3 八个 step 的 Teacher 格式错误数均为 0。

### 7.3 Gold Token-F1 Bonus V2

| Step | Rep1 总 reward | Rep1 stable base | Rep1 bonus | Rep1 bonus应用率 | Rep2 总 reward | Rep2 stable base | Rep2 bonus | Rep2 bonus应用率 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1253 | 0.1164 | 0.0089 | 0.1055 | 0.1389 | 0.1328 | 0.0061 | 0.0742 |
| 2 | 0.1130 | 0.1049 | 0.0082 | 0.1074 | 0.1115 | 0.1049 | 0.0066 | 0.0898 |
| 3 | 0.1309 | 0.1191 | 0.0117 | 0.1621 | 0.1329 | 0.1205 | 0.0124 | 0.1777 |
| 4 | 0.1917 | 0.1840 | 0.0077 | 0.1094 | 0.2083 | 0.2000 | 0.0083 | 0.1230 |
| 5 | 0.1763 | 0.1602 | 0.0162 | 0.2148 | 0.1872 | 0.1721 | 0.0152 | 0.2109 |
| 6 | 0.2133 | 0.2053 | 0.0080 | 0.1191 | 0.2408 | 0.2314 | 0.0094 | 0.1289 |
| 7 | 0.1990 | 0.1908 | 0.0082 | 0.1172 | 0.2183 | 0.2135 | 0.0049 | 0.0781 |
| 8 | 0.2179 | 0.2121 | 0.0057 | 0.0859 | 0.2085 | 0.2016 | 0.0069 | 0.1035 |

- Rep1 最后 3 step：总 reward 0.2101，stable base 0.2027，bonus 0.0073。
- Rep2 最后 3 step：总 reward 0.2225，stable base 0.2155，bonus 0.0071。

Repeat 3：

| Step | 总 reward | stable base | Actor EM | bonus | bonus 应用率 | step 秒 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1102 | 0.1027 | 0.0762 | 0.0074 | 0.0957 | 291.60 |
| 2 | 0.1179 | 0.1107 | 0.0898 | 0.0072 | 0.1016 | 290.35 |
| 3 | 0.1734 | 0.1619 | 0.1328 | 0.0115 | 0.1582 | 275.15 |
| 4 | 0.1940 | 0.1828 | 0.1563 | 0.0112 | 0.1523 | 288.08 |
| 5 | 0.1720 | 0.1563 | 0.1211 | 0.0158 | 0.2188 | 287.45 |
| 6 | 0.2214 | 0.2145 | 0.1953 | 0.0070 | 0.1055 | 281.01 |
| 7 | 0.1856 | 0.1805 | 0.1621 | 0.0051 | 0.0781 | 284.23 |
| 8 | 0.1966 | 0.1914 | 0.1719 | 0.0052 | 0.0898 | 275.81 |

- Rep3 最后 3 step：总 reward 0.2012，stable base 0.1954，bonus 0.0057，bonus 应用率 0.0911。
- 额外 bonus 均值本身不大，但 `norm=false` 后不会被每组独立缩放到与 EM 信号相同的单位方差。
- Rep3 末段训练 reward 最低，3500 F1 也最低；但 Rep1/Rep2 的末段 reward 与评估排序仍不完全可靠，三个点不足以建立可泛化的相关性。
- Rep3 仅 Step 1 出现 1 次 Teacher 格式错误（367 次 Teacher 调用中的 `0.2725%`），Step 2-8 均为 0；该单条按 reward 规则不获得 Teacher 奖励。

## 8. 3500 评估设置

### 8.1 数据集

```text
data/global_train_eval_data/3500e/co_search_ablation.eval.parquet
```

- 样本数：3500；
- SHA256：`bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283`；
- 数据源：2WikiMultihopQA 563、Bamboogle 125、HotpotQA 562、MuSiQue 562、NQ 562、PopQA 563、TriviaQA 563。

### 8.2 推理参数

| 项目 | 值 |
|---|---:|
| Actor temperature | 0 |
| Actor vLLM 数据并行副本 | 6，NPU 0-5 |
| Recall 后端 | 2，NPU 6-7 |
| 评估 batch size | 384 |
| vLLM `max_num_seqs` | 64 |
| trace flush interval | 500 |
| Recall top_n / top_m | 50 / 5 |
| 最大交互轮数 | 6 |

九组评估均为 `3500/3500` 成功、`0` 失败。前六组推理与首次聚合总墙钟约 2 小时 04 分钟；新增三组从 `21:09:18` 到 `22:06:58`，墙钟 57 分 40 秒，之后复用已落盘 trace 完成最终聚合。

## 9. 评估产物

### 9.1 单模型报告

```text
reports/eval/agenticIterRag/
260713-newdata3500-search-r1-512-normfalse-rep1-run1.report.md
260713-newdata3500-search-r1-512-normfalse-rep2-run1.report.md
260713-newdata3500-search-r1-512-normfalse-rep3-run1.report.md
260713-newdata3500-spad-512-stable-normfalse-rep1-run1.report.md
260713-newdata3500-spad-512-stable-normfalse-rep2-run1.report.md
260713-newdata3500-spad-512-stable-normfalse-rep3-run1.report.md
260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep1-run1.report.md
260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep2-run1.report.md
260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep3-run1.report.md
```

对应完整 trace 与 runtime log 位于：

```text
log/eval/agenticIterRag/<task-name>/trace/
log/eval/agenticIterRag/<task-name>/runtime_logs/
```

### 9.2 聚合与 bootstrap

```text
reports/eval/agenticIterRag/260713-newdata3500-normfalse-512-aggregate/report.md
reports/eval/agenticIterRag/260713-newdata3500-normfalse-512-aggregate/summary.json
reports/eval/agenticIterRag/260713-newdata3500-normfalse-512-three-repeats-aggregate/report.md
reports/eval/agenticIterRag/260713-newdata3500-normfalse-512-three-repeats-aggregate/summary.json
```

paired bootstrap 设置：

- 对同一问题的两模型结果配对；
- 10,000 次重采样；
- bootstrap seed 42；
- 报告 95% CI；
- 差值方向统一为 `right - left`。

新增三组评估在 `22:06:58` 已全部完成。初次自动聚合随后因 spec 把三个不同训练 checkpoint 错写成同一模型的三次推理重复而被模型指纹/`repeat_id` 校验拒绝；该失败发生在读取 trace 的校验阶段，没有影响任何评估产物。修正为 13 个独立 checkpoint 模型后，直接复用九份既有 trace 于 `23:15:20` 完成最终聚合，没有重跑推理。

训练编排日志：

```text
log/agenticIterRag/260713-normfalse-512-six-trainings-orchestrator/runner.log
log/agenticIterRag/260713-normfalse-512-three-rep3-trainings-orchestrator/runner.log
```

评估编排日志：

```text
log/eval/agenticIterRag/260713-normfalse-512-six-3500evals-orchestrator/runner.log
log/eval/agenticIterRag/260713-normfalse-512-three-rep3-3500evals-orchestrator/runner.log
```

## 10. 与历史训练结果对比

### 10.1 历史基线

| 历史模型 | norm | inflight | Reward / 关键语义 | EM | F1 | 完整答案率 |
|---|---:|---:|---|---:|---:|---:|
| Search-R1-512 | true | 不适用 | 原始 EM | 0.1180 | 0.1965 | 0.6271 |
| SPAD-512 stable 最佳历史 | true | 1 | `spad_em_teacher_backoff` | 0.1360 | 0.2265 | 0.6989 |
| SPAD-512 stable 同 inflight 对照 | true | 2 | `spad_em_teacher_backoff` | 0.1054 | 0.1798 | 0.5900 |
| SPAD-512 Gold Token-F1 V1 | true | 2 | 较宽松 bonus eligibility | 0.1231 | 0.2046 | 0.6220 |

相关历史报告：

```text
docs/AgenticIterRag_v1/work_reports/260712-16a_四模型3500单次评估与推理加速报告.md
docs/AgenticIterRag_v1/work_reports/260712-17a_SPAD-512_Stable_Stage1重复训练与Inflight消融3500评估报告.md
docs/AgenticIterRag_v1/work_reports/260713-18a_SPAD_GoldTokenF1Bonus新Reward_512与5100训练及3500评估报告.md
```

### 10.2 本轮三次均值相对历史值

| 对比 | EM 差值 | F1 差值 | 解读 |
|---|---:|---:|---|
| Search norm=false 三次均值 - 历史 norm=true | -0.0049 | -0.0040 | 三次均值略低；Rep1 提高而 Rep2/Rep3 降低 |
| Stable norm=false 三次均值 - 历史 inflight2 norm=true | +0.0165 | +0.0257 | 三次都提高，是本轮最一致的相对关系 |
| Stable norm=false 三次均值 - 历史 inflight1 最佳 | -0.0141 | -0.0211 | 仍未恢复历史最佳 stable |
| Gold V2 norm=false 三次均值 - 历史 Gold V1 norm=true | -0.0026 | -0.0017 | 平均略低；同时改变 V2 eligibility 与 norm，不能单因素归因 |
| Gold V2 norm=false 三次均值 - Stable norm=false 三次均值 | -0.0013 | -0.0026 | 排序随第三次结果翻转，差值小于训练重复波动 |

`stream_group_max_inflight` 是必须保留的关键差别：历史 stable 最佳使用 `1`，本轮六个 SPAD checkpoint 均使用 `2`。它不改变 Teacher 单次采样参数和 reward 公式，但会改变 Teacher 请求排队、流式回传及异步调度节奏。当前结果再次说明，不应把 inflight=1 的历史最佳与 inflight=2 实验直接当成只差 norm 的严格消融。

## 11. Paired Bootstrap 关键结果

下表均为同一 3500 问题上的 paired bootstrap，差值为右侧模型减左侧模型。

| 对比 | EM 差值及 95% CI | F1 差值及 95% CI |
|---|---:|---:|
| 历史 Search norm=true -> Search false Rep1 | +0.0091 [0.0020, 0.0166] | +0.0142 [0.0065, 0.0220] |
| 历史 Search norm=true -> Search false Rep2 | -0.0163 [-0.0240, -0.0086] | -0.0195 [-0.0280, -0.0109] |
| 历史 Search norm=true -> Search false Rep3 | -0.0074 [-0.0149, 0.0000] | -0.0069 [-0.0150, 0.0012] |
| Search false Rep1 -> Rep2 | -0.0254 [-0.0337, -0.0174] | -0.0337 [-0.0425, -0.0249] |
| Search false Rep1 -> Rep3 | -0.0166 [-0.0243, -0.0091] | -0.0211 [-0.0295, -0.0129] |
| Search false Rep2 -> Rep3 | +0.0089 [0.0029, 0.0149] | +0.0126 [0.0063, 0.0188] |
| 历史 Stable inflight2 true -> Stable false Rep1 | +0.0074 [0.0011, 0.0137] | +0.0106 [0.0043, 0.0168] |
| 历史 Stable inflight2 true -> Stable false Rep2 | +0.0243 [0.0157, 0.0329] | +0.0376 [0.0287, 0.0464] |
| 历史 Stable inflight2 true -> Stable false Rep3 | +0.0177 [0.0103, 0.0254] | +0.0288 [0.0209, 0.0366] |
| 历史 Stable inflight1 最佳 -> Stable false Rep3 | -0.0129 [-0.0209, -0.0046] | -0.0180 [-0.0263, -0.0094] |
| Stable false Rep1 -> Rep2 | +0.0169 [0.0086, 0.0251] | +0.0270 [0.0182, 0.0355] |
| Stable false Rep1 -> Rep3 | +0.0103 [0.0029, 0.0177] | +0.0182 [0.0106, 0.0261] |
| Stable false Rep2 -> Rep3 | -0.0066 [-0.0137, 0.0009] | -0.0089 [-0.0163, -0.0014] |
| 历史 Gold V1 -> Gold V2 false Rep1 | -0.0057 [-0.0143, 0.0029] | -0.0013 [-0.0107, 0.0079] |
| 历史 Gold V1 -> Gold V2 false Rep2 | +0.0054 [-0.0029, 0.0143] | +0.0065 [-0.0024, 0.0156] |
| 历史 Gold V1 -> Gold V2 false Rep3 | -0.0074 [-0.0157, 0.0009] | -0.0101 [-0.0185, -0.0016] |
| Gold V2 false Rep1 -> Rep2 | +0.0111 [0.0040, 0.0186] | +0.0078 [0.0004, 0.0153] |
| Gold V2 false Rep1 -> Rep3 | -0.0017 [-0.0103, 0.0069] | -0.0088 [-0.0181, 0.0006] |
| Gold V2 false Rep2 -> Rep3 | -0.0129 [-0.0214, -0.0046] | -0.0166 [-0.0255, -0.0078] |
| Stable false Rep1 -> Gold V2 false Rep1 | +0.0046 [-0.0043, 0.0131] | +0.0128 [0.0035, 0.0222] |
| Stable false Rep2 -> Gold V2 false Rep2 | -0.0011 [-0.0091, 0.0071] | -0.0064 [-0.0145, 0.0019] |
| Stable false Rep3 -> Gold V2 false Rep3 | -0.0074 [-0.0149, 0.0000] | -0.0141 [-0.0220, -0.0064] |
| Search false Rep3 -> Stable false Rep3 | +0.0126 [0.0040, 0.0211] | +0.0189 [0.0095, 0.0283] |
| Search false Rep3 -> Gold V2 false Rep3 | +0.0051 [-0.0040, 0.0143] | +0.0048 [-0.0048, 0.0148] |

这些 CI 只衡量固定 checkpoint 在 3500 个问题上的采样不确定性。它们不覆盖“重新训练一次会得到另一个 checkpoint”的训练不确定性。因此：

- 单个 checkpoint 的差值 CI 不跨 0，不等于训练方法稳定更好；
- Search Rep1 相对历史提高，Rep2 下降，Rep3 的 CI 触及或跨过 0，是固定配置重复训练不稳定的直接证据；
- Stable 三个 norm=false checkpoint 相对同 inflight=2 历史模型均提升，但仍不能用单 checkpoint CI 代替跨训练重复的统计结论；
- 对训练方法下结论时，应把 repeat 间差异放在单 checkpoint bootstrap 之前考虑。

## 12. 分数据源 F1

| 模型 | 2Wiki | Bamboogle | Hotpot | MuSiQue | NQ | PopQA | TriviaQA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Search false Rep1 | 0.1086 | 0.2404 | 0.2022 | 0.0925 | 0.2788 | 0.2789 | 0.2968 |
| Search false Rep2 | 0.1011 | 0.1263 | 0.1757 | 0.0508 | 0.2424 | 0.2438 | 0.2597 |
| Search false Rep3 | 0.0924 | 0.1446 | 0.1955 | 0.0556 | 0.2609 | 0.2590 | 0.2844 |
| Stable false Rep1 | 0.0869 | 0.1806 | 0.2011 | 0.0729 | 0.2345 | 0.2772 | 0.2717 |
| Stable false Rep2 | 0.1031 | 0.2530 | 0.2163 | 0.0808 | 0.2929 | 0.2861 | 0.3172 |
| Stable false Rep3 | 0.1026 | 0.2189 | 0.2153 | 0.0859 | 0.2663 | 0.2783 | 0.3007 |
| Gold V2 false Rep1 | 0.1057 | 0.2087 | 0.1951 | 0.0739 | 0.2695 | 0.2821 | 0.2915 |
| Gold V2 false Rep2 | 0.1050 | 0.2426 | 0.2149 | 0.0943 | 0.2627 | 0.2866 | 0.2954 |
| Gold V2 false Rep3 | 0.1053 | 0.2122 | 0.2096 | 0.0760 | 0.2422 | 0.2614 | 0.2683 |

观察：

- Search Rep3 大多落在 Rep1/Rep2 之间，但 2Wiki 继续降低，说明第三轮没有简单复现任一前序 checkpoint；
- Stable Rep3 在七个数据源上都位于或接近前两次范围中部，整体分数也位于 Rep1/Rep2 之间；
- Gold V2 三次在 2Wiki 上接近，Rep3 的 NQ、PopQA、TriviaQA 则明显低于前两次，这些单跳源主导了 Rep3 整体 F1 下降；
- MuSiQue 仍是所有模型最弱的数据源，单次整体分数的提升没有解决复杂多跳问题上的瓶颈。

## 13. 训练 Reward 与评估 Reward 的关系

本轮结果不支持用最后几步训练 reward 直接选择泛化更好的 checkpoint：

| 方法 | 末 3 step reward：Rep1 / Rep2 / Rep3 | 3500 F1：Rep1 / Rep2 / Rep3 | 排序是否同向 |
|---|---:|---:|---|
| Search-R1 | 0.2057 / 0.2122 / 0.2070 | 0.2108 / 0.1771 / 0.1896 | 否，恰好反向 |
| SPAD stable | 0.1896 / 0.1952 / 0.1948 | 0.1904 / 0.2174 / 0.2086 | 是 |
| Gold V2 | 0.2101 / 0.2225 / 0.2012 | 0.2032 / 0.2110 / 0.1944 | 是 |

Stable 与 Gold V2 的三个点呈同向排序，Search-R1 却恰好反向；样本数只有三个，不能建立可靠相关性。训练 reward 是当前 512 数据、当前 rollout 分布上的在线指标；3500 评估覆盖不同问题分布，并以 temperature=0 运行，两者并非同一目标分布。

## 14. 为什么关闭 Std 归一化没有得到清晰单调提升

### 14.1 它修正的是信号尺度，不是信号质量

`norm=false` 可以让 `0.1` backoff 比 `1.0` EM 弱，但不会保证 Teacher backoff 本身总是正确，也不会消除 Teacher 对证据歧义、gold 别名质量及检索噪声的依赖。

### 14.2 Search-R1 的大幅波动说明问题不只在 Teacher

Search-R1 完全不使用 Teacher，三个 repeat 仍出现 EM 极差 0.0254、F1 极差 0.0337。这说明异步 rollout 与优化路径差异本身就是主要方差源之一。

### 14.3 512 数据只训练 8 step，早期路径依赖很强

每个 step 都覆盖较大比例的数据和更新量。早期 rollout 中少量高 reward 轨迹的差异，会影响后续策略分布；8 step 内没有足够长的训练过程把这种差异平均掉。

### 14.4 历史最佳还有 inflight 差异

历史 stable 最佳是 `stream_group_max_inflight=1`，本轮是 `2`。inflight 会影响异步 Teacher 请求顺序。虽然它不改变 reward 公式，但在无法做到严格确定性调度时可能改变实际获得的训练轨迹和更新顺序。

### 14.5 Gold V2 同时改了 eligibility

历史 Gold V1 与本轮 Gold V2 不只是 norm 开关不同。V2 限制 bonus 只在 Actor 合法闭合答案时发放，这一语义修正理论上更对齐 Actor 目标，但它减少了 bonus 覆盖率。因此不能用 V1/V2 的近似持平直接判断 norm 开关无效。

## 15. 后续实验建议

建议按以下顺序继续：

1. 固定 `stream_group_max_inflight=2`，对 stable reward 的 norm=true/false 各做至少 3 个显式不同 seed；这是最干净的 advantage 消融。
2. 对 Gold V2 同样做 norm=true/false，不再与历史 Gold V1 混比，以拆开 eligibility 与 advantage normalization 两个变量。
3. 每个 seed 同时记录数据顺序、Actor rollout seed，以及可能影响异步请求顺序的 worker/service 配置。
4. 训练中增加固定小型 validation，但不要仅按训练 reward 选择 checkpoint。
5. 在结论稳定前，保留历史 inflight=1 最佳模型作为上界参照，不把本轮任一 checkpoint 替换为新的唯一 stable。
6. 若重点是验证 `0.1` backoff 的绝对尺度，优先比较 stable reward；Search-R1 主要用于估计训练系统自身方差。

## 16. 复现实验入口

训练：

```bash
bash tasks/train_tasks/agenticIterRag/run_260713_normfalse_512_stage1_six_trainings.sh
bash tasks/train_tasks/agenticIterRag/run_260713_normfalse_512_stage1_three_rep3_trainings.sh
```

评估与聚合：

```bash
bash tasks/eval_tasks/agenticIterRag/run_260713_normfalse_512_six_3500evals.sh
bash tasks/eval_tasks/agenticIterRag/run_260713_normfalse_512_three_rep3_3500evals.sh
```

再次执行前必须修改 task/run 名，避免覆盖本报告对应的正式产物。

## 17. 最终状态

- 九次训练：全部完成；
- 九次 3500 评估：全部完成，均为 3500 成功、0 失败；
- 聚合与 10,000 次 paired bootstrap：完成；
- checkpoint、训练日志、推理 trace、单模型报告、聚合报告：均已落盘；
- 当前没有仍需跟踪的训练或评估任务；评估服务已经退出并释放 NPU。

本轮最稳妥的结论是：`norm_adv_by_std_in_grpo=false` 已按设计恢复了 reward 的绝对尺度，但相同 `data_seed=42` 下的三次重复仍不足以证明它稳定改善泛化。Stable 的三次均值目前最高，Gold V2 的离散程度最小，但两者差距小于训练重复方差；下一步仍需要严格的不同 seed、同 inflight、同 reward 版本的因子消融。
