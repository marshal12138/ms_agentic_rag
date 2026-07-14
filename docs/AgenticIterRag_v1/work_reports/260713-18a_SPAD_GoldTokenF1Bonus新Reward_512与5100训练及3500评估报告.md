# SPAD Gold Token-F1 Bonus 新 Reward：512 / 5100 Stage1 训练与 3500 评估报告

日期：2026-07-13

> 后续代码修订（2026-07-13）：本报告中的 512/5100 checkpoint 使用初版 bonus 条件，即只要求
> Teacher 输出合法。实验结束后，同名 reward 的 extra bonus 条件已收紧为：Actor 必须输出完整非空的
> `<answer>...</answer>`，且 Teacher evidence status 必须为 `supported_answer` 或
> `ambiguous_evidence`。原 `spad_em_teacher_backoff` 基础奖励不变。新审计版本为
> `actor_answer_closed_teacher_supported_v2`；本报告的历史指标不代表该修订版本的效果。

## 1. 结论

本轮独立实现了 reward `spad_em_teacher_backoff_gold_token_f1_bonus`，完成了 512 和 5100
两个规模的 SPAD Stage1 正式训练，并分别在同一份 3500e 数据上进行一次确定性评估。两次训练
都只执行 Stage1；两次评估均为 3500/3500 成功、失败 0。

| 模型 | Reward | Inflight | EM | F1 | 完整答案率 | 平均搜索数 | 重复查询率 | Max-turn 率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 512 历史最佳 | stable | 1 | **0.1360** | **0.2265** | **0.6989** | 2.3391 | 0.3340 | 0.2426 |
| 512 旧 reward 同调度复训 | stable | 2 | 0.1054 | 0.1798 | 0.5900 | 2.3566 | 0.3466 | 0.2363 |
| 512 新 reward | token-F1 bonus | 2 | 0.1231 | 0.2046 | 0.6220 | 2.4654 | 0.3820 | 0.2511 |
| 5100 旧 reward | stable | 2 | **0.1923** | **0.2700** | **0.7220** | 2.6557 | 0.5906 | 0.2443 |
| 5100 新 reward | token-F1 bonus | 2 | 0.1837 | 0.2576 | 0.6334 | 3.0071 | 0.5763 | 0.3589 |

主要结论：

1. 512 新 reward 相对同为 `inflight=2` 的旧 reward 复训，EM `+0.0177`、F1 `+0.0248`，
   3500 题 paired bootstrap 置信区间均不跨 0；但仍显著低于更早的历史最佳 512 checkpoint。
2. 5100 新 reward 相对旧 reward 没有改善：EM `-0.0086` 的区间跨 0，F1 `-0.0125` 的区间
   不跨 0；完整答案率下降 `-0.0886`，Max-turn 率从 0.2443 上升到 0.3589。
3. 在新 reward 内部，5100 相对 512 的 EM/F1 提高 `+0.0606/+0.0530`，区间均不跨 0，说明
   扩大训练规模仍然有效；但新 reward 没有超过同规模的旧 reward。
4. 因此当前不能采用新 reward 替代 stable reward。它在一次 512 复训对照上有收益，但在更有代表性
   的 5100 规模上出现 F1 和答案完整性退化。下一步应先做 bonus 权重/触发条件消融，而不是直接扩大训练。

这里的 bootstrap 只刻画两个既定 checkpoint 在 3500 个问题上的逐题差异，不包含重新训练方差。
actor rollout 仍是随机采样，单次训练对单次训练不能证明 reward 的总体因果效应。

## 2. 新 Reward 原理

新 reward 完整保留 `spad_em_teacher_backoff` 的基础行为：

- 每题采样 8 条 rollout，并按题分组。
- 组内只要存在一条 actor answer 命中 gold EM，各轨迹仍只按自身 EM 得 1 或 0。
- 只有整组 actor EM 全为 0 时才调用 Teacher；Teacher 判断证据支持或存在歧义时，stable base
  reward 给 0.1，否则给 0。

在 stable base reward 计算结束后，新模块只对同时满足下列条件的轨迹追加 bonus：

- `group_all_em_zero=true`；
- Teacher 确实被调用；
- Teacher 输出成功按约定格式解析；
- 没有 Teacher format error；
- Teacher answer 非空。

公式为：

```text
teacher_gold_token_f1 = max(token_f1(teacher_answer, gold_alias_i))
bonus = 0.1 * teacher_gold_token_f1
final_reward = stable_base_reward + bonus
```

token-F1 同时考虑 precision 和 recall。Teacher answer 比 gold 更长时不会像 EM 一样直接归零，但
无关扩写会降低 precision，因此比“gold token 被包含即可”的单向覆盖率更保守。

## 3. 实现独立性

stable reward 文件没有被改写为新公式。新模块先调用 stable 的公开 batch 入口，再对结果副本追加
bonus 和审计字段：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_gold_match_bonus.py
```

新增审计字段包括：

- `base_reward`
- `teacher_gold_token_f1`
- `teacher_gold_token_f1_bonus`
- `teacher_gold_token_f1_bonus_applied`
- `teacher_gold_token_f1_bonus_applied_count`
- `teacher_gold_token_f1_bonus_weight`

路由在 `search_policy_rl.py` 中按精确 reward type 选择独立模块和函数；stable type 仍走原模块。
独立配置块位于 `AgenticIterRag/config/agent_training/spad_rag_base.yaml`。本轮 overlay 只选择新 type，
不覆盖旧 reward 实现。

相关回归测试共 17 项通过，覆盖：基础分不变、多个 alias 取最大 token-F1、包含/扩写情形、
非 eligible 记录不加分、非法负权重、配置路由和 prompt version 传播。Python 编译检查也通过。

另外，`spad_rag_base.yaml` 的默认停止点已经改为 `search_policy_rl`。因此后续 SPAD 训练默认只执行
Stage1；Stage2/Stage3 只有在显式 overlay 改写停止点和 enable 配置时才会执行。

## 4. 训练配置与随机性

| 参数 | 512 | 5100 |
|---|---:|---:|
| 初始模型 | Qwen3-1.7B Base | Qwen3-1.7B Base |
| train max samples | 512 | 5100 |
| VERL steps | 8 | 79 |
| 每 step 问题数 | 64 | 64 |
| 每题 rollout | 8 | 8 |
| actor temperature / top_p | 1 / 1 | 1 / 1 |
| data shuffle / seed | true / 42 | true / 42 |
| reward | token-F1 bonus | token-F1 bonus |
| `stream_group_max_inflight` | 2 | 2 |
| Stage2 / Stage3 | 未执行 | 未执行 |

Teacher 共同参数：GLM-4.7-Flash，NPU 4-5，TP=2，BF16，temperature=0，top_p=1，
max_tokens=512，timeout=180 秒，thinking=false，batch workers=16。

训练数据：

```text
512:  data/global_train_eval_data/512t/co_search_ablation.train.parquet
      SHA256 2f9eb86fb40fbb69fab2aca7f6a4e4a05d6879e6dbbcd0fbe1d73e1a1a010558
5100: data/global_train_eval_data/5100t/co_search_ablation.train.parquet
      SHA256 6e9307a8b3a866ecd045170bc0e92048e7e00fba0a0098b4ced5dd227ba9b09c
```

512 实际记录 8 x 64 = 512 个 prompt slot、4096 条 rollout。5100 配置沿用历史正式设置，
VERL 实际记录 79 x 64 = 5056 个 prompt slot、40448 条 rollout；因此“5100”是 train-max-samples
实验名，不应误写成实际完成了 5100 个 prompt update。

已固定数据 seed 和 Teacher 确定性解码，但 actor 使用 temperature=1 的随机 rollout，Ray/vLLM
异步调度也没有为每个 `question + rollout_index + step` 绑定独立 seed，所以不是位级可复现训练。

## 5. Run 与 Checkpoint

| 实验 | Run | HF checkpoint | 权重 SHA256 |
|---|---|---|---|
| 512 新 reward | `260713-011350-061908-...newdata_512_gold_token_f1_bonus_stage1` | `checkpoints/AIR/260713-011350-061908-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_bonus_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `0a802732...0361a3` |
| 5100 新 reward | `260713-022724-631051-...newdata_5100_gold_token_f1_bonus_stage1` | `checkpoints/AIR/260713-022724-631051-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_bonus_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` | `e33af413...a86565` |
| 512 历史最佳 | `260711-103304-616277-...newdata_512` | `checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `df6385a0...87c4ea` |
| 512 旧 reward inflight=2 | `260712-143738-025140-...inflight2_ablation` | `checkpoints/AIR/260712-143738-025140-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_stage1_inflight2_ablation/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8` | `d9ca546b...55af93` |
| 5100 旧 reward | `260711-235953-727858-...newdata_5100` | `checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79` | `41f869b0...fc25d0` |

两个新 run 的 manifest 均只选择并完成 `search_policy_rl`，HF finalizer 完成分片合并、Transformers
本地加载校验和原子落盘；训练和评估结束后 8 张 NPU 均已释放。

## 6. 训练 Reward 审计

| 训练 | 平均 base | 平均 bonus | 平均 final | Actor EM | Actor F1 | 正 bonus | Teacher 调用 | 格式错误 | 单 step 平均 | step 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.1489 | 0.0131 | 0.1620 | 0.1189 | 0.2129 | 735/4096 | 3014/4096 | 5 | 305.6 秒 | 40分44秒 |
| 5100 | 0.2957 | 0.0079 | 0.3035 | 0.2722 | 0.3460 | 4679/40448 | 23347/40448 | 16 | 286.4 秒 | 6小时17分03秒 |

| 训练 | 最后 3 步 base | 最后 3 步 bonus | 最后 3 步 final | 最后 3 步 EM | 最后 3 步 F1 | 最后 3 步平均耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 512（steps 6-8） | 0.1716 | 0.0115 | 0.1831 | 0.1419 | 0.2255 | 320.6 秒 |
| 5100（steps 77-79） | 0.3122 | 0.0058 | 0.3180 | 0.2891 | 0.3500 | 306.5 秒 |

512 中正 bonus 覆盖率为 17.9%，5100 为 11.6%；Teacher 调用率也从 73.6% 降到 57.7%。
这符合触发条件：策略的 actor EM 提高后，`group_all_em_zero` 组减少，新 bonus 会自然变稀疏。
因此固定权重 0.1 在训练前段/小规模上的相对作用更强，在 5100 后段只贡献约 0.0058 的均值。

## 7. 3500 评估协议

评估数据：

```text
data/global_train_eval_data/3500e/co_search_ablation.eval.parquet
SHA256 bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283
```

统一协议：no-ranker、Recall Top N=50、模型可见 Top M=5、最多 6 轮 assistant、
temperature=0、top_p=1、6 个 Actor vLLM replica、2 个 Recall replica、infer batch=384、
flush every N=500。每个 checkpoint 只评估一次；差值使用按题 paired bootstrap 10000 次、seed 42。

两个新模型的单次报告：

```text
reports/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-bonus-stage1-run1.report.md
reports/eval/agenticIterRag/260713-newdata3500-spad-5100-gold-token-f1-bonus-stage1-run1.report.md
```

512 推理墙钟 813.2 秒；5100 推理墙钟 922.8 秒。两次均保存 metrics、完整 trace、summary、
run config 和带 checkpoint/data 哈希的 eval manifest。

## 8. Paired Bootstrap

| 比较（右减左） | EM 差值及 95% CI | F1 差值及 95% CI | 完整答案率差值及 95% CI |
|---|---:|---:|---:|
| 512 历史最佳 inflight=1 -> 512 新 reward inflight=2 | -0.0129 [-0.0203, -0.0051] | -0.0220 [-0.0295, -0.0142] | -0.0769 [-0.0917, -0.0617] |
| 512 旧 reward inflight=2 -> 512 新 reward inflight=2 | +0.0177 [0.0097, 0.0257] | +0.0248 [0.0163, 0.0330] | +0.0320 [0.0166, 0.0477] |
| 5100 旧 reward inflight=2 -> 5100 新 reward inflight=2 | -0.0086 [-0.0177, 0.0009] | -0.0125 [-0.0224, -0.0025] | -0.0886 [-0.1046, -0.0726] |
| 512 新 reward -> 5100 新 reward | +0.0606 [0.0497, 0.0720] | +0.0530 [0.0420, 0.0644] | +0.0114 [-0.0071, 0.0303] |
| 512 历史最佳 -> 5100 旧 reward | +0.0563 [0.0457, 0.0674] | +0.0435 [0.0330, 0.0547] | +0.0231 [0.0069, 0.0394] |

512 的“同 inflight 对照”比历史最佳更接近 reward 消融，但仍是两次独立随机训练；历史最佳对比又
同时包含 `inflight=1 -> 2` 和代码时间点差异。二者应同时报告，不能只选择有利的一组。

## 9. 分数据源 F1

| 数据源 | 512 历史最佳 | 512 旧 inflight=2 | 512 新 | 5100 旧 | 5100 新 |
|---|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 0.1107 | 0.0905 | 0.1043 | 0.1497 | 0.1731 |
| Bamboogle | 0.2485 | 0.1742 | 0.2595 | 0.2701 | 0.2807 |
| HotpotQA | 0.2443 | 0.1879 | 0.2021 | 0.3265 | 0.3187 |
| MuSiQue | 0.0967 | 0.0656 | 0.0884 | 0.0886 | 0.0962 |
| NQ | 0.2966 | 0.2179 | 0.2696 | 0.3637 | 0.3287 |
| PopQA | 0.2831 | 0.2587 | 0.2694 | 0.3262 | 0.3088 |
| TriviaQA | 0.3227 | 0.2593 | 0.2813 | 0.3654 | 0.3147 |

5100 新 reward 在 2Wiki、Bamboogle、MuSiQue 上提高，但在 HotpotQA 和三个 single-hop 数据源上
下降，尤其 NQ/TriviaQA。这说明退化不是所有数据源一致的小噪声，而更像策略行为发生偏移。

## 10. 为什么 5100 没有提升

以下是由训练/评估统计支持的机制推断，不是多 seed 因果证明：

1. **bonus 只作用于全零 EM 的困难组，并随训练变稀疏。** 5100 全程 bonus 均值只有 0.0079，
   最后三步降至 0.0058；它对后期梯度排序的影响有限。
2. **Teacher-answer token-F1 奖励的是 Teacher 与 gold 的接近程度，不直接奖励 actor 最终答案完整。**
   它可能提高“证据足以让 Teacher 生成相关答案”的轨迹分数，却没有额外约束 actor 及时输出完整答案。
3. **评估行为显示搜索/停止策略变差。** 5100 新模型平均搜索数从 2.6557 增至 3.0071，Max-turn
   从 0.2443 增至 0.3589，完整答案率从 0.7220 降至 0.6334。F1 下降与未及时完成答案一致。
4. **bonus 的相对影响在困难样本上更集中。** Teacher 被调用时说明 8 条 actor rollout 全部 EM=0；
   对这些组按 Teacher answer 与 gold 的匹配度加分，可能强化“继续搜索、让证据更可判定”的轨迹，
   但未区分这些轨迹最终能否由 actor 自己正确闭合。
5. **训练 reward 不能直接预测确定性评估。** 5100 的 rollout final reward/Actor EM 较高，但这是
   temperature=1、训练数据上的统计；评估是 temperature=0 的未见 3500 题，目标和分布并不等价。

更稳妥的下一轮消融：保持其余配置不变，先比较 bonus weight 0.02/0.05/0.1；或者只在 Teacher
判断 `supported_answer` 且 actor 已合法闭合答案时加 bonus。每个设置至少跑多个训练 seed，并同时把
完整答案率、Max-turn、平均搜索数设为主要指标，不能只看训练 reward。

## 11. 产物位置

训练入口与 overlay：

```text
tasks/train_tasks/agenticIterRag/run_260713_AIR_spad_qwen3_1_7b_glm47_gold_match_bonus_512_stage1.sh
tasks/train_tasks/agenticIterRag/run_260713_AIR_spad_qwen3_1_7b_glm47_gold_match_bonus_5100_stage1.sh
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_gold_match_bonus_512_stage1_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_gold_match_bonus_5100_stage1_overlay.yaml
```

聚合规格和结果：

```text
tasks/eval_tasks/agenticIterRag/newdata_model_eval_run_spec.260713_gold_token_f1_bonus.json
reports/eval/agenticIterRag/260713-gold-token-f1-bonus-3500-aggregate/report.md
reports/eval/agenticIterRag/260713-gold-token-f1-bonus-3500-aggregate/summary.json
```

本报告：

```text
docs/AgenticIterRag_v1/work_reports/260713-18a_SPAD_GoldTokenF1Bonus新Reward_512与5100训练及3500评估报告.md
```
