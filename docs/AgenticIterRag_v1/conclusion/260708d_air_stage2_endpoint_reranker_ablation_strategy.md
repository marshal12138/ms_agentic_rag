# AIR Stage2 Reranker End-Point 消融策略说明

日期：2026-07-08

## 背景

前期 first-point hard subset 的问题在于：训练样本取的是 agent 第 0 个 search point，但 label/reward 使用的是原始 question 的 final gold answer。对于多跳问题，第 0 步 search 很可能本来就不应该直接命中 final answer，因此用 final gold answer 判断第 0 步 rerank 是否正确会引入 label 噪声。

因此本轮消融切换到 end-point 数据。end-point 更接近最终回答前的检索状态，top5 排序结果对 frozen agent continuation answer F1 的影响更直接，也更适合作为 stage2 reranker 的训练切入点。

## 数据策略

本轮使用 end-point branch dataset，并通过配置化 stage `filter_reranker_branch_dataset` 过滤 hard/improvable 子集。

过滤条件：

- `top50_hit=True`：end-point 的 top50 候选中至少有文档包含 final gold answer 字符串。
- `top5_hit=False`：原始 top5 中没有包含 final gold answer 的文档，说明 reranker 有可改善空间。
- `baseline_reward=0.0`：原始 agent continuation final answer reward 为 0，说明该样本在当前策略下没有答对。

过滤后数据：

- manifest：`data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason_hard_top50hit_top5miss_baseline0/manifest.json`
- 样本数：453
- step policy：end-point

这类样本是 reranker stage2 最值得优先训练的机会样本：当前 top5 错过了含答案证据，但 top50 内还有可救回的候选。

## 消融目标

本轮消融不直接追求长训收益，而是先判断 stage2 reward 和训练配置是否能产生稳定、可学习的正向信号。核心问题有三个：

1. 只用 final answer reward 是否已经能学到排序策略。
2. evidence-hit 辅助 reward 是否能降低 final answer F1 的稀疏性和噪声。
3. evidence-hit 权重过低或过高时，是否会冲淡 final answer reward。

## 固定训练配置

为保证各组可比，三组消融固定以下训练设置：

- 数据：同一个 end-point hard subset，453 条。
- rollout：`rollout.n=4`。
- prompt：修正后的 short-reason rerank prompt，并作为默认 prompt。
- max response length：256。
- learning rate：`5e-6`。
- KL：`0.02`。
- total training steps：3。
- train batch size：64。
- continuation frozen agent：stage1 checkpoint frozen agent。

`rollout.n=4` 是本轮的保守选择：比 n=1/2 更能降低单样本采样噪声，同时训练成本仍可接受。正式训练仍沿用 n=4，避免在策略尚未完全验证时直接放大到 n=8。

`max_response_length=256` 的目的不是压缩有效信息，而是约束 reranker 输出只完成结构化排序和短理由，避免旧 prompt 下长输出、格式不闭合、clip ratio 上升、训练时间膨胀等问题。实际三组消融中 response length 均值约 65-80 token，clip ratio 全程为 0，说明 256 足够。

## Reward 消融组

### A1：answer-only

配置：

- `reward_strategy=answer_reward`
- `evidence_hit_weight=0.0`

目的：

验证纯 final answer reward 是否已经能推动 reranker 学习。该组最贴近最终目标，但 reward 稀疏且受 frozen agent continuation 能力影响大。

结果：

```text
critic/score/mean:
step1 0.1229
step2 0.1309
step3 0.1587
```

观察：

- reward 稳定上升。
- clip ratio 全程为 0。
- step1 有 2 条格式罚分，step2/3 为 0。
- 正分样本数从 39/256 增加到 50/256。

结论：

answer-only reward 本身不是无效的，reranker 确实能从 end-point hard subset 中学到排序信号。但 final answer reward 稀疏，上升幅度相对慢。

### A2：answer + evidence-hit，w=0.1

配置：

- `reward_strategy=answer_reward_plus_evidence_hit`
- `evidence_hit_weight=0.1`

其中 evidence-hit 定义为：reranker 输出 top5 中至少有一篇文档包含 normalized final gold answer 字符串，则给 evidence bonus。

目的：

验证较弱 evidence bonus 是否能作为辅助 shaping signal，降低 final answer reward 的稀疏性，同时不明显干扰最终 answer 目标。

结果：

```text
critic/score/mean:
step1 0.1499
step2 0.1633
step3 0.1459
```

观察：

- step1/step2 高于 A1，但 step3 回落。
- evidence bonus 带来了大量 0.1 小正分。
- step2 的 1.0 样本为 29，step3 降到 22。
- clip ratio 全程为 0。

结论：

w=0.1 能增加低强度正反馈，但没有稳定提升 final-answer 高分样本。它更像是在缓解稀疏性，而不是稳定改善最终 answer reward，因此不作为正式训练首选。

### A3：answer + evidence-hit，w=0.2

配置：

- `reward_strategy=answer_reward_plus_evidence_hit`
- `evidence_hit_weight=0.2`

目的：

验证更强 evidence bonus 是否能提供足够的排序学习信号，并观察是否会冲淡 final answer reward。

结果：

```text
critic/score/mean:
step1 0.1866
step2 0.1921
step3 0.2195
```

观察：

- reward 三步连续上升。
- clip ratio 全程为 0。
- 格式罚分从 3 条降到 1 条。
- step3 正分样本数达到 131/256。
- 1.0 高分样本没有崩：step1 为 22，step2 为 31，step3 为 24。
- 0.8 及以上样本在 step3 明显增加。

结论：

w=0.2 是三组里最稳定的策略。它确实引入了更多 evidence 小正分，但没有明显牺牲 final answer 高分样本，并且总 reward 趋势最稳定。

## 判定标准

本轮没有只看 reward 均值，而是同时检查以下指标：

- `critic/score/mean` 是否连续上升。
- `response_length/clip_ratio` 是否为 0。
- 输出是否稳定包含 `<rerank>` 和 `</rerank>`。
- 格式罚分是否下降或保持低位。
- 1.0 高分样本是否崩塌。
- 正分样本数是否增加。
- KL 和 grad 是否没有异常放大。
- 单 step 耗时是否可接受。

这个判定标准用于避免两类误判：

1. reward 上升只是格式罚分减少造成的假象。
2. evidence bonus 抬高均值，但 final answer 高分样本下降。

## 最终选择

正式训练采用 A3：

```text
reward_strategy = answer_reward_plus_evidence_hit
evidence_hit_weight = 0.2
rollout.n = 4
max_response_length = 256
total_training_steps = 8
save_freq = 4
```

正式训练 run：

```text
260708-124753-407577-pipeline-agentic_iter_rag_v1_endpoint_hard_short_reason_ans_ev_w02_n4_1epoch
```

正式训练启动前确认：

- filter stage 已完成。
- 使用 end-point hard subset，453 条。
- `trainer.total_training_steps=8`。
- `actor_rollout_ref.rollout.n=4`。
- `data.max_response_length=256`。
- `reward_strategy=answer_reward_plus_evidence_hit`。
- `evidence_hit_weight=0.2`。

## 风险和后续验证

A3 的训练 reward 最稳定，但它仍然是训练期 reward，不等价于最终 eval answer F1。由于 evidence-hit bonus 基于 final gold answer 字符串命中，它可能提升 top5 evidence 命中率，但最终是否转化为 agent answer F1，还取决于 frozen/online agent 是否能利用证据。

因此正式训练完成后需要做两类验证：

1. reranker 层面：top5 answer-hit / evidence-hit 是否提升。
2. agent 层面：真实评估集的 final answer F1 是否提升，且原本已答对样本是否没有明显被误排伤害。

本轮消融的结论是：end-point hard subset 上，stage2 reranker 训练是有学习信号的；在当前配置下，`answer_reward_plus_evidence_hit` 且 `evidence_hit_weight=0.2` 是最适合进入正式训练的策略。
