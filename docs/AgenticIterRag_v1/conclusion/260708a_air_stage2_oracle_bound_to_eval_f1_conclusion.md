# AIR stage2 reranker oracle bound 到全局 eval F1 的换算结论

日期：2026-07-08

## 背景

本结论来自 fixed prompt 之后的 stage2 reward-bound 诊断实验。实验目标不是训练 reranker，而是用固定策略估计：如果 reranker 能把包含 gold answer 的 doc 提前到 top5，最终 continuation answer F1 在真实 agent 流程中大约还有多少可见收益。

相关数据：

- branch train/eval 集总样本数：5100
- hard/improvable 机会样本切片：`top50_hit_top5_miss_baseline0`
- 机会样本数：396 / 5100 = 7.76%
- reward-bound 诊断输出：`outputs/agenticIterRag/reward_bound_diagnosis/260708_fixed_prompt_top50miss_n100`

诊断结果，过滤条件为 `top50_hit_top5_miss_baseline0`：

```text
baseline mean: 0.0000
identity mean: 0.0408
random   mean: 0.1503
oracle   mean: 0.3545

oracle improved: 37 / 98
oracle better than identity: 33 / 98
oracle worse than identity: 0 / 98
oracle max_turns: 21 / 100
oracle answered: 77 / 100
```

## 换算公式

最终 eval F1 提升可以粗略按下面方式估算：

```text
全局 eval F1 提升 ~= 机会样本占比 * 机会样本上的平均可提升幅度
```

其中：

```text
机会样本占比 = 396 / 5100 = 0.0776
oracle 相对 identity 的平均提升 = 0.3545 - 0.0408 = 0.3137
```

所以，对比当前原始 agent/top5 identity 策略，全局 answer F1 的 oracle 可见收益上限约为：

```text
0.0776 * 0.3137 = 0.0244
```

也就是约：

```text
+2.4 F1 points
```

如果更乐观地从 `baseline0` 直接看 oracle 绝对收益：

```text
0.0776 * 0.3545 = 0.0275
```

也就是约：

```text
+2.7 F1 points
```

## 结论

当前诊断给出的 stage2 reranker 在全局最终 answer F1 上的 oracle 可见收益上限，大约是：

```text
+2.4 到 +2.8 F1 points
```

这不是训练后 reranker 的预期收益，而是“oracle reranker 能做到的上限估计”。真实训练后通常会低于这个值，原因包括：

- 模型不可能 100% 学到 oracle top5。
- reranker 可能在原本已答对的样本上误排，带来负收益。
- 包含 gold answer 字符串的 doc 不一定包含足够推理证据。
- 多跳问题中，第一步 doc 正确也不保证后续 search 和 answer 正确。
- frozen agent 可能没有正确利用 observation。
- continuation 可能走到 max_turns。
- final answer F1 reward 比较严格，答案表述偏一点也会丢分。

因此，后续训练的现实目标可以设为：

```text
短期有效：+0.5 到 +1.0 F1
比较理想：+1.0 到 +1.5 F1
接近 oracle 上限：+2.0 F1 以上
```

关键判断：

```text
reranker 确实有提升空间，但空间主要集中在约 8% 的 hard/improvable 样本里；
全量 eval 的最终 F1 上限自然不会特别大。
```

## 对后续实验的含义

不建议继续做全量盲训作为主线。更合理的 stage2 消融顺序是：

1. 先在 `top50_hit_top5_miss_baseline0` hard 子集上训练，确认 reranker 能学习到可迁移的排序信号。
2. 引入 hard/improvable 加权采样，而不是让大量无收益样本稀释 reward。
3. 在 final answer F1 之外补充 top5 answer-hit/evidence-hit 辅助 reward，降低 continuation answer reward 的噪声。
4. 确认 hard 子集训练有效后，再做全量混合训练，并重点监控全局 F1、hard slice F1、原本 answered 样本的退化率。
