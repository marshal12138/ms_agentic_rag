# AIR stage2 end-point hard slice 分布与 reward-bound 结论

日期：2026-07-08

## 目标

在 first-point hard subset 被确认存在 step-level evidence 偏差后，本轮改为分析 end-point：

1. 每条 trajectory 取最后一个 search/rerank 点。
2. 分析 top50/top5 对 final gold answer 的命中分布。
3. 对 end-point hard slice 跑固定策略 reward-bound continuation，换算最终全局 answer F1 可见提升。

## 数据构造

新增 end-point branch dataset overlay：

```text
tasks/train_tasks/agenticIterRag/configs/rebuild_branch_260704e_endpoint_short_reason_overlay.yaml
```

生成的 end-point branch manifest：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/manifest.json
```

配置：

```text
sample_count = 5100
step_policy = end_point
candidate_top_n = 50
visible_top_m = 5
prompt_template_version = cosearch_rerank_topm_v1_short_reason_fixed_example
```

筛选出的 end-point hard slice：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason_hard_top50hit_top5miss_baseline0/manifest.json
```

筛选条件：

```text
top50_hit == true
top5_hit == false
baseline_reward == 0.0
```

结果：

```text
sample_count = 453
baseline_reward = 0.0 for all 453
```

step index 分布：

```text
step_index 0: 3
step_index 1: 178
step_index 2: 227
step_index 3: 42
step_index 4: 3
```

这说明 end-point hard slice 不再固定取第 0 个 search 点，而是覆盖多跳轨迹的最后搜索点。

## end-point top50/top5 分布

全量 5100 条 end-point 样本，按当前 answer normalize 统计：

```text
top50_hit = 3897 / 5100 = 76.41%
top5_hit  = 3255 / 5100 = 63.82%
top50_hit_top5_miss = 642 / 5100 = 12.59%
top50_hit_top5_miss_baseline0 = 453 / 5100 = 8.88%
top50_miss = 1203 / 5100 = 23.59%
baseline0 = 1945 / 5100 = 38.14%
baseline1 = 2236 / 5100 = 43.84%
```

从纯分布看，最干净的 reranker 机会集是：

```text
top50_hit_top5_miss_baseline0 = 453 / 5100 = 8.88%
```

理论 count ceiling：

```text
如果这 453 条全从 F1=0 救到 F1=1:
global F1 ceiling = 453 / 5100 = +8.88 F1 points
```

若把全部 `top50_hit_top5_miss` 642 条中的 partial-baseline 样本也纳入，理论 ceiling 为：

```text
sum(1 - baseline_reward) / 5100 = +9.93 F1 points
```

但这只是字符串命中和 baseline 分布给出的极限上限，不代表 frozen agent continuation 真的能答对。

## reward-bound 实验

实验路径：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/reward_bound/260708_endpoint_hard_top50hit_top5miss_baseline0_n100_identity_oracle
```

命令核心配置：

```text
branch_manifest = end-point hard slice manifest
sample_count = 100
sample_mode = random
strategies = identity, oracle
reward_strategy = answer_reward
workers = 8
```

输出文件：

```text
reward_bound_summary.json
reward_bound_results.jsonl
reward_bound_report.md
```

结果：

```text
baseline mean = 0.0000

identity mean = 0.0400
oracle   mean = 0.2405
oracle - identity = 0.2005

oracle improved = 23 / 100
oracle worse    = 2 / 100
oracle same     = 75 / 100

identity answered = 80 / 100
identity max_turns = 20 / 100

oracle answered = 83 / 100
oracle max_turns = 17 / 100
```

关键观察：

1. end-point oracle 明显优于 identity，但提升没有纯 top50/top5 分布暗示得那么大。
2. oracle p50 仍然是 0，说明很多样本即使把含 final answer 的 doc 放进 top5，frozen agent 也未必能答对。
3. 有 17% oracle 仍走到 max_turns。
4. 有 2 个样本 oracle 比 identity 更差，说明替换 top5 仍可能破坏已有上下文或触发不同 continuation 路径。

## 换算到全局最终 F1

换算公式：

```text
global F1 lift ~= hard_slice_share * mean_reward_gain_on_hard_slice
```

其中：

```text
hard_slice_share = 453 / 5100 = 0.0888235
mean_reward_gain = oracle_mean - identity_mean = 0.2405 - 0.0400 = 0.2005
```

所以：

```text
global F1 lift ~= 0.0888235 * 0.2005
                = 0.01781
```

也就是：

```text
oracle reranker 相对 identity 的可见提升约 +1.78 F1 points
```

如果从原始 baseline0 绝对收益看：

```text
0.0888235 * 0.2405 = 0.02136
```

也就是：

```text
oracle reranker 相对原始 baseline0 的绝对可见收益约 +2.14 F1 points
```

因此 end-point hard slice 的真实 continuation bound 比纯分布上限低很多：

```text
理论 count ceiling: +8.88 到 +9.93 F1
真实 n=100 oracle continuation bound: +1.78 到 +2.14 F1
```

## 结论

end-point 比 first-point 在 label 语义上更合理，因为最后一次 search 更接近 answer-seeking step。但真实 reward-bound 显示：

```text
stage2 reranker 的最终全局 F1 oracle 可见收益大约在 +1.8 到 +2.1 F1 points。
```

这意味着后续训练目标应更保守：

```text
短期有效: +0.4 到 +0.8 F1
比较理想: +0.8 到 +1.2 F1
接近 oracle: +1.5 F1 以上
```

建议下一步不要直接全量训练，而是在 end-point hard slice 上做短步消融：

1. answer-only vs answer+evidence-hit。
2. evidence-hit 对 end-point 更合理，可以保留 `w=0.2` 作为主候选。
3. 使用 short-reason prompt、rollout.n=4、lr=5e-6、KL=0.02、max_response_length=256 的稳定配置。
4. 如果短步训练 reward 均值能超过 identity bound 并接近 oracle bound 的一部分，再进入 end-point 正式训练。
