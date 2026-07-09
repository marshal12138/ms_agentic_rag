# AIR stage2 reranker 消融与正式训练结论

日期：2026-07-08

## 目标

本轮工作目标是先通过 reranker stage2 消融确认训练策略有效，再启动正式训练实验。

约束：

- rollout-n 最低 4，最高 8。
- 先消融，确认策略稳定有效后再正式训练。
- 训练前服务准备阶段高频探查，训练阶段按 3-5 分钟节奏探查。

## 前置诊断结论

reward-bound 诊断已经说明 stage2 answer reward 不是无效的：

- hard slice：`top50_hit_top5_miss_baseline0 = 396 / 5100 = 7.76%`
- n=100 reward-bound：
  - identity mean：0.0408
  - random mean：0.1503
  - oracle mean：0.3545
  - oracle improved：37 / 98

换算到全局最终 answer F1，oracle reranker 可见收益上限约为 `+2.4` 到 `+2.8` F1 points。真实训练可达目标应更保守，短期有效目标约 `+0.5` 到 `+1.0` F1。

因此本轮不做全量盲训，而是先在 hard/improvable 子集上训练。

## 关键实现与数据修正

### prompt 修正

原始 reranker prompt 容易诱导模型逐条分析候选 passage，导致输出过长、格式不闭合、stage2 训练不稳定。

已将默认 prompt 改为 short-reason 版本：

- `<reason>` 最多 2 个短句。
- `<reason>` 最多 60 个英文词。
- 只总结已选证据类型。
- 不逐条分析 candidates。
- 不提未选择 passage。
- 不在 reason 中写 index。

相关文件：

- `AgenticIterRag/agentic_iter_rag/llm_reranker/format.py`
- `AgenticIterRag/config/reranker_training/llm_reranker_grpo_branch.yaml`

注意：旧 manifest 中的 prompt 是预渲染字段，所以仅修改 prompt 模板不会影响已存在 dataset。为此已新增重渲染工具并生成 short-reason hard dataset。

### short-reason hard dataset

数据集：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_short_reason_hard_top50hit_top5miss_baseline0/manifest.json
```

样本数：396。

prompt version：

```text
cosearch_rerank_topm_v1_short_reason_fixed_example
```

重渲染工具：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/rerender_branch_dataset_prompt.py
```

### evidence-hit 辅助 reward

已支持三种 stage2 reward strategy：

- `answer_reward`
- `evidence_hit_reward`
- `answer_reward_plus_evidence_hit`

最终采用：

```text
answer_reward_plus_evidence_hit
evidence_hit_weight = 0.2
format_invalid_score = -0.5
```

相关文件：

- `AgenticIterRag/agentic_iter_rag/reranker_training/continuation_reward.py`
- `AgenticIterRag/agentic_iter_rag/reranker_training/trainer_entry.py`

## 消融实验结果

### A1：answer-only，hard subset，n=4，3 steps

run：

```text
260708-012354-178184-pipeline-agentic_iter_rag_v1_stage2_hard_answer_n4_3step
```

结果：

```text
step1 score = 0.013811
step2 score = 0.203385
step3 score = 0.070262
mean score = 0.095819
```

结论：answer-only 可以训练，但 reward 噪声较大，平均收益弱。

### A2：answer + evidence-hit w=0.2，hard subset，n=4，3 steps

run：

```text
260708-020423-273000-pipeline-agentic_iter_rag_v1_stage2_hard_answer_evidence_w02_n4_3step
```

结果：

```text
step1 score = 0.115684
step2 score = 0.241797
step3 score = 0.108329
mean score = 0.155270
```

格式闭合：

```text
step1 close = 245 / 256
step2 close = 256 / 256
step3 close = 254 / 256
```

结论：加入 evidence-hit 后明显优于 answer-only，但旧 prompt 仍有输出长度和格式不稳定风险。

### n=8 正式尝试

run：

```text
260708-024039-526299-pipeline-agentic_iter_rag_v1_stage2_hard_answer_evidence_w02_n8_1epoch
```

结果：

```text
step1 score = 0.105130
```

随后在 wake_up 阶段触发 NPU OOM。

结论：n=8 符合 rollout-n 上限，但当前资源配置下不可稳定训练。正式实验应回退到 n=4。

### 旧 prompt n=4 正式尝试

run：

```text
260708-031010-068331-pipeline-agentic_iter_rag_v1_stage2_hard_answer_evidence_w02_n4_1epoch
```

结果：

```text
step1 score = 0.091620
step2 score = -0.295680
step2 response_length/clip_ratio = 0.7461
step2 close = 66 / 256
```

结论：失败原因不是 reward 方向，而是 prompt 诱导长输出和格式崩溃。需要 short-reason prompt、降低 LR、加强 KL、限制 response length。

### short-reason 稳定性消融，n=4，3 steps

overlay：

```text
tasks/train_tasks/agenticIterRag/configs/stage2_hard_short_reason_answer_evidence_stable_3step_overlay.yaml
```

run：

```text
260708-040125-046381-pipeline-agentic_iter_rag_v1_stage2_hard_short_reason_ans_ev_w02_lr5e6_kl02_n4_3step
```

关键配置：

```text
reward_strategy = answer_reward_plus_evidence_hit
evidence_hit_weight = 0.2
learning_rate = 5e-6
use_kl_loss = true
kl_loss_coef = 0.02
max_response_length = 256
rollout.n = 4
```

结果：

```text
step1 score = 0.146204
step2 score = 0.272786
step3 score = 0.199077
mean score = 0.206023
response_length/clip_ratio = 0.0 / 0.0 / 0.0
```

rollout 格式：

```text
step1 close = 256 / 256, bad_format_score = 1
step2 close = 256 / 256, bad_format_score = 3
step3 close = 256 / 256, bad_format_score = 8
```

结论：策略有效且稳定，可以进入正式训练。

## 正式训练结果

overlay：

```text
tasks/train_tasks/agenticIterRag/configs/stage2_hard_short_reason_answer_evidence_stable_n4_1epoch_overlay.yaml
```

run：

```text
260708-043304-766388-pipeline-agentic_iter_rag_v1_stage2_hard_short_reason_ans_ev_w02_lr5e6_kl02_n4_1epoch
```

实际训练配置确认：

```text
sample_count = 396
total_training_steps = 6
train_batch_size = 64
actor_rollout_ref.rollout.n = 4
actor_rollout_ref.rollout.val_kwargs.n = 4
reward_strategy = answer_reward_plus_evidence_hit
evidence_hit_weight = 0.2
learning_rate = 5e-6
kl_loss_coef = 0.02
max_response_length = 256
```

注意：`stages/train_llm_reranker/manifest.json` 的 `outputs.n_samples_per_prompt` 顶层字段显示为 8，但 phase 内字段和 Hydra overrides 均确认实际训练为 `rollout.n=4`。后续引用训练配置时应以 phase 内 `n_samples_per_prompt=4` 和 Hydra override 为准。

### training metrics

```text
step1 score = 0.159038, clip_ratio = 0.0, response_len_mean = 74.41
step2 score = 0.247031, clip_ratio = 0.0, response_len_mean = 79.02
step3 score = 0.236589, clip_ratio = 0.0, response_len_mean = 78.62
step4 score = 0.216809, clip_ratio = 0.0, response_len_mean = 77.65
step5 score = 0.261504, clip_ratio = 0.0, response_len_mean = 89.05
step6 score = 0.205403, clip_ratio = 0.0, response_len_mean = 95.47

mean score = 0.221062
min score = 0.159038
max score = 0.261504
```

### rollout 格式核验

```text
step1 rows = 256, close = 256, open = 256, bad_format_score = 1
step2 rows = 256, close = 256, open = 256, bad_format_score = 2
step3 rows = 256, close = 256, open = 256, bad_format_score = 1
step4 rows = 256, close = 256, open = 256, bad_format_score = 1
step5 rows = 256, close = 256, open = 256, bad_format_score = 3
step6 rows = 256, close = 256, open = 256, bad_format_score = 1
```

总计：

```text
rollout rows = 1536
format close = 1536 / 1536
bad_format_score = 9 / 1536
```

结论：正式训练没有复现旧 prompt 的格式崩溃，训练过程稳定。

### checkpoint

最终可用 checkpoint：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/agenticIterRag/260708-043304-766388-pipeline-agentic_iter_rag_v1_stage2_hard_short_reason_ans_ev_w02_lr5e6_kl02_n4_1epoch/stages/train_llm_reranker/reranker_model_verl/stage2_agentic/global_step_6
```

`global_step_3` 目录存在，但 checkpoint manager 已删除其 `actor` 子目录；因此当前直接可用的是 `global_step_6`。

训练报告：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/agenticIterRag/260708-043304-766388-pipeline-agentic_iter_rag_v1_stage2_hard_short_reason_ans_ev_w02_lr5e6_kl02_n4_1epoch/stages/train_llm_reranker/training_reports/stage2_agentic/air_llm_reranker.metrics.jsonl
```

rollout 数据：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/agenticIterRag/agenticIterRag/260708-043304-766388-pipeline-agentic_iter_rag_v1_stage2_hard_short_reason_ans_ev_w02_lr5e6_kl02_n4_1epoch/runtime_logs/train_llm_reranker/stage2_agentic/rollout_data
```

## 结论

本轮确认的有效 stage2 reranker 训练策略是：

```text
hard/improvable subset
short-reason default prompt
answer_reward_plus_evidence_hit
evidence_hit_weight = 0.2
rollout.n = 4
learning_rate = 5e-6
kl_loss_coef = 0.02
max_response_length = 256
```

相对 answer-only 消融，answer+evidence-hit 明显更强；相对旧 prompt，short-reason prompt 解决了长输出和格式不闭合问题。n=8 在当前 NPU 资源配置下触发 OOM，当前可稳定正式训练的 rollout-n 是 4。

下一步应使用 `global_step_6` 在真实 eval set 上跑 `agent + llm reranker` answer F1 评估，确认能否把 hard-subset reward 提升转化为全局最终 answer F1。期望收益仍应按 reward-bound 约束保守估计：短期有效目标为 `+0.5` 到 `+1.0` F1，理想为 `+1.0` 到 `+1.5` F1。
