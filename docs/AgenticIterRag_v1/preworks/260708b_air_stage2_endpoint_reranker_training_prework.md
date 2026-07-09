# AIR Stage2 End-Point Reranker Training Prework

记录时间：2026-07-08

本文件用于恢复 AIR LLM reranker stage2 后续工作，记录此前消融经验、当前正式训练策略、从轨迹到训练集的数据加工链路，以及相关代码/配置结构。

## 当前核心结论

1. first-point hard subset 不适合继续作为主训练集。它取 agent 第 0 个 search point，但 reward/label 用原始问题的 final gold answer；多跳问题里第 0 步 query 很可能本来就不应该命中 final answer，因此会引入 step-level label 噪声。
2. end-point 更适合作为本轮 stage2 reranker 训练切入点。end-point 是每条 trajectory 的最后一个 search/rerank 点，离最终回答最近，top5 排序对 continuation answer F1 的影响更直接。
3. end-point hard slice 的真实 continuation oracle bound 不高但有效。当前最干净机会集是 `top50_hit_top5_miss_baseline0 = 453 / 5100 = 8.88%`；n=100 oracle continuation bound 换算到全局 answer F1 约 `+1.78` 到 `+2.14` points。
4. 消融中最稳定策略是 `answer_reward_plus_evidence_hit`，`evidence_hit_weight=0.2`，`rollout.n=4`，`max_response_length=256`。当前正式训练沿用该策略，并跑 `1.5` epoch。

## 从轨迹到训练数据

### 1. 原始训练入口数据

当前任务仍从已有 AIR 轨迹产物恢复，而不是重新生成轨迹：

```text
data/AgenticIterRag/source/co_search_ablation.train.parquet
```

对应 enhanced trajectory manifest：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/trajectory/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79/manifest.json
```

enhanced trajectory 每条样本主要包含：

- `question`
- `gold_answers`
- `baseline_final_answer`
- `baseline_reward`
- `steps`
- 每个 step 的 `sub_query`
- 每个 step 的 `recall_topn_docs`
- 每个 step 的 `doc_id_order`
- 每个 step 的 `messages_before_tool_response`

其中 `messages_before_tool_response` 是 stage2 continuation reward 的关键上下文：reranker 重新排序 top5 后，会把新的 tool observation 拼回这些历史消息，再交给 frozen agent 继续回答。

### 2. 构造 branch dataset

stage：

```text
build_reranker_branch_dataset
```

代码入口：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/branch_dataset.py
```

核心配置位置：

```yaml
reranker_training:
  branch_dataset:
    step_policy: end_point
    candidate_top_n: 50
    visible_top_m: 5
    max_doc_chars: 2000
    prompt_template_version: cosearch_rerank_topm_v1_short_reason_fixed_example
```

`step_policy` 已支持：

```text
first_point: 取第一个 search step
end_point: 取最后一个 search step
random_point: 用 trajectory_id + seed 做稳定 hash，取一个随机 search step
```

当前正式训练使用：

```text
step_policy = end_point
```

branch builder 会做强校验：

- `step.sub_query` 必须等于 `tool_call.arguments.query`
- `recall_topn_docs` 必须有 `candidate_top_n=50` 篇
- doc id 顺序必须和 enhanced step 的 `doc_id_order` 一致
- `messages_before_tool_response` 必须非空，且最后一条是 assistant tool call

每条 branch sample 会写入：

- `prompt`：top50 reranker prompt，要求模型输出 top5 编号
- `reward_model.ground_truth.target`：原始 trajectory 的 `gold_answers`
- `extra_info.candidate_docs`：top50 候选文档
- `extra_info.candidate_index_to_doc_id`：模型输出编号到 doc_id 的映射
- `extra_info.messages_before_tool_response`：continuation 拼接上下文
- `extra_info.baseline_reward`：原始 agent final answer reward
- `extra_info.step_index`
- `extra_info.step_policy`

当前 end-point branch dataset：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/manifest.json
```

关键信息：

```text
sample_count = 5100
candidate_top_n = 50
visible_top_m = 5
prompt_template_version = cosearch_rerank_topm_v1_short_reason_fixed_example
```

### 3. 过滤 hard/improvable subset

stage：

```text
filter_reranker_branch_dataset
```

代码入口：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/filter_branch_dataset.py
```

该 stage 已进入 pipeline/manifest 体系，不再依赖手动命令。策略配置比较轻量，支持：

```text
kind: builtin
kind: python_callable
kind: script
```

当前正式训练使用 builtin hard filter：

```yaml
branch_filter:
  enabled: true
  version: 260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason_hard_top50hit_top5miss_baseline0
  overwrite: true
  max_samples: -1
  sample_mode: none
  random_seed: 20260708
  strategy:
    kind: builtin
    name: top50_hit_top5_miss_baseline0
    builtin_name: top50_hit_top5_miss_baseline0
    kwargs:
      require_top50_hit: true
      require_top5_miss: true
      require_baseline_zero: true
      target_key: reward_model.ground_truth.target
      candidate_docs_key: extra_info.candidate_docs
      baseline_reward_key: extra_info.baseline_reward
```

筛选语义：

- `top50_hit=True`：top50 中至少一篇文档包含 normalized final gold answer 字符串
- `top5_hit=False`：原始 top5 没有文档包含 final gold answer 字符串
- `baseline_reward=0.0`：原始 agent 这条 trajectory final answer reward 为 0

当前 hard subset：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason_hard_top50hit_top5miss_baseline0/manifest.json
```

关键信息：

```text
sample_count = 453
sample_count_before_filter = 5100
source_step_policy = end_point
```

step index 分布：

```text
step_index 0: 3
step_index 1: 178
step_index 2: 227
step_index 3: 42
step_index 4: 3
```

## 消融经验

### first-point 经验

最初 first-point hard subset 结果看起来有上限：

```text
top50_hit_top5_miss_baseline0 = 396 / 5100 = 7.76%
identity mean = 0.0408
random mean = 0.1503
oracle mean = 0.3545
oracle improved = 37 / 98
```

但后来确认这个统计有语义问题：first-point 的 sub-query 是 agent 的中间搜索 query，不一定以 final gold answer 为目标。对多跳问题来说，用 final gold answer 判断第 0 步 rerank 是否正确，会把 bridge-step 样本误当成 answer-step 样本。

结论：first-point 不能作为当前 answer reward stage2 的主训练集。如果以后要训练 bridge-step，需要单独设计 step-level label，例如下一跳实体、桥接证据、或 query-specific evidence，而不是直接用 final gold answer。

### end-point reward-bound

end-point 全量 5100 条分布：

```text
top50_hit = 3897 / 5100 = 76.41%
top5_hit  = 3255 / 5100 = 63.82%
top50_hit_top5_miss = 642 / 5100 = 12.59%
top50_hit_top5_miss_baseline0 = 453 / 5100 = 8.88%
top50_miss = 1203 / 5100 = 23.59%
baseline0 = 1945 / 5100 = 38.14%
baseline1 = 2236 / 5100 = 43.84%
```

end-point hard slice n=100 oracle continuation：

```text
identity mean = 0.0400
oracle mean = 0.2405
oracle - identity = 0.2005
oracle improved = 23 / 100
oracle worse = 2 / 100
oracle max_turns = 17 / 100
```

换算：

```text
453 / 5100 * (0.2405 - 0.0400) = +1.78 F1 points
453 / 5100 * 0.2405 = +2.14 F1 points
```

含义：reranker 有空间，但 final answer F1 的真实可见收益上限中等，不能期待全局大幅提升。短期有效目标应设为 `+0.4` 到 `+0.8` F1，比较理想是 `+0.8` 到 `+1.2` F1。

### short ablation

三组 end-point short ablation 使用同一个 hard subset，固定：

```text
rollout.n = 4
train_batch_size = 64
max_response_length = 256
learning_rate = 5e-6
KL = 0.02
total_training_steps = 3
```

A1：answer-only

```text
critic/score/mean:
step1 0.1229
step2 0.1309
step3 0.1587
```

结论：answer-only 能学到信号，但 final answer reward 稀疏，提升慢。

A2：answer + evidence-hit，w=0.1

```text
critic/score/mean:
step1 0.1499
step2 0.1633
step3 0.1459
```

结论：w=0.1 能提供弱 shaping signal，但没有稳定提升 final-answer 高分样本。

A3：answer + evidence-hit，w=0.2

```text
critic/score/mean:
step1 0.1866
step2 0.1921
step3 0.2195
```

观察：

- reward 三步连续上升
- clip ratio 全程为 0
- 格式罚分从 3 条降到 1 条
- 正分样本数 step3 达到 131 / 256
- 1.0 高分样本没有崩

结论：`evidence_hit_weight=0.2` 是三组中最稳定的策略。

### n=8 经验

曾尝试 `rollout.n=8`，但在 wake_up 阶段触发 NPU OOM。当前资源配置下，n=8 不适合作为正式训练默认值。正式训练回退并固定为 n=4。

## 当前正式训练策略

训练入口：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh
```

该脚本显式在 bash CLI 中写 overlay，避免环境变量或隐式默认值导致运行记录不一致：

```bash
--OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_base_overlay.yaml
--OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_answer_evidence_w02_n4_1p5epoch_overlay.yaml
```

当前正式 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_base_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_answer_evidence_w02_n4_1p5epoch_overlay.yaml
```

正式实验名：

```text
agentic_iter_rag_v1_endpoint_hard_short_reason_ans_ev_w02_n4_1p5epoch
```

正式训练 run：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/agenticIterRag/260708-145444-973727-pipeline-agentic_iter_rag_v1_endpoint_hard_short_reason_ans_ev_w02_n4_1p5epoch
```

正式训练日志：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/agenticIterRag/260708-145444-973727-pipeline-agentic_iter_rag_v1_endpoint_hard_short_reason_ans_ev_w02_n4_1p5epoch/stages/train_llm_reranker/runtime_services/stage2_agentic/verl_train.log
```

tmux session：

```text
air_endpoint_1p5_260708
```

关键训练参数：

```text
sub_strategy = answer_reward_plus_evidence_hit
evidence_hit_weight = 0.2
rollout.n = 4
val_rollout.n = 4
train_batch_size = 64
ppo_mini_batch_size = 64
learning_rate = 5e-6
KL = 0.02
max_response_length = 256
max_prompt_length = 16384
rollout_max_model_len = 16640
total_epochs = 1.5
save_freq = 10
```

注意：用户层正式 overlay 只写 `total_epochs: 1.5`，不写 `total_training_steps`。由于 VERL trainer 原生使用 `range(total_epochs)`，不能直接吃小数 epoch，`trainer_entry.py` 会把小数 epoch 解析成底层 step cap：

```text
sample_count = 453
train_batch_size = 64
steps_per_epoch = 7
1.5 epoch => ceil(7 * 1.5) = 11 training steps
```

dry-run 已确认：

```text
requested_total_epochs = 1.5
resolved_total_epochs = 2
resolved_total_training_steps = 11
fractional_epoch_resolved = true
explicit_total_training_steps = false
```

## Reward 策略

代码入口：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/continuation_reward.py
AgenticIterRag/agentic_iter_rag/reranker_training/rewards/agentic_rag_rollout_reward.py
```

batch reward 入口：

```text
compute_agentic_rag_rollout_reward_batch
```

当前正式 reward：

```text
score = (1 - evidence_hit_weight) * answer_score + evidence_hit_weight * evidence_hit_score
evidence_hit_weight = 0.2
```

其中：

- `answer_score`：reranker 选出 top5 后，frozen agent continuation 的 final answer F1
- `evidence_hit_score`：reranker 输出 top5 中只要有一篇文档包含 normalized final gold answer 字符串，则为 1，否则为 0
- `continuation_status`：continuation 结束状态，例如 answered/max_turns/format_error 等

新增 reward dump 字段：

```text
answer_score
evidence_hit_score
continuation_status
```

这些字段由 reward 函数返回 dict，再由 VERL `RerankerRewardManager` 写入 `reward_extra_info`，后续可用于训练报告和排查 reward 上升是否来自 answer 或 evidence shaping。

## 代码结构

### Pipeline

```text
scripts/agenticIterRag_v1/01_pipeline_launcher.sh
scripts/agenticIterRag_v1/assets/compile_config.py
scripts/agenticIterRag_v1/assets/run_pipeline.py
```

`run_pipeline.py` 中和本任务相关的 stage：

```text
build_reranker_branch_dataset
filter_reranker_branch_dataset
train_llm_reranker
```

当前 pipeline 从 `build_reranker_branch_dataset` 恢复，到 `train_llm_reranker` 停止：

```yaml
pipeline:
  resume_from_stage: build_reranker_branch_dataset
  stop_after_stage: train_llm_reranker
  force_rerun_stages:
    - build_reranker_branch_dataset
    - filter_reranker_branch_dataset
    - train_llm_reranker
```

### Dataset

```text
AgenticIterRag/agentic_iter_rag/reranker_training/branch_dataset.py
AgenticIterRag/agentic_iter_rag/reranker_training/filter_branch_dataset.py
```

`branch_dataset.py` 负责从 enhanced trajectory 选择 search step，并渲染 reranker prompt。

`filter_branch_dataset.py` 负责从 branch dataset 过滤 hard/improvable 子集，支持 builtin/python_callable/script 三类策略。

### Prompt

```text
AgenticIterRag/agentic_iter_rag/llm_reranker/format.py
```

当前 prompt version：

```text
cosearch_rerank_topm_v1_short_reason_fixed_example
```

要求模型输出短理由和 top5 rerank tag，避免长 reasoning 导致 response clip、格式不闭合和训练耗时膨胀。

### Training Entry

```text
AgenticIterRag/main_train_llm_reranker.py
AgenticIterRag/agentic_iter_rag/reranker_training/trainer_entry.py
```

`trainer_entry.py` 负责：

- 读取 pipeline final config
- 根据 active phase 合并 phase config 到 trainer config
- 构造 VERL hydra overrides
- 启动 recall service
- 启动 frozen-agent vLLM multi-instance proxy
- 启动 VERL GRPO 训练
- 写 checkpoint manifest 和 training reports

小数 epoch 解析逻辑在：

```text
resolve_training_schedule
```

### VERL Trainer

```text
AgenticIterRag/verl/verl/trainer/ppo/uni_search_r1_reranker_ray_trainer.py
```

已修正 final checkpoint 保存逻辑：

- `save_freq` 表示每 N step 周期保存一次 checkpoint
- 最后一步必须保存 checkpoint，不受 `save_freq` 限制
- dataloader exhausted fallback 也会尝试保存最后 checkpoint

当前 `save_freq=10`，正式训练解析为 11 step，因此预期至少有：

```text
global_step_10: 周期 checkpoint
global_step_11: final checkpoint
```

实际以训练完成后的 manifest 和目录为准。

### Reward Manager

```text
AgenticIterRag/verl/verl/workers/reward_manager/reranker_reward_manager.py
```

dict reward 会写入：

```text
reward_extra_info
```

因此 `answer_score/evidence_hit_score/continuation_status` 可以进入训练统计链路。

## 配置结构

### 训练脚本

```text
tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh
```

约束：

- 不要把该脚本复杂化
- overlay 必须显式写在 bash CLI 中
- 不通过环境变量临时切 overlay

### Base overlay

```text
tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_base_overlay.yaml
```

负责：

- 指定 260704e enhanced trajectory
- 指定 end-point branch dataset
- 指定 hard filter 策略
- 配置 stage2 基础训练超参
- 配置 recall/frozen-agent 服务资源

### Formal overlay

```text
tasks/train_tasks/agenticIterRag/configs/endpoint_hard_short_reason_answer_evidence_w02_n4_1p5epoch_overlay.yaml
```

只覆盖正式实验差异：

```yaml
main_run:
  project:
    experiment_name: agentic_iter_rag_v1_endpoint_hard_short_reason_ans_ev_w02_n4_1p5epoch

reranker_training:
  training_phases:
    stage2_agentic:
      sub_strategy: answer_reward_plus_evidence_hit
      evidence_hit_weight: 0.2
      total_epochs: 1.5
```

不要在正式 overlay 中写 `total_training_steps`。正式实验以 epoch 作为用户层语义，小数 epoch 由 `trainer_entry.py` 转换为底层 VERL step cap。

## 服务资源结构

当前 stage2 continuation 需要：

```text
recall proxy: 127.0.0.1:8130
recall backend: 0.0.0.0:8131, NPU 7
frozen-agent proxy: 127.0.0.1:8140
frozen-agent instances:
  127.0.0.1:8141, NPU 4
  127.0.0.1:8142, NPU 5
  127.0.0.1:8143, NPU 6
reranker actor: NPU 0-3
```

当前正式训练启动检查时这些服务已拉起，VERL run script 已开始执行。

## 后续验证建议

正式训练完成后，不要只看训练 reward，需要做固定 eval 对比：

1. base reranker
2. 消融 step checkpoint
3. formal final checkpoint
4. oracle bound

至少需要比较：

- global final answer F1
- hard slice final answer F1
- top5 evidence-hit / answer-hit
- 原本 baseline 已答对样本是否被误排伤害
- `answer_score` 和 `evidence_hit_score` 是否同步提升
- `continuation_status` 中 max_turns 是否下降

如果训练 reward 上升但 final eval F1 不升，优先检查：

- evidence-hit bonus 是否只是提高字符串命中而没有改善 answer_score
- frozen agent 是否无法利用被 reranker 提前的证据
- end-point hard subset 是否过窄，导致泛化差
- 原本 top5 已命中的样本是否在全局 eval 被误排
- final checkpoint 是否确实保存并用于转换/eval
