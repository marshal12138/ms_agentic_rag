# AIR Stage2 Reranker Hard-Subset Prework

记录时间：2026-07-08

本文件用于中断后恢复 AIR LLM reranker stage2 策略优化工作。当前工作按用户要求已停止，尚未启动新的 reranker 消融训练。

## 当前目标

基于 reward-bound 诊断结论，下一阶段不再全量盲训 stage2，而是先做 hard/improvable 子集上的短步消融：

1. A1：hard 子集 + 纯 `answer_reward`
2. A2：hard 子集 + `answer_reward_plus_evidence_hit`
3. 只有确认短步消融有效后，再进入正式 stage2 训练

## 已完成工作

### 1. Prompt 修正

已修正 AIR reranker prompt 的错误示例。

旧示例中存在越界 index，例如 `[233]`、`[105]`、`[729]`、`[688]`。现在示例已改为：

```text
<rerank>[27] > [23] > [10] > [7] > [6]</rerank>
```

相关文件：

```text
AgenticIterRag/agentic_iter_rag/llm_reranker/format.py
AgenticIterRag/config/reranker_training/llm_reranker_grpo_branch.yaml
tasks/train_tasks/agenticIterRag/configs/from_existing_260704e_traj_to_reranker_training_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/stage2_from_260707144331_data_260707072935_gs40_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/stage2_ablation_reranker_efficiency_5step_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/rebuild_branch_260704e_fixed_prompt_overlay.yaml
```

当前默认 prompt version：

```text
cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example
```

### 2. Fixed prompt 全量 branch dataset 已重建

Manifest：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example/manifest.json
```

关键信息：

```text
sample_count: 5100
prompt_template_version: cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example
candidate_top_n: 50
visible_top_m: 5
```

### 3. Reward-bound 诊断已完成

诊断脚本：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/reward_bound_diagnosis.py
```

主要结果目录：

```text
outputs/agenticIterRag/reward_bound_diagnosis/260708_fixed_prompt_top50miss_n100
```

过滤条件：

```text
top50_hit_top5_miss_baseline0
```

n=100 结果：

```text
baseline mean: 0.0000

identity mean: 0.0408
random   mean: 0.1503
oracle   mean: 0.3545

oracle improved: 37 / 98
oracle better than identity: 33 / 98
oracle worse than identity: 0 / 98
```

结论：

1. stage2 answer reward 不是无效的。
2. oracle top5 有明显收益，但上限不高。
3. 很多样本即使 top5 中放入含答案 doc，frozen agent 仍会答错或 max_turns。
4. 下一步应训练 hard/improvable 子集，或提高这类样本权重。

### 4. Hard/improvable 子集已生成

新增过滤工具：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/filter_branch_dataset.py
```

已生成 hard 子集 manifest：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_fixed_example_hard_top50hit_top5miss_baseline0/manifest.json
```

关键信息：

```text
sample_count: 396
filter: top50_hit_top5_miss_baseline0
baseline_nonzero: 0
prompt_template_version: cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example
dataset_parquet: exists
```

生成命令：

```bash
source src/env_manage/compatible_python.sh
PYTHONPATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/AgenticIterRag:${PYTHONPATH:-} "$PY" \
  -m agentic_iter_rag.reranker_training.filter_branch_dataset \
  --source-manifest /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example/manifest.json \
  --filter top50_hit_top5_miss_baseline0 \
  --out-version 260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_fixed_example_hard_top50hit_top5miss_baseline0
```

### 5. Evidence-hit 辅助 reward 能力已实现，但尚未训练验证

已在 continuation reward 中增加：

```text
evidence_hit_reward
answer_reward_plus_evidence_hit
evidence_hit_weight
```

相关文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/continuation_reward.py
AgenticIterRag/agentic_iter_rag/reranker_training/trainer_entry.py
```

当前语义：

```text
answer_reward:
  score = final answer F1

evidence_hit_reward:
  score = 1 if selected top5 docs contain normalized gold answer substring else 0

answer_reward_plus_evidence_hit:
  score = (1 - evidence_hit_weight) * answer_score + evidence_hit_weight * evidence_hit_score
```

trainer 已透传：

```text
AIR_CONTINUATION_EVIDENCE_HIT_WEIGHT
custom_reward_function.reward_kwargs.evidence_hit_weight
```

## 已通过的检查

编译检查已通过：

```bash
python -m py_compile \
  AgenticIterRag/agentic_iter_rag/reranker_training/continuation_reward.py \
  AgenticIterRag/agentic_iter_rag/reranker_training/filter_branch_dataset.py \
  AgenticIterRag/agentic_iter_rag/reranker_training/trainer_entry.py
```

当前端口检查干净：

```text
8130, 8131, 8140, 8141, 8142, 8143: no listener
```

## 尚未完成

用户要求中断时，下面工作尚未开始：

1. 尚未新增 A1/A2 hard 子集训练 overlay。
2. 尚未 dry-run 编译 A1/A2 final config。
3. 尚未启动新的 stage2 消融训练。
4. 尚未验证 `answer_reward_plus_evidence_hit` 的训练效果。
5. 尚未进入正式训练。

## 恢复时建议步骤

### Step 1：新增两个 overlay

建议新增：

```text
tasks/train_tasks/agenticIterRag/configs/stage2_hard_answer_3step_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/stage2_hard_answer_evidence_3step_overlay.yaml
```

共同配置：

```yaml
branch_dataset_manifest: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_fixed_example_hard_top50hit_top5miss_baseline0/manifest.json

stage1_format.enabled: false
stage2_agentic.enabled: true
stage2_agentic.init_model: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B
stage2_agentic.train_batch_size: 64
stage2_agentic.ppo_mini_batch_size: 64
stage2_agentic.n_samples_per_prompt: 4
stage2_agentic.val_n_samples_per_prompt: 4
stage2_agentic.total_training_steps: 3
stage2_agentic.rollout_temperature: 1.0
stage2_agentic.rollout_top_p: 1.0
stage2_agentic.sampling_stop:
  - "</rerank>"
```

A1：

```yaml
stage2_agentic.sub_strategy: answer_reward
stage2_agentic.evidence_hit_weight: 0.0
```

A2：

```yaml
stage2_agentic.sub_strategy: answer_reward_plus_evidence_hit
stage2_agentic.evidence_hit_weight: 0.2
```

### Step 2：先 dry-run

示例：

```bash
bash scripts/agenticIterRag_v1/01_pipeline_launcher.sh --dry-run \
  --main-run-config agentic_iter_rag_main \
  --DATA_CONFIG=co_search_ablation \
  --PIPELINE_CONFIG=offline_two_stage \
  --RESOURCE_CONFIG=local_8gpu_0_7 \
  --INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm \
  --INFER_BUDGET_CONFIG=air_aligned_budget \
  --RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch \
  --MODEL_CONFIG=qwen3_4b \
  --ROLLOUT_CONFIG=air_async_qwen3_4b \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/stage2_hard_answer_3step_overlay.yaml
```

重点确认：

```text
branch_dataset_sample_count = 396
dataset_files 指向 hard subset parquet
custom_reward_function.reward_kwargs.reward_strategy 正确
evidence_hit_weight 正确
n_samples_per_prompt = 4
total_training_steps = 3
```

### Step 3：再跑短步训练

先跑 A1，再跑 A2。每轮完成后看：

```text
reward_mean
format_valid_rate
response_length_mean
response_clip_ratio
pg_loss / grad_norm
timing_s/gen
timing_s/reward
timing_s/update_actor
是否出现 step collapse 到 -0.5
```

### Step 4：判断策略是否有效

有效标准建议：

1. 3 step 内不出现全量 `reward=-0.5`。
2. response length 不坍缩到极短。
3. format pass rate 稳定。
4. A2 的 reward 分布比 A1 更密，且没有明显诱导模型只学 evidence-hit 而忽略 final answer。
5. 如果 A2 有效，再扩到 5 step。

### Step 5：正式训练

只有 A1/A2 短步消融确认后，再创建正式 overlay：

```text
stage2_hard_answer_evidence_formal_overlay.yaml
```

正式训练不建议立即全量 5100。更稳妥路径：

1. hard subset 396 训练 1 epoch 或固定若干 step。
2. 再混入全量样本，使用 hard oversampling 或 curriculum。
3. 同时保留 oracle-bound 分桶评估。

## 当前停止状态

按用户要求，本轮工作已停止。没有继续创建 overlay，也没有启动训练任务。
