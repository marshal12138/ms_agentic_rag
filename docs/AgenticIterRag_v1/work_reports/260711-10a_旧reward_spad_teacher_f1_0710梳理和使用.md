# 旧 reward：`spad_teacher_f1_0710` 梳理和使用

日期：2026-07-11；北京时间09点

## 1. 结论

2026-07-10 SPAD Stage1 正式训练所用 reward 已命名为：

```text
spad_teacher_f1_0710
```

该方案与当前 `spad_em_teacher_backoff` 组级新 reward 使用独立 Python 模块、独立函数入口和独立
运行时路由。切换时只允许修改 `agent_training.reward.type` 或加载本文给出的 overlay，不应手工混配
reward manager、stop token 或 stream 开关。

## 2. 0710 实际运行证据

来源 run：

```text
log/agenticIterRag/260710-021433-474200-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_formal_500_offlinebatch_260710
```

运行时 `verl_command.argv` 明确记录：

- 函数：`compute_spad_search_policy_reward_details`。
- `reward_model.use_reward_loop=True`。
- `reward_model.reward_manager=naive`。
- stop：`['</tool_call>', '<answer>']`，即 actor 在答案开标签停止。
- teacher：GLM-4.7-Flash，`temperature=0`、`top_p=1`、`max_tokens=512`、thinking 关闭。
- 每 prompt 8 rollout，逐 rollout 异步调用 teacher。

0710 的 7 个有效 step 共 3584 条 rollout，产物反查结果为：

| 指标 | 数值 |
| --- | ---: |
| teacher called | 2617 |
| actor 格式无效 | 964 |
| 无检索证据 | 3 |
| bad-stop applied | 1231 |
| teacher XML 格式错误 | 2 |

这些统计来自 7 个 `rollout_data/{1..7}.jsonl`，不是用当前 reward 重放得到的估计值。

## 3. 奖励语义

该方案对每条 rollout 独立判定，不先看同 UID 的其他 7 条结果。

1. 无检索证据：`-0.5`。
2. actor XML/action 格式错误或未合法停在 `<answer>`：`-0.5`。
3. 合法停止后，teacher 仅依据 actor 实际看到的检索证据输出短答案。
4. teacher XML 格式错误：`-0.1`。
5. teacher 回答 `证据不足无法作答` 且搜索预算未耗尽：`-0.35`。
6. 搜索预算耗尽仍证据不足：`-0.15`。
7. 其余情况按 teacher answer 对 gold 的最大 token F1 计分，并加入成本项：

```text
reward = teacher_answer_f1
         - 0.02 * max(search_count - 1, 0)
         - 0.10 * duplicate_query_count
         - 0.02 * missing_reason_count
```

首搜免费。0710 rollout 中可观察到 `teacher_f1=1`、`search_count=1`、最终 `score=0.98` 的样例，
其 `-0.02` 来自缺失 reason，而非首搜成本。

## 4. 独立代码

冻结模块：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward_0710.py
```

唯一训练入口：

```text
compute_spad_teacher_f1_0710_details
```

模块内部固化了0710的 teacher system prompt、evidence 排版、teacher answer XML parser、answer F1、
bad-stop 和成本公式。它不调用当前新 reward 的 UID 分组、EM 或 teacher status/backoff 分支。

当前新 reward 仍位于：

```text
AgenticIterRag/agentic_iter_rag/agent_training/spad/rewards/search_policy_teacher_reward.py
```

稳定入口为 `compute_spad_em_teacher_backoff_batch`；历史通用入口
`compute_spad_search_policy_reward_batch` 仅保留兼容。两者不能在同一训练 run 内混用。

## 5. 自动防错路由

`search_policy_rl.py` 根据 reward 名称自动选择完整运行契约：

| 配置 | `spad_teacher_f1_0710` | `spad_em_teacher_backoff` |
| --- | --- | --- |
| Python 模块 | `search_policy_teacher_reward_0710.py` | `search_policy_teacher_reward.py` |
| Python 入口 | `compute_spad_teacher_f1_0710_details` | `compute_spad_em_teacher_backoff_batch` |
| manager | `naive` | `batch` |
| reward loop | 开 | 关 |
| UID 流式组奖励 | 关 | 开 |
| answer stop | `<answer>` | `</answer>` |
| teacher 目标 | 生成短答案并算 F1 | 全零 EM 组证据状态回退 |

即使 base trainer 默认写了 `reward_manager: batch` 和 `stream_group_reward: true`，选择0710名称后也会
强制改为 naive/非流式，避免“旧公式配新调度”或“旧 opening-stop 配完整答案 parser”。

## 6. 使用方法

新 512t 数据上的0710 reward 对比 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_512_reward_0710_overlay.yaml
```

它必须放在正式 SPAD 和 512 scale overlay 之后：

```bash
bash scripts/agenticIterRag_v1/01_pipeline_launcher.sh \
  --main-run-config agentic_iter_rag_main \
  --DATA_CONFIG=co_search_ablation \
  --PIPELINE_CONFIG=offline_two_stage \
  --RESOURCE_CONFIG=local_8gpu_0_7 \
  --INFER_RUNTIME_CONFIG=agentic_iter_rag_vllm \
  --INFER_BUDGET_CONFIG=air_aligned_budget \
  --RERANKER_TRAINING_CONFIG=llm_reranker_grpo_branch \
  --AGENT_TRAINING_CONFIG=spad_rag_base \
  --MODEL_CONFIG=qwen3_1_7b \
  --ROLLOUT_CONFIG=air_async_qwen3_1_7b \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_formal_overlay.yaml \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_512_scale_overlay.yaml \
  --OVERLAY_YAML=tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_512_reward_0710_overlay.yaml
```

正式启动前应先追加 `--dry-run`，并在 `verl_command_plan.json` 检查以下五项：

```text
reward_model.reward_manager=naive
+reward_model.use_reward_loop=True
+reward_model.stream_group_reward=False
custom_reward_function.name=compute_spad_teacher_f1_0710_details
+actor_rollout_ref.rollout.stop=['</tool_call>','<answer>']
```

## 7. 验证

新增测试覆盖：

- 名称到独立模块、naive manager、reward loop、非流式和 opening-stop 的自动路由。
- supported answer 的 teacher F1。
- `证据不足无法作答` 的 `-0.35` bad-stop。
- teacher XML 错误的 `-0.1`。

定向测试结果：`8/8` 通过。完整测试结果应随当前代码实现报告持续更新。
