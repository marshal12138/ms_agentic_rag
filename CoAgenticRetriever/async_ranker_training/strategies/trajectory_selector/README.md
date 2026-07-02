# 异步轨迹选择策略

本目录包含 async-ranker-training 使用的轨迹选择器。轨迹选择器运行在 rollout 之后、LLM-as-judge 请求构造之前。它的输出是一组 `ToolCallContext` 对象，而不是对比学习样本。

异步训练流程如下：

```text
rollout DataProto
-> build_fresh_trajectories_from_dataproto(...)
-> trajectory_selector.select(...)
-> build_requests_from_contexts(...)
-> async LLM judge
-> signal_builder
-> sample_builder
-> replay buffer / ranker update
```

`trajectory_selector` 只决定哪些 rollout 中的工具调用上下文会进入 judge 请求候选池。单个 global step 最终提交给 judge 的请求数，后续仍会受到以下配置限制：

```yaml
ranker_training:
  async_ranker_training:
    max_sub_query: 10
    sub_query_selection_policy: random
```

例如：如果 `select_all` 返回 512 个工具调用上下文，而 `max_sub_query: 10`，那么 `build_requests_from_contexts(...)` 在该 global step 中只会选择 10 个上下文。

## 可用策略

### `best_and_worst_f1`

文件：

```text
async_ranker_training.strategies/trajectory_selector/best_and_worst_f1.py
```

行为：

```text
fresh trajectories
-> 解析为 RankerTrajectory
-> 保留有效且 score >= min_final_reward 的轨迹
-> 按 score 降序排序
-> 选择 top_k 条最优轨迹
-> 选择 bottom_n 条最差轨迹
-> 去重
-> 将选中的轨迹展开为 ToolCallContext 对象
```

默认配置：

```yaml
ranker_training:
  async_ranker_training:
    trajectory_selector:
      type: best_and_worst_f1
      top_k: 1
      bottom_n: 2
      min_final_reward: 0.0
```

当 judge 预算有限，并且希望从同一个 rollout batch 中挑出少量高价值对比信号时，可以使用这个策略：它会同时选取表现强的轨迹和表现弱的轨迹。

### `select_all`

文件：

```text
async_ranker_training.strategies/trajectory_selector/select_all.py
```

行为：

```text
fresh trajectories
-> 解析为 RankerTrajectory
-> 保留所有 is_valid == True 的轨迹
-> 不按 score 排序
-> 不应用 top_k / bottom_n
-> 不应用 min_final_reward
-> 将所有有效轨迹展开为 ToolCallContext 对象
```

`select_all` 并不表示每个上下文都会提交给 judge。它的含义是：所有有效轨迹都会进入候选池。最终提交的请求数量仍由 `max_sub_query` 和 `sub_query_selection_policy` 控制。

每个被选中上下文都会附加以下元数据：

```text
trajectory_selection_strategy: select_all
trajectory_selection_role: all
trajectory_selection_rank: 当前 rollout batch 中有效轨迹的顺序
trajectory_selection_score: 原始轨迹分数
```

当目标信号来源是“所有成功的 agent 行为”，而不是“按最终 F1 选出的最好/最差行为”时，适合使用这个策略。

## 如何配置训练

### 方案 1：编辑 async ranker training 策略 YAML

对于 DeepSeek-Flash async-ranker-training 训练任务，编辑：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

设置：

```yaml
ranker_training:
  async_ranker_training:
    trajectory_selector:
      type: select_all
```

显式保留请求数量限制：

```yaml
ranker_training:
  async_ranker_training:
    max_sub_query: 10
    sub_query_selection_policy: random
```

然后运行现有训练任务：

```bash
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_ranker_training_ds_flash_mix_signal.sh
```

### 方案 2：在 shell 中使用 Hydra override

训练脚本会把 `COAGENTIC_EXTRA_ARGS` 追加到 Hydra 命令中。如果只想切换这个策略，可以运行：

```bash
COAGENTIC_EXTRA_ARGS='ranker_training.async_ranker_training.trajectory_selector.type=select_all' \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_ranker_training_ds_flash_mix_signal.sh
```

如果想扩大候选请求预算：

```bash
COAGENTIC_EXTRA_ARGS='ranker_training.async_ranker_training.trajectory_selector.type=select_all ranker_training.async_ranker_training.max_sub_query=32' \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_ranker_training_ds_flash_mix_signal.sh
```

对于 mix-signal batch 实验，这个配置与下面的配置相互独立：

```yaml
ranker_training:
  async_ranker_training:
    sample_builder_request_batch: 3
```

`sample_builder_request_batch` 控制 signal_builder/sample_builder 运行前需要有多少条已完成的 judge 信号。它不控制单个 global step 中有多少 rollout 上下文会提交给 judge。

## 预期指标

使用 `select_all` 时，以下指标相比 `best_and_worst_f1` 可能会上升：

```text
async_ranker_training/selector_contexts
async_ranker_training/selected_tool_calls
async_ranker_training/built_requests
async_ranker_training/labeler_submitted_count
```

如果 `selector_contexts` 很大，但 `selected_tool_calls` 仍接近 `max_sub_query`，说明请求数量限制正在按预期生效。

Ranker 更新使用的样本量由另一组配置单独控制：

```yaml
ranker_training:
  batch_size: 16
  sample_builder:
    num_groups_per_step: 32
```

异步 sample builder 可以从已完成的 judge 信号 batch 中构造 `num_groups_per_step` 条新样本，而 trainer 可以抽取 `batch_size` 条样本用于实际的 ranker 更新。
