# CoAgenticRetriever ranker 训练模式与配置方法

本文记录 CoAgenticRetriever 中 agent LLM 与 dense ranker 的训练耦合方式。重点区分三个概念：

- `training/global_step`：主 agent LLM 的训练 step。
- `ranker/local_update_step`：dense ranker 真正执行 `optimizer.step()` 的次数。
- `created_global_step` / `max_glb_step_lag`：async ranker training 请求产生时的主训练 step，用于控制过旧 label signal 是否还可用。

## 1. 结论

当前不是简单只有 `background_ranker_thread: true/false` 两种总模式，而是先分为两条 ranker 数据来源路径：

1. 非 async ranker training 的同步 contrastive 路径。
2. async ranker training 路径。

在 async ranker training 路径里，再由 `background_ranker_thread` 分成两种 ranker update 执行方式：

1. `background_ranker_thread: false`：LLM judge 请求是异步的，但 ranker update 由主训练循环每个 global step 尝试触发固定次数。
2. `background_ranker_thread: true`：ranker update 由后台线程持续消费 completed judge signals，主训练循环只提交请求和采集 metrics。

因此，需要特别注意：

- `trainer.ranker_steps_per_global_step` 只控制非 async ranker training 的同步 contrastive 路径。
- 在 async ranker training 路径中，`background_ranker_thread: false` 时，ranker 不是由 `trainer.ranker_steps_per_global_step` 控制，而是由 `ranker_training.async_ranker_training.ranker_updates_per_global_step` 控制每个 `training/global_step` 最多尝试多少次前台 async ranker update。
- 在 async ranker training 路径中，`background_ranker_thread: true` 时，主循环不再主动触发 ranker update；ranker 的 `local_update_step` 不再被 global step 的调用次数直接限制。

## 2. 相关代码位置

主 trainer：

```text
CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py
```

async ranker 后台训练器：

```text
CoAgenticRetriever/async_ranker_training/ranker_async_trainer.py
```

当前 flash async ranker training 策略 YAML：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

基础 async ranker training 配置：

```text
CoAgenticRetriever/config/async_ranker_training.yaml
```

注意：当前训练脚本通过策略 YAML 覆盖基础配置时，应优先修改实际被训练脚本加载的策略 YAML；只改基础配置不一定会影响该训练任务。

## 3. 模式 A：非 async ranker training 同步 contrastive

### 3.1 语义

该模式不走 LLM judge async ranker training。主训练每个 global step 完成 rollout 后，从当前 rollout 的 search tool traces 中构造 ranker contrastive 样本，然后在主训练循环内执行 ranker update。

此模式下，agent LLM 与 ranker 是 step-coupled 的：主循环会在同一个 global step 内执行 ranker 更新，并等待相关逻辑完成后再进入后续 step。

### 3.2 控制 ranker 每个 global step 更新次数的配置

```yaml
trainer:
  ranker_trainable: true
  ranker_update_mode: contrastive
  ranker_steps_per_global_step: 2

ranker_training:
  signal_source: rollout
```

具体 `signal_source` 名称以当前策略实现为准；关键是不能满足 async ranker training 启用条件：

```python
ranker_training.async_ranker_training.enable == true
ranker_training.signal_source == "async_ranker_training"
```

### 3.3 `ranker_steps_per_global_step` 的作用

在非 async ranker training 路径里，trainer 会执行类似逻辑：

```python
steps_per_global = trainer.ranker_steps_per_global_step
for ranker_step_idx in range(steps_per_global):
    process_ranker_contrastive_step(...)
```

因此：

- `ranker_steps_per_global_step=1`：每个 global step 做 1 次 ranker optimizer step。
- `ranker_steps_per_global_step=2`：每个 global step 做 2 次 ranker optimizer step。
- 这里的 step 是 `ranker/local_update_step` 的增量，不是 micro batch，也不是 gradient accumulation step。

## 4. 模式 B：async ranker training + 前台 opportunistic ranker update

### 4.1 配置

```yaml
ranker_training:
  signal_source: async_ranker_training
  async_ranker_training:
    enable: true
    background_ranker_thread: false
    ranker_updates_per_global_step: 2
```

当前 flash 策略文件中对应位置：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

### 4.2 语义

该模式下，LLM judge 请求是异步的：

1. 每个 global step rollout 后，trainer 从 trajectory 中选择若干 tool call。
2. trainer 构造 `AsyncLabelRequest` 并提交给 `AsyncLabeler`。
3. `AsyncLabeler` 的 worker 后台向 judge 服务发请求，完成后把 signal 放入 completed buffer。

但是 dense ranker update 不是完全后台独立执行。主训练循环每个 global step 会按配置尝试若干次：

```python
for _ in range(ranker_updates_per_global_step):
    try_train_once(wait=False, timeout=0.0)
```

每次尝试会从 completed buffer 中取 signal；如果当时没有可用 signal，则提前停止本 global step 的 ranker update 尝试。

### 4.3 关键限制

此模式下：

- ranker update 机会仍受 `training/global_step` 限制。
- 每个 global step 最多主动触发 `ranker_updates_per_global_step` 次 ranker update。
- `trainer.ranker_steps_per_global_step` 不控制该路径。
- 如果 judge signal 来得慢，某些 global step 中 `ranker/async_updated_this_step` 会是 `0`。

因此，这个模式可以理解为：

```text
LLM judge labeling 是异步的；
ranker optimizer update 是主循环 step-coupled 的 opportunistic update。
```

它不是最彻底的 agent LLM / ranker 异步训练。

### 4.4 相关 metrics

常看指标：

```text
ranker/async_background_thread: 0
ranker/async_attempted_update
ranker/async_updated_this_step
ranker/async_updates_this_step
ranker/async_updates_per_global_step
ranker/async_updates
ranker/local_update_step
async_ranker_training/labeler_completed_count
async_ranker_training/labeler_completed_buffer_size
timing_s/ranker_async_update_for_step
```

## 5. 模式 C：async ranker training + 后台 ranker 训练线程

### 5.1 配置

```yaml
ranker_training:
  signal_source: async_ranker_training
  async_ranker_training:
    enable: true
    background_ranker_thread: true
    ranker_updates_per_global_step: 1
```

当前 flash 策略已按该模式配置：

```yaml
ranker_training:
  async_ranker_training:
    background_ranker_thread: true
    ranker_updates_per_global_step: 1
```

`ranker_updates_per_global_step` 在后台线程模式下不参与调度；它仍建议显式配置，方便同一份 YAML 在切回 `background_ranker_thread: false` 时语义明确。

### 5.2 语义

该模式下，主训练循环只负责：

1. rollout。
2. 生成并提交 async ranker training 请求。
3. 采集 ranker async trainer metrics。
4. 继续主 agent PPO/GRPO 更新和后续 global step。

dense ranker update 由 `RankerAsyncTrainer` 后台线程执行：

```python
while not closed:
    try_train_once(wait=True, timeout=1.0)
```

后台线程会持续从 completed buffer 中消费 judge signals，构造 contrastive samples，并调用：

```python
ranker_wg.update_ranker_contrastive(batch)
```

### 5.3 与 global step 的关系

此模式下，`ranker/local_update_step` 不再被 global step 的调用次数直接限制。

也就是说，在两个相邻 `training/global_step` 之间，ranker 可能发生：

- 0 次 update：没有足够 completed signals，或 judge 服务较慢。
- 1 次 update：刚好消费到一批 signals。
- 多次 update：completed signals 积压较多，ranker GPU 有吞吐余量。

但是 global step 仍然间接影响 ranker：

- rollout 只在 global step 中产生新 trajectories。
- async ranker training 请求也由 global step 提交。
- `max_glb_step_lag` 会限制过旧 signal 是否可用。

因此更准确的说法是：

```text
background_ranker_thread=true 后，ranker optimizer update 不再被每个 global step 一次调用限制；
但 ranker 的训练数据来源仍由主训练 rollout/global step 产生。
```

### 5.4 控制后台 ranker update 速度的配置

后台线程模式下，ranker update 频率主要由这些配置和资源决定：

```yaml
ranker_training:
  batch_size: 16
  gradient_accumulation_steps: 2

  async_ranker_training:
    sample_builder_request_batch: 1
    max_sub_query: 10
    request_queue_size: 2048
    completed_buffer_size: 4096
    num_workers: 4
    max_glb_step_lag: 3
    ranker_updates_per_global_step: 1

    sample_builder:
      num_groups_per_step: 32
      neg_per_pos: 15
```

含义：

- `sample_builder_request_batch`：一次 ranker update 前，从 completed buffer 消费多少条 completed judge signal。
- `sample_builder.num_groups_per_step`：每次构造多少组 contrastive training groups。
- `ranker_training.batch_size`：每次 ranker optimizer step 采样多少条 contrastive samples。
- `ranker_training.gradient_accumulation_steps`：一次 optimizer step 内切成多少个 micro batch 做梯度累计；它不增加 optimizer step 数。
- `num_workers`：async ranker training labeler 并发 judge 请求 worker 数。
- `max_glb_step_lag`：过旧 signal 的容忍 step lag。
- `ranker_updates_per_global_step`：只在 `background_ranker_thread=false` 时生效；后台线程模式下不控制 ranker update 次数。

### 5.5 相关 metrics

常看指标：

```text
ranker/async_background_thread: 1
ranker/async_attempted_update: 0
ranker/async_updates
ranker/async_updates_since_last_log
ranker/async_updated_this_step
ranker/local_update_step
ranker/loss
ranker/num_queries
ranker/gradient_accumulation_steps
ranker/micro_batch_size
async_ranker_training/labeler_completed_count
async_ranker_training/labeler_failed_count
async_ranker_training/labeler_completed_buffer_size
```

其中：

- `ranker/async_updates` 是后台线程累计完成的 ranker optimizer update 次数。
- `ranker/async_updates_since_last_log` 是两次主循环 metrics 采集之间新增的 ranker update 次数。
- `ranker/local_update_step` 是 dense ranker worker 内部真实 optimizer step 计数。

## 6. 共享推理 ranker 的同步配置

无论前台还是后台 async ranker update，只要启用共享推理 ranker，都需要关注：

```yaml
ranker_training:
  shared_inference_ranker:
    enable: true
    actor_name: coagentic_shared_dense_ranker
    actor_namespace: null
    sync_interval: 2
```

`sync_interval` 按 `ranker/local_update_step` 计数，不按 `training/global_step` 计数。

例如：

- `sync_interval: 2` 表示每完成 2 次 ranker optimizer step，就把训练 ranker 参数同步到共享推理 ranker。
- 如果后台线程在一个 global step 间隔内完成了 3 次 ranker update，则可能在同一个主训练 step 间隔内触发一次同步。

共享推理 ranker 用于 rollout/search tool 的 dense rerank；训练 ranker 用于 optimizer update。二者不是同一个 Python 对象，需要靠该同步机制传递参数。

## 7. 推荐配置

如果目标是更接近真正的 agent LLM / ranker 异步训练，推荐：

```yaml
ranker_training:
  signal_source: async_ranker_training
  shared_inference_ranker:
    enable: true
    sync_interval: 2
  async_ranker_training:
    enable: true
    background_ranker_thread: true
    ranker_updates_per_global_step: 1
    sample_builder_request_batch: 1
```

如果目标是便于调试、减少后台线程复杂性，使用：

```yaml
ranker_training:
  signal_source: async_ranker_training
  async_ranker_training:
    enable: true
    background_ranker_thread: false
    ranker_updates_per_global_step: 2
```

如果目标是不使用 LLM judge async ranker training，而是按每个 global step 固定更新 ranker，则使用非 async ranker training 同步 contrastive：

```yaml
trainer:
  ranker_steps_per_global_step: 2

ranker_training:
  signal_source: rollout
  async_ranker_training:
    enable: false
```

## 8. 排查口径

判断当前跑的是哪种模式：

1. 看 Hydra 最终配置：

```text
ranker_training.signal_source
ranker_training.async_ranker_training.enable
ranker_training.async_ranker_training.background_ranker_thread
ranker_training.async_ranker_training.ranker_updates_per_global_step
trainer.ranker_steps_per_global_step
```

2. 看训练 metrics：

```text
ranker/async_mode
ranker/async_background_thread
ranker/async_attempted_update
ranker/async_updates_since_last_log
ranker/steps_per_global_step
ranker/local_update_step
```

判断标准：

- 出现 `ranker/steps_per_global_step`：通常是非 async ranker training 同步 contrastive 路径。
- `ranker/async_background_thread: 0` 且 `ranker/async_attempted_update > 0`：async ranker training 前台 opportunistic update。
- `ranker/async_background_thread: 1` 且 `ranker/async_attempted_update: 0`：async ranker training 后台 ranker 线程。
