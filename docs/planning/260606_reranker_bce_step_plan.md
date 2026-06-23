# Reranker BCE Step 改造方案

本文档整理两个需求的实现方案：

1. 将 reranker 当前的 GRPO step 替换成对比学习 BCE loss 的计算和参数更新流程，即 BCE step。
2. 每个 global step 中，main agent 仍执行 1 次 GRPO step；reranker 执行 N 次 BCE step，用于缓解 BCE 监督训练收敛较慢的问题。

目标代码以当前 `verl` 框架为准，主要涉及：

- `verl/trainer/ppo/search_r1_reranker_reward_ray_trainer.py`
- `verl/workers/fsdp_workers.py`
- `verl/workers/actor/dp_actor.py`
- `verl/experimental/agent_loop/search_r1_dual_agent_loop.py`
- 相关 config / launch script

如果在当前 `CoSearch_derevitives` 合并副本中的 `CoAgenticRetriever` 路径落地，路径前缀应切换到：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/
```

## 当前机制

当前 dual-agent trainer 的主循环大致是：

1. 从 dataloader 取一个 batch。
2. `async_rollout_manager.generate_sequences(...)` 生成：
   - `main_batch`
   - `reranker_batch`
3. main agent 走 `process_single_agent_ppo_step(...)`。
4. reranker 也走同一个 `process_single_agent_ppo_step(...)`。
5. 在 `process_single_agent_ppo_step(...)` 内：
   - 计算 `old_log_probs`
   - 计算 reward / token_level_rewards
   - 调用 `compute_advantage(... adv_estimator=config.algorithm.adv_estimator ...)`
   - 调用 `actor_rollout_wg.update_actor(batch)`

因此 reranker 当前本质上仍是 policy optimization：

```text
reranker rollout -> outcome reward -> GRPO advantage -> PPO/GRPO actor update
```

这个路径的问题是 reranker 的学习信号长、稀疏、噪声大，并且要依赖生成格式、答案 reward、counterfactual continuation 等链路。对 reranker 这种“给 query/context + candidate docs，判断文档相关性并排序”的模块，可以引入更直接的 BCE 监督信号。

## 目标机制

main agent 继续保持现有 GRPO：

```text
main rollout -> reward -> GRPO advantage -> update_actor
```

reranker 改成 BCE step：

```text
reranker training examples -> score candidate docs -> BCEWithLogits loss -> optimizer step
```

每个 trainer global step 的调度变成：

```text
for global_step:
    rollout once
    main:      1 x GRPO step
    reranker:  N x BCE step
    global_step += 1
```

其中 `N` 由配置控制，例如：

```yaml
trainer:
  reranker_update_mode: bce
  reranker_bce_steps_per_global_step: 4
```

`global_steps` 仍按 main agent 的 step 计数；reranker 的 BCE update step 单独记录为 `reranker_bce/update_step` 指标。

## BCE 训练任务定义

推荐第一版使用 pointwise contrastive BCE，不额外引入分类头，直接复用 causal LM 参数和 LoRA/FSDP 训练链路。

### 输入样本

每个 reranker 样本来自一次 Search-R1 工具调用位置，至少需要：

- `initial_query`: 原始问题
- `sub_query`: 当前 Search-R1 发出的检索 query
- `candidate_docs`: top-N 检索文档
- `golden_answers`: 标准答案列表
- 可选 `positive_doc_ids`: 如果已有更可靠的 oracle/evidence 标注，优先使用
- 可选 `trajectory_uid`, `turn_id`, `tool_call_id`: 用于去重、日志和 replay buffer

当前 `search_r1_dual_agent_loop.py` 已经在构造 reranker pipeline 时拿到 top documents、query、trajectory 等上下文。改造时应在构造 `reranker_batch` 的同时，把 BCE 需要的字段放进 `non_tensor_batch` 或单独的 replay buffer sample 中。

### 标签构造

第一版标签可以按优先级选择：

1. 如果样本里有 `positive_doc_ids` / evidence doc 标注，则这些 doc 为正样本。
2. 否则使用 answer-string heuristic：文档 `title + text + contents` 中包含任一 normalized golden answer，则标为正。
3. 如果一个 query 下没有正样本：
   - 默认跳过该 query 的 BCE 样本；
   - 或配置允许 `allow_all_negative=true` 时保留负样本，但需要降低权重。

负样本从同一 query 的 candidate docs 中采样，建议默认比例：

```text
positive : negative = 1 : 3
```

如果正样本很多，最多保留 `max_pos_per_query`；如果负样本很多，最多保留 `max_neg_per_query`。

### 打分方式

为了避免改模型结构，使用 yes/no token delta 作为 logit：

```text
score = logit("yes") - logit("no")
loss = BCEWithLogitsLoss(score, label)
```

每个候选文档构造成一个 scoring prompt：

```text
You are a reranker for open-domain QA.

Initial question:
{initial_query}

Search query:
{sub_query}

Candidate passage:
{title}
{contents}

Does this passage contain evidence useful for answering the search query and final question?
Answer yes or no.
```

模型只需在最后一个位置上给 `yes/no` 概率。训练时不采样生成，不需要解析 `<rerank>` 标签。

这种方案的优点：

- 复用 `AutoModelForCausalLM`，不增加 classification head。
- 复用现有 FSDP、optimizer、LoRA checkpoint。
- BCE loss 和推理排序都可以从同一个 score 定义得到。

代价：

- 如果线上 reranker 仍使用 `<rerank>...</rerank>` 生成式输出，仅训练 yes/no scoring prompt 不一定直接提升生成式排序格式。
- 因此建议同步支持 reranker inference 的 `score` 模式：对 top-N docs 批量打分，按 score 排序取 top-M。

## 数据结构建议

新增一个 BCE batch schema，和 PPO/GRPO 的 `responses/advantages/old_log_probs` 解耦。

Tensor fields：

```text
input_ids:      [B, L]
attention_mask: [B, L]
position_ids:   [B, L]
labels:         [B] float, 0/1
loss_weights:   [B] float, optional
```

Non-tensor fields：

```text
query_uid
doc_id
initial_query
sub_query
golden_answers
label_source
```

注意：BCE step 不需要以下 PPO 字段：

```text
responses
response_mask
old_log_probs
advantages
returns
ref_log_prob
token_level_rewards
```

这也是建议新增 `update_reranker_bce(...)` API 的原因，避免和 `update_actor(...)` 的 PPO 数据契约混在一起。

## Worker 改造点

### `DataParallelPPOActor`

在 `verl/workers/actor/dp_actor.py` 中新增：

```python
def update_reranker_bce(self, data: DataProto):
    self.actor_module.train()
    # select input_ids / attention_mask / position_ids / labels / loss_weights
    # split mini-batch and micro-batch
    # forward causal LM
    # take final non-pad token logits
    # score = yes_logit - no_logit
    # loss = BCEWithLogitsLoss(reduction="none")(score, labels)
    # weighted mean, backward, optimizer step
    # return metrics
```

核心 loss：

```python
last_idx = attention_mask.sum(dim=-1) - 1
last_logits = logits[torch.arange(batch_size), last_idx]
score = last_logits[:, yes_token_id] - last_logits[:, no_token_id]
loss_vec = F.binary_cross_entropy_with_logits(score, labels.float(), reduction="none")
loss = (loss_vec * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
```

需要返回的指标：

```text
reranker_bce/loss
reranker_bce/pos_loss
reranker_bce/neg_loss
reranker_bce/score_pos_mean
reranker_bce/score_neg_mean
reranker_bce/acc_at_0
reranker_bce/num_pos
reranker_bce/num_neg
reranker_bce/grad_norm
reranker_bce/lr
```

### `FSDP worker`

在 `verl/workers/fsdp_workers.py` 中新增 worker group 可调用接口：

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
@DistProfiler.annotate(color="red", role="reranker_bce_update")
def update_reranker_bce(self, data: DataProto):
    assert self._is_actor
    # offload/load 与 update_actor 一致
    # data.to("cpu")
    # self.actor.update_reranker_bce(data)
    # scheduler.step()
    # return DataProto(meta_info={"metrics": metrics})
```

这样 trainer 可以调用：

```python
self.reranker_actor_rollout_wg.update_reranker_bce(bce_batch)
```

而 main agent 仍调用：

```python
self.actor_rollout_wg.update_actor(main_batch)
```

## Trainer 改造点

### 新增 reranker BCE step 函数

在 `search_r1_reranker_reward_ray_trainer.py` 中新增类似 remote function：

```python
@ray.remote
def process_reranker_bce_step(
    batch: DataProto,
    reranker_actor_wg: RayWorkerGroup,
    tokenizer,
    config,
    global_steps: int,
    bce_step_idx: int,
):
    # build / validate BCE batch
    # set meta_info:
    #   yes_token_id, no_token_id
    #   global_steps
    #   bce_step_idx
    #   micro_batch_size
    # call reranker_actor_wg.update_reranker_bce(bce_batch)
    # reduce metrics
    # return metrics/timing
```

不要在这个函数中调用：

```python
compute_log_prob
compute_ref_log_prob
compute_advantage
update_actor
```

这些属于 GRPO/PPO 路线，BCE step 应完全绕开。

### 主循环调度

当前逻辑是 main 和 reranker 都启动 `process_single_agent_ppo_step.remote(...)`：

```python
main_futures = process_single_agent_ppo_step.remote(...)
reranker_futures = process_single_agent_ppo_step.remote(...)
main_results, reranker_results = ray.get([main_futures, reranker_futures])
```

改成：

```python
main_future = process_single_agent_ppo_step.remote(...)

reranker_bce_results = []
if reranker_train_enabled and config.trainer.reranker_update_mode == "bce":
    for bce_i in range(config.trainer.reranker_bce_steps_per_global_step):
        bce_batch = reranker_bce_buffer.sample_or_build(...)
        reranker_future = process_reranker_bce_step.remote(
            batch=bce_batch,
            reranker_actor_wg=self.reranker_actor_rollout_wg,
            tokenizer=self.reranker_tokenizer,
            config=self.config,
            global_steps=self.global_steps,
            bce_step_idx=bce_i,
        )
        reranker_bce_results.append(ray.get(reranker_future))
elif reranker_train_enabled:
    reranker_future = process_single_agent_ppo_step.remote(...)

main_results = ray.get(main_future)
```

说明：

- main worker group 和 reranker worker group 在不同资源池上，main GRPO future 可以先启动，reranker BCE steps 可并行占用 reranker GPU。
- 同一个 reranker worker group 的多次参数更新应按顺序执行，所以建议在 driver 上逐次 `ray.get`，不要一次性提交 N 个更新。
- `global_steps` 只在 main step 后递增一次；BCE 的子步用 `bce_step_idx` 和独立 metric 记录。

### Replay buffer

如果 N > 1，仅重复使用同一批 reranker samples 容易过拟合当前 batch。建议增加轻量 replay buffer：

```text
RerankerBCEReplayBuffer
  max_size
  add(samples_from_current_rollout)
  sample(batch_size, pos_neg_ratio)
```

行为：

1. 每个 global step 的 rollout 产出一批 fresh reranker BCE samples。
2. fresh samples 进入 buffer。
3. 第 1 个 BCE step 优先用 fresh samples。
4. 后续 BCE step 从 buffer 混采 fresh + historical samples。

默认配置建议：

```yaml
trainer:
  reranker_bce:
    replay_size: 200000
    batch_size: 256
    fresh_ratio: 0.5
    pos_neg_ratio: 0.333333
```

## Config 设计

建议新增：

```yaml
trainer:
  reranker_update_mode: bce      # grpo | bce | off
  reranker_bce_steps_per_global_step: 4
  reranker_bce_start_step: 0
  reranker_bce_stop_step: null
  reranker_bce:
    batch_size: 256
    micro_batch_size_per_gpu: 8
    replay_size: 200000
    fresh_ratio: 0.5
    max_pos_per_query: 5
    max_neg_per_query: 15
    allow_all_negative: false
    label_source: answer_string  # answer_string | oracle_doc_ids | mixed
    score_mode: yes_no_delta
    positive_weight: auto
    yes_token: " yes"
    no_token: " no"
```

保持兼容：

- `reranker_update_mode=grpo` 时走当前 reranker GRPO 路线。
- `reranker_update_mode=bce` 时 reranker 不再创建/使用 ref policy，也不需要 reranker reward GRPO。
- `reranker_update_mode=off` 时 reranker 只 rollout/inference，不更新。

## Inference 路线建议

如果 reranker 改为 BCE scoring 训练，最好同步支持 scorer inference：

```text
for each candidate doc:
    score = logit_yes - logit_no
sort docs by score desc
return top-M
```

需要改的位置：

- `search_r1_dual_agent_loop.py` 中 reranker 调用逻辑
- 或新增 `RerankerScorer` / `reranker_score_documents(...)`

短期可以保留生成式 reranker 用于兼容，但配置上区分：

```yaml
trainer:
  reranker_inference_mode: generate  # generate | score
```

推荐评估时至少跑两组：

1. `reranker_update_mode=bce`, `reranker_inference_mode=score`
2. `reranker_update_mode=bce`, `reranker_inference_mode=generate`

如果第 1 组提升而第 2 组不提升，说明 BCE 主要学到了 scorer 能力，生成格式没有被显式训练到。

## 指标和日志

每个 global step 记录：

```text
main_actor/*
main_reward/*
reranker_bce/loss
reranker_bce/loss_step_0
reranker_bce/loss_step_N
reranker_bce/acc_at_0
reranker_bce/score_pos_mean
reranker_bce/score_neg_mean
reranker_bce/num_pos
reranker_bce/num_neg
reranker_bce/buffer_size
reranker_bce/fresh_samples
reranker_bce/update_steps_per_global_step
```

timing：

```text
timing/main_grpo
timing/reranker_bce_total
timing/reranker_bce_step_mean
timing/rollout
```

这可以帮助判断 N 增大后是否真的提高 reranker 学习速度，还是只是拖慢 overall throughput。

## 验证计划

### 单元测试

1. label builder：
   - answer 命中文档时产生正样本。
   - 无正样本时默认跳过。
   - pos/neg 采样比例正确。

2. BCE loss：
   - `yes_logit - no_logit` shape 正确。
   - 全正、全负、混合 batch 不报错。
   - `loss_weights` 生效。

3. replay buffer：
   - capacity 裁剪正确。
   - fresh/historical 混采比例正确。

### Smoke run

先用极小参数：

```yaml
trainer:
  total_training_steps: 2
  reranker_update_mode: bce
  reranker_bce_steps_per_global_step: 2
  reranker_bce:
    batch_size: 16
    micro_batch_size_per_gpu: 1
```

检查：

- main GRPO step 正常完成。
- 每个 global step 记录 2 个 reranker BCE 子步。
- reranker checkpoint 正常保存。
- loss、grad_norm、lr 都有数值。
- 没有再对 reranker 调用 `compute_advantage` / `update_actor`。

### 对照实验

最小三组：

1. 当前 baseline：main GRPO + reranker GRPO。
2. main GRPO + reranker BCE，N=1。
3. main GRPO + reranker BCE，N=4 或 N=8。

重点看：

- reranker BCE loss 是否稳定下降。
- reranker score 正负样本 margin 是否拉开。
- final QA reward / EM / F1 是否提升。
- 每 step wall time 增量是否可接受。

## 风险和处理

### 标签噪声

answer-string heuristic 会漏掉 paraphrase，也可能误标。

处理：

- 支持 `label_source=mixed`，优先 oracle/evidence doc，其次 answer-string。
- 对 answer-string 标签降低权重。
- 记录 `label_source` 指标，区分不同来源样本的 loss。

### 类别不平衡

top-N 文档中负样本远多于正样本。

处理：

- 采样控制 pos/neg 比例。
- `BCEWithLogitsLoss(pos_weight=...)` 或 per-example `loss_weights`。
- 对 all-negative query 默认跳过。

### 训练目标和推理目标不一致

BCE scorer 训练不必然提升生成式 `<rerank>` 输出。

处理：

- 推荐新增 reranker scorer inference。
- 如果必须保留生成式输出，可以将 BCE 作为辅助 loss，而不是完全替换；但这会扩大改造范围。

### N 个 BCE step 导致 stale rollout

同一个 global step 内，reranker 参数更新 N 次，但 rollout 只生成一次，后面的 BCE step 使用的数据相对当前 reranker 更旧。

处理：

- N 从 2 或 4 起步，不建议一开始设太大。
- replay buffer 混采，降低单 batch 过拟合。
- 后续若收益明显，再考虑每 K 个 BCE step 重新采样 reranker batch。

### Ray worker 调用顺序

同一个 reranker worker group 上连续提交多个 update，不能假设并发更新是安全的。

处理：

- driver 逐次 `ray.get`，保证 BCE step 顺序更新。
- main GRPO 可和 reranker BCE 序列并行，因为二者使用不同 worker group。

## 推荐实施顺序

1. 增加 BCE sample builder 和 label builder，先离线从已有 trajectory / reranker_batch 构造样本并统计正负比例。
2. 在 `DataParallelPPOActor` 和 `FSDP worker` 增加 `update_reranker_bce`，用 fake batch 跑通单步 loss。
3. 在 trainer 中增加 `reranker_update_mode=bce` 分支，先 N=1 跑通。
4. 增加 `reranker_bce_steps_per_global_step=N` 调度和 replay buffer。
5. 加入 scorer inference mode，并做 N=1 / N=4 / baseline 对照。
6. 根据指标决定是否进一步做辅助生成 loss 或更强标签构造。

## 最小可审版本

如果只做第一版最小闭环，建议范围控制为：

1. `search_r1_reranker_reward_ray_trainer.py`
   - 新增 `process_reranker_bce_step`
   - 主循环增加 `reranker_update_mode=bce` 分支
   - 支持 `reranker_bce_steps_per_global_step`

2. `fsdp_workers.py`
   - 新增 `update_reranker_bce`

3. `dp_actor.py`
   - 新增 `update_reranker_bce`
   - 使用 yes/no token delta BCE

4. `search_r1_dual_agent_loop.py`
   - 在 reranker batch / BCE sample 中带出 candidate docs、query、answers

5. config / script
   - 增加 BCE 相关配置项
   - 默认 `reranker_update_mode=grpo`，显式开启 BCE，避免破坏现有实验

这个版本可以先不改 scorer inference，只验证 BCE step 是否稳定工作。但如果要验证最终效果，建议尽快补上 `reranker_inference_mode=score`。
