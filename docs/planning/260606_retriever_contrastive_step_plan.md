# Retriever Contrastive Step 改造方案

本文档记录新的 CoAgenticRetriever 改造目标：使用 E5-base-v2 作为可训练 retriever encoder，并在当前 CoSearch/CoAgenticRetriever 的 VERL 训练框架中，同步训练 agent LLM 和 retriever。

核心目标：

1. agent LLM 继续按当前 CoSearch 路线执行 GRPO/PPO step。
2. retriever 不再只是外部检索服务，而是 E5-base-v2 模型本体参与训练。
3. retriever 训练使用对比学习，新增 `process_retriever_contrastive_step`。
4. 每执行 1 次 agent LLM 梯度计算和参数更新，就执行 N 次 retriever 对比学习梯度计算和参数更新，默认 `N=2`。
5. 保留策略空间：轨迹选择、监督信号构造、正负例采样、loss 类型、replay buffer、在线服务同步策略都应可配置。

当前落地路径：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/
```

## 当前代码事实

当前 agent LLM 训练使用 PPO/GRPO step 函数：

```python
process_single_agent_ppo_step(...)
```

该函数的职责是处理 agent LLM 的 policy optimization：

```text
main_batch
  -> process_single_agent_ppo_step
  -> actor_rollout_wg.update_actor
```

retriever 对比学习和 agent PPO/GRPO 的数据结构、worker 类型、loss、资源分配都不同，因此不复用旧的双 agent trainer 作为改造基底。新的目标是新增独立 retriever trainer 脚本，并在其中编排 agent PPO/GRPO step 与 retriever contrastive step：

```text
CoAgenticRetriever/verl/verl/trainer/ppo/search_r1_retriever_contrastive_ray_trainer.py
```

该 trainer 中新增 retriever step：

```text
retriever_contrastive_batch
  -> process_retriever_contrastive_step
  -> retriever_wg.update_retriever_contrastive
```

## 目标机制

训练主循环应变为：

```text
for global_step:
    rollout once with agent LLM
    collect tool-call trajectories / retrieved top-N docs

    main agent:
        1 x process_single_agent_ppo_step

    retriever:
        N x process_retriever_contrastive_step

    global_step += 1
```

其中：

- `global_step` 仍按 agent LLM 的 update step 计数。
- retriever 的子步单独记录为 `retriever/update_step` 和 `retriever/step_idx_in_global`。
- N 由配置控制，默认 2。

配置建议：

```yaml
trainer:
  retriever_trainable: true
  retriever_update_mode: contrastive   # off | contrastive
  retriever_steps_per_global_step: 2
  retriever_start_step: 0
  retriever_stop_step: null
```

第一版已确认的代码边界：

1. `process_retriever_contrastive_step` 是 retriever 对比学习训练 step 的编排函数，不复用 PPO/GRPO 的 `compute_log_prob`、`compute_advantage`、`update_actor` 语义。
2. `process_retriever_contrastive_step` 放在和 `process_single_agent_ppo_step` 同级的位置，建议新增独立文件：

```text
CoAgenticRetriever/verl/verl/trainer/ppo/retriever_contrastive_step.py
```

3. 轨迹选择、监督信号构造、正负例采样、replay buffer、collator 放到 trainer 外部，形成可替换策略模块。目录使用正确拼写：

```text
CoAgenticRetriever/retriever_strategies/
  __init__.py
  schemas.py
  trajectory_selector.py
  signal_builder.py
  sample_builder.py
  replay_buffer.py
  collator.py
```

4. 默认 selector 选择 F1 得分最高的合法轨迹。合法轨迹数量做成可选参数，默认 1。
5. 被选中的每条合法轨迹返回其内部所有合法 search/tool-call context，而不是只返回一个 tool call。

第一版已确认的具体设计：

1. `process_single_agent_ppo_step` 本身不改返回值；retriever 训练所需数据从 rollout 输出侧新增 `fresh_trajectories`。
2. `fresh_trajectories` 使用本文档定义的完整 trajectory schema。
3. `max_selected_trajectories` 默认 1，并从当前 rollout batch 全局选择 final answer F1 最高的合法轨迹。
4. 样本不足 32 组时，允许重复使用 positive，并为每次重复重新随机采样 negatives。
5. `update_retriever_contrastive` 第一版只训练 query encoder，doc encoder frozen，避免现有 FAISS doc index 立即失效。

## GPU 资源分配

8 卡默认分配：

```text
GPU 0-3: agent LLM 训练 + rollout
GPU 4:   retriever encoder 训练
GPU 5:   retriever online service
GPU 6-7: 待定 / 预留
```

这里需要明确区分两个 retriever 副本：

```text
GPU 4: train retriever encoder
GPU 5: serve retriever encoder
```

第一版建议接受“服务模型滞后训练模型”的事实，不要每步刷新在线服务。推荐配置：

```yaml
retriever_training:
  service_refresh_interval: 10
```

第一版也不建议刷新全量 doc index。E5-base-v2 当前 FAISS index 是固定 doc embedding 空间，如果训练 doc encoder，旧 index 会失效。因此第一版建议：

```yaml
retriever_model:
  path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  train_query_encoder: true
  train_doc_encoder: false
  doc_index_refresh: false
```

优先只训练 query encoder，让 query embedding 更好地匹配固定 doc embedding/index。

## `process_retriever_contrastive_step` 的构造原则

`process_retriever_contrastive_step` 应负责 retriever 训练 step 的编排，但不要混入 PPO/GRPO 的数据契约。

不要在 retriever contrastive step 中调用：

```text
compute_log_prob
compute_ref_log_prob
compute_advantage
update_actor
```

这些属于 LLM policy optimization。

推荐职责划分：

```text
process_retriever_contrastive_step:
  1. 选择用于构造对比学习样本的轨迹数据
  2. 构造对比学习监督信号
  3. 选取正负例并构造 contrastive samples
  4. 从 fresh samples / replay buffer 中采样训练 batch
  5. collate/tokenize 成 retriever DataProto
  6. 调用 retriever_wg.update_retriever_contrastive
  7. 汇总 metrics / timing

retriever_wg.update_retriever_contrastive:
  1. E5 encode query / docs
  2. 计算 similarity scores
  3. 计算 contrastive loss
  4. backward
  5. optimizer step
  6. 返回训练指标
```

策略性、可替换的部分放在 step 的 CPU/driver 侧；真正依赖 GPU 的 forward/backward/update 放在 retriever worker 中。

确认后的函数签名：

```python
process_retriever_contrastive_step(
    fresh_trajectories,
    retriever_wg,
    replay_buffer,
    selector,
    signal_builder,
    sample_builder,
    collator,
    config,
    global_steps,
    retriever_step_idx,
)
```

确认后的内部流程：

```python
selected_contexts = selector.select(fresh_trajectories)

labeled_docs = signal_builder.build(selected_contexts)

contrastive_samples = sample_builder.build(labeled_docs)

if retriever_step_idx == 0:
    replay_buffer.add(contrastive_samples, source_step=global_steps)

train_samples = replay_buffer.sample(
    batch_size=config.retriever_training.batch_size,
    fresh_ratio=config.retriever_training.fresh_ratio,
)

batch = collator(train_samples)

output = retriever_wg.update_retriever_contrastive(batch)

return output.metrics
```

注意：如果每个 agent global step 后执行 N 次 retriever update，fresh samples 不能在 N 个子步里重复加入 replay buffer。第一版约定只在 `retriever_step_idx == 0` 时 add，或者由 replay buffer 基于 `(global_steps, trajectory_id, tool_call_id)` 做去重。

## `fresh_trajectories` 形态和 agent step 输出改造

`fresh_trajectories` 来自本轮 agent rollout，而不是来自 agent PPO update 的梯度结果。agent 的 `process_single_agent_ppo_step` 本身暂时不需要改变 PPO/GRPO 更新逻辑，但 agent rollout / trainer 主循环需要额外保留并传出 retriever 训练所需的轨迹结构。

第一版建议 `fresh_trajectories` 为 list[dict]，每个元素对应一条 origin query 的一次完整 rollout：

```python
{
    "trajectory_id": str,
    "origin_query": str,
    "golden_answers": list[str],
    "final_answer": str,
    "score": float,
    "score_type": "f1",
    "is_valid": bool,
    "messages": list[dict],
    "tool_calls": [
        {
            "tool_call_id": str,
            "turn_idx": int,
            "tool_name": "search",
            "sub_query": str,
            "retrieved_passages": [
                {
                    "doc_id": str,
                    "rank": int,
                    "title": str | None,
                    "text": str,
                    "retriever_score": float | None,
                    "metadata": dict,
                },
                ...
            ],
        },
        ...
    ],
    "metadata": {
        "dataset": str | None,
        "qid": str | None,
        "global_steps": int,
    },
}
```

其中：

- `origin_query` 来自训练数据原问题。
- `sub_query` 来自 agent 在 rollout 中发起 search/tool call 时生成的检索 query。
- `retrieved_passages` 至少保留 top50，供 signal builder 和 sample builder 使用。
- `score` 当前默认为 final answer F1，用于 trajectory selector。
- `messages` 用于必要时追溯 prompt / answer / tool 格式，但不进入默认对比学习 collator。

对 agent 过程的最小改造点：

1. rollout manager / agent loop 在每条轨迹结束时，额外构造上述 retriever trajectory record。
2. trainer 主循环在拿到 rollout batch 的同时，拿到 `fresh_trajectories`。
3. `process_single_agent_ppo_step` 仍处理 `main_batch`，不强行塞入 retriever 字段。
4. `process_retriever_contrastive_step` 消费 `fresh_trajectories`，与 PPO batch schema 解耦。

## 对比学习任务构造策略空间

对比学习样本构造应拆成三层，避免写死在 trainer 里。

### 轨迹选择策略

从 rollout 轨迹中选择哪些 tool call 用于 retriever 训练。

```yaml
retriever_training:
  trajectory_selector:
    type: top_f1_trajectories
    max_selected_trajectories: 1
    min_final_reward: 0.0
```

可选策略：

- `top_f1_trajectories`：按 final answer F1 从高到低选择合法轨迹，默认选择 1 条。
- `all_tool_calls`：所有合法 tool call 都进入候选池。
- `successful_answer`：只选择 final answer reward 达标的轨迹，标签噪声较低。
- `failed_answer`：选择失败轨迹，用于挖 hard cases。
- `reward_weighted`：按 final reward 给样本加权。
- `hard_cases`：优先选择 top-N 中有答案但 agent 仍答错的样本。

第一版确认：

```text
top_f1_trajectories，max_selected_trajectories 默认 1。
对每条被选中轨迹，返回其中所有合法 search/tool-call context。
```

合法 tool-call context 的最低要求：

```text
origin_query 非空
sub_query 非空
retrieved_passages 数量 > 0
retrieved_passages 中可解析 doc_id/rank/text
```

### 监督信号构造策略

从 origin query、sub query、candidate docs、golden answers 中构造正负信号。

```yaml
retriever_training:
  signal_builder:
    type: topk_pseudo_rank  # topk_pseudo_rank | answer_string | oracle_evidence | llm_judge | reward_delta | mixed
    positive_top_k: 5
    allow_all_negative: false
```

可选策略：

- `topk_pseudo_rank`：将当前 retriever 召回排序的 top-K 文档标为 positive，其余标为 negative。
- `answer_string`：如果 doc title/content/contents 包含 normalized golden answer，则标为 positive。
- `oracle_evidence`：如果数据中已有 evidence doc id，优先使用。
- `llm_judge`：用 LLM 判断 passage 是否支持 origin query / sub query。
- `reward_delta`：通过 counterfactual remove/replace doc 后 final reward 变化构造信号。
- `mixed`：多策略融合，按置信度加权。

第一版确认：

```text
topk_pseudo_rank，positive_top_k 默认 5。
```

默认数据格式：

```python
{
    "trajectory_id": str,
    "tool_call_id": str,
    "turn_idx": int,
    "origin_query": str,
    "sub_query": str,
    "trajectory_score": float,
    "score_type": "f1",
    "passages": [
        {
            "doc_id": str,
            "rank": int,
            "title": str | None,
            "text": str,
            "retriever_score": float | None,
            "label": 1,  # rank <= positive_top_k
            "label_source": "top5_pseudo_rank",
            "metadata": dict,
        },
        ...
    ],
}
```

风险：`topk_pseudo_rank` 是伪标签，会强化当前 retriever 的排序偏置。第一版接受它作为 bootstrap 策略，但必须保留 `label_source`，并为后续 `answer_string`、`llm_judge`、`reward_delta` 预留替换空间。

### 正负例采样策略

从候选文档中构造 contrastive sample。

```yaml
retriever_training:
  sample_builder:
    type: random_negative_repeat
    num_groups_per_step: 32
    neg_per_pos: 15
    allow_repeat_negative_sampling: true
    use_in_batch_negatives: false
```

可选策略：

- `random_negative_repeat`：每个 positive 随机搭配固定数量 negatives；样本不足时允许重复负采样。
- `topn_hard_negative`：从当前 retriever top-N 高排名但非 positive 的 docs 中采负例。
- `in_batch_negative`：同一 batch 中其他 query 的 positive docs 作为负例。
- `mixed`：top-N hard negative + in-batch negative 结合。

第一版确认：

```text
label=1 的 passage 作为正样本。
label=0 的 passage 作为负样本。
每个正样本随机匹配 15 个负样本，形成 1+15 的 contrastive group。
默认构造 32 组。
如果 fresh samples 不足 32 组，sample_builder 默认通过重复负采样补足。
```

Agentic-R 对齐点：

```text
query_input = origin_query + " [SEP] " + sub_query
passages = [positive] + negatives
loss = cross_entropy(query_emb @ passage_emb.T / temperature, label=0)
```

第一版确认固定使用 Agentic-R 风格 query 输入：

```text
query_input = origin_query + " [SEP] " + sub_query
```

不直接只用 `sub_query`，原因是 origin query 提供了完整任务上下文，且该格式和 Agentic-R 最小对比学习实现一致。

第一版 contrastive sample schema：

```python
{
    "sample_id": str,
    "query_input": str,
    "origin_query": str,
    "sub_query": str,
    "positive": {
        "doc_id": str,
        "rank": int,
        "title": str | None,
        "text": str,
    },
    "negatives": [
        {
            "doc_id": str,
            "rank": int,
            "title": str | None,
            "text": str,
        },
        ...
    ],
    "positive_doc_index": 0,
    "label_source": "top5_pseudo_rank",
    "trajectory_id": str,
    "tool_call_id": str,
}
```

## Retriever contrastive batch schema

建议不要复用 PPO batch 字段。新增 retriever contrastive schema。

Tensor fields：

```text
query_input_ids:       [B, Lq]
query_attention_mask:  [B, Lq]
doc_input_ids:         [B, K, Ld]
doc_attention_mask:    [B, K, Ld]
positive_doc_index:    [B]
loss_weights:          [B] optional
```

Non-tensor fields：

```text
origin_query
sub_query
doc_ids
golden_answers
trajectory_uid
turn_id
tool_call_id
label_source
sample_source
```

不需要以下 PPO 字段：

```text
responses
response_mask
old_log_probs
advantages
returns
ref_log_prob
token_level_rewards
```

## Loss 设计

第一版推荐 InfoNCE / MultipleNegativesRankingLoss：

```text
query_emb = normalize(E5(query))
doc_emb   = normalize(E5(doc))
score     = dot(query_emb, doc_emb) / temperature
loss      = cross_entropy(scores, positive_doc_index)
```

配置：

```yaml
retriever_training:
  loss:
    type: info_nce  # info_nce | pairwise_margin
    temperature: 0.05
```

第一版统一使用 `process_retriever_contrastive_step`，loss 默认 `info_nce`。

## `update_retriever_contrastive` 计算和更新细节

`update_retriever_contrastive` 是 retriever worker 暴露给 trainer 的 GPU 训练接口，负责完整的 forward / loss / backward / optimizer step。

第一版 batch 输入：

```text
B = config.retriever_training.batch_size
K = 1 + config.retriever_training.sample_builder.neg_per_pos

query_input_ids:       [B, Lq]
query_attention_mask:  [B, Lq]
doc_input_ids:         [B, K, Ld]
doc_attention_mask:    [B, K, Ld]
positive_doc_index:    [B]，第一版全为 0
loss_weights:          [B] optional
```

第一版 forward：

```python
query_emb = retriever.encode_query(
    input_ids=query_input_ids,
    attention_mask=query_attention_mask,
)  # [B, H]

flat_doc_input_ids = doc_input_ids.reshape(B * K, Ld)
flat_doc_attention_mask = doc_attention_mask.reshape(B * K, Ld)

doc_emb = retriever.encode_doc(
    input_ids=flat_doc_input_ids,
    attention_mask=flat_doc_attention_mask,
)  # [B*K, H]

doc_emb = doc_emb.reshape(B, K, H)

query_emb = normalize(query_emb, dim=-1)
doc_emb = normalize(doc_emb, dim=-1)

scores = einsum("bh,bkh->bk", query_emb, doc_emb)
logits = scores / config.retriever_training.loss.temperature
```

第一版 loss：

```python
labels = positive_doc_index  # [B], default zeros
per_sample_loss = cross_entropy(logits, labels, reduction="none")

if loss_weights is not None:
    loss = (per_sample_loss * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
else:
    loss = per_sample_loss.mean()
```

第一版梯度更新：

```python
optimizer.zero_grad(set_to_none=True)
loss.backward()
grad_norm = clip_grad_norm_(trainable_parameters, max_norm=config.retriever_training.max_grad_norm)
optimizer.step()
scheduler.step()
```

第一版参数冻结策略：

```yaml
retriever_model:
  train_query_encoder: true
  train_doc_encoder: false
```

如果 query encoder 和 doc encoder 共享同一个 E5 encoder 参数，则不能只通过一次模型 forward 同时训练 query 侧、冻结 doc 侧。第一版推荐实现成“训练 query encoder + 固定 doc encoder”两个权重对象：

```text
query_encoder: train mode, requires_grad=True
doc_encoder:   eval mode, requires_grad=False
```

`doc_encoder` 初始从同一个 E5-base-v2 权重加载，输出固定 doc embedding 空间，以避免 FAISS index 立即失效。

如果第一版为了简化只使用单 encoder 计算 query/doc，则必须明确风险：doc encoder 参数更新后，在线 FAISS index 的 doc embedding 会和训练后的 encoder 不一致。这个模式只能用于离线 smoke，不建议作为正式训练默认。

第一版返回指标：

```python
pred = logits.argmax(dim=-1)
acc_at_1 = (pred == labels).float().mean()

pos_scores = scores.gather(1, labels[:, None]).squeeze(1)
neg_mask = torch.ones_like(scores, dtype=torch.bool)
neg_mask.scatter_(1, labels[:, None], False)
neg_scores = scores[neg_mask].reshape(B, K - 1)

metrics = {
    "retriever/loss": loss.item(),
    "retriever/acc@1": acc_at_1.item(),
    "retriever/pos_score_mean": pos_scores.mean().item(),
    "retriever/neg_score_mean": neg_scores.mean().item(),
    "retriever/score_margin": (pos_scores.mean() - neg_scores.mean()).item(),
    "retriever/num_queries": B,
    "retriever/num_docs_per_query": K,
    "retriever/lr": scheduler.get_last_lr()[0],
    "retriever/grad_norm": grad_norm.item(),
}
```

## Worker 改造点

新增 retriever worker，而不是复用 LLM actor worker。

建议新增模块：

```text
verl/verl/workers/retriever/
  e5_retriever_worker.py
  retriever_contrastive_actor.py
```

worker 暴露接口：

```python
def update_retriever_contrastive(self, data: DataProto) -> DataProto:
    # load data to GPU
    # encode query / docs
    # compute contrastive loss
    # backward
    # optimizer step
    # return DataProto(meta_info={"metrics": metrics})
```

返回指标：

```text
retriever/loss
retriever/acc@1
retriever/mrr
retriever/pos_score_mean
retriever/neg_score_mean
retriever/score_margin
retriever/num_queries
retriever/num_pos
retriever/num_neg
retriever/buffer_size
retriever/lr
retriever/grad_norm
```

## Retriever trainer 改造点

不直接修改旧双 agent trainer，也不复用其第二个 LLM actor 的训练结构。新增独立 retriever trainer：

```text
CoAgenticRetriever/verl/verl/trainer/ppo/search_r1_retriever_contrastive_ray_trainer.py
```

该 trainer 中新增 remote step：

```python
@ray.remote
def process_retriever_contrastive_step(
    fresh_trajectories: list,
    retriever_wg: RayWorkerGroup,
    replay_buffer,
    selector,
    signal_builder,
    sample_builder,
    collator,
    config,
    global_steps: int,
    retriever_step_idx: int,
):
    # 1. select trajectories
    # 2. build labels/signals
    # 3. build contrastive samples
    # 4. add fresh samples to buffer
    # 5. sample train batch
    # 6. collate/tokenize to DataProto
    # 7. call retriever_wg.update_retriever_contrastive(batch)
    # 8. return metrics/timing
```

主循环目标结构：

```python
main_future = process_single_agent_ppo_step.remote(...)

retriever_results = []
if retriever_train_enabled:
    for i in range(config.trainer.retriever_steps_per_global_step):
        result = ray.get(
            process_retriever_contrastive_step.remote(
                fresh_trajectories=retriever_trajectories,
                retriever_wg=self.retriever_wg,
                replay_buffer=self.retriever_replay_buffer,
                selector=self.retriever_trajectory_selector,
                signal_builder=self.retriever_signal_builder,
                sample_builder=self.retriever_sample_builder,
                collator=self.retriever_collator,
                config=self.config,
                global_steps=self.global_steps,
                retriever_step_idx=i,
            )
        )
        retriever_results.append(result)

main_results = ray.get(main_future)
```

注意：

- main agent 和 retriever 可以并行，因为使用不同 GPU 资源池。
- 同一个 retriever worker group 的 N 次 update 应顺序执行，不要并发提交 N 个更新。
- `global_step` 只在 main step 后递增一次。

## Replay buffer

如果每个 global step 做 N 次 retriever update，仅重复 fresh samples 容易过拟合当前 rollout。建议新增轻量 replay buffer：

```text
RetrieverContrastiveReplayBuffer
  max_size
  add(fresh_samples)
  sample(batch_size, fresh_ratio)
```

配置：

```yaml
retriever_training:
  replay_buffer:
    enable: true
    max_size: 200000
    batch_size: 256
    fresh_ratio: 0.5
```

行为：

1. 每个 global step 的 rollout 产出 fresh retriever samples。
2. fresh samples 进入 buffer。
3. 第 1 个 retriever contrastive step 优先使用 fresh samples。
4. 后续 step 从 buffer 混采 fresh + historical samples。

## 在线 retriever 服务同步

训练和服务建议解耦：

```text
GPU 4: retriever train worker
GPU 5: retriever online service
```

第一版只训练 query encoder，并保持 FAISS doc index 固定。在线服务刷新方式可配置：

```yaml
retriever_training:
  service_refresh_interval: 10
  service_refresh_mode: reload_weights  # reload_weights | restart_server | none
```

不建议第一版做：

```text
每 step 重启服务
每 step 重建 doc index
训练 doc encoder 后立即刷新全量 index
```

## 指标和日志

每个 global step 记录：

```text
main_actor/*
main_reward/*
retriever/loss
retriever/loss_step_0
retriever/loss_step_N
retriever/acc@1
retriever/mrr
retriever/score_margin
retriever/num_queries
retriever/num_pos
retriever/num_neg
retriever/buffer_size
retriever/fresh_samples
retriever/steps_per_global_step
```

timing：

```text
timing/main_grpo
timing/retriever_contrastive_total
timing/retriever_contrastive_step_mean
timing/rollout
timing/retriever_service_refresh
```

## 验证计划

### 单元测试

1. trajectory selector：
   - 成功/失败轨迹筛选正确。
   - tool call 抽取数量正确。

2. signal builder：
   - answer string 命中文档时产生 positive。
   - 无 positive 时默认跳过。
   - label_source 正确记录。

3. sample builder：
   - 正负例比例正确。
   - hard negatives 从 top-N 中采样。
   - in-batch negatives 生效。

4. retriever loss：
   - scores shape 正确。
   - InfoNCE loss 可 backward。
   - acc@1 / MRR 指标正确。

5. replay buffer：
   - capacity 裁剪正确。
   - fresh/historical 混采比例正确。

### Smoke run

极小参数：

```yaml
trainer:
  total_training_steps: 2
  retriever_trainable: true
  retriever_update_mode: contrastive
  retriever_steps_per_global_step: 2
retriever_training:
  replay_buffer:
    batch_size: 16
```

检查：

- agent LLM 的 GRPO step 正常完成。
- 每个 global step 记录 2 个 retriever contrastive 子步。
- retriever loss、grad_norm、lr 有数值。
- 不再创建第二个 LLM actor worker。
- 不再对 retriever 调用 `process_single_agent_ppo_step` 或 `update_actor`。

### 对照实验

最小三组：

1. baseline：agent GRPO + 固定 E5 retriever。
2. agent GRPO + retriever contrastive，N=1。
3. agent GRPO + retriever contrastive，N=2 或 N=4。

重点看：

- retriever loss 是否下降。
- positive/negative score margin 是否拉开。
- top-N recall 是否提升。
- final QA EM/F1 是否提升。
- 每 step wall time 增量是否可接受。

## 风险和处理

### 标签噪声

answer-string heuristic 会漏掉 paraphrase，也可能误标。

处理：

- 支持 `signal_builder=mixed`。
- answer-string 标签给较低权重。
- 后续加入 LLM judge 或 reward-delta 信号。

### Doc encoder / index 不一致

如果训练 doc encoder，现有 FAISS index 立即过期。

处理：

- 第一版只训练 query encoder。
- doc encoder 冻结。
- index 不刷新。

### 服务模型滞后训练模型

GPU 5 在线服务使用的 retriever 可能落后 GPU 4 训练权重。

处理：

- 设置 `service_refresh_interval`。
- 先从 10 或 50 个 global step 起步。
- 记录 refresh timing 和服务版本号。

### N 次 retriever update 使用 stale rollout

同一个 global step 内，retriever 参数更新 N 次，但 rollout 只生成一次。

处理：

- N 从 2 起步，不要一开始设太大。
- 使用 replay buffer 混采历史样本。
- 后续再考虑每 K 个 retriever step 重新 rollout。

## 推荐实施顺序

1. 新增独立 trainer：`search_r1_retriever_contrastive_ray_trainer.py`，不要直接改造旧双 agent trainer。
2. 在 agent rollout 输出中导出 retriever contrastive 所需的 tool-call 轨迹字段，形成 `fresh_trajectories`。
3. 实现 trajectory selector / signal builder / sample builder，并先离线统计正负比例。
4. 实现 E5 retriever worker 和 `update_retriever_contrastive`，用 fake batch 跑通。
5. 在新 retriever trainer 中新增 `process_retriever_contrastive_step`。
6. 在新 retriever trainer 中实现每个 agent step 后 N 次 retriever contrastive update。
7. 接入 GPU 5 在线 retriever 服务同步。
8. 做 baseline / N=1 / N=2 对照实验。

## 最小可审版本

第一版最小闭环建议只做：

1. `search_r1_retriever_contrastive_ray_trainer.py`
   - 新增独立 trainer 类
   - 编排 agent `process_single_agent_ppo_step`
   - 新增 `process_retriever_contrastive_step`
   - 支持 `retriever_steps_per_global_step`

2. agent rollout / agent loop
   - 从 tool call 轨迹中带出 origin_query、sub_query、top-N docs、golden answers、final reward

3. 新增 retriever training 模块
   - trajectory selector
   - signal builder
   - sample builder
   - replay buffer
   - collator

4. 新增 retriever worker
   - `update_retriever_contrastive`
   - InfoNCE loss
   - query encoder trainable，doc encoder frozen

5. config / launch script
   - GPU 0-3 agent
   - GPU 4 retriever train
   - GPU 5 retriever service
   - GPU 6-7 reserved
   - 默认 `retriever_steps_per_global_step=2`

这个版本先不刷新 doc index，只验证 agent LLM + retriever query encoder 同步训练能否稳定跑通。
