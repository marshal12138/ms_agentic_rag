# Retriever Contrastive Step 修正方案 Fix Ver2

本文档用于修正 `260606_retriever_contrastive_step_plan.md` 中 retriever 训练规划不准确的问题，并作为后续代码改造的确认版计划。

确认时间：2026-06-10

## 1. 问题复盘

上一版规划的核心问题是：把 retriever 训练设计成“一个 retriever 训练逻辑下拆分 query encoder 和 doc encoder，并只训练 query encoder”的方案。

当前代码事实是：

```text
LocalRetrieverContrastiveWorker
  query_encoder = AutoModel.from_pretrained(query_encoder_path)
  doc_encoder   = AutoModel.from_pretrained(doc_encoder_path)
  train_doc_encoder = false

训练时：
  query -> query_encoder -> query_emb
  docs  -> frozen doc_encoder -> doc_emb
  logits = query_emb @ doc_emb.T / temperature
  只更新 query_encoder
```

这等价于让可训练 query encoder 去适配一个固定 doc embedding 空间。这个方案有两个明显问题：

1. 它不是 Agentic-R / Tevatron 风格的共享 encoder dense retriever 训练。
2. 它把 retriever 训练目标和在线召回服务混在一起考虑，导致“训练模型”和“召回模型”的职责不清。

本次修正后，系统中应明确存在两个 retriever 模型角色，而不是一个模型的 query/doc 两侧拆分训练。

## 2. 新目标图景

整个 CoAgenticRetriever 流程中存在两个 retriever 模型，二者都初始化自 `e5-base-v2`，但职责完全不同。

```text
GPU 05: recall retriever / 召回模型
  - e5-base-v2
  - 不训练
  - 使用固定 query/doc encoder 和固定 FAISS doc index
  - 负责在线 top50 召回

GPU 04: rank retriever / 精排模型
  - e5-base-v2
  - 训练
  - query 和 doc 共享同一个 encoder
  - 对 recall retriever 返回的 top50 docs 重新打分
  - 选择 top5 docs 进入 agent 上下文或记录为训练样本
  - 使用对比学习训练
```

关键约束：

1. 召回模型和精排模型都来自 `e5-base-v2`，但训练后只有精排模型发生变化。
2. 召回模型保持 frozen，不随精排模型训练刷新。
3. 召回模型的 FAISS doc index 固定，不由精排模型更新。
4. 精排模型不负责全库 ANN 检索，只负责 top50 内 rerank。
5. 精排模型使用一个共享 encoder 同时编码 query 和 doc，不再拆成独立的 `query_encoder` / `doc_encoder`。

## 3. 修正后的端到端流程

训练主流程应改为：

```text
for global_step:
    agent rollout
        -> 调用调度脚本启动的 recall retriever
        -> 得到 recall_top50_docs
        -> 调用配置传入的 rank retriever
        -> 对 query + recall_top50_docs 重新打分
        -> 得到 rerank_top50_docs / ranked_passages
        -> 选择 rerank_top5_docs 给 agent
        -> 记录完整 tool_call / rerank trace

    agent LLM:
        -> process_single_agent_ppo_step
        -> actor_rollout_wg.update_actor

    rank retriever:
        -> 从 rollout trace 构造 contrastive samples
        -> process_retriever_contrastive_step
        -> rank_retriever_wg.update_retriever_contrastive
```

其中：

- `global_step` 仍按 agent LLM update step 计数。
- `retriever/update_step` 单独记录 rank retriever 的更新次数。
- recall retriever 在训练期间只服务，不反向传播。
- rank retriever 每个 agent global step 可执行 N 次更新，默认仍保留 `N=2`。

## 4. 模型边界

### 4.1 Recall Retriever

召回模型是 frozen online retriever：

```yaml
recall_retriever:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  device: cuda:5
  trainable: false
  top_k: 50
  index_refresh: false
```

物理默认位置通过配置默认值和本地调度脚本指定为 GPU05；核心框架 Python
代码只接收 `device` 配置值，不在代码中写死 GPU 编号。

职责：

1. 接收 agent 发出的 search query。
2. 用固定 e5-base-v2 query encoder 编码 query。
3. 使用固定 FAISS index 做 top50 召回。
4. 返回 top50 docs，包括 `doc_id`、`rank`、`retriever_score`、`title`、`text`、metadata。

### 4.2 Rank Retriever

精排模型是 trainable shared-encoder reranker：

```yaml
rank_retriever:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  device: cuda:4
  trainable: true
  shared_encoder: true
  rerank_top_k: 5
```

物理默认位置通过配置默认值和本地调度脚本指定为 GPU04。

职责：

1. 接收 query 和 recall retriever 返回的 top50 docs。
2. 使用同一个 encoder 分别编码 query 和 docs。
3. 计算相似度分数。
4. 对 top50 docs 排序并选择 top5。
5. 训练时对共享 encoder 做 forward/backward/update。

后续代码命名建议：

```text
不要继续使用：
  query_encoder
  doc_encoder
  train_doc_encoder

建议使用：
  recall_retriever
  rank_retriever
  rank_encoder
  shared_encoder
  rerank_top_k
```

## 5. Agentic-R 参考原则

Agentic-R / Tevatron dense retriever 的核心思想是：query 和 passage 复用同一个 encoder。

抽象形式：

```python
q_reps = encoder(query_tokens)
p_reps = encoder(passage_tokens)
scores = q_reps @ p_reps.T
loss = cross_entropy(scores / temperature, target)
```

在 Tevatron 代码中，`DenseModel.encode_passage` 复用 `encode_query`：

```python
def encode_passage(self, psg):
    return self.encode_query(psg)
```

本项目修正后的 rank retriever 应遵循这个原则：

```text
一个 AutoModel / encoder 实例
  encode_query(query)
  encode_docs(docs)
  query/doc 共享参数
  contrastive loss 反向更新同一个 encoder
```

## 6. Rank Retriever 对比学习训练

第一版修正只保留最小稳定目标：InfoNCE / cross entropy over grouped docs。

单条训练样本结构：

```python
{
    "sample_id": str,
    "trajectory_id": str,
    "tool_call_id": str,
    "origin_query": str,
    "sub_query": str,
    "query_input": str,
    "recall_top50_docs": list[dict],
    "rerank_top50_docs": list[dict],
    "ranked_passages": list[dict],
    "positive": dict,
    "negatives": list[dict],
    "positive_doc_index": int,
    "label_source": str,
    "sample_source": "fresh" | "replay",
}
```

训练 batch 语义：

```text
batch_size = B
docs_per_query = 1 positive + M negatives

query_input_ids:      [B, query_len]
query_attention_mask: [B, query_len]
doc_input_ids:        [B, docs_per_query, doc_len]
doc_attention_mask:   [B, docs_per_query, doc_len]
positive_doc_index:   [B]
```

forward/update：

```text
query_emb = shared_encoder(query_input_ids)
doc_emb   = shared_encoder(flatten(doc_input_ids))
scores    = dot(query_emb, doc_emb)
logits    = scores / temperature
loss      = cross_entropy(logits, positive_doc_index)
```

第一版不加入 RankNet / distillation loss。原因是先修正模型边界和资源边界，确保共享 encoder 精排训练跑通。后续可以增加：

```yaml
rank_retriever_training:
  loss:
    type: info_nce_plus_ranknet
    ranknet_weight: 0.2
```

## 7. 样本构造策略

上一版的策略模块可以保留，但字段语义要修正为 recall/rank 两阶段。

仍然保留：

```text
trajectory_selector
signal_builder
sample_builder
replay_buffer
collator
logging_utils
```

需要修正：

1. `signal_builder` 的输入应来自 rank retriever 排序后的 `ranked_passages` / `rerank_top50_docs`。
2. `sample_builder` 应从 `rerank_top50_docs` 中选择 positive/negative，而不是从 recall 排序中选正负例。
3. `collator` 输出仍可复用 query/doc token batch，但语义改为 rank retriever 训练 batch。
4. construction logger 必须打印：
   - origin query
   - sub query
   - recall top50 的前若干项
   - rerank top50 的前若干项
   - rerank top5
   - positive
   - negatives
   - label source
   - positive rank in recall top50
   - positive rank in rerank top5, if available

默认策略建议：

```yaml
rank_retriever_training:
  trajectory_selector:
    type: top_f1_trajectories
    max_selected_trajectories: 1
    min_final_reward: 0.0

  signal_builder:
    type: topk_pseudo_rank
    positive_top_k: 5
    source: recall_top50

  sample_builder:
    type: random_negative_repeat
    num_groups_per_step: 32
    neg_per_pos: 15
    negative_pool: recall_top50_exclude_positive
    allow_repeat_negative_sampling: true
```

## 8. GPU 资源分配

修正后的默认资源分配：

```text
GPU 0-3: agent LLM 训练 + rollout
GPU 4:   rank retriever shared encoder 训练 + rerank forward
GPU 5:   recall retriever online service + FAISS retrieval
GPU 6-7: 预留
```

重要边界：

1. GPU05 上的 recall retriever 不加载训练 checkpoint。
2. GPU04 上的 rank retriever 不刷新 GPU05 的 FAISS index。
3. GPU04 rank retriever 可以保存 checkpoint，用于后续 rerank eval 或离线部署。
4. 如果未来要让 trained rank retriever 影响召回，需要单独设计 index rebuild / service swap，不属于本版目标。

## 9. 配置建议

新增或修正配置命名：

```yaml
trainer:
  rank_retriever_trainable: true
  rank_retriever_update_mode: contrastive
  rank_retriever_steps_per_global_step: 2
  rank_retriever_start_step: 0
  rank_retriever_stop_step: null

recall_retriever:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  device: cuda:5
  top_k: 50
  trainable: false
  index_refresh: false
  service_url: http://127.0.0.1:8030/retrieve

rank_retriever:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  device: cuda:4
  shared_encoder: true
  rerank_top_k: 5
  max_query_length: 192
  max_doc_length: 256

rank_retriever_training:
  batch_size: 32
  max_grad_norm: 1.0
  log_every_n_steps: 10
  log_first_sample: true
  construction_log_jsonl: null

  loss:
    type: info_nce
    temperature: 0.05

  replay_buffer:
    enable: true
    max_size: 200000
    fresh_ratio: 0.5

  optim:
    lr: 2.0e-5
    weight_decay: 0.01
    warmup_steps: 0
    total_steps: 1000
```

兼容性建议：

- 旧字段 `retriever_model.train_doc_encoder` 应废弃。
- 旧字段 `retriever_model.query_encoder_path` / `doc_encoder_path` 不应再作为 rank retriever 主配置。
- 若为了兼容历史 checkpoint 保留字段，应在代码中标记 deprecated，并避免新脚本使用。

## 10. 代码改造计划

### 10.1 Worker 改造

目标文件：

```text
CoAgenticRetriever/verl/verl/workers/retriever/e5_retriever_worker.py
```

改造目标：

1. 将当前 `LocalRetrieverContrastiveWorker` 从 query/doc 双 encoder 改为 shared encoder。
2. 删除 `doc_encoder`、`query_encoder_path`、`doc_encoder_path`、`train_doc_encoder` 的核心训练路径。
3. 新增或重命名为：

```text
LocalE5RankRetrieverWorker
  encoder
  encode_query
  encode_docs
  rerank_topk
  update_retriever_contrastive
```

4. `save_checkpoint` 保存共享 encoder：

```text
checkpoint_dir/
  rank_encoder/
  tokenizer files
```

### 10.2 Rerank 接口

新增 rank retriever 推理接口：

```python
rerank_topk(
    query: str,
    docs: list[dict],
    top_k: int = 5,
) -> list[dict]
```

输出应保留 recall 信息和新增 rank 信息：

```python
{
    "doc_id": str,
    "title": str,
    "text": str,
    "recall_rank": int,
    "recall_score": float,
    "rank_score": float,
    "rank_rank": int,
}
```

### 10.3 Trainer 主循环

目标文件：

```text
CoAgenticRetriever/verl/verl/trainer/ppo/search_r1_retriever_contrastive_ray_trainer.py
CoAgenticRetriever/verl/verl/trainer/ppo/retriever_contrastive_step.py
```

改造目标：

1. trainer 中显式区分 `recall_retriever` 和 `rank_retriever_wg`。
2. `process_retriever_contrastive_step` 继续作为 rank retriever 训练 step 的编排函数。
3. 不把 recall retriever 的服务刷新和 rank retriever 训练混在一个配置里。
4. fresh trajectories 中保留 recall top50 和 rerank top5。

### 10.4 Tool / Rollout Trace

需要保证 `CoSearchTool` 或 rollout trace 保存：

```python
{
    "tool_call_id": str,
    "sub_query": str,
    "recall_top50_docs": list[dict],
    "rerank_top50_docs": list[dict],
    "ranked_passages": list[dict],
    "rerank_top5_docs": list[dict],
    "rank_model_checkpoint": str | None,
}
```

如果第一版暂时不把 rank retriever 插入在线 agent 上下文，则 smoke/inference 至少要验证：

```text
recall top50 -> rank retriever rerank -> output top5
```

## 11. 本地脚本改造计划

目标目录：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local
```

训练脚本应支持：

```bash
RECALL_GPU_ID=5
RANK_GPU_ID=4
RECALL_TOP_K=50
RERANK_TOP_K=5
TOTAL_STEPS=2
```

推理脚本应验证：

```text
1. 调度脚本默认启动在 GPU05 的 recall retriever 返回 top50
2. GPU04 rank retriever 对 top50 rerank
3. 输出 top5
4. MAX_EVAL_STEPS=1
```

本地验证可分为两层：

1. `co-training`：默认训练模式，启动完整 CoSearch/VERL 路径，同时训练 agent LLM 和 dense rank retriever。
2. `dense-reranker-only`：不启动完整 LLM，默认通过 00 脚本启动 GPU05 recall service 获取 top50，并在 GPU04 上验证 rank retriever 共享 encoder 训练。
3. `service_smoke`：启动或调用 GPU05 recall service，再调用 GPU04 rank retriever 完成 top50 -> top5。

物理 GPU 编排允许写在 YAML 配置默认值和 `scripts/coagenticRetriever_local`
调度脚本中；核心框架 Python 代码只接受配置传入的 device，不写死 GPU 编号。

## 12. 验证标准

实现完成后必须验证：

```text
训练：
  - 使用真实 GPU
  - rank retriever 在 GPU04
  - 只跑 2 step
  - 日志显示 shared encoder
  - 日志显示 query/doc 均由同一 encoder 编码
  - 输出 loss / acc@1 / mrr / score_margin

推理：
  - 使用真实 GPU
  - recall retriever 在 GPU05
  - rank retriever 在 GPU04
  - 只跑 1 step
  - 输出 recall_top50 摘要
  - 输出 rerank_top5
```

日志要求：

```text
每 10 step 打印 1 组完整 contrastive construction 数据。
smoke 验证时第 1 step 也打印 1 组完整样本。
```

完整样本日志必须包含：

```text
origin_query
sub_query
recall_top50 sample
rerank_top5 sample
positive doc
negative docs
positive_doc_index
label_source
sample_source
```

## 13. 非目标

本版不做：

1. 不训练 recall retriever。
2. 不刷新 recall FAISS doc index。
3. 不把 rank retriever checkpoint 自动同步到 recall service。
4. 不做全库 rerank。
5. 不在第一版引入复杂 distillation / RankNet 混合 loss。
6. 不复用 PPO/GRPO 的 `compute_log_prob`、`compute_advantage`、`update_actor` 语义来训练 retriever。

## 14. 结论

新的修正方向是：

```text
召回模型 frozen，部署 GPU05，负责 top50。
精排模型 trainable，部署 GPU04，使用 e5-base-v2 共享 encoder，对 top50 rerank 成 top5。
精排模型使用 Agentic-R 风格的共享 encoder 对比学习训练。
```

后续代码实现应以这个边界为准，废弃上一版“只训练 query encoder、doc encoder frozen”的 retriever 训练路线。
