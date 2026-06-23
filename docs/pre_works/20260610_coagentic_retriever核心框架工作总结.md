# CoAgenticRetriever 核心框架工作总结

记录日期：2026-06-10  
工作目录：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives`  
核心代码目录：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever`  
本地脚本目录：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local`  
核心设计文档：`docs/framework_CoAgenticRetriever.md`  
修正计划文档：`docs/planning/260610_retriever_contrastive_step_plan_fix_ver2.md`

本文档记录 CoAgenticRetriever 核心框架的实现工作，供后续接手人员快速理解：当前框架为什么引入两个 retriever、代码入口在哪里、训练样本如何构造、哪些字段语义必须严格区分、训练/推理 smoke 如何验证。

## 一句话状态

当前已经完成 CoAgenticRetriever 的核心框架实现和 Fix Ver2 修正：

1. 系统中明确区分 frozen recall retriever 和 trainable rank retriever。
2. recall retriever 负责全库 top50 召回，不训练，默认部署在 GPU05。
3. rank retriever 负责对 recall top50 做精排，训练，默认部署在 GPU04。
4. rank retriever 使用 `e5-base-v2` 共享 encoder，query/doc 不再拆成两个独立 encoder。
5. 对比学习样本从 rank retriever 的完整 top50 排序结果构造。
6. `ranked_passages` / `rerank_top50_docs` 是 rank retriever 排序后的 50 个 chunk。
7. `rerank_top5_docs` 只用于 agent 下一步决策，不作为完整训练样本池。
8. 本地训练脚本已用真实 GPU 跑通 2 step。
9. 本地推理脚本已用真实 GPU 跑通 1 step。
10. 日志系统会按间隔打印完整的对比学习任务构造样例。

## 核心目标

本次工作不是复现官方 CoSearch 的 LLM PPO 主流程，而是在 CoSearch/VERL 框架旁边补上一个可训练的 dense rank retriever，使 agent rollout 产生的检索轨迹可以反过来训练一个共享 encoder 的精排模型。

修正后的目标图景：

```text
agent rollout
  -> recall retriever: frozen e5-base-v2, GPU05, FAISS top50
  -> rank retriever: trainable e5-base-v2 shared encoder, GPU04, rerank top50
  -> rerank_top5_docs 给 agent 继续推理
  -> rerank_top50_docs / ranked_passages 进入日志和训练样本构造

agent LLM update:
  -> 保持原 PPO/GRPO 路径

rank retriever update:
  -> trajectory_selector
  -> signal_builder
  -> sample_builder
  -> replay_buffer
  -> collator
  -> rank_retriever_worker.update_retriever_contrastive
```

## 与上一版错误规划的区别

上一版规划里曾把 retriever 训练理解成一个模型内部拆成 query encoder 和 doc encoder，并且只训练 query encoder。这个设计已被废弃。

当前代码语义是：

```text
recall retriever:
  - e5-base-v2
  - 不训练
  - 负责在线召回 top50
  - 固定 FAISS/index

rank retriever:
  - e5-base-v2
  - 训练
  - query/doc 共享同一个 encoder
  - 只在 recall top50 内做 rerank
  - 用 InfoNCE 对比学习训练
```

因此后续不要再恢复或扩展这些旧概念：

```text
query_encoder / doc_encoder 拆分训练
train_doc_encoder=false
只训练 query encoder
用 recall 排名直接构造 rank retriever 正负例
```

## 关键目录和文件

核心框架文件：

```text
CoAgenticRetriever/
  config/retriever_contrastive.yaml

  retriever_strategies/
    schemas.py
    config.py
    collator.py
    replay_buffer.py
    logging_utils.py
    trajectory_selector/top_f1.py
    signal_builder/topk_pseudo_rank.py
    sample_builder/random_negative_repeat.py

  verl/verl/trainer/ppo/retriever_contrastive_step.py
  verl/verl/trainer/ppo/search_r1_retriever_contrastive_ray_trainer.py
  verl/verl/workers/retriever/e5_retriever_worker.py

  main_coagentic_retriever.py
```

本地验证脚本：

```text
scripts/coagenticRetriever_local/
  00_start_dense_retriever_server.sh
  01_train_qwen3_4b_ablation_1epoch_timing.sh
  02_infer_qwen3_4b_ablation_val_only.sh
  README.md
  assets/
    01_retriever_contrastive_smoke.py
    02_retriever_infer_smoke.py
```

`00_start_dense_retriever_server.sh` 是本地 frozen recall service 的唯一入口；
它直接调用 `src/retrievers/gpu_dense_retriever_server.py`，默认
`CUDA_VISIBLE_DEVICES=5`、进程内 `DEVICE=cuda`，并将 e5 flat doc embedding
矩阵加载到 GPU torch tensor 中。该入口不再调用 legacy Search-R1 CPU retrieval
server。

说明文档：

```text
docs/framework_CoAgenticRetriever.md
docs/planning/260610_retriever_contrastive_step_plan_fix_ver2.md
scripts/coagenticRetriever_local/README.md
```

## 策略模块职责

`retriever_strategies/schemas.py`

定义策略模块之间共享的数据结构。重点字段：

```text
recall_top50_docs:
  recall retriever 返回的原始 top50

ranked_passages:
  rank retriever 对 recall top50 重排后的完整 50 个 passage

rerank_top50_docs:
  ranked_passages 的 dict 形式，用于日志和样本构造

rerank_top5_docs:
  rank retriever top5，用于 agent 下一步上下文
```

`retriever_strategies/trajectory_selector/top_f1.py`

从 rollout trace 中选择有效轨迹，并把 tool call 归一化成 `ToolCallContext`。当前会分别解析 recall 和 rank 两阶段字段：

```text
recall_top50_docs -> recall 侧候选
rerank_top50_docs / ranked_passages -> rank 侧完整排序结果
rerank_top5_docs -> agent 侧 top5 视图
```

注意：代码中仍可能出现 `retrieved_passages` 字符串，但只作为旧 trace 兼容 fallback，不是当前训练主字段。

`retriever_strategies/signal_builder/topk_pseudo_rank.py`

基于 `context.ranked_passages` 构造伪标签：

```text
rank 1..positive_top_k -> positive
rank positive_top_k+1..50 -> negative
```

当前默认 `positive_top_k=5`，也就是 rank retriever top5 为正例，top5 之外为负例。

`retriever_strategies/sample_builder/random_negative_repeat.py`

从 labeled ranked passages 中构造对比学习 group。每个 group 包含：

```text
query_input
positive
negatives
positive_doc_index
recall_top50_docs
rerank_top50_docs
rerank_top5_docs
label_source
```

`retriever_strategies/collator.py`

把 `ContrastiveSample` 转成 rank retriever worker 可消费的 batch：

```text
query_input_ids
query_attention_mask
doc_input_ids
doc_attention_mask
positive_doc_index
loss_weight
non_tensor trace metadata
```

`retriever_strategies/replay_buffer.py`

维护 fresh/replay 样本池。避免同一个 step 内重复加入相同 sample，并支持 fresh/replay 混合采样。

`retriever_strategies/logging_utils.py`

打印对比学习任务构造过程。默认每 10 step 打印 1 组完整样例，smoke 首步也会打印。日志重点包括：

```text
origin_query
sub_query
recall_top50_sample
rerank_top50_sample
rerank_top5_sample
positive
negatives
label_source
```

## Rank Retriever Worker

核心实现文件：

```text
CoAgenticRetriever/verl/verl/workers/retriever/e5_retriever_worker.py
```

当前 worker 名称：

```text
LocalE5RankRetrieverWorker
LocalRetrieverContrastiveWorker = LocalE5RankRetrieverWorker  # 兼容旧引用
```

关键语义：

```text
self.encoder = AutoModel.from_pretrained(...)
encode_queries(...) -> self.encoder
encode_docs(...)    -> self.encoder
```

也就是说 query 和 doc 共享同一个 encoder 实例，forward/backward 都更新同一套参数。

训练 loss：

```text
query_emb = shared_encoder(query)
doc_emb   = shared_encoder(docs)
scores    = dot(query_emb, doc_emb)
logits    = scores / temperature
loss      = cross_entropy(logits, positive_doc_index)
```

checkpoint 保存位置：

```text
.../retriever/rank_encoder/
```

Worker 已移除 CPU fallback。CUDA 不可见或 device 非 CUDA 时会 fail fast，避免误用 CPU 跑出无意义验证。

## 字段语义修正：ranked_passages

最近一次关键修正是把内部变量 `context.retrieved_passages` 改为 `context.ranked_passages`。

修正原因：

```text
retrieved_passages 语义容易被理解成 recall retriever 的召回结果。
但 signal_builder 需要的是 rank retriever 已排序的结果。
```

当前正确语义：

```text
context.ranked_passages:
  rank retriever 对 recall top50 的完整排序结果，长度应为 50

passage.rank:
  rank retriever 排名，也就是 rank_rank

passage.metadata["recall_rank"]:
  recall retriever 原始排名

passage.retriever_score:
  rank retriever 分数，也就是 rank_score
```

必须避免的错误：

```text
用 recall_retriever 的 rank 标 top5 正例
只记录 rerank_top5_docs 而丢掉 rerank_top50_docs
把 recall_rank 写入 passage.rank
```

## 本地训练脚本

入口：

```text
scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

当前默认模式是 full co-training：

```text
RUN_MODE=co-training
```

默认语义：

```text
1. 启动完整 CoSearch/VERL 训练路径。
2. agent LLM 走原 PPO/GRPO update。
3. dense rank retriever 走 contrastive update。
4. rollout trace 中的 recall top50 会被当前 rank retriever 补写为 rerank_top50_docs / ranked_passages。
5. 对比学习样本从 rank retriever 的完整 top50 排序中构造。
```

默认资源意图：

```text
GPU00-03: agent LLM VERL workers
GPU04: trainable dense rank retriever
GPU05: frozen recall retrieval service
```

注意：物理 GPU 编排允许写在 YAML 配置默认值和
`scripts/coagenticRetriever_local` 调度脚本；01/02 脚本也可以通过环境变量覆盖
这些配置。`CoAgenticRetriever` 核心 Python 代码只读取
`rank_retriever.device` / `recall_retriever.device` 等最终配置值，不写死 GPU
编号。

该模式只接受 `RUN_MODE=co-training`。

## Dense Reranker Only 验证模式

核心 smoke 实现：

```text
scripts/coagenticRetriever_local/assets/01_retriever_contrastive_smoke.py
```

该模式需要显式指定：

```text
RUN_MODE=dense-reranker-only
```

该模式只接受 `RUN_MODE=dense-reranker-only`。

该模式做的事情：

1. 读取现有训练 parquet 和 wiki corpus。
2. 默认通过 `00_start_dense_retriever_server.sh` 启动 GPU05 frozen recall service 并获取 top50。
3. 用 trainable rank retriever 对 top50 重新排序，得到 rerank top50。
4. 取 rerank top5 给模拟 agent trace。
5. 用 rerank top50 构造 ranked_passages。
6. 通过 strategy pipeline 构造对比学习样本。
7. 更新 rank retriever shared encoder。

真实 GPU 2 step 验证命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives

RUN_NAME=coagentic_rank_retriever_ranked_passages_gpu_smoke_v2 \
RUN_MODE=dense-reranker-only \
TOTAL_STEPS=2 \
RETRIEVER_CONTRASTIVE_BATCH_SIZE=4 \
RETRIEVER_NEG_PER_POS=3 \
RECALL_TOP_K=50 \
RERANK_TOP_K=5 \
RANK_GPU_ID=4 \
RECALL_GPU_ID=5 \
AUTO_START_RECALL_SERVICE=1 \
RETRIEVER_DEVICE_TRAIN=cuda:0 \
RECALL_RETRIEVER_DEVICE=cuda:1 \
bash scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

说明：

```text
CUDA_VISIBLE_DEVICES=4,5 时：
  cuda:0 -> 物理 GPU04，用于 rank retriever
  cuda:1 -> 物理 GPU05，用于 recall retriever
```

验证结果：

```text
训练通过 2 steps
rank-retriever-worker device=cuda:0
recall-retriever device=cuda:1
label_source=rank_retriever_top5_pseudo_rank
日志中存在 rerank_top50_sample 和 rerank_top5_sample
正例来自 rank retriever top5
负例来自 rank retriever top5 之外
checkpoint 保存到 retriever/rank_encoder
```

成功 checkpoint：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coagentic_rank_retriever_ranked_passages_gpu_smoke_v2/retriever/rank_encoder
```

## 本地推理脚本

入口：

```text
scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

核心 smoke 实现：

```text
scripts/coagenticRetriever_local/assets/02_retriever_infer_smoke.py
```

本地推理做的事情：

1. 读取 eval parquet。
2. 加载训练后的 `rank_encoder`。
3. frozen recall retriever 返回 top50。
4. rank retriever 返回完整 rerank top50。
5. `rerank_top5 = rerank_top50[:5]` 作为 agent 侧结果。
6. 写出 jsonl，包含 recall top50 sample、rerank top50 sample、rerank top5。

真实 GPU 1 step 验证命令：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives

RUN_NAME=coagentic_rank_retriever_ranked_passages_infer_gpu_smoke_v2 \
MAX_EVAL_STEPS=1 \
CHECKPOINT_DIR=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coagentic_rank_retriever_ranked_passages_gpu_smoke_v2/retriever \
RECALL_TOP_K=50 \
RECALL_CANDIDATE_DOCS=4096 \
TOP_K=5 \
RANK_GPU_ID=4 \
RECALL_GPU_ID=5 \
RETRIEVER_DEVICE_TRAIN=cuda:0 \
RECALL_RETRIEVER_DEVICE=cuda:1 \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

验证结果：

```text
推理通过 1 step
rank worker 从 retriever/rank_encoder 加载 checkpoint
frozen recall worker 运行在 cuda:1
输出包含 recall_top50_sample
输出包含 rerank_top50_sample
输出包含 rerank_top5
rank_rank 与 recall_rank 独立记录
```

推理输出：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/<TASK_NAME>/retriever_infer_smoke.jsonl
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/<TASK_NAME>/runtime_logs/coagentic_retriever_infer_smoke.infer.log
```

## 策略配置点

主要配置文件：

```text
CoAgenticRetriever/config/retriever_contrastive.yaml
```

核心开关：

```yaml
trainer:
  retriever_trainable: true
  retriever_update_mode: contrastive
  retriever_steps_per_global_step: 2
```

Recall retriever：

```yaml
recall_retriever:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  device: cuda:5
  top_k: 50
  trainable: false
  index_refresh: false
```

`device` 可以作为配置默认值保留；核心 Python 代码不写死物理 GPU 编号。

Rank retriever：

```yaml
rank_retriever:
  model_path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
  device: cuda:4
  shared_encoder: true
  rerank_top_k: 5
```

Trajectory selector：

```yaml
rank_retriever_training:
  trajectory_selector:
    type: top_f1_trajectories
    max_selected_trajectories: 1
    min_final_reward: 0.0
```

Signal builder：

```yaml
rank_retriever_training:
  signal_builder:
    type: topk_pseudo_rank
    positive_top_k: 5
```

Sample builder：

```yaml
rank_retriever_training:
  sample_builder:
    type: random_negative_repeat
    num_groups_per_step: 32
    neg_per_pos: 15
```

Loss：

```yaml
rank_retriever_training:
  loss:
    type: info_nce
    temperature: 0.05
```

Replay buffer：

```yaml
rank_retriever_training:
  replay_buffer:
    enable: true
    max_size: 200000
    fresh_ratio: 0.5
```

Construction logger：

```yaml
rank_retriever_training:
  construction_log:
    enable: true
    every_n_steps: 10
    max_negatives_to_print: 15
```

## 已执行的静态验证

已完成：

```text
python -m compileall
bash -n 训练脚本
bash -n 推理脚本
策略 import 检查
ranked_passages 字段检查
rank_rank / recall_rank 解析检查
```

重点检查项：

```text
context.retrieved_passages 不再作为信号构造主路径
context.ranked_passages 存在于 ToolCallContext
ranked passage 的 passage.rank 来自 rank_rank
recall_rank 保存在 metadata 中
```

## 后续接手注意事项

1. 不要把 `rerank_top5_docs` 当成训练样本池。训练和日志需要完整 `rerank_top50_docs`。
2. 不要从 `recall_top50_docs` 的 rank 直接构造正负例。正负例必须来自 rank retriever 排序结果。
3. 如果未来接入真实 agent tool call，需要保证 tool trace 同时记录：
   - `recall_top50_docs`
   - `rerank_top50_docs`
   - `rerank_top5_docs`
4. 如果未来扩展 loss，建议在共享 encoder 语义稳定后再加入 RankNet 或 distillation。
5. 当前 recall index 不随 rank retriever 训练刷新，这是 Fix Ver2 的明确边界。
6. 当前 worker 是 CUDA-only，真实训练/推理验证需要在可访问 GPU 的环境下执行。
7. 本地 smoke 的 `cuda:0/cuda:1` 是经过 `CUDA_VISIBLE_DEVICES=4,5` 后的逻辑编号，不等同于物理卡号。

## 快速阅读顺序

后续人员建议按以下顺序理解代码：

```text
1. docs/planning/260610_retriever_contrastive_step_plan_fix_ver2.md
2. docs/framework_CoAgenticRetriever.md
3. CoAgenticRetriever/retriever_strategies/schemas.py
4. CoAgenticRetriever/retriever_strategies/trajectory_selector/top_f1.py
5. CoAgenticRetriever/retriever_strategies/signal_builder/topk_pseudo_rank.py
6. CoAgenticRetriever/retriever_strategies/sample_builder/random_negative_repeat.py
7. CoAgenticRetriever/verl/verl/workers/retriever/e5_retriever_worker.py
8. scripts/coagenticRetriever_local/assets/01_retriever_contrastive_smoke.py
9. scripts/coagenticRetriever_local/assets/02_retriever_infer_smoke.py
```
