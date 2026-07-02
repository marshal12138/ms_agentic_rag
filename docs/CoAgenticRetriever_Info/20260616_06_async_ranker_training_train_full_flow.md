# 06 异步排序标签训练策略全流程说明

本文梳理 CoAgenticRetriever 新增 async ranker training 训练策略的完整运行链路。重点是把训练入口、服务启动、Hydra 配置注入、rollout 后的 LLM judge 请求、ranker contrastive 更新、checkpoint 和日志产物串起来，便于后续复现实验和排查问题。

当前说明对应 2026-06-16 的代码状态。重点路径：

- 任务入口：`tasks/train_tasks/train_CAR_async_ranker_training_ds_flash.sh`
- 主训练脚本：`scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`
- 共享 VERL launcher：`scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh`
- Python 入口：`CoAgenticRetriever/main_coagentic_retriever.py`
- Trainer：`CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py`
- async ranker training 框架：`CoAgenticRetriever/async_ranker_training`
- async ranker training 策略 YAML：`scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml`
- LLM judge 服务 YAML：`CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml`
- LLM judge 启动脚本：`CoAgenticRetriever/scripts/launch_llm_as_judge.sh`
- 默认 prompt：`CoAgenticRetriever/async_ranker_training/prompts/llm_judge_rank50_v1.md`

## 1. 总体链路

async ranker training 不是替换 agent GRPO 训练，而是替换 ranker contrastive 样本信号构造方式。

旧的同步 ranker contrastive 链路：

```text
rollout tool_call_details
  -> trajectory_selector
  -> pseudo-rank signal_builder
  -> sample_builder
  -> replay_buffer
  -> collator
  -> ranker_wg.update_ranker_contrastive
```

新的异步链路：

```text
rollout tool_call_details
  -> enrich ranker top50 trace
  -> build AsyncLabelRequest
  -> AsyncLabeler.submit(...)
  -> agent GRPO/PPO update continues

background async ranker training labeler:
  request queue
  -> LLMJudgeRank50Stage
  -> CandidateSignalData
  -> completed buffer

ranker trainer:
  completed buffer pop_latest(n=sample_builder_request_batch)
  -> signal_builder
  -> sample_builder
  -> replay_buffer
  -> collator
  -> ranker_wg.update_ranker_contrastive
```

核心区别：

1. GRPO 主训练不等待 LLM judge。
2. GRPO 主训练不等待 ranker sample_builder。
3. ranker contrastive step 可以落后于 agent global step。
4. LLM judge 只输出排序分数，不输出正负例标签。
5. `signal_builder` 负责把 judge 排序转成正负标签。
6. `sample_builder` 负责把已带标签的 context 复采样成固定数量的对比学习样本组。

## 2. 任务入口脚本

文件：

```text
tasks/train_tasks/train_CAR_async_ranker_training_ds_flash.sh
```

这个脚本是 async ranker training 训练的推荐入口。它复用 01 主训练脚本，但预设 async ranker training 相关环境变量。

关键默认值：

```bash
EXP_NAME=CAR_async_ranker_training_ds_flash_v1
ENABLE_ASYNC_RANKER_TRAINING=1
ASYNC_RANKER_TRAINING_YAML=scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
AUTO_START_LLM_JUDGE=1
AUTO_STOP_LLM_JUDGE=1
LLM_JUDGE_SERVICE_CONFIG=CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
LLM_JUDGE_ENDPOINT=http://127.0.0.1:8067/v1/chat/completions
LLM_JUDGE_PREFLIGHT=1
TOP_N=50
TOP_M=5
```

资源默认分工：

```bash
AGENT_GPU_IDS=0,1,2,3
RANK_GPU_ID=4
RECALL_GPU_ID=5
LLM judge GPU=6,7
```

注意：

- `TOP_N=50` 表示工具 trace 保存 dense ranker 排序后的 top50，用于 LLM judge。
- `TOP_M=5` 表示 agent LLM 实际看到 rank 后 top5，用于决策。
- 不能把 `TOP_N=50` 理解成直接使用 recall top50。
- LLM judge 对 50 个 chunk 打分，与 agent 只看 top5 是两条不同用途的链路。

## 3. 主训练脚本阶段

文件：

```text
scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

01 脚本新增了 async ranker training 相关阶段：

```text
write_env
check_paths
validate_async_ranker_training_config
if DRY_RUN: exit
ensure_llm_judge_service
ensure_recall_service
start nvidia-smi sampler
start training reporter
export env
run assets/00_run_agentic_iter_rag_verl.sh
```

### 3.1 validate_async_ranker_training_config

当 `ENABLE_ASYNC_RANKER_TRAINING=1` 时，脚本会检查：

1. `ASYNC_RANKER_TRAINING_YAML` 必须存在。
2. `LLM_JUDGE_SERVICE_CONFIG` 必须存在。
3. async ranker training YAML 可以被转换为 Hydra dotlist。
4. LLM judge stage 必须配置 `prompt.path`。
5. `prompt.path` 指向的文件必须存在。
6. prompt 文件必须包含 `## system:` 和 `## user:`。
7. `launch_llm_as_judge.sh --dry-run` 能解析服务侧 YAML。
8. 创建 `<run-log-dir>/async_ranker_training` 和 `<run-log-dir>/async_ranker_training/judge_server`。

这一步的目标是让配置问题在训练前失败，而不是等到 rollout 后才失败。

### 3.2 ensure_llm_judge_service

当 `ENABLE_ASYNC_RANKER_TRAINING=1` 时，01 会先检查 LLM judge endpoint：

```text
http://127.0.0.1:8067/v1/models
```

如果可用，直接复用已有服务。

如果不可用：

- `AUTO_START_LLM_JUDGE=0` 且 `LLM_JUDGE_PREFLIGHT=1`：直接报错退出。
- `AUTO_START_LLM_JUDGE=1`：调用 `CoAgenticRetriever/scripts/launch_llm_as_judge.sh` 自动启动服务。

自动启动时会把 judge 服务日志写到当前 run 目录：

```text
log/train_logs/coAgenticRetriever/<run-name>/async_ranker_training/judge_server/
  vllm_gpu06_07_8067.log
  vllm_gpu06_07_8067.pid
```

`AUTO_STOP_LLM_JUDGE=1` 时，只会停止本脚本记录到 `LLM_JUDGE_PID` 的进程。多人共用同一个 judge 服务时，建议显式设为：

```bash
AUTO_START_LLM_JUDGE=0
AUTO_STOP_LLM_JUDGE=0
LLM_JUDGE_PREFLIGHT=1
```

## 4. LLM judge 服务

启动脚本：

```text
CoAgenticRetriever/scripts/launch_llm_as_judge.sh
```

服务侧配置：

```text
CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

核心参数：

```yaml
model:
  model_path: /data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash
  served_model_name: DeepSeek-V4-Flash

server:
  host: 0.0.0.0
  port: 8067
  endpoint: http://127.0.0.1:8067/v1/chat/completions

runtime:
  python: /data04/envs/ms/deepseek_v4/bin/python
  vllm: /data04/envs/ms/deepseek_v4/bin/vllm
  cuda_visible_devices: "6,7"
  tensor_parallel_size: 2
  gpu_memory_utilization: 0.95
  max_model_len: 11000
  kv_cache_dtype: fp8
  disable_custom_all_reduce: true
```

配置分工：

- 服务侧 YAML 只描述 vLLM 如何启动。
- 训练侧 YAML 只描述训练如何调用 endpoint、如何限流、如何构造样本。
- 不要把 `model_path`、GPU、`max_model_len` 这类服务参数塞进训练侧 YAML。

## 5. Hydra 配置注入

CoAgenticRetriever 默认 config 已加载：

```yaml
defaults:
  - ranker_contrastive
  - async_ranker_training
  - _self_
```

默认 async ranker training 配置文件：

```text
CoAgenticRetriever/config/async_ranker_training.yaml
```

默认值是关闭：

```yaml
ranker_training:
  signal_source: pseudo_rank
  async_ranker_training:
    enable: false
```

任务脚本通过：

```bash
ASYNC_RANKER_TRAINING_YAML=scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

把策略 YAML 注入到 Hydra。共享 launcher 中会收集：

```bash
HYDRA_OVERRIDE_YAMLS
RANKER_STRATEGY_YAML
ASYNC_RANKER_TRAINING_YAML
```

然后转换成 Hydra dotlist，插入 `main_coagentic_retriever.py` 的启动参数。

优先级从低到高：

```text
Hydra 默认配置
< launcher 固定参数
< COAGENTIC_DEFAULT_EXTRA_ARGS
< HYDRA_OVERRIDE_YAMLS / RANKER_STRATEGY_YAML / ASYNC_RANKER_TRAINING_YAML
< 用户 COAGENTIC_EXTRA_ARGS
< 用户直接传给脚本的 "$@"
```

因此如果用户在 `COAGENTIC_EXTRA_ARGS` 中覆盖了 async ranker training 字段，会覆盖策略 YAML。

## 6. 训练侧 async ranker training 策略 YAML

文件：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

关键配置：

```yaml
ranker_training:
  signal_source: async_ranker_training

  async_ranker_training:
    enable: true
    max_sub_query: 10
    max_glb_step_lag: 3
    request_queue_size: 2048
    completed_buffer_size: 4096
    sample_builder_request_batch: 1
    drop_policy: drop_oldest
    num_workers: 4

    stages:
      - type: llm_as_judge
        endpoint: http://127.0.0.1:8067/v1/chat/completions
        model: DeepSeek-V4-Flash
        max_docs_per_request: 50
        temperature: 0.0
        max_tokens: 1024
        request_timeout_seconds: 600
        prompt:
          path: CoAgenticRetriever/async_ranker_training/prompts/llm_judge_rank50_v1.md
          version: llm_judge_rank50_v1
          max_chunk_chars: 512

    sample_builder:
      type: random_negative_repeat
      num_groups_per_step: 32
      neg_per_pos: 15
      allow_repeat_negative_sampling: true
      strategy_kwargs:
        signal_builder_type: llm_judge_topk
        positive_top_k: 5
        label_source: llm_judge_top5
```

字段含义：

- `signal_source: async_ranker_training`：让 trainer 进入 async ranker 信号链路。
- `enable: true`：开启 async ranker training labeler，并让 ranker 从 completed buffer 构造训练样本。
- `max_sub_query: 10`：每个 global step 最多提交 10 个 search tool call 给 judge。
- `max_glb_step_lag: 3`：request 超过 3 个 global step 仍未处理则过期丢弃。
- `sample_builder_request_batch: 1`：进入 sample_builder 前需要攒够的 completed judge signal 数。
- `max_docs_per_request: 50`：每个 judge request 评估 50 个 chunk。
- `max_tokens: 1024`：judge response 的最大输出 token。
- `max_chunk_chars: 512`：每个 chunk 注入 prompt 的最大字符数。
- `signal_builder_type: llm_judge_topk`：把 judge 排名前 K 的 docs 标为 positive。
- `positive_top_k: 5`：默认 judge rank 1..5 是 positive，rank 6..50 是 negative。
- `num_groups_per_step: 32`：一次 sample_builder 调用固定构造 32 组 contrastive samples。
- `neg_per_pos: 15`：每个 positive 配 15 个 negative。

注意：`sample_builder_request_batch` 约束的是一次 sample_builder 调用前需要攒够多少条 completed judge signals；`num_groups_per_step` 约束的是这一次 sample_builder 调用最终产出多少组 contrastive samples。默认 `sample_builder_request_batch=1` 表示一条 completed signal 就触发一次 sample_builder；如果改成 3，则必须攒够 3 条 completed signals 后一起进入 sample_builder，但最终仍然只产出固定 `num_groups_per_step` 组样本。

## 7. Rollout 后的数据流

一个 global step 中，trainer 的主流程在 `fit()` 中执行：

```text
main_batch = async_rollout_manager.generate_sequences(...)
if ranker_train_enabled:
  _enrich_tool_calls_with_ranker(main_batch)
  if async_ranker_training_enabled:
    _submit_async_ranker_training_requests(main_batch)
agent GRPO update continues
```

### 7.1 enrich ranker top50 trace

函数：

```text
_enrich_tool_calls_with_ranker(main_batch)
```

它读取每个 search tool call 的 recall docs，然后用当前 trainable dense ranker 重新排序，写回：

```text
rank_top50_docs
rank_top5_docs
ranked_passages
```

语义：

- `rank_top50_docs`：ranker 排序后的完整 top50，用于 async judge 和 ranker 训练信号。
- `rank_top5_docs`：ranker 排序后的 top5，用于 agent 决策可见内容。
- `ranked_passages`：兼容旧字段名，语义同 rank 后 top50。

主 metrics 中可以检查：

```text
ranker/trace_enriched_tool_calls
ranker/trace_ranked_docs
```

如果一个 step 有 512 个 tool call，`ranker/trace_ranked_docs` 应接近：

```text
512 * 50 = 25600
```

### 7.2 build AsyncLabelRequest

函数：

```text
_submit_async_ranker_training_requests(main_batch)
  -> build_requests_from_dataproto(...)
```

`build_requests_from_dataproto` 从：

```text
main_batch.non_tensor_batch["tool_call_details"]
```

构造 `AsyncLabelRequest`。

每个 request 包含：

```text
request_id
created_global_step
origin_query
sub_query
trajectory_id
tool_call_id
ranked_chunk_list: 50 chunks
prompt_version
```

强校验：

```text
ranked_chunk_list 长度必须等于 50
doc_id 不能为空
doc_id 不能重复
text 不能为空
```

如果失败，会增加：

```text
async_ranker_training/invalid_requests
```

正常训练中这个值应该为 0 或接近 0。若大量 invalid，优先检查 `TOP_N`、tool trace 保存字段和 `_enrich_tool_calls_with_ranker`。

## 8. AsyncLabeler 处理流程

核心类：

```text
CoAgenticRetriever/async_ranker_training/labeler.py
```

训练初始化时：

```text
AsyncLabeler(config, log_dir=<run-log-dir>/async_ranker_training)
AsyncLabeler.start()
```

提交请求时：

```text
AsyncLabeler.submit(requests)
```

这是非阻塞入队。若 request queue 满，会丢弃新请求并增加 `dropped_count`。

后台 worker：

```text
request queue
  -> 检查 current_global_step - created_global_step 是否超过 max_glb_step_lag
  -> LLMJudgeRank50Stage.score(request)
  -> judge 失败则写 failures.jsonl 并丢弃
  -> 再次检查是否过期
  -> 写 CandidateSignalData 到 completed buffer
```

LLM judge 失败不会抛到主训练，也不会阻塞 GRPO。失败只通过：

```text
async_ranker_training/labeler_failed_count
async_ranker_training/failures.jsonl
```

观察。

## 9. LLM judge prompt 和输出

prompt 文件：

```text
CoAgenticRetriever/async_ranker_training/prompts/llm_judge_rank50_v1.md
```

读取方式：

```text
stage config prompt.path
  -> MarkdownSystemUserPrompt
  -> render_messages(request)
  -> OpenAI-compatible chat completions messages
```

prompt 文件必须包含：

```text
## system:
## user:
```

代码会替换这些占位符：

```text
{{原始查询问题}}
{{规范化后的查询问题}}
{{允许的所有段落ID列表}}
{{段落ID}}
{{段落文本片段}}
```

LLM judge 期望输出：

```json
{"ranked_ids": ["doc_id_1", "...", "doc_id_50"]}
```

`LLMJudgeRank50Stage` 会校验：

```text
ranked_ids 正好 50 个
ranked_ids 不重复
ranked_ids 不包含未知 id
ranked_ids 不缺少 request 中的 id
```

通过后转为：

```text
JudgeChunkScore(doc_id, judge_rank, judge_score)
judge_score = (50 - judge_rank + 1) / 50
```

注意：judge 不输出 positive/negative。positive/negative 必须由 sample_builder 决定。

## 10. Completed Buffer

核心类：

```text
CoAgenticRetriever/async_ranker_training/buffer.py
```

写入：

```text
CompletedSignalBuffer.push(CandidateSignalData)
```

消费：

```text
CompletedSignalBuffer.pop_latest(n=sample_builder_request_batch, wait=True, timeout=1.0)
```

当前语义：

- buffer 满时按 `drop_policy` 丢弃。
- `pop_latest` 会等待 completed buffer 中至少有 `n` 条 judge signals，攒够后一次取最新 `n` 条。
- 被取出的 `CandidateSignalData` 会从 buffer 中移除。
- buffer 中不足 `n` 条时，ranker async trainer 后台线程等待更多 completed signals。
- 这个等待只发生在 ranker 后台线程，不阻塞 agent GRPO。

不要把消费逻辑改成读 `buffer[-N:]`，否则会反复训练同一批最新 signal。

## 11. ranker contrastive 更新

核心类：

```text
CoAgenticRetriever/async_ranker_training/ranker_async_trainer.py
```

启动时机：

```text
_init_ranker_components()
  -> AsyncLabeler.start()
  -> RankerAsyncTrainer.start()
```

训练循环：

```text
signals = completed.pop_latest(n=sample_builder_request_batch, wait=True, timeout=1.0)
labeled_contexts = signal_builder.build(signals)
samples = sample_builder.build(labeled_contexts)
replay_buffer.add(samples)
train_samples = replay_buffer.sample(batch_size, fresh_ratio)
batch = collator(train_samples)
ranker_wg.update_ranker_contrastive(batch)
```

这段循环中的一次 `_train_once(...)` 对应一次 ranker contrastive update。这里的 step 指一次 ranker update，不是一次 rollout global step，也不是一次 judge request。

当前默认实现不是后台 CUDA 线程，而是在主训练循环中 opportunistic 执行一次 `try_train_once(wait=False)`。LLM judge 仍然异步运行；ranker update 和 ranker top50 enrichment/checkpoint 保存共用同一个 ranker worker，并在主线程内串行执行。

保留了显式开关：

```yaml
ranker_training:
  async_ranker_training:
    background_ranker_thread: false
```

只有手动改成 `true` 时才会启动后台 ranker 线程。默认关闭的原因是：后台 Python 线程执行 CUDA ranker update 时，如果在 `_train_once` 内卡住，主训练仍会继续推进，completed buffer 会持续增长，但 `ranker/local_update_step` 不再增长，问题容易被 metrics 掩盖。

为了避免 update、rank top50 enrichment 和 checkpoint 保存并发访问 ranker，代码使用 `ranker_lock`：

```text
ranker_wg.update_ranker_contrastive
ranker_wg.save_checkpoint
```

都应在 lock 保护下执行。

## 12. 一个 contrastive step 的前向和反向

本节把一次 ranker contrastive update 拆开说明。当前 async 默认链路是：

```text
sample_builder_request_batch 条 completed judge signals
  -> signal_builder 标注 top5 positive / 其余 negative
  -> sample_builder 构造 32 组 fresh samples
  -> replay_buffer 采样训练 batch
  -> collator tokenize
  -> ranker forward
  -> contrastive loss
  -> backward / optimizer step
```

### 12.1 signal_builder：judge 排序转正负标签

当前默认策略：

```text
CoAgenticRetriever/async_ranker_training.strategies/signal_builder/llm_judge_topk.py
```

默认配置：

```yaml
strategy_kwargs:
  signal_builder_type: llm_judge_topk
  positive_top_k: 5
  label_source: llm_judge_top5
```

假设 LLM judge 对 50 个文档给出排序：

```text
rank 1: d01
rank 2: d02
rank 3: d03
rank 4: d04
rank 5: d05
rank 6: d06
...
rank 50: d50
```

`llm_judge_topk` 会输出一个 `LabeledRankingContext`：

```text
positive docs: d01, d02, d03, d04, d05
negative docs: d06 ... d50
```

也就是：

```text
label = 1: judge_rank <= positive_top_k
label = 0: judge_rank > positive_top_k
```

这里的职责边界很重要：LLM judge 只给排序；`signal_builder` 把排序转成正负标签；`sample_builder` 不再决定 top1/top5 这种标签语义。

### 12.2 sample_builder：带标签 context 转对比学习样本组

当前默认策略：

```text
CoAgenticRetriever/async_ranker_training.strategies/sample_builder/random_negative_repeat.py
```

默认参数：

```text
num_groups_per_step = 32
neg_per_pos = 15
allow_repeat_negative_sampling = true
```

它的输入是一个或多个 `LabeledRankingContext`。每个 context 已经有 positive/negative 标签。它先构造候选项：

```text
对每个 context:
  positives = label == 1 的 passages
  negatives = label == 0 的 passages
  对每个 positive:
    candidate = (context, positive, negatives)
```

然后循环 candidate，直到产出 `num_groups_per_step` 组样本。每组样本固定是：

```text
1 positive + neg_per_pos negatives
```

样本内文档顺序固定为：

```text
documents = [positive, negative_1, negative_2, ..., negative_15]
positive_doc_index = 0
```

`query_input` 的构造方式：

```text
origin_query + " [SEP] " + sub_query
```

### 12.3 一条 judge signal 的样本分布例子

假设一条 completed judge signal 经过 `llm_judge_topk` 后得到：

```text
positive: P1, P2, P3, P4, P5
negative: N1 ... N45
```

sample_builder 会形成 5 个 candidate：

```text
candidate 1: P1 + 45 negatives
candidate 2: P2 + 45 negatives
candidate 3: P3 + 45 negatives
candidate 4: P4 + 45 negatives
candidate 5: P5 + 45 negatives
```

如果 `num_groups_per_step = 32`，它会 round-robin 复用这 5 个 candidate，最终 positive 使用次数是：

```text
P1: 7 组
P2: 7 组
P3: 6 组
P4: 6 组
P5: 6 组
```

每组会从 45 个 negatives 中随机抽 15 个。单组内部在 negative 数量足够时不重复；不同组之间可以复用同一个 negative。

一个具体 sample 可能是：

```text
query_input = origin_query + " [SEP] " + sub_query

documents = [
  P3,
  N7,
  N2,
  N41,
  ...
  N28
]

positive_doc_index = 0
label_source = llm_judge_top5
sample_source = fresh
```

因此，一条 judge signal 已经足够构造 32 组 contrastive samples，不需要等待 32 条 judge signals。

### 12.4 三条 judge signal 的当前策略和可选策略

当前默认处理粒度是：

```text
sample_builder_request_batch = 1
completed.pop_latest(n=sample_builder_request_batch)
```

也就是一条 completed judge signal 立即触发一次 sample_builder 调用。若 3 条 signal 依次完成并被依次处理：

```text
S1 -> 32 组 fresh samples
S2 -> 32 组 fresh samples
S3 -> 32 组 fresh samples

合计 96 组 fresh samples
```

这只是默认配置，不是 sample_builder 的本质限制。

如果把 `sample_builder_request_batch` 改成 3，则会等待 completed buffer 中攒够 3 条 signals，然后一起进入 sample_builder：

```text
sample_builder_request_batch = 3
[S1, S2, S3] -> sample_builder -> 32 组 fresh samples
```

如果每条 signal 都有 5 个 positive，那么一共有 15 个 candidate。`num_groups_per_step = 32` 时，32 组样本会分布在这 15 个 candidate 上，而不是每条 signal 各自产出 32 组。粗略分布是：

```text
前 2 个 candidate: 各 3 组
剩下 13 个 candidate: 各 2 组
```

signal 级别大致是：

```text
S1: 约 10-11 组
S2: 约 10-11 组
S3: 约 10-11 组
```

所以准确语义是：

```text
默认配置：1 条 signal -> 1 次 sample_builder -> 32 组样本
sample_builder_request_batch=3：3 条 signals -> 1 次 sample_builder -> 仍然固定 32 组样本
```

### 12.5 replay_buffer：fresh samples 转训练 batch

sample_builder 产出的 fresh samples 会加入 replay buffer。随后从 replay buffer 采样训练 batch：

```text
train_samples = replay_buffer.sample(batch_size, fresh_ratio)
```

典型配置下：

```text
batch_size = 32
fresh_ratio = 0.5
```

这表示一次 ranker update 的训练 batch 可能混合：

```text
16 条 fresh samples
16 条 replay samples
```

如果刚启动、历史样本很少，replay 部分也可能来自刚加入的样本池。这里的 batch 采样和 judge signal 数量没有一一对应关系。

### 12.6 collator：样本文本转模型输入张量

collator 输入是 `train_samples`。假设：

```text
B = 32
K = 16
```

其中：

```text
B: 本次 ranker update 的样本组数
K: 每组文档数 = 1 positive + 15 negatives
```

collator 生成：

```text
query_texts: 长度 B
doc_texts: 长度 B * K
```

tokenize 后得到：

```text
query_input_ids:      [B, Lq]
query_attention_mask: [B, Lq]

doc_input_ids:        [B, K, Ld]
doc_attention_mask:   [B, K, Ld]

positive_doc_index:   [B]
loss_weights:         [B]
```

当前 sample_builder 总是把 positive 放在第 0 个位置，所以：

```text
positive_doc_index = 0
```

对所有样本都成立。

### 12.7 ranker forward：编码和打分

ranker worker 对 query 和 docs 分别用 E5 encoder 编码。

query：

```text
query_input_ids [B, Lq]
  -> encoder
  -> last_hidden_state [B, Lq, H]
  -> mean_pool
  -> query_emb [B, H]
```

docs：

```text
doc_input_ids [B, K, Ld]
  -> reshape [B*K, Ld]
  -> encoder
  -> last_hidden_state [B*K, Ld, H]
  -> mean_pool
  -> reshape
  -> doc_emb [B, K, H]
```

随后对 query/doc embedding 做 L2 normalize。归一化后点积就是 cosine similarity：

```text
scores = einsum("bh,bkh->bk", query_emb, doc_emb)
scores shape = [B, K]
```

例如某一组：

```text
documents = [P3, N7, N2, N41, ...]
positive_doc_index = 0
```

模型可能输出：

```text
scores[i] = [
  0.81,  # P3
  0.73,  # N7
  0.64,  # N2
  0.52,
  ...
]
```

再除以 temperature 得到 logits：

```text
logits = scores / temperature
```

若 `temperature = 0.05`：

```text
0.81 / 0.05 = 16.2
0.73 / 0.05 = 14.6
0.64 / 0.05 = 12.8
```

### 12.8 loss 和 backward

loss 是组内 cross entropy：

```text
loss_i = cross_entropy(logits[i], target=positive_doc_index[i])
```

当前 target 通常是 0。等价于：

```text
loss_i = -log(
  exp(logit_positive) /
  sum(exp(logit_doc_j) for all docs in the same group)
)
```

也就是要求：

```text
positive doc 的分数高于同组 15 个 negative docs
```

如果有 `loss_weights`：

```text
loss = sum(loss_i * weight_i) / sum(weight_i)
```

否则是普通平均。

反向过程只更新 ranker encoder，不更新 LLM judge，也不更新 agent LLM：

```text
optimizer.zero_grad()
loss.backward()
clip_grad_norm
optimizer.step()
scheduler.step()
local_update_step += 1
```

梯度效果是：

```text
提高 sim(query, positive)
降低 sim(query, negatives)
```

## 13. 默认 sample_builder 策略

默认 builder：

```text
CoAgenticRetriever/async_ranker_training.strategies/sample_builder/random_negative_repeat.py
```

策略：

```text
对每个 LabeledRankingContext:
  label=1 的 passages 作为 positives
  label=0 的 passages 作为 negative pool

循环构造样本:
  query_input = origin_query + " [SEP] " + sub_query
  每个 sample = 1 positive + neg_per_pos negatives
  允许重复负采样
  直到 num_groups_per_step
```

默认参数：

```text
num_groups_per_step=32
neg_per_pos=15
```

如果需要调整 top1/top3/top5 的正样本定义，应优先新增或配置 `signal_builder`。如果需要调整每组负采样、hard/easy negative、按 rank bucket 采样等组样本策略，应新增或配置 `sample_builder`。

通过 YAML 切换 sample_builder：

```yaml
sample_builder:
  type: your_new_builder
```

不要把这些策略写进 LLM judge stage。

## 14. Checkpoint

主 checkpoint 仍跟随 agent global step 保存。

async 模式下，trainer 的 `_save_checkpoint()` 会：

```text
super()._save_checkpoint()
ranker_async_trainer.save_checkpoint(ranker_path)
```

ranker checkpoint 路径：

```text
checkpoints/qwen3_4b_probe/coAgenticRetriever/<run-name>/global_step_<step>/ranker/
```

ranker 保存频率跟随 `trainer.save_freq`，不是按 async ranker update 次数单独保存。

## 15. 日志产物

一个 async ranker training run 的日志目录：

```text
log/train_logs/coAgenticRetriever/<run-name>/
```

主训练日志和报告：

```text
<run-name>.train.log
<run-name>.metrics.jsonl
<run-name>.search_timing.jsonl
<run-name>.nvidia_smi.csv
<run-name>.timing_report.latest.md
<run-name>.training_metrics_report.latest.md
<run-name>.detailed_metrics_report.latest.md
<run-name>.contrastive_construction.jsonl
```

async ranker training 子目录：

```text
async_ranker_training/
  requests.jsonl
  completed_signals.jsonl
  failures.jsonl
  metrics.jsonl
  judge_server/
    vllm_gpu06_07_8067.log
    vllm_gpu06_07_8067.pid
```

文件含义：

- `requests.jsonl`：提交给 async ranker training labeler 的请求摘要。
- `completed_signals.jsonl`：成功完成 judge 并写入 buffer 的 `CandidateSignalData`。
- `failures.jsonl`：judge 格式失败、过期、请求失败等记录。
- `metrics.jsonl`：async ranker training labeler 队列、完成、失败、丢弃统计。
- `judge_server/vllm_gpu06_07_8067.log`：vLLM judge 服务日志。

## 16. 主 metrics 检查项

每个 step 应重点看：

```text
ranker/enabled
ranker/trace_enriched_tool_calls
ranker/trace_ranked_docs
async_ranker_training/candidate_tool_calls
async_ranker_training/invalid_requests
async_ranker_training/built_requests
async_ranker_training/accepted_requests
ranker/async_mode
ranker/async_wait_empty
ranker/async_updates
ranker/async_consumed_signals
ranker/async_built_samples
ranker/local_update_step
async_ranker_training/labeler_submitted_count
async_ranker_training/labeler_completed_count
async_ranker_training/labeler_failed_count
async_ranker_training/labeler_expired_count
async_ranker_training/labeler_request_queue_size
async_ranker_training/labeler_completed_buffer_size
async_ranker_training/labeler_completed_buffer_dropped_count
```

正常信号：

- `ranker/trace_ranked_docs = ranker/trace_enriched_tool_calls * 50`。
- `async_ranker_training/invalid_requests` 为 0 或接近 0。
- `async_ranker_training/accepted_requests` 接近 `max_sub_query`。
- `async_ranker_training/labeler_completed_count` 随 step 增长。
- `ranker/local_update_step` 在 completed signal 能构造出有效样本后增长。

需要排查的信号：

- `invalid_requests` 大量增加：通常是没有保存 rank top50。
- `request_queue_size` 长期接近上限：judge 太慢或 worker 太少。
- `completed_buffer_size` 长期为 0：judge 失败或 endpoint 不通。
- `failed_count` 增长过快：prompt 输出格式不稳定或 max_tokens/max_chunk_chars 不合适。
- `expired_count` 增长过快：judge 延迟超过 `max_glb_step_lag`。
- `ranker/async_updates` 长期为 0：没有 completed judge signal，或 signal_builder/sample_builder 无法构造有效正负样本。

## 17. 已验证 smoke

10 step GPU smoke 目录：

```text
log/train_logs/coAgenticRetriever/260615-2342-async-ds-flash-10step-CAR_async_ranker_training_ds_flash_gpu10
```

关键结果：

```text
completed_train_steps: 10
step10 ranker/trace_ranked_docs: 25600
step10 async_ranker_training/invalid_requests: 0
step10 async_ranker_training/labeler_submitted_count: 100
step10 async_ranker_training/labeler_completed_count: 68
step10 async_ranker_training/labeler_failed_count: 22
step10 async_ranker_training/labeler_expired_count: 0
step10 ranker/local_update_step: 2
```

耗时：

```text
avg train step: 189.657s
p50 train step: 183.682s
p90 train step: 204.685s
max train step: 226.276s
```

这个 smoke 只改了训练步数，batch size、rollout N、并发规模等参数保持正式脚本默认规模。

## 18. 推荐运行方式

dry-run：

```bash
DRY_RUN=1 \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_ranker_training_ds_flash.sh
```

10 step smoke：

```bash
TOTAL_STEPS=10 \
RUN_STAMP=260616-async-smoke \
EXP_NAME=CAR_async_ranker_training_ds_flash_smoke \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_ranker_training_ds_flash.sh
```

正式训练：

```bash
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_ranker_training_ds_flash.sh
```

如果复用别人已经启动的 judge 服务：

```bash
AUTO_START_LLM_JUDGE=0 \
AUTO_STOP_LLM_JUDGE=0 \
LLM_JUDGE_PREFLIGHT=1 \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_ranker_training_ds_flash.sh
```

## 19. 常见问题

### 19.1 为什么 ranker 没有每个 global step 都更新

这通常表示 completed judge signal 还没有进入有效样本构造链路。如果 judge 慢、失败率高、请求过期，或 signal_builder/sample_builder 无法得到有效正负样本，ranker update 会落后于 GRPO。

### 19.2 为什么 step 里 submitted_count 增长但 local_update_step 不增长

说明请求已经提交，但还没有形成可训练的 completed signal/sample。检查：

```text
async_ranker_training/labeler_completed_count
async_ranker_training/labeler_failed_count
async_ranker_training/labeler_completed_buffer_size
```

### 19.3 为什么 failures.jsonl 有 judge 失败

DeepSeek-V4-Flash 有时会返回 49/51 个 id、重复 id、非 JSON 或缺失 id。当前策略是记录失败并丢弃，不阻塞训练。

### 19.4 为什么不能让 judge 直接给正负例

因为正负例定义是 ranker 训练策略，应放在 `signal_builder`。这样后续可以只换 `signal_builder` 来实验 top1/top3/top5，不需要改 judge prompt 和 judge stage。负采样和样本组构造策略则放在 `sample_builder`。

### 19.5 为什么不能使用 recall top50

async ranker training 的目标是给当前 trainable ranker 的排序结果提供外部监督信号。因此输入必须是 ranker 对 recall top50 重排后的 top50。recall top50 是候选池，不是当前 ranker 排序结果。

## 20. 接手检查清单

启动前：

1. `DRY_RUN=1` 是否通过。
2. `ASYNC_RANKER_TRAINING_YAML` 是否指向正确策略 YAML。
3. `LLM_JUDGE_SERVICE_CONFIG` 中模型路径、GPU、端口是否正确。
4. `prompt.path` 是否存在并包含 `## system:` / `## user:`。
5. `TOP_N=50`、`TOP_M=5` 是否保持默认。

训练中：

1. `ranker/trace_ranked_docs` 是否等于 tool call 数乘 50。
2. `async_ranker_training/invalid_requests` 是否接近 0。
3. `async_ranker_training/labeler_completed_count` 是否增长。
4. `async_ranker_training/labeler_failed_count` 是否可接受。
5. `ranker/local_update_step` 是否在 completed signal 能构造有效样本后增长。

训练后：

1. `async_ranker_training/completed_signals.jsonl` 是否存在。
2. `async_ranker_training/failures.jsonl` 的失败原因是否符合预期。
3. `timing_report.latest.md` 中 step 耗时是否异常。
4. checkpoint 中是否保存了 `global_step_<step>/ranker/`。
5. 如果自动启动了 judge 服务，确认 `AUTO_STOP_LLM_JUDGE` 是否按预期处理进程。
