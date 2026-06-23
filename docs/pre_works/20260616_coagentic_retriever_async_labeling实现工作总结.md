# CoAgenticRetriever Async Labeling 实现工作总结

记录日期：2026-06-16  
工作目录：`/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives`  
计划文档：`docs/planning/260615_async_data_labelling.md`  
训练入口：`tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh`

本文档记录 CoAgenticRetriever 异步排序标签框架的实现工作。重点不是复述设计计划，而是说明当前代码已经如何落地、关键入口在哪里、哪些语义不能改错，以及后续接手人员需要避开的实现坑。

## 一句话状态

当前已经完成 async labeling 训练链路的代码实现，并用真实 GPU 跑通过 10 step smoke：

1. rollout 后从 tool call 中抽取 `origin_query + sub_query + ranker 排序后的 top50 chunk`。
2. `AsyncLabeler.submit()` 非阻塞提交 LLM judge 请求。
3. LLM judge 服务默认使用 GPU06/GPU07 上的 DeepSeek-V4-Flash vLLM endpoint。
4. LLM judge 只返回 50 个 doc id 的排序分数，不产出正负例标签。
5. completed signal 进入 async buffer。
6. 后台 `RankerAsyncTrainer` 一旦从 buffer 取到 completed judge signal，就立即调用 `signal_builder/sample_builder` 构造对比学习样本。
7. ranker contrastive update 在后台执行，等待 buffer 只阻塞 ranker 链路，不阻塞 GRPO / agent LLM 更新。
8. 主训练日志下新增 `async_labeling/` 子目录，专门记录 async labeling 的观察日志。

## 核心语义

这几个语义后续不要改错：

1. `rank_top50_docs` / `ranked_chunk_list` 必须是 dense ranker 对 recall top50 重排后的 top50。
2. 绝对不能把 recall top50 直接送给 LLM judge 当训练信号。
3. `TOP_N=50` 表示工具 trace 中保存 rank 后 top50，用于 async judge。
4. `TOP_M=5` 表示 agent LLM 实际看到 rank 后 top5，用于 agent 决策。
5. rank 后 top5 给 agent 决策，与 LLM judge 对 50 个 chunk 打分是两条不同用途的链路。
6. LLM judge 只给排序/分数，不给 positive/negative 标签。
7. positive/negative、重复负采样、样本数补齐/截断全部属于 `sample_builder` 策略。
8. judge 失败时只计数并丢弃，不阻塞主训练。
9. completed buffer 中的 `CandidateSignalData` 被消费后应从 buffer 中移除；buffer 不足时只让 ranker async trainer 等待。
10. ranker 可以落后于 GRPO，这是预期行为；通过 `max_sub_query`、`max_glb_step_lag` 和指标观察控制滞后。

## 新增代码结构

核心目录：

```text
CoAgenticRetriever/async_labeling/
  __init__.py
  schemas.py
  config.py
  request_builder.py
  buffer.py
  labeler.py
  ranker_async_trainer.py
  prompt.py
  logging.py
  metrics.py
  worker.py

  stages/
    base.py
    llm_judge_rank50.py
    extra_scorer_stub.py

  sample_builder/
    base.py
    config.py
    random_negative_repeat_from_signal.py

  prompts/
    llm_judge_rank50_v1.md

  configs/
    llm_judge_vllm_deepseek_flash_gpu06_07.yaml

  utils/
    id_utils.py
    jsonl.py
    time_utils.py
    validation.py
```

配置入口：

```text
CoAgenticRetriever/config/async_labeling.yaml
scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml
CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

脚本入口：

```text
tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
CoAgenticRetriever/scripts/launch_llm_as_judge.sh
```

trainer 接入点：

```text
CoAgenticRetriever/verl/verl/trainer/ppo/coagentic_ranker_contrastive_ray_trainer.py
```

## 关键模块职责

`schemas.py`

定义 async labeling 独立数据结构，不依赖 VERL runtime：

```text
AsyncLabelRequest:
  LLM judge 请求输入，必须包含 ranker 排序后的 50 个 chunk。

CandidateSignalData:
  completed buffer 中的单位元素，保存 judge_scores/final_scores。

ContrastiveSample:
  sample_builder 产出的 ranker contrastive 样本。
```

`AsyncLabelRequest.validate_rank50()` 会强校验：

```text
ranked_chunk_list 长度必须等于 50
每个 chunk 必须有 doc_id
doc_id 不能重复
每个 chunk 必须有 text
```

这个校验很重要。之前 smoke 中发现过工具 trace 只保存 top10 导致全部 invalid 的问题，根因就是训练配置没有保存 rank 后 top50。

`request_builder.py`

从 rollout 的 `main_batch.non_tensor_batch["tool_call_details"]` 构造 `AsyncLabelRequest`：

```text
tool_call_details
  -> flatten 所有 search tool call
  -> select_tool_calls(..., max_sub_query)
  -> build_request_from_tool_call(...)
  -> validate_rank50()
```

当前 `max_sub_query` 限制的是每个 global step 提交的 tool call 数，不是 trajectory 数。即使一个 step 中有数百个 tool call，默认也只提交 10 个给 LLM judge。

`labeler.py`

`AsyncLabeler` 是主训练侧的非阻塞 facade：

```text
submit(requests)       非阻塞入队
start()                启动后台 worker thread
update_global_step()   更新当前 global step，用于过期判断
get_metrics()          写 async metrics 并返回当前状态
close()                关闭 worker
```

worker 处理逻辑：

```text
request queue
  -> max_glb_step_lag 过期检查
  -> LLMJudgeRank50Stage.score()
  -> 失败则写 failures.jsonl 并丢弃
  -> 再次做 max_glb_step_lag 过期检查
  -> CandidateSignalData 写入 completed buffer
```

`buffer.py`

completed buffer 是 ranker 链路消费的队列。当前关键语义是：

```text
有 completed judge signal 就立即 destructive pop 最新 1 条 CandidateSignalData
取出后从 buffer 中移除
buffer 为空时只由 ranker async trainer 后台线程等待
```

不要改成反复读取 `buffer[-N:]`，否则会让 ranker 重复训练同一批最新 signal。

`stages/llm_judge_rank50.py`

LLM judge stage 调用 OpenAI-compatible chat completions endpoint：

```text
endpoint: http://127.0.0.1:8067/v1/chat/completions
model: DeepSeek-V4-Flash
temperature: 0.0
max_tokens: 1024
request_timeout_seconds: 600
```

返回解析只接受 JSON 中的 `ranked_ids`：

```json
{"ranked_ids": ["doc_id_1", "...", "doc_id_50"]}
```

强校验：

```text
ranked_ids 必须正好 50 个
不能重复
不能包含未知 doc_id
不能缺少请求中的 doc_id
```

通过后转换为线性分数：

```text
judge_rank = 1..50
judge_score = (50 - rank + 1) / 50
```

注意：DeepSeek 偶尔会返回 49/51 个 id、重复 id 或非 JSON。当前设计是计入 `failed_count` 并丢弃，不做样本构造。

`sample_builder/random_negative_repeat_from_signal.py`

默认策略对应旧框架的 `ranker_strategies/sample_builder/random_negative_repeat.py` 思路：

```text
按 final_scores/judge_scores 排序
judge rank1 作为 positive
其余作为 negative pool
重复负采样直到达到 num_groups_per_step
每组样本包含 1 positive + neg_per_pos negatives
query_input = origin_query + " [SEP] " + sub_query
```

当前默认参数：

```yaml
num_groups_per_step: 32
neg_per_pos: 15
allow_repeat_negative_sampling: true
```

后续如果要做 top3/top5/hard negative/easy negative 等策略，不要把策略细节塞进 LLM judge。应新增 `sample_builder` 实现，并通过 `sample_builder.type` 注册切换。

`ranker_async_trainer.py`

当前实现使用本地 daemon thread，不是 Ray remote actor：

```text
completed buffer pop_latest(n=1)
  -> signal_builder.build(signals)
  -> sample_builder.build(labeled_contexts)
  -> replay_buffer.add(samples)
  -> replay_buffer.sample(batch_size, fresh_ratio)
  -> collator
  -> ranker_wg.update_ranker_contrastive(batch)
```

ranker worker 和 checkpoint 保存共享同一个 `ranker_lock`。这样可以避免后台 ranker update 和主线程 save checkpoint 同时访问 ranker。

## Trainer 接入方式

`coagentic_ranker_contrastive_ray_trainer.py` 中的关键改动：

1. `_init_ranker_components()` 中初始化原有 ranker 组件后，读取 `ranker_training.async_labeling` 配置。
2. 当 `ranker_training.signal_source == "async_labeling"` 且 `enable=true` 时：
   - 创建 `AsyncLabeler`
   - 创建 `RankerAsyncTrainer`
   - 启动两个后台线程
3. rollout 后先调用 `_enrich_tool_calls_with_ranker(main_batch)`。
4. `_enrich_tool_calls_with_ranker()` 会对每个 tool call 写入：
   - `rank_top50_docs`
   - `rank_top5_docs`
   - `ranked_passages`
5. async 模式下调用 `_submit_async_labeling_requests(main_batch)`，只提交请求，不等待结果。
6. async 模式下跳过旧的同步 `process_ranker_contrastive_step(...)`。
7. `logger.log(...)` 会合并 `ranker_async_trainer.get_metrics()` 返回的 async/ranker 指标。
8. `_save_checkpoint()` 中如果 async trainer 存在，则通过 async trainer 带锁保存 ranker checkpoint。

主训练时序：

```text
rollout generate_sequences
  -> enrich ranker top50 trace
  -> submit async label requests
  -> agent GRPO update
  -> 不等待 judge
  -> 不等待 ranker sample

background:
  async labeler worker 写 completed buffer
  ranker async trainer 消费 buffer 并更新 ranker
```

## 配置关系

任务脚本：

```text
tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
```

重要默认值：

```bash
ENABLE_ASYNC_LABELING=1
ASYNC_LABELING_YAML=scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml
AUTO_START_LLM_JUDGE=1
AUTO_STOP_LLM_JUDGE=1
LLM_JUDGE_SERVICE_CONFIG=CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
LLM_JUDGE_ENDPOINT=http://127.0.0.1:8067/v1/chat/completions
TOP_N=50
TOP_M=5
```

底层训练脚本：

```text
scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

底层脚本默认 async labeling 是关闭的：

```bash
ENABLE_ASYNC_LABELING=0
AUTO_START_LLM_JUDGE=0
AUTO_STOP_LLM_JUDGE=0
```

因此正式使用 async labeling 时，优先使用任务脚本 `train_CAR_async_labeling_ds_flash.sh`，不要直接裸跑底层脚本后忘记传环境变量。

训练侧 YAML：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_labeling_deepseek_flash.yaml
```

核心字段：

```yaml
ranker_training:
  signal_source: async_labeling
  async_labeling:
    enable: true
    max_sub_query: 10
    max_glb_step_lag: 3
    request_queue_size: 2048
    completed_buffer_size: 4096
    num_workers: 4
    stages:
      - type: llm_as_judge
        max_docs_per_request: 50
        max_tokens: 1024
        request_timeout_seconds: 600
        prompt:
          path: CoAgenticRetriever/async_labeling/prompts/llm_judge_rank50_v1.md
          max_chunk_chars: 512
    sample_builder:
      num_groups_per_step: 32
      neg_per_pos: 15
```

LLM judge 服务侧 YAML：

```text
CoAgenticRetriever/async_labeling/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

核心字段：

```yaml
model.model_path: /data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash
server.port: 8067
runtime.cuda_visible_devices: "6,7"
runtime.tensor_parallel_size: 2
runtime.gpu_memory_utilization: 0.95
runtime.max_model_len: 11000
runtime.kv_cache_dtype: fp8
```

模型路径、GPU、端口、`--max-model-len` 等 vLLM 启动参数都写在服务侧 YAML，不写在训练侧 async labeling YAML。

## Prompt 管理

默认 prompt 文件：

```text
CoAgenticRetriever/async_labeling/prompts/llm_judge_rank50_v1.md
```

该文件内容来自早期 planning prompt，但已经复制到源码目录作为默认 prompt。代码中不硬编码 prompt 文本，只从磁盘读取。

`config.validate_prompt_path()` 会校验：

```text
prompt.path 必须存在
prompt 文件必须包含 "## system:" 和 "## user:" section
```

后续新增 prompt 时建议继续放在：

```text
CoAgenticRetriever/async_labeling/prompts/
```

然后只改 YAML：

```yaml
prompt:
  path: CoAgenticRetriever/async_labeling/prompts/your_prompt.md
  version: your_prompt_version
```

不要在 Python 代码中写死 prompt。

## 日志产物

async labeling 的观察日志不直接耦合现有训练 logger，但目录归属于当前 run：

```text
log/train_logs/coAgenticRetriever/<run-name>/async_labeling/
```

例如 10 step smoke：

```text
log/train_logs/coAgenticRetriever/260615-2342-async-ds-flash-10step-CAR_async_labeling_ds_flash_gpu10/async_labeling/
  requests.jsonl
  completed_signals.jsonl
  failures.jsonl
  metrics.jsonl
  judge_server/
    vllm_gpu06_07_8067.log
    vllm_gpu06_07_8067.pid
```

文件含义：

```text
requests.jsonl:
  被提交给 async labeler 的请求摘要。

completed_signals.jsonl:
  judge 成功并写入 completed buffer 的 CandidateSignalData。

failures.jsonl:
  judge 失败、格式不合规、过期丢弃等失败记录。

metrics.jsonl:
  AsyncLabeler.get_metrics() 输出的队列、完成、失败、丢弃统计。

judge_server/vllm_gpu06_07_8067.log:
  LLM judge vLLM 服务日志。
```

主训练指标仍写入：

```text
<run-name>.metrics.jsonl
<run-name>.timing_report.latest.md
<run-name>.training_metrics_report.latest.md
<run-name>.detailed_metrics_report.latest.md
```

主 metrics 中新增关注项：

```text
async_labeling/candidate_tool_calls
async_labeling/invalid_requests
async_labeling/built_requests
async_labeling/accepted_requests
async_labeler/submitted_count
async_labeler/completed_count
async_labeler/failed_count
async_labeler/expired_count
async_labeler/request_queue_size
async_labeler/completed_buffer_size
ranker/async_mode
ranker/async_updates
ranker/async_wait_empty
ranker/async_consumed_signals
ranker/async_built_samples
ranker/local_update_step
```

## 验证记录

10 step GPU smoke run：

```text
log/train_logs/coAgenticRetriever/260615-2342-async-ds-flash-10step-CAR_async_labeling_ds_flash_gpu10
```

关键结果：

```text
completed_train_steps: 10
step10 ranker/trace_ranked_docs: 25600
step10 async_labeling/invalid_requests: 0
step10 async_labeler/submitted_count: 100
step10 async_labeler/completed_count: 68
step10 async_labeler/failed_count: 22
step10 async_labeler/expired_count: 0
step10 ranker/local_update_step: 2
step10 ranker/async_updates: 1
step10 ranker/async_consumed_signals: 32
step10 ranker/async_built_samples: 32
```

`ranker/trace_ranked_docs=25600` 对应 `512 tool calls * 50 docs`，说明 smoke 中 async request 使用的是 ranker 排序后的 top50 trace，不是 recall top50，也不是 top10。

耗时报告：

```text
avg train step: 189.657s
p50 train step: 183.682s
p90 train step: 204.685s
max train step: 226.276s
```

ranker async update 本身很短：

```text
timing/ranker_async_update_total: 2.57s / 3.33s 左右
```

主要耗时仍在 rollout/tool calls 和 actor update，不在 ranker async update。

## 推荐运行方式

配置 dry-run：

```bash
DRY_RUN=1 bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
```

10 step smoke：

```bash
TOTAL_STEPS=10 \
RUN_STAMP=260616-async-smoke \
EXP_NAME=CAR_async_labeling_ds_flash_smoke \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
```

注意：smoke 只建议改 `TOTAL_STEPS` / `RUN_STAMP` / `EXP_NAME`。不要为了 smoke 改 batch size、rollout N、并发数等规模参数，否则验证结果不能代表正式脚本。

如果多人共用已经启动的 judge 服务，建议显式关闭自动停止：

```bash
AUTO_START_LLM_JUDGE=0 \
AUTO_STOP_LLM_JUDGE=0 \
LLM_JUDGE_PREFLIGHT=1 \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/train_CAR_async_labeling_ds_flash.sh
```

如果需要本脚本自动启动并在退出时停止 judge 服务，保持任务脚本默认值即可：

```bash
AUTO_START_LLM_JUDGE=1
AUTO_STOP_LLM_JUDGE=1
```

## 已踩过的坑

### 1. top50 必须是 rank 后 top50

最容易犯的错误是把 `TOP_N` 理解成 recall top50。这里不是。

正确链路：

```text
recall retriever top50
  -> dense ranker rerank
  -> rank_top50_docs 保存完整 50 个
  -> rank_top5_docs 给 agent
  -> rank_top50_docs 给 async judge
```

如果 tool trace 只保存 top10，`AsyncLabelRequest.validate_rank50()` 会让请求全部 invalid。修复方式不是改 judge 接受 top10，而是保证训练工具保存 rank 后 top50。

### 2. LLM judge 不做标签策略

不要让 judge 输出 positive/negative。judge 只输出排序：

```json
{"ranked_ids": ["...", "..."]}
```

正负例划分属于 `sample_builder`。未来换 top3/top5/hard-negative 策略时，只新增 sample builder，不改 judge stage 的职责。

### 3. prompt 必须磁盘注入

默认 prompt 已放到：

```text
CoAgenticRetriever/async_labeling/prompts/llm_judge_rank50_v1.md
```

不要把 prompt 文本写进 `llm_judge_rank50.py`。这样后续 prompt 迭代只改 markdown 和 YAML，不改代码。

### 4. LLM judge 格式失败是正常观测项

DeepSeek-V4-Flash 在 50 文档排序时可能返回：

```text
49/51 个 id
重复 id
未知 id
缺失 id
非 JSON
```

当前实现会写入 `failures.jsonl` 并增加 `failed_count`。这不是训练崩溃条件。后续如果失败率过高，优先改 prompt、max_tokens、max_chunk_chars 或增加格式修复 stage，而不是让主训练等待。

### 5. async ranker trainer 当前是 thread，不是 Ray actor

计划讨论中过 Ray remote 方案，但当前实现采用本地 daemon thread。原因是现有 ranker worker 已在 trainer 进程中本地封装，线程加锁接入更小。

因此需要注意：

```text
ranker_wg.update_ranker_contrastive
ranker_wg.save_checkpoint
```

都必须走 `ranker_lock`，避免 checkpoint 与后台 update 并发。

### 6. ranker 落后 GRPO 是预期行为

如果 judge 慢或失败率高，可能出现 GRPO 已经更新多步，ranker 还没有更新的情况。负面影响是：

```text
ranker 信号滞后
ranker 与 agent policy 的分布不同步
ranker update 次数减少
contrastive 样本更偏旧策略
```

但这比阻塞 GRPO 更符合当前目标。需要通过以下指标观察：

```text
async_labeler/request_queue_size
async_labeler/completed_buffer_size
async_labeler/failed_count
async_labeler/expired_count
ranker/async_updates
ranker/local_update_step
```

### 7. max_sub_query 限制的是 tool call

默认 `max_sub_query=10`。它不是每个 trajectory 数量，而是每个 global step 最多提交给 async labeler 的 search tool call 数。

这个限制用于避免一个 step 中几百个 tool call 全部进入 judge，导致 ranker 链路严重落后。

### 8. max_glb_step_lag 会在两处检查

当前过期检查发生在：

```text
worker 从 queue 取出 request 时
judge 返回结果准备写 buffer 前
```

过期后写 failure/expired 计数，不写 completed buffer。

### 9. 日志目录必须跟 run-name 走

async labeling 日志不要写到全局散落目录。默认应在：

```text
log/train_logs/coAgenticRetriever/<run-name>/async_labeling/
```

这样一个训练 run 的主日志、metrics、judge 服务日志、async request/signal/failure 都在同一个 run-name 目录下。

### 10. 服务侧配置和训练侧配置不要混在一起

训练侧 YAML 描述策略：

```text
max_sub_query
max_glb_step_lag
sample_builder
prompt.path
endpoint
```

服务侧 YAML 描述 vLLM 启动：

```text
model_path
cuda_visible_devices
tensor_parallel_size
max_model_len
kv_cache_dtype
port
```

不要把 `model_path`、`--max-model-len` 这类 vLLM 参数塞进训练侧策略 YAML。

## 后续扩展建议

### sample_builder 策略扩展

新增策略应放在：

```text
CoAgenticRetriever/async_labeling/sample_builder/
```

并在 `sample_builder/config.py` 中注册。建议保持统一接口：

```python
builder.build(signals: list[CandidateSignalData]) -> list[ContrastiveSample]
```

可配置参数建议保持少量通用字段：

```yaml
num_groups_per_step
neg_per_pos
strategy_kwargs
```

具体 topK、hard/easy negative、重复采样细节可以放进 `strategy_kwargs`，不要过早把所有策略参数固定成顶层 schema。

### extra_scorer 扩展

当前 `extra_scorer_stub.py` 只是预留接口。后续增加新的评分阶段时，应放在 `stages/` 下，并让 `CandidateSignalData` 中：

```text
judge_scores
extra_scores
final_scores
score_version
```

保持可解释。`final_scores` 是 sample_builder 默认读取的最终排序分数。

### judge 输出修复

如果后续发现 judge 失败率过高，可以新增一个轻量修复逻辑：

```text
解析 ranked_ids
去重
过滤未知 id
把缺失 id 按原 ranker 顺序补到末尾
记录 repair_count 和 repair_policy
```

但第一版没有这么做，是为了保持 judge 信号严格可解释。是否启用修复应成为配置项，不能静默修改。

## 接手检查清单

接手或改动前先检查：

1. `TOP_N=50` 是否仍然生效。
2. `ranker/trace_ranked_docs` 是否约等于 `tool_calls * 50`。
3. `async_labeling/invalid_requests` 是否为 0 或接近 0。
4. `completed_signals.jsonl` 中每条 signal 是否有 50 个 `ranked_chunk_list`。
5. `failures.jsonl` 的主要失败原因是否只是 judge 格式问题。
6. `ranker/local_update_step` 是否随训练推进增长。
7. `async_labeler/completed_buffer_size` 是否长期为 0。
8. `async_labeler/request_queue_size` 是否长期堆满。
9. `AUTO_STOP_LLM_JUDGE` 是否会误停别人共用的 judge 服务。
10. prompt 是否从 `CoAgenticRetriever/async_labeling/prompts/` 读取，而不是代码硬编码。

