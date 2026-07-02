# CoAgenticRetriever Async Data Labelling 计划

确认时间：2026-06-15

本文档记录 CoAgenticRetriever 中为 dense ranker contrastive step 增加异步样本信号构造能力的设计方案。当前内容用于实现前确认，不代表代码已经落地。

## 1. 背景与目标

当前 CoAgenticRetriever 通过 contrastive step 优化 trainable dense ranker。现有同步链路是：

```text
fresh trajectories
  -> trajectory_selector
  -> signal_builder
  -> sample_builder
  -> replay_buffer
  -> collator
  -> ranker_wg.update_ranker_contrastive
```

默认 `signal_builder` 主要基于当前 ranker 排名构造 pseudo label，监督信号较弱。新目标是在 GPU06/GPU07 上启动 DeepSeek-Flash LLM-as-judge 服务，异步为 `origin_query + sub_query + ranked_chunk_list` 生成排序信号，再由可配置 `sample_builder` 构造 ranker contrastive samples。

设计目标：

1. `async_ranker_training` 是可配置的 ranker 训练策略。
2. LLM judge 只产生排序/分数信号，不产生正负例标签。
3. 正负例划分、hard/easy negative、重复采样和补齐逻辑全部属于 `sample_builder`。
4. `async_ranker_training` 不阻塞 GRPO / agent LLM 主训练链路。
5. `sample_builder` 不等待 completed buffer；ranker async trainer 只在后台等待下一条 completed signal。
6. ranker contrastive step 允许落后于 GRPO step，但必须通过限流、过期丢弃和指标监控控制滞后。

非目标：

- 不让 LLM judge 同步阻塞 `signal_builder.build()` 或 GRPO 主循环。
- 不直接引入 Redis/Kafka 等外部队列。
- 不废弃现有 pseudo-rank fallback。
- 不改变 ranker 的 InfoNCE loss 形式。
- 不让 GRPO 主链路等待 ranker sample 或 ranker update。

## 2. 总体架构

建议拆成三条链路：

```text
GRPO / rollout 主链路
  -> rollout 产生 tool_call_details
  -> 提取 origin_query + sub_query + ranked_chunk_list
  -> async_ranker_training_labeler.submit(...)
  -> 继续 agent GRPO/PPO update
  -> 不等待 judge，不等待 ranker sample

async_ranker_training 链路
  -> request queue
  -> LLM-as-judge stage
  -> optional extra scoring stage
  -> score merger
  -> completed CandidateSignalData buffer

ranker contrastive 链路
  -> completed buffer 一有 signal 就 destructive pop 最新 1 个 CandidateSignalData
  -> signal_builder 转成 labeled context
  -> sample_builder 通过重复复采样构造 num_groups_per_step 个 ContrastiveSample
  -> ranker contrastive update
```

关键原则：

- `AsyncLabeler.submit()` 必须非阻塞。
- `CandidateSignalBuffer.pop_latest()` 可以阻塞，但只能由 ranker async trainer 调用。
- ranker contrastive step 应从主训练循环拆出，变成 Ray actor、后台进程或等价异步 trainer。
- 如果 `ranker_step.remote()` 后立刻 `ray.get()`，本质上仍会阻塞 GRPO，不符合该设计。

## 3. 配置设计

框架级默认配置建议新增：

```text
CoAgenticRetriever/config/async_ranker_training.yaml
```

主训练配置加载方式：

```yaml
# CoAgenticRetriever/config/coagentic_retriever_trainer.yaml
defaults:
  - ranker_contrastive
  - async_ranker_training
  - _self_
```

本地 DeepSeek-Flash / GPU06-GPU07 覆盖配置建议放在：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

配置分两层：

- 训练侧 async ranker training 配置：描述训练过程如何提交请求、限流、消费 completed buffer、调用哪个 judge endpoint、使用哪个 prompt 和 sample_builder。
- 服务侧 LLM judge 启动配置：描述 vLLM 服务如何启动，包括模型地址、GPU、端口、`--max-model-len`、tensor parallel、KV cache dtype 等。

服务侧启动配置建议放在：

```text
CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

`CoAgenticRetriever/scripts/launch_llm_as_judge.sh` 只读取该服务侧 YAML，也允许通过显式环境变量覆盖少量字段；训练侧 YAML 不直接保存 `model_path`、`--max-model-len` 等 vLLM 启动参数，只保存 endpoint 和 served model name。

建议配置：

```yaml
ranker_training:
  async_ranker_training:
    enable: true

    # 每个 global_step 最多提交多少个 sub_query/tool call。
    max_sub_query: 10

    # 请求允许滞后的最大 global step 数。
    max_glb_step_lag: 3

    request_queue_size: 2048
    completed_buffer_size: 4096
    drop_policy: drop_oldest

    num_workers: 4
    request_timeout_seconds: 60
    max_retries: 2
    sub_query_selection_policy: high_value_first

    stages:
      - type: llm_as_judge
        endpoint: http://127.0.0.1:8067/v1/chat/completions
        model: deepseek-flash
        score_schema: ranked_ids_top50
        max_docs_per_request: 50
        temperature: 0.0
        max_tokens: 1024
        request_timeout_seconds: 600
        max_retries: 2
        prompt:
          path: CoAgenticRetriever/async_ranker_training/prompts/llm_judge_rank50_v1.md
          version: llm_judge_rank50_v1
          format: markdown_system_user_template
          max_chunk_chars: 512
          include_title: true
          include_scores: true
          shuffle_passages: false
          output_mode: no_think_json

      - type: extra_scorer
        enable: false
        weight: 0.3

    sample_builder:
      type: random_negative_repeat
      num_groups_per_step: 32
      neg_per_pos: 15
      allow_repeat_negative_sampling: true
      seed: 42

    logging:
      enable: true
      log_dir: null  # null 表示使用 <TRAIN_LOG_DIR>/async_ranker_training
      write_request_text: false
      max_text_chars: 512
      metrics_interval_seconds: 30
      sample_preview_every_n: 50
```

### 3.1 `max_sub_query`

默认值：`10`。

含义：

- 每个 `global_step` 进入 async ranker training labeler 的 sub-query / tool-call 数不能超过该值。
- 限制粒度是 tool call，不是 trajectory。
- 即使 `trajectory_selector` 只选中 3 条轨迹，如果每条轨迹有 8 个 search 动作，也最多只提交 10 个请求。

第一版提交流程：

```text
rollout tool_call_details
  -> flatten 成 tool_call candidates
  -> 按 sub_query_selection_policy 排序
  -> 只取前 max_sub_query 个
  -> submit 到 async_ranker_training_labeler
```

`high_value_first` 的具体排序规则仍待确认。

### 3.2 `max_glb_step_lag`

默认值：`3`。

含义：

- 处理请求时检查 `current_global_step - request.created_global_step`。
- 如果超过 `max_glb_step_lag`，请求过期，直接跳过，不再调用 LLM judge。

该检查发生两次：

1. worker 从 request queue 取出请求时。
2. LLM judge 返回结果、准备写入 completed buffer 前。

## 4. 数据结构

### 4.1 `AsyncLabelRequest`

`async_ranker_training_labeler` 的输入不是 `ContrastiveSample`，也不是已经 label 好的 `LabeledRankingContext`，而是从 rollout `tool_call_details` 提取出的候选排序上下文。

```text
AsyncLabelRequest:
  request_id
  created_global_step
  trajectory_id
  tool_call_id
  turn_idx

  origin_query
  sub_query
  golden_answers

  ranked_chunk_list:
    [
      {
        doc_id
        title
        text
        recall_rank
        recall_score
        rank_rank
        rank_score
        metadata
      }
    ]

  trajectory_score
  score_type
  trace_metadata
  label_policy
  prompt_version
```

`ranked_chunk_list` 必须来自当前 ranker 的 `rank_top50_docs`，且第一版要求正好 50 个 chunk。缺失、不足 50 个、id 缺失或字段不完整时应直接报错并记录 `invalid_request` / `missing_ranked_chunk_list` 计数；该 request 丢弃，不 fallback 到 `recall_top50_docs`，避免把 recall 阶段数据混入 judge 信号。

### 4.2 `CandidateSignalData`

`CandidateSignalData` 是 completed candidate signal buffer 的单位元素。它表示一组已经完成异步打分、可被 `sample_builder` 消费的候选信号数据。

```text
CandidateSignalData:
  signal_id
  request_id
  created_global_step
  completed_global_step
  completed_at

  trajectory_id
  tool_call_id
  turn_idx
  origin_query
  sub_query
  golden_answers

  ranked_chunk_list

  scores:
    [
      {
        doc_id
        judge_rank
        judge_score
        extra_score
        final_score
        confidence
        reason_code
      }
    ]

  label_source
  score_version
  prompt_version
  judge_model

  status
  error
  latency_ms
  raw_response_ref
```

注意：

- `CandidateSignalData` 不包含 `positives` / `negatives`。
- LLM judge stage 的职责到 `scores` 为止。
- 正负例划分、hard/easy negative 选择和补齐规则都属于 `sample_builder` 策略内部逻辑。

## 5. Completed Buffer 语义

训练用 completed buffer 采用 destructive queue 语义：

```text
CandidateSignalBuffer.pop_latest(n=1, wait=True)
  -> 返回最新 1 个 CandidateSignalData
  -> 返回的数据从 active buffer 中移除
  -> sample_builder 使用这些数据构造 ContrastiveSample
```

约束：

- buffer 中被消费掉的 `CandidateSignalData` 从 active buffer 消失。
- sample_builder 不等待 completed buffer；等待只发生在 ranker async trainer 后台线程取下一条 completed signal 时。
- ranker contrastive step 等待不应导致 GRPO step 等待。

为兼顾可追溯性，区分两类存储：

```text
completed_signal_queue    # 训练消费队列，pop 后消失
signal_audit_store        # append-only JSONL/SQLite，永久保留用于复盘
```

## 6. LLM-as-Judge 设计

### 6.1 服务形态

第一版使用 OpenAI-compatible vLLM 服务：

```text
endpoint: http://127.0.0.1:8067/v1/chat/completions
model: deepseek-flash 或 DeepSeek-V4-Flash
gpu: 6,7
temperature: 0.0
max_tokens: 1024
```

`max_tokens` 和 `max_chunk_chars` 的默认值按 `pipelines/temp/deepseek_v4_flash_judge/` 的实测结果收紧：

- 临时 vLLM 服务使用 `--max-model-len 11000`。
- `01_prepare_chunk_ranking_prompts.py` 中 no-think 请求使用 `max_tokens=1024`。
- `results/latest/summary_no_think.json` 中 100 个 50-chunk 请求全部成功，平均 `prompt_tokens` 约 `7813`，平均 `completion_tokens` 约 `254`，平均总 token 约 `8067`。
- 因此第一版正式配置继续使用 `max_tokens=1024`，同时把 `prompt.max_chunk_chars` 设为 `512`，避免在线训练遇到更长 chunk 时挤爆 11000 context。
- 如果后续 prompt 变长、打开 think 模式、或需要保留更长 chunk 文本，应先重新跑 50-chunk judge smoke/benchmark，再上调 `max_tokens` 或 `max_chunk_chars`。

服务启动脚本建议：

```text
CoAgenticRetriever/scripts/launch_llm_as_judge.sh --config CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml
```

服务侧配置文件示例：

```yaml
service:
  name: llm_judge_deepseek_flash_gpu06_07
  backend: vllm

model:
  model_path: /data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash
  served_model_name: DeepSeek-V4-Flash
  trust_remote_code: true

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
  linear_backend: auto
  moe_backend: auto
  disable_custom_all_reduce: true

logs:
  log_dir: log/train_logs/<GROUP_NAME>/<RUN_NAME>/async_ranker_training/judge_server
  log_file: vllm_gpu06_07_8067.log
  pid_file: vllm_gpu06_07_8067.pid
```

启动脚本职责：

- 读取服务侧 YAML，组装 `vllm serve` 命令。
- 检查 `model.model_path`、`runtime.vllm`、端口可用性和 `/v1/models` readiness。
- 将 vLLM stdout/stderr 写入当前 run 的 `async_ranker_training/judge_server/` 子目录。
- 失败时直接退出并返回非零 exit code，不静默 fallback 到其他模型或端口。

可以参考 `pipelines/temp/deepseek_v4_flash_judge/00_start_vllm_gpu06_07.sh` 的启动参数和日志经验，但正式 async ranker training 框架不能 import、source、subprocess 调用或依赖该 temp pipeline 的代码。

### 6.2 Prompt

默认 prompt 文件：

```text
CoAgenticRetriever/async_ranker_training/prompts/llm_judge_rank50_v1.md
```

该默认文件内容来自 `docs/planning/260615_llm_judge_prompt.md`，但运行时不读取 planning 文档。正式代码默认读取 `CoAgenticRetriever/async_ranker_training/prompts/llm_judge_rank50_v1.md`。

规则：

- prompt 文本不能硬编码在 Python 代码中。
- `LLMJudgeRank50Stage` 初始化时读取 `prompt.path` 指向的模板文件。
- prompt 文件路径通过配置注入，可以被本地实验 YAML 覆盖。
- prompt version 通过 `prompt.version` 记录到 `CandidateSignalData.prompt_version` 和观察日志。
- 如果 prompt 文件不存在、缺少 system/user 段或变量渲染失败，应在初始化或 smoke 阶段 fail fast。

模板变量映射：

```text
{{原始查询问题}} -> request.origin_query
{{规范化后的查询问题}} -> request.sub_query
{{允许的所有段落ID列表}} -> request.ranked_chunk_list[*].doc_id
{{段落ID}} -> chunk.doc_id
{{段落标题}} -> chunk.title
{{检索排名}} -> chunk.recall_rank 或 chunk.rank_rank
{{检索分数}} -> chunk.recall_score 或 chunk.rank_score
{{段落文本片段}} -> chunk.text 截断到 max_chunk_chars
```

后续稳定 prompt 均放在：

```text
CoAgenticRetriever/async_ranker_training/prompts/
```

`docs/planning/` 中的 prompt 只作为讨论和演化记录。

### 6.3 Judge Stage

`LLMJudgeRank50Stage` 负责：

- 组装 OpenAI-compatible `messages`。
- 调用 `/v1/chat/completions`。
- 解析 JSON-only `{"ranked_ids": [...]}`。
- 校验正好 50 个 id、无重复、无未知 id。
- 将 `ranked_ids` 转换为 `judge_rank` / `judge_score`。

建议接口：

```text
LLMJudgeRank50Stage.score(request, context) -> StageResult
```

`StageResult`：

```text
ok: bool
scores: list[JudgeChunkScore]
raw_response_ref: str | null
usage: dict
latency_ms: float
error_type: str | null
error_message: str | null
```

失败时：

- `ok=false`
- 不生成 `CandidateSignalData`
- 不进入 completed buffer
- 只写 failures 日志和 metrics

### 6.4 `ranked_ids` 校验

校验步骤：

1. 从模型输出中解析一个 JSON object。
2. 读取 `ranked_ids`。
3. 确认 `ranked_ids` 是 list。
4. 确认长度为 50。
5. 确认没有重复 id。
6. 确认所有 id 都来自 request 的 passage id 集合。
7. 确认 request 中每个 passage id 都出现一次。

任一失败都视为 judge failure，只计数后丢弃。

`judge_score` 第一版建议：

```text
judge_score = (50 - judge_rank + 1) / 50
final_score = judge_score
```

`extra_scorer` 第一版只预留接口，不实现额外评分：

```text
extra_score = null
```

### 6.5 请求生命周期

```text
AsyncLabelRequest
  -> check lag before judge
  -> render prompt
  -> call judge endpoint
  -> parse content
  -> validate ranked_ids
  -> check lag before completed buffer write
  -> build CandidateSignalData
  -> push completed buffer
  -> write observation logs
```

judge 失败只计数后丢弃，不写入 completed signal queue，不阻塞 GRPO。

### 6.6 与 temp pipeline 的关系

`pipelines/temp/deepseek_v4_flash_judge/` 只作为参考实现和踩坑记录。

可参考：

- vLLM GPU06/GPU07 启动参数。
- 50 chunk ranking prompt 的任务描述和 JSON-only 输出约束。
- OpenAI-compatible client、timeout、并发和 latency 统计方式。
- vLLM 启动和运行问题排查日志。

必须在正式框架中重新实现：

- prompt builder。
- judge client。
- `ranked_ids` 解析和校验。
- latency / token usage / failure 记录。
- async ranker training 观察日志。
- smoke 检查脚本。

## 7. Sample Builder 策略框架

`sample_builder` 是独立、可配置的策略层。LLM judge 只提供 `scores`；正样本、负样本、hard/easy negative、采样次数和补齐逻辑都属于 `sample_builder`。

建议目录：

```text
CoAgenticRetriever/async_ranker_training/sample_builder/
  __init__.py
  base.py
  config.py
  random_negative_repeat.py
```

后续新增策略：

```text
topk_rest_negative.py
graded_hard_easy.py
pairwise_margin.py
```

核心接口：

```text
class CandidateSignalSampleBuilder:
    def build(signals: list[CandidateSignalData]) -> list[ContrastiveSample]
```

输入输出契约：

```text
input:
  1 个或少量 CandidateSignalData；当前训练线程一有 completed signal 就消费 1 条

output:
  目标 num_groups_per_step 个 ContrastiveSample
  每组 ContrastiveSample = 1 positive + neg_per_pos negatives
```

通用配置只保留跨策略稳定的重要参数：

```yaml
ranker_training:
  async_ranker_training:
    sample_builder:
      type: random_negative_repeat
      num_groups_per_step: 32
      neg_per_pos: 15
      allow_repeat_negative_sampling: true
      seed: 42
```

参数语义：

- `num_groups_per_step`：输出侧参数。每次 ranker update 需要多少组 `ContrastiveSample`。
- `neg_per_pos`：单组样本结构参数。每个 positive 搭配多少 negatives。
- `allow_repeat_negative_sampling`：negative 不足时是否允许重复采样。

构造数量规则：

```text
signals = buffer.pop_latest(n=1, wait=True)
labeled_contexts = signal_builder.build(signals)
candidate_groups = sample_builder.enumerate_possible_groups(labeled_contexts)

if len(candidate_groups) > num_groups_per_step:
  按策略采样/截断到 num_groups_per_step

if len(candidate_groups) < num_groups_per_step:
  通过重复 positive、重新负采样、跨 signal 采样等方式补齐

if 仍然不足:
  记录 insufficient_samples，按策略等待更多 signals、跳过该 ranker step 或 fallback
```

### 7.1 默认策略

默认策略：`random_negative_repeat`。

它应与当前 `ranker_strategies/sample_builder/random_negative_repeat.py` 保持一致的核心行为：

- 从每个 signal 内部根据 judge rank/score 派生 positive pool 和 negative pool。
- 为每个 positive 建立候选 `(signal, positive, negatives)`。
- 轮转候选 positive，直到构造出 `num_groups_per_step` 组样本。
- 每组随机采样 `neg_per_pos` 个 negatives。
- negative 不足时，如果 `allow_repeat_negative_sampling=true`，允许重复采样 negative 补齐。
- 每组样本使用 `query_input = origin_query + " [SEP] " + sub_query`。
- `positive_doc_index=0`，documents 顺序保持 `[positive, *negatives]`。
- `sample_source="fresh"` 或后续由 buffer/replay 逻辑决定。

默认策略中的 positive/negative 派生规则作为策略内部默认值，不提前暴露到公共配置。后续 top3、top5、hard/easy negative 等行为应通过新增 strategy 或 strategy 私有 `strategy_kwargs` 扩展。

策略扩展示例：

```yaml
ranker_training:
  async_ranker_training:
    sample_builder:
      type: graded_hard_easy
      num_groups_per_step: 64
      neg_per_pos: 15
      strategy_kwargs:
        ...
```

`strategy_kwargs` 只由对应 strategy 解释，核心框架不理解其内部含义。

## 8. 新增代码架构

建议新增独立 package：

```text
CoAgenticRetriever/
  async_ranker_training/
    __init__.py
    schemas.py
    config.py
    request_builder.py
    labeler.py
    worker.py
    buffer.py
    logging.py
    metrics.py

    configs/
      llm_judge_vllm_deepseek_flash_gpu06_07.yaml

    prompts/
      llm_judge_rank50_v1.md

    stages/
      __init__.py
      base.py
      llm_judge_rank50.py
      extra_scorer_stub.py

    sample_builder/
      __init__.py
      base.py
      config.py
      random_negative_repeat.py

    utils/
      __init__.py
      jsonl.py
      time_utils.py
      id_utils.py
      validation.py
```

模块职责：

- `schemas.py`：定义 `AsyncLabelRequest`、`CandidateChunk`、`JudgeChunkScore`、`CandidateSignalData`、`AsyncLabelFailure` 等轻量 schema。
- `config.py`：读取和规范化 `ranker_training.async_ranker_training` 配置。
- `request_builder.py`：从 rollout `tool_call_details` 构造 `AsyncLabelRequest`，应用 `max_sub_query` 和 selection policy。
- `labeler.py`：对 trainer 暴露 `AsyncLabeler.submit()`、`get_metrics()`、`close()`。
- `worker.py`：后台 worker loop，负责取请求、调用 stages、写 completed buffer 和观察日志。
- `buffer.py`：维护 destructive completed queue 和 append-only audit store。
- `stages/`：维护 LLM judge 和后续 scorer stage。
- `sample_builder/`：维护从 `CandidateSignalData` 到 `ContrastiveSample` 的策略。
- `logging.py`：维护 async ranker training 观察日志。
- `metrics.py`：聚合 async ranker training labeler 和 sample builder 指标。

## 9. Trainer 集成

当前同步链路大致是：

```text
rollout
  -> _enrich_tool_calls_with_ranker
  -> process_main_agent_ppo_step.remote(...)
  -> build_fresh_trajectories_from_dataproto
  -> process_ranker_contrastive_step(...)
  -> ray.get(main_futures)
```

新框架：

```text
init_workers()
  -> 初始化 AsyncLabeler
  -> 初始化 RankerAsyncTrainer actor

each global step:
  -> rollout
  -> _enrich_tool_calls_with_ranker
  -> async_ranker_training_labeler.submit(main_batch.tool_call_details, global_step)
  -> process_main_agent_ppo_step.remote(...)
  -> 不在主循环内等待 ranker samples
  -> ray.get(main_futures)

background RankerAsyncTrainer:
  -> completed_signal_queue.pop_latest(n=1, wait=True)
  -> signal_builder.build(...)
  -> sample_builder.build(...)
  -> ranker_wg.update_ranker_contrastive(...)
```

`RankerAsyncTrainer` 建议作为 Ray actor：

```text
RankerAsyncTrainer
  init(config, ranker_checkpoint_init)
  start()
  notify_agent_global_step(global_step)
  save_checkpoint(agent_global_step)
  get_metrics()
  stop()
```

checkpoint 规则：

- ranker checkpoint 保存频率跟随 agent global step。
- agent trainer 触发 checkpoint 时，通知 `RankerAsyncTrainer.save_checkpoint(agent_global_step)`。
- 保存路径继续使用 agent checkpoint step 下的 ranker 子目录：

```text
checkpoints/<project>/<run>/global_step_<N>/ranker/
```

## 10. 训练脚本配置与运行方式

新框架应兼容旧训练入口的使用方式。旧入口示例是：

```text
tasks/train_tasks/train_CAR_naive_acce.sh
  -> scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
  -> scripts/coagenticRetriever_local/assets/00_run_agentic_iter_rag_verl.sh
  -> CoAgenticRetriever/main_coagentic_retriever.py
```

新框架不要另起一套完全不同的训练入口，而是在现有入口上增加 async ranker training 的可选配置。目标是让用户仍然通过 `tasks/train_tasks/*.sh` 启动实验，只多设置少量环境变量和 YAML 路径。

建议新增 task 脚本：

```text
tasks/train_tasks/train_CAR_async_ranker_training_ds_flash.sh
```

该脚本只负责实验级配置：

```bash
export EXP_NAME="CAR_async_ranker_training_ds_flash_v1"

export ENABLE_ASYNC_RANKER_TRAINING=1
export ASYNC_RANKER_TRAINING_YAML="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml"

export AUTO_START_LLM_JUDGE=1
export AUTO_STOP_LLM_JUDGE=0
export LLM_JUDGE_SERVICE_CONFIG="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml"
export LLM_JUDGE_ENDPOINT="http://127.0.0.1:8067/v1/chat/completions"

export AGENT_GPU_IDS="0,1,2,3"
export RANK_GPU_ID="4"
export RECALL_GPU_ID="5"

export COAGENTIC_EXTRA_ARGS="actor_rollout_ref.rollout.max_num_batched_tokens=32768 actor_rollout_ref.rollout.multi_turn.max_parallel_calls=2 ++data.apply_chat_template_kwargs.enable_thinking=False"

bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

### 10.1 Launcher 新增环境变量

`scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh` 建议新增：

```bash
ENABLE_ASYNC_RANKER_TRAINING="${ENABLE_ASYNC_RANKER_TRAINING:-0}"
ASYNC_RANKER_TRAINING_YAML="${ASYNC_RANKER_TRAINING_YAML:-}"
AUTO_START_LLM_JUDGE="${AUTO_START_LLM_JUDGE:-0}"
AUTO_STOP_LLM_JUDGE="${AUTO_STOP_LLM_JUDGE:-0}"
LLM_JUDGE_SERVICE_CONFIG="${LLM_JUDGE_SERVICE_CONFIG:-}"
LLM_JUDGE_ENDPOINT="${LLM_JUDGE_ENDPOINT:-http://127.0.0.1:8067/v1/chat/completions}"
LLM_JUDGE_PREFLIGHT="${LLM_JUDGE_PREFLIGHT:-1}"
ASYNC_RANKER_TRAINING_LOG_DIR="${ASYNC_RANKER_TRAINING_LOG_DIR:-${LOG_DIR}/async_ranker_training}"
```

这些变量写入 `${LOG_DIR}/${RUN_NAME}.env`，便于复盘。

### 10.2 Hydra YAML 注入方式

现有 launcher 已支持：

```text
HYDRA_OVERRIDE_YAMLS
RANKER_STRATEGY_YAML
COAGENTIC_EXTRA_ARGS
```

async ranker training 应复用同一套机制。launcher 收集 YAML 时追加 `ASYNC_RANKER_TRAINING_YAML`：

```bash
hydra_collect_yaml_override_files hydra_yaml_files \
  "${HYDRA_OVERRIDE_YAMLS:-}" \
  "${RANKER_STRATEGY_YAML:-}" \
  "${ASYNC_RANKER_TRAINING_YAML:-}"
```

这样 async ranker training 主配置可以放在：

```text
scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml
```

而不是把复杂配置塞进 `COAGENTIC_EXTRA_ARGS`。`COAGENTIC_EXTRA_ARGS` 仍保留给临时、小范围的 dotlist override。

### 10.3 Judge 服务生命周期

judge 服务由 shell launcher 管理，不由 Python trainer 隐式启动。建议新增函数：

```text
ensure_llm_judge_service()
cleanup_llm_judge_service()
check_llm_judge_service()
```

规则：

- `ENABLE_ASYNC_RANKER_TRAINING=1` 且 `AUTO_START_LLM_JUDGE=1` 时，训练 launcher 调用 `CoAgenticRetriever/scripts/launch_llm_as_judge.sh --config "${LLM_JUDGE_SERVICE_CONFIG}"`。
- `LLM_JUDGE_PREFLIGHT=1` 时，在训练前请求 `/v1/models` 或最小 chat completion smoke。
- judge 服务日志写入 `${LOG_DIR}/async_ranker_training/judge_server/`。
- 如果服务不可用，直接报错退出，不 fallback 到同步 pseudo label。
- `AUTO_STOP_LLM_JUDGE=0` 为默认值，因为 judge 服务可能被多个实验复用；只有明确开启时训练结束才停止该服务。

GPU 分配原则：

- agent LLM 使用 `AGENT_GPU_IDS`，默认 `0,1,2,3`。
- dense ranker 使用 `RANK_GPU_ID`，默认 `4`。
- recall retriever 使用 `RECALL_GPU_ID`，默认 `5`。
- LLM judge 服务使用服务侧 YAML 中的 `cuda_visible_devices: "6,7"`。
- 训练进程的 `GPU_IDS` 不应包含 `6,7`，避免 Ray / VERL 抢占 judge GPU。

### 10.4 训练配置边界

训练侧 YAML 只描述训练策略：

```text
endpoint
served model name
prompt path/version
max_sub_query
max_glb_step_lag
request queue / completed buffer
sample_builder
logging
```

服务侧 YAML 只描述 vLLM 部署：

```text
model_path
served_model_name
host / port
cuda_visible_devices
tensor_parallel_size
gpu_memory_utilization
max_model_len
kv_cache_dtype
backend flags
judge server log path
```

不要在训练侧 YAML 保存 `model_path`、`--max-model-len` 等 vLLM 启动参数；也不要在服务侧 YAML 保存 `sample_builder` 等训练策略参数。

### 10.5 启用开关

第一版不建议新增新的 trainer mode。继续使用：

```text
trainer.ranker_update_mode=contrastive
```

async ranker training 作为 ranker contrastive 的 signal source：

```yaml
ranker_training:
  signal_source: async_ranker_training
  async_ranker_training:
    enable: true
```

`ENABLE_ASYNC_RANKER_TRAINING=0` 时旧任务完全不受影响，仍走当前 `signal_builder` / pseudo-rank 构造逻辑。

## 11. 与当前代码框架的区别

当前框架：

```text
ranker_strategies/
  trajectory_selector
  signal_builder
  sample_builder
  replay_buffer
  collator

process_ranker_contrastive_step:
  同步完成 select/signal/sample/update
```

新框架：

```text
async_ranker_training/
  request_builder
  async worker
  scoring stages
  completed signal queue
  random_negative_repeat sample builder
  async ranker training observation logs

RankerAsyncTrainer:
  等待 completed signal
  独立执行 ranker update
```

核心差异：

- 当前 `signal_builder` 在训练 step 内同步产出标签；新框架由异步服务提前/延迟产出 `CandidateSignalData`。
- 当前 `sample_builder` 消费 `LabeledRankingContext`；新框架的 async sample builder 消费 `CandidateSignalData`。
- 当前 ranker update 在主 trainer loop 中执行；新框架中 ranker update 作为后台 actor 运行。
- 当前 replay buffer 存放 `ContrastiveSample`；新框架新增 completed signal queue，存放 `CandidateSignalData`，消费后再构造 sample。
- 新框架的 async ranker training 观察日志依附现有 train run 目录，但不接入正式 report schema。

## 11. 观察日志

新增日志只服务 async ranker training 策略调试。它复用现有训练日志系统创建的 run 目录，在 run 目录下新增 `async_ranker_training/` 子目录；但不接入 `src/logs/report_system` 的 report schema，不生成正式 train report。

默认目录：

```text
log/train_logs/<GROUP_NAME>/<RUN_NAME>/async_ranker_training/
```

示例：

```text
log/train_logs/coAgenticRetriever/260613-004352-CAR_mem_speed_no_think_v1/async_ranker_training/
```

建议产物：

```text
log/train_logs/<GROUP_NAME>/<RUN_NAME>/async_ranker_training/
  async_ranker_training.env.json
  requests.jsonl
  completed_signals.jsonl
  failures.jsonl
  queue_events.jsonl
  metrics.jsonl
  samples_preview.jsonl
```

文件职责：

- `async_ranker_training.env.json`：记录配置、judge endpoint、prompt version、score version、`max_sub_query`、`max_glb_step_lag`。
- `requests.jsonl`：append-only 请求日志，记录 `AsyncLabelRequest` 轻量版。
- `completed_signals.jsonl`：append-only 成功信号日志，记录 `CandidateSignalData` 或轻量版。
- `failures.jsonl`：失败日志，记录 request id、失败类型、错误信息、latency、created global step。
- `queue_events.jsonl`：记录 enqueue、drop_by_lag、drop_by_queue_full、pop_latest 等队列事件。
- `metrics.jsonl`：周期性写入 async ranker training labeler 和 sample builder 指标快照。
- `samples_preview.jsonl`：低频采样写入由 signal 构造出的 contrastive sample 预览。

写入原则：

- 不阻塞训练主链路。
- 写失败不能影响训练。
- 大文本字段默认截断，避免日志膨胀。
- 路径依附现有 train run 目录，便于和 `.train.log`、`.metrics.jsonl`、GPU 采样、checkpoint 对齐查看。
- judge 失败只进入 `failures.jsonl` 和 metrics，不进入 completed buffer。

## 12. 指标

建议新增指标：

```text
async_ranker_training/labeler_submitted_count
async_ranker_training/selected_tool_calls
async_ranker_training/labeler_expired_count
async_ranker_training/labeler_request_queue_size
async_ranker_training/labeler_completed_buffer_size
async_ranker_training/labeler_completed_count
async_ranker_training/labeler_failed_count
async_ranker_training/labeler_failed_count
async_ranker_training/labeler_avg_lag_steps
async_ranker_training/labeler_max_lag_steps
async_ranker_training/labeler_avg_latency_ms
ranker/async_consumed_signals
ranker/async_wait_seconds
ranker/async_sample_builder_consumed_signals
ranker/async_sample_builder_candidate_groups
ranker/async_sample_builder_output_groups
ranker/async_sample_builder_repeated_positive_groups
ranker/async_sample_builder_repeated_negative_count
ranker/async_sample_builder_insufficient_samples
ranker/update_per_agent_step
ranker/agent_step_lag
```

## 13. 滞后风险与缓解

如果 async ranker training 变慢，可能出现 agent LLM 连续更新多次，而 ranker contrastive step 长时间没有更新。这是异步设计允许出现的情况，但有负面影响：

- ranker 滞后于 agent query 分布。
- tool 质量改善延迟。
- 候选信号过期。
- 训练节奏失衡，系统退化为主要训练 agent，ranker 基本旁路。

缓解策略：

- 使用 `max_glb_step_lag=3` 丢弃过期请求和过期结果。
- 使用 `max_sub_query=10` 控制每个 global step 的请求量。
- 只消费最新 signals，过旧 signals 从 active queue 丢弃。
- 当 judge 队列积压时，按 selection policy 选择高价值 tool call。
- 保留 pseudo-rank fallback。

## 14. 第一版落地边界

第一版按最小闭环实现：

1. 新增 `CoAgenticRetriever/config/async_ranker_training.yaml`。
2. 新增 `CoAgenticRetriever/async_ranker_training/configs/llm_judge_vllm_deepseek_flash_gpu06_07.yaml`，保存模型地址、GPU、端口、`max_model_len` 等 vLLM 服务启动参数。
3. 补齐 `CoAgenticRetriever/scripts/launch_llm_as_judge.sh`，通过 `--config` 读取服务侧 YAML 启动 judge 服务。
4. 新增 `AsyncLabelRequest` 和 `CandidateSignalData` schema。
5. 新增 async ranker training labeler request queue、completed signal queue 和 audit store。
6. 新增 LLM-as-judge scorer stage，调用 GPU06/GPU07 上的 DeepSeek-Flash endpoint，对每个 request 的 50 个 chunk 返回 `ranked_ids`。
7. 新增 `max_sub_query=10` 和 `max_glb_step_lag=3` 的限流/过期逻辑。
8. 新增从 `tool_call_details` 构造 label request 的 request builder。
9. 新增 `random_negative_repeat` sample builder。
10. 将 ranker contrastive update 拆成后台 Ray actor 或等价异步 trainer。
11. 保留现有 pseudo-rank fallback。
12. `extra_scorer` 第一版只预留接口，不实现额外评分。
13. ranker checkpoint 保存频率跟随 agent global step。
14. judge 失败时只计数后丢弃。
15. 新增 async ranker training 观察日志子目录，默认写入 `log/train_logs/<GROUP_NAME>/<RUN_NAME>/async_ranker_training/`。
16. DeepSeek-Flash judge 实现只参考 `pipelines/temp/deepseek_v4_flash_judge/`，正式代码必须在 `async_ranker_training/` 中重新实现，不能调用 temp pipeline 代码。

第一版验证目标：

- GPU06/GPU07 的 DeepSeek-Flash judge 服务能稳定处理请求。
- async ranker training labeler 能将 rollout tool calls 转成 `CandidateSignalData`。
- completed queue 能以 destructive pop 方式在有 signal 时立即提供最新 completed signal。
- sample builder 能稳定输出 `num_groups_per_step` 组 `ContrastiveSample`。
- ranker async trainer 能等待 signal 并更新 ranker。
- GRPO step 和 ranker contrastive step 可以解耦运行。
- async ranker training 观察日志能够支持排查请求、失败、队列积压和样本构造质量。

## 15. 待继续确认的问题

1. `sub_query_selection_policy=high_value_first` 的具体排序规则。
2. judge 失败是否需要写 audit record；训练语义已确认是只计数后丢弃，不进入 completed buffer。
3. 第一版 `RankerAsyncTrainer` 是否必须直接使用 Ray actor，还是允许先用本地 thread 做 smoke。
4. `requests.jsonl` 和 `completed_signals.jsonl` 是否允许写完整 chunk text，还是默认只写截断文本。
