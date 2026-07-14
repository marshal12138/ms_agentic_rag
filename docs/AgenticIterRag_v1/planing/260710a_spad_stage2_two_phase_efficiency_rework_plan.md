# SPAD Stage2 两阶段高吞吐重构计划

日期：2026-07-10

关联执行计划：

- `docs/AgenticIterRag_v1/planing/260709a_spad_rag_qwen17_glm47_execution_plan.md`
- `docs/AgenticIterRag_v1/preworks/260709c_spad_rag_implementation_status_prework.md`

## 1. 背景和问题

当前 SPAD Stage2 `answer_refresh_data` 已能跑通基本链路，但实现形态不满足正式数据生产要求：

1. actor rollout、search、teacher labeling 串在一个单样本循环中，整体吞吐低。
2. actor vLLM 只启动单实例，无法吃满多卡。
3. teacher 和 actor 在同一个 Stage2 流程内交错调用，资源规划不清晰，难以把 NPU 利用率拉高。
4. 现有配置仍以单实例 `gpu_ids/port` 为中心，不适合多 replica 调度。
5. 失败样本处理和统计不足，单样本错误不应导致整批数据失败，但也不能静默丢失。
6. Stage2 最终产物必须严格满足 Stage3 DPO 数据消费契约。

本计划覆盖旧的 Stage2 单实例设计。重构后不保留旧单实例执行路径；旧配置结构只用于启动前失败提示，不作为兼容路径运行。

## 2. 目标

Stage2 拆分为两个 phase：

1. Phase A：`trajectory_rollout`
   - 使用 Stage1 产出的 actor 模型。
   - 多 actor replica 并发执行 multi-turn agent rollout。
   - 调用 recall/search，收集 evidence。
   - 输出 actor final answer、rejected answer、messages_before_final_answer、evidence_steps、search_count、错误/跳过原因。
   - 不调用 teacher，不生成 chosen，不写 Stage3 DPO pair。

2. Phase B：`teacher_labeling`
   - 读取 Phase A 的 trajectory JSONL。
   - 过滤无 actor final answer、无 evidence、格式不合法的样本。
   - 多 teacher replica 并发生成 evidence-grounded chosen answer。
   - 复用现有 teacher prompt、teacher answer 解析、F1/filter 逻辑。
   - 输出完整处理记录和 Stage3 可直接消费的 DPO pair 数据。

重构后的 Stage2 必须满足：

1. actor rollout 与 teacher labeling 分 phase 执行，资源可以分别最大化。
2. Phase A 和 Phase B 都按批量提交、bounded inflight 调度，不允许默认单条串行。
3. 单样本失败只过滤并记录统计；只有系统性失败才使 Stage2 失败。
4. 最终 `answer_distill_pairs.jsonl` 可被 Stage3 DPO 直接读取。
5. 配置默认就是多 replica 高吞吐配置，不保留低效单实例默认。

## 3. Stage2 最终产物契约

Stage2 最终必须写出：

```text
outputs/stages/train_agent/spad_rag/answer_refresh_data/
  answer_refresh_actor_trajectories.jsonl
  refresh_rollouts.jsonl
  answer_distill_pairs.jsonl
  answer_distill_dataset_manifest.json
  stage2_stats.json
  stage2_resource_monitor.jsonl
```

### 3.1 Phase A 产物

`answer_refresh_actor_trajectories.jsonl` 每行是一条 query 的 actor rollout 结果。所有样本都写入，包括失败和跳过样本。

必需字段：

```json
{
  "index": 0,
  "status": "completed|skipped|failed",
  "skip_reason": null,
  "question": "...",
  "gold_answers": ["..."],
  "messages_before_final_answer": [],
  "actor_final": "...",
  "actor_answer": "...",
  "rejected": "...",
  "evidence_steps": [],
  "search_count": 0,
  "sub_queries": [],
  "actor_elapsed_s": 0.0,
  "errors": []
}
```

字段要求：

1. `messages_before_final_answer` 是 actor 生成 final answer 前的完整 chat messages，用于 Stage3 prompt 构造。
2. `actor_answer` 和 `rejected` 指向同一份 actor final answer 文本。
3. `evidence_steps` 保存 teacher 可见 evidence，包括每轮 sub query 和 visible top docs。
4. 失败样本也必须包含 `index/question/status/skip_reason/errors`，方便复跑和统计。

### 3.2 Phase B 产物

`refresh_rollouts.jsonl` 是完整 teacher 处理记录，包括 kept、skipped、failed。

`answer_distill_pairs.jsonl` 只包含 Stage3 可用样本。每行必须包含：

```json
{
  "index": 0,
  "question": "...",
  "gold_answers": ["..."],
  "messages_before_final_answer": [],
  "chosen": "...",
  "rejected": "...",
  "evidence_steps": [],
  "teacher_reason": "...",
  "teacher_f1": 0.0,
  "filter_status": "kept"
}
```

Stage3 DPO 硬约束：

1. `messages_before_final_answer` 非空且是 list。
2. `chosen` 非空。
3. `rejected` 非空。
4. `chosen` 不允许包含 `<status>`。
5. `chosen` 与 `rejected` 不应完全相同；完全相同则过滤并计数。
6. `answer_distill_pairs.jsonl` 为空时 Stage2 失败，避免 Stage3 才报错。

`answer_distill_dataset_manifest.json` 至少包含：

```json
{
  "dataset_jsonl": ".../answer_distill_pairs.jsonl",
  "total": 200,
  "kept": 0,
  "phase_a_output": ".../answer_refresh_actor_trajectories.jsonl",
  "phase_b_output": ".../refresh_rollouts.jsonl",
  "stats_json": ".../stage2_stats.json"
}
```

## 4. 失败过滤和统计

单样本错误不使整个 Stage2 失败。样本级失败必须写入 JSONL，并进入统计。

### 4.1 Phase A 样本级过滤

过滤/失败原因：

1. `actor_no_finish`：actor 未输出合法 final answer。
2. `actor_missing_tool_call`：需要搜索但没有有效 tool call。
3. `actor_invalid_tool_call`：tool call JSON 或字段非法。
4. `no_search_evidence`：无可用 evidence。
5. `actor_timeout`：actor 请求超时。
6. `search_timeout`：search 请求超时。
7. `search_error`：retrieval 服务返回错误。
8. `max_turns_exceeded`：达到最大轮数仍未完成。

Phase A 统计字段：

```json
{
  "phase": "trajectory_rollout",
  "total": 200,
  "trajectory_completed": 0,
  "skipped_actor_no_finish": 0,
  "skipped_actor_missing_tool_call": 0,
  "skipped_actor_invalid_tool_call": 0,
  "skipped_no_search_evidence": 0,
  "actor_errors": 0,
  "search_errors": 0,
  "timeout_errors": 0,
  "written_trajectories": 0,
  "avg_search_count": 0.0,
  "avg_actor_elapsed_s": 0.0
}
```

### 4.2 Phase B 样本级过滤

过滤/失败原因：

1. `no_actor_final`：Phase A 没有 actor final answer。
2. `no_evidence`：Phase A 没有 evidence。
3. `teacher_format_error`：teacher 输出无法解析为 `<reason>/<answer>`。
4. `teacher_answer_empty`：chosen answer 为空。
5. `teacher_answer_has_status`：chosen answer 含 `<status>`。
6. `teacher_f1_below_threshold`：teacher answer 与 gold F1 不达标。
7. `chosen_equals_rejected`：chosen 与 rejected 完全相同。
8. `teacher_timeout`：teacher 请求超时。
9. `teacher_error`：teacher 服务错误。
10. `schema_invalid_pair`：写入 DPO pair 前 schema 校验失败。

Phase B 统计字段：

```json
{
  "phase": "teacher_labeling",
  "total_trajectories": 200,
  "eligible_for_teacher": 0,
  "teacher_completed": 0,
  "kept": 0,
  "skipped_no_actor_final": 0,
  "skipped_no_evidence": 0,
  "skipped_teacher_format": 0,
  "skipped_evidence_insufficient": 0,
  "skipped_teacher_f1": 0,
  "skipped_chosen_equals_rejected": 0,
  "teacher_errors": 0,
  "timeout_errors": 0,
  "schema_invalid_pairs": 0,
  "avg_teacher_elapsed_s": 0.0
}
```

### 4.3 Stage2 系统级失败条件

以下情况才使 Stage2 失败：

1. actor/teacher/recall 服务启动失败。
2. 所有 actor replica 不可用。
3. 所有 teacher replica 不可用。
4. Phase A 输入 dataset 为空或读取失败。
5. Phase A 完成后无任何可进入 Phase B 的 trajectory。
6. Phase B 完成后 `answer_distill_pairs.jsonl` kept 数为 0。
7. resume JSONL 损坏且无法定位到安全续跑点。
8. manifest 或 stats 文件写入失败。

## 5. 配置设计

需要修改的配置文件：

1. `AgenticIterRag/config/agent_training/spad_rag_base.yaml`
2. `AgenticIterRag/config/resource/local_8gpu_0_7.yaml`
3. `tasks/train_tasks/agenticIterRag/configs/spad_qwen3_1_7b_glm47_formal_overlay.yaml`

不新增旧版兼容 overlay，不保留单实例 Stage2 配置。

### 5.1 Stage2 phase 配置

在 `agent_training.sub_stages.answer_refresh_data` 下新增：

```yaml
phase_order:
  - trajectory_rollout
  - teacher_labeling
resume_from_phase: null
stop_after_phase: null
resume_existing: true
```

`resume_from_phase` 用于下一次 smoke 直接从 Stage2 的指定 phase 开始。

`stop_after_phase` 用于只跑 Phase A 或只跑 Phase B 的消融。

`resume_existing` 为 true 时，已经写出的 JSONL 按 `index` 去重续跑。

### 5.2 Phase A scheduler 配置

```yaml
phases:
  trajectory_rollout:
    scheduler:
      trajectory_submit_batch_size: 96
      inflight_per_actor: 16
      max_inflight_per_actor: 24
      retrieval_query_batch_size: 32
      retrieval_flush_interval_ms: 50
      progress_log_interval: 20
      request_timeout_s: 180
```

含义：

1. `trajectory_submit_batch_size=96`：一次向调度器提交 96 条 query。
2. `inflight_per_actor=16`：每个 actor replica 默认保持 16 条并发 trajectory。
3. `max_inflight_per_actor=24`：允许短时间冲到 24，用于吸收 search/后处理抖动。
4. 6 个 actor replica 时，目标 actor inflight 为 96，峰值 144。
5. `retrieval_query_batch_size=32`：search proxy 聚合子查询后批量送 recall。
6. `retrieval_flush_interval_ms=50`：不足 batch 时 50ms 刷出，避免长尾阻塞。

### 5.3 Phase B scheduler 配置

```yaml
phases:
  teacher_labeling:
    scheduler:
      teacher_submit_batch_size: 32
      inflight_per_teacher: 4
      max_inflight_per_teacher: 6
      progress_log_interval: 20
      request_timeout_s: 240
```

含义：

1. `teacher_submit_batch_size=32`：一次向 teacher 调度器提交 32 条 eligible trajectory。
2. `inflight_per_teacher=4`：每个 teacher replica 默认保持 4 条并发请求。
3. `max_inflight_per_teacher=6`：允许短时峰值并发。
4. 4 个 teacher replica 时，目标 teacher inflight 为 16，峰值 24。
5. teacher 是大模型 TP=2，单请求成本高，默认并发低于 actor。

### 5.4 Resource 配置

Phase A 资源：

```yaml
services:
  actor_vllm:
    replicas:
      - name: spad-refresh-actor-0
        gpu_ids: [0]
        port: 8340
        tensor_parallel_size: 1
      - name: spad-refresh-actor-1
        gpu_ids: [1]
        port: 8341
        tensor_parallel_size: 1
      - name: spad-refresh-actor-2
        gpu_ids: [2]
        port: 8342
        tensor_parallel_size: 1
      - name: spad-refresh-actor-3
        gpu_ids: [3]
        port: 8343
        tensor_parallel_size: 1
      - name: spad-refresh-actor-4
        gpu_ids: [4]
        port: 8344
        tensor_parallel_size: 1
      - name: spad-refresh-actor-5
        gpu_ids: [5]
        port: 8345
        tensor_parallel_size: 1
    common:
      max_num_seqs: 32
      max_num_batched_tokens: 32768
      gpu_memory_utilization: 0.80
      enable_prefix_caching: true
      enable_chunked_prefill: true
```

Phase A recall 资源沿用双 worker：

```yaml
services:
  recall_service:
    accelerator_backend:
      gpu_ids: [6, 7]
    instance_count: 2
    query_batch_size: 32
```

Phase B 资源：

```yaml
services:
  teacher_answerer:
    replicas:
      - name: spad_glm47_vllm_8067
        gpu_ids: [0, 1]
        port: 8067
        tensor_parallel_size: 2
      - name: spad_glm47_vllm_8068
        gpu_ids: [2, 3]
        port: 8068
        tensor_parallel_size: 2
      - name: spad_glm47_vllm_8069
        gpu_ids: [4, 5]
        port: 8069
        tensor_parallel_size: 2
      - name: spad_glm47_vllm_8070
        gpu_ids: [6, 7]
        port: 8070
        tensor_parallel_size: 2
```

资源切换方式：

1. Phase A 启动 actor replicas + recall workers。
2. Phase A 完成后关闭 actor 和 recall。
3. Phase B 启动 teacher replicas。
4. Phase B 完成后关闭 teacher。

由于 actor 和 teacher 不同时运行，可以在 Phase B 把 8 张 NPU 全部给 teacher 使用。

### 5.5 配置校验

启动 Stage2 前必须校验：

1. `actor_vllm.replicas` 存在且数量大于 0。
2. `teacher_answerer.replicas` 存在且数量大于 0。
3. 不接受旧的 `actor_vllm.gpu_ids + actor_vllm.port` 单实例配置。
4. 不接受旧的 `teacher_answerer.gpu_ids + teacher_answerer.port` 单实例配置。
5. actor replica 端口不重复。
6. teacher replica 端口不重复。
7. teacher container name 不重复。
8. replica `tensor_parallel_size` 与 `gpu_ids` 数量一致。
9. `trajectory_submit_batch_size >= actor_replica_count * inflight_per_actor`，否则报警或失败。
10. `teacher_submit_batch_size >= teacher_replica_count * inflight_per_teacher`，否则报警或失败。

## 6. 调度实现设计

### 6.1 ServiceManager 改造

文件：

- `AgenticIterRag/agentic_iter_rag/agent_training/spad/service_manager.py`

新增：

1. `start_actor_vllm_replicas(...)`
2. `start_teacher_replicas(...)`
3. `stop_service_group(group_name)`
4. replica config 校验函数。

旧的 Stage2 单实例启动函数不再被 Stage2 调用。若代码仍进入旧路径，直接报错。

### 6.2 Phase A 调度

文件：

- `AgenticIterRag/agentic_iter_rag/agent_training/spad/refresh_rollout.py`

新增：

1. `_run_trajectory_rollout_phase(...)`
2. `ActorReplicaPool`
3. `RetrievalBatcher`
4. `JsonlAppendWriter`
5. `Stage2StatsCollector`

调度策略：

1. 读取输入 query，按 `trajectory_submit_batch_size` 切块。
2. 建立 bounded queue，目标全局 inflight 为 `actor_replica_count * inflight_per_actor`。
3. 使用 least-inflight 分配 actor replica。
4. 每个 trajectory 在同一个 actor replica 上完成多轮生成，避免一次 trajectory 在不同模型实例之间切换。
5. search 请求进入 `RetrievalBatcher`，按 `retrieval_query_batch_size` 或 `retrieval_flush_interval_ms` 刷出。
6. 每完成一个 query 立即追加写 `answer_refresh_actor_trajectories.jsonl`。
7. 日志每 `progress_log_interval` 条打印吞吐、成功数、失败数、平均 search count、actor inflight、recall queue 长度。

### 6.3 Phase B 调度

新增：

1. `_run_teacher_labeling_phase(...)`
2. `TeacherReplicaPool`
3. `DpoPairValidator`

调度策略：

1. 流式读取 `answer_refresh_actor_trajectories.jsonl`。
2. 先做本地 eligibility 过滤：无 final answer、无 evidence、schema 缺失直接跳过。
3. eligible 样本按 `teacher_submit_batch_size` 提交。
4. 使用 least-inflight 分配 teacher replica。
5. teacher 输出后复用现有 `_extract_teacher_result`、F1 和 filter 逻辑。
6. 每条样本写 `refresh_rollouts.jsonl`。
7. kept 样本写 `answer_distill_pairs.jsonl`。
8. 所有 kept 样本写入前必须通过 `DpoPairValidator`。

## 7. Stage3 兼容性

Stage3 DPO 当前依赖：

- `messages_before_final_answer`
- `chosen`
- `rejected`

重构后的 `answer_distill_dataset_manifest.json` 必须继续设置：

```json
{
  "outputs": {
    "dataset_jsonl": ".../answer_distill_pairs.jsonl"
  }
}
```

或者兼容当前 reader 支持的：

```json
{
  "dataset_jsonl": ".../answer_distill_pairs.jsonl"
}
```

推荐两者都写，降低后续改动风险。

Phase B 完成后立刻执行 schema validation：

1. 逐行读取 `answer_distill_pairs.jsonl`。
2. 统计 invalid row，不因单行 invalid 直接终止前面处理。
3. invalid row 不写入最终 DPO pair。
4. 如果最终 kept 为 0，则 Stage2 失败。
5. 校验报告写入 `stage2_stats.json`。

## 8. 资源监控

Stage2 运行期间写：

```text
docs/AgenticIterRag_v1/work_report/260710_spad_stage2_resource_monitor.md
outputs/stages/train_agent/spad_rag/answer_refresh_data/stage2_resource_monitor.jsonl
```

采样字段：

```json
{
  "ts": "2026-07-10T00:00:00+08:00",
  "phase": "trajectory_rollout",
  "actor_posts": 0,
  "teacher_posts": 0,
  "retrieval_posts": 0,
  "npu": [
    {"id": 0, "hbm_mb": 0, "aicore_util": 0}
  ]
}
```

Phase A 重点观察：

1. actor replicas 的 HBM 是否均衡。
2. actor AICore 是否由之前接近 0% 提升到有持续利用率。
3. recall NPU6/7 是否成为瓶颈。
4. search queue 是否积压。

Phase B 重点观察：

1. 四个 teacher TP=2 replica 是否都在服务请求。
2. teacher AICore 是否比单实例阶段显著提升。
3. teacher request latency 和 kept rate。

## 9. 实施步骤

1. 修改配置：
   - 覆盖 Stage2 单实例 actor/teacher 配置为 replicas。
   - 增加 phase、scheduler、filter、resume 配置。
   - 增加配置校验。

2. 改造 ServiceManager：
   - 支持 actor replicas。
   - 支持 teacher replicas。
   - 支持 group stop，保证 Phase A 和 Phase B 资源切换干净。

3. 改造 `refresh_rollout.py`：
   - 拆分 Phase A 和 Phase B。
   - 实现 actor pool、teacher pool、retrieval batcher。
   - 实现 JSONL 追加、resume、去重。
   - 实现样本级失败过滤和统计。

4. 增加 Stage3 schema validator：
   - kept 样本写入前校验。
   - kept=0 作为系统级失败。

5. 增加测试：
   - 旧单实例配置应失败。
   - 单条 actor 失败不会导致 Stage2 失败。
   - 单条 teacher format error 不会导致 Stage2 失败。
   - invalid pair 被过滤并计数。
   - manifest 可被 Stage3 DPO reader 读取。

6. 执行 smoke：
   - 直接从 Stage2 开始。
   - actor 模型使用上一次 Stage1 产出的 `global_step_2`。
   - Stage2 使用 200 query。
   - Stage3 只跑 DPO phase，1 epoch。

## 10. 验收标准

功能验收：

1. Stage2 能从 Stage1 `global_step_2` actor checkpoint 直接启动。
2. Phase A 生成 `answer_refresh_actor_trajectories.jsonl`。
3. Phase B 生成 `refresh_rollouts.jsonl`、`answer_distill_pairs.jsonl`、`answer_distill_dataset_manifest.json`。
4. Stage3 DPO 能读取 Stage2 manifest 并进入训练。

效率验收：

1. actor 不再只有 NPU0 工作，Phase A actor HBM 分布到 NPU0-5。
2. Phase A actor 请求吞吐显著高于旧单实例串行。
3. Phase B teacher 使用 4 个 TP=2 replica。
4. Stage2 资源监控报告记录 HBM 和 AICore 利用率。

数据质量验收：

1. stats 能解释每类样本去向。
2. `answer_distill_pairs.jsonl` 只包含 kept DPO pair。
3. schema invalid pair 为 0，或全部被过滤且有计数。
4. kept=0 时 Stage2 明确失败。

## 11. 下次 smoke 命令约束

下次 smoke 从 Stage2 开始：

1. 使用上一次 Stage1 产出的 actor checkpoint：

```text
checkpoints/AIR/260709-231614-279722-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_e2e_stage1_2step_stage2_200_stage3_dpo1epoch/stages/train_agent/spad_rag/search_policy_rl/actor_model_verl/global_step_2
```

2. Stage2 仍只使用 200 query。
3. Stage3 仍只跑 DPO phase，1 epoch。
4. 不改动其他训练和推理参数。

## 12. 当前旧实现必须替换的点

必须删除或绕开以下低效行为：

1. Stage2 启动单个 actor vLLM。
2. Stage2 在一个 for-loop 内串行处理 query。
3. actor rollout 后立即同步调用 teacher。
4. 只有处理完全部样本后才写最终文件。
5. 配置仍允许 `actor_vllm.gpu_ids/port` 作为 Stage2 默认路径。
6. 配置仍允许 `teacher_answerer.gpu_ids/port` 作为 Stage2 默认路径。

替换后的默认行为必须是：

1. Phase A：6 actor replicas + recall workers + bounded inflight + batched retrieval。
2. Phase B：4 teacher replicas + bounded inflight。
3. 样本级失败过滤、完整统计、最终 kept DPO pair manifest。
