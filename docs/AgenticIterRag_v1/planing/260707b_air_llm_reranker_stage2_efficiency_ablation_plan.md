# AIR LLM Reranker Stage2 训练效率消融计划

## 1. 这次消融到底想解决什么

这次消融只针对 AIR LLM reranker 的 `stage2_agentic` 训练阶段。

说得直接一点：现在 stage2 的训练链路已经不是单纯让 reranker 生成一个排序，然后本地算格式分。它后面还要接 frozen search agent continuation rollout，再通过 agent 最终答案算 reward。所以 stage2 的单步耗时会同时受下面几件事影响：

1. reranker actor/rollout 自己生成排序的速度。
2. reranker 输出是否短、是否能及时命中 `</rerank>` stop sequence。
3. reward 侧 frozen agent continuation 的请求并发能力。
4. retriever 是否成为 continuation search 的瓶颈。
5. PPO/ref/update_actor 这些训练侧 forward/backward 是否有 offload、显存峰值和 token 动态合批问题。

这次消融的目标不是“先把 batch 降下来跑得快一点”，而是在固定底线参数的前提下，把真实训练效率调上去。

固定底线是：

1. `train_batch_size = 64`
2. `rollout_n = 8`
3. 每个 step 的 reranker rollout 数量是 `64 * 8 = 512`
4. `max_response_length = 512`
5. stage1 和 stage2 使用同一版 CoSearch 对齐 prompt

当前优先级是：

1. 不降低 `batch_size=64` 和 `rollout_n=8`。
2. 尽量关闭 actor 侧 offload，减少 CPU/NPU 来回搬运。
3. 尽量提高 reranker rollout vLLM 的显存利用率。
4. 在不 OOM 的前提下，让 NPU 计算利用率尽量打满。
5. 每轮消融只跑 5 个 step，先看趋势，不做长时间盲跑。

## 2. 这次消融不做什么

这篇文档只记录 stage2 训练效率消融，不讨论下面这些事情：

1. 不重新设计 reward 语义。
2. 不重新跑 `generate_traces`。
3. 不重新构建已有全量 branch dataset，除非后面确认 prompt 或 schema 需要重新物化。
4. 不把 `train_batch_size` 降到 64 以下。
5. 不把 `rollout_n` 降到 8 以下。
6. 不展开 stage1 format reward 训练效果讨论。
7. 不把 CPU retriever 作为本轮 stage2 默认方案，因为当前 stage2 优先验证 NPU retriever + frozen agent 并发池。

## 3. 当前 prompt 和数据状态

当前 AIR LLM reranker prompt 已经改成和 CoSearch reranker 行为对齐：

1. 输入是 top50 candidate passages。
2. 输出只要求 top5 index。
3. 输出格式仍然是 `<reason>...</reason>` 和 `<rerank>...</rerank>`。
4. `<rerank>` 中要求输出 exactly 5 个不同 index。
5. index 范围仍然是 `[1, 50]`。
6. AIR 只比 CoSearch 原版 prompt 多一句限制：`Do NOT analyze all 50 candidate passages one by one.`

当前配置里使用的 prompt 版本是：

```text
cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example
```

当前 stage2 消融使用的数据集 manifest 是：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example/manifest.json
```

当前数据规模：

```text
5100 branch samples
```

2026-07-08 已修正 prompt 示例：旧 prompt 中示例 `<rerank>` 出现 `[233]`、`[105]`、`[729]`、`[688]` 等越界 index，会污染模型对合法输出范围的学习。当前 fixed_example 版本示例为：

```text
<rerank>[27] > [23] > [10] > [7] > [6]</rerank>
```

并且已经重新物化 branch dataset。stage2 消融直接消费 fixed_example manifest，不再使用旧 `no_analyze50_eval` 数据集。

## 4. 当前正在跑的有效消融 run

当前有效的 stage2 5-step 消融 run 是：

```text
260707-215901-978960-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_5step
```

pipeline 日志目录：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/agenticIterRag/agenticIterRag/260707-215901-978960-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_5step
```

pipeline 输出目录：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/agenticIterRag/260707-215901-978960-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_5step
```

VERL 训练日志：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/agenticIterRag/260707-215901-978960-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_5step/stages/train_llm_reranker/runtime_services/stage2_agentic/verl_train.log
```

训练报告目录：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/outputs/agenticIterRag/agenticIterRag/260707-215901-978960-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_5step/stages/train_llm_reranker/training_reports/stage2_agentic
```

rollout 明细目录：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/agenticIterRag/agenticIterRag/260707-215901-978960-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_5step/runtime_logs/train_llm_reranker/stage2_agentic/rollout_data
```

当前观察到的状态：

1. 进程仍在。
2. stage2 的 VERL 命令已经启动。
3. `metrics.jsonl` 暂时还没有产出，说明还没有完成一个可记录的 training step。
4. stage manifest 暂时还没有写出，说明训练 stage 尚未结束。
5. 日志里已经看到 `step=1 started total=512 workers=64`。
6. 已看到 `step=1 trajectories=416/512 prompts~=52/64 workers_done=52/64 elapsed_s=136.7`。
7. 后续日志主要是 tokenizer warning 和 profiler warning，暂时没有完整 step 的 `timing_s/*` 指标。

所以当前不能把这轮消融称为“已完成”。它只是第一个有效 stage2 消融 run，正在跑，尚未给出完整 5-step 结果。

## 5. 当前消融 overlay

当前 5-step 消融 overlay 是：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/agenticIterRag/configs/stage2_ablation_reranker_efficiency_5step_overlay.yaml
```

这个 overlay 的意图很明确：

1. 只跑 `train_llm_reranker`。
2. 关闭 `stage1_format`。
3. 只开启 `stage2_agentic`。
4. 不重跑数据生产和数据集构建。
5. 使用当前 CoSearch topM prompt 的 branch dataset。
6. 每轮真实训练只跑 5 step。

核心训练参数：

```yaml
train_batch_size: 64
ppo_mini_batch_size: 64
n_samples_per_prompt: 8
val_n_samples_per_prompt: 8
total_training_steps: 5
total_epochs: 1
max_prompt_length: 16384
max_response_length: 512
rollout_max_model_len: 16896
```

核心 offload 参数：

```yaml
actor_activation_offload: false
actor_optimizer_offload: false
actor_param_offload: false
ref_param_offload: true
```

核心 vLLM rollout 参数：

```yaml
rollout_gpu_memory_utilization: 0.80
max_num_seqs: 160
max_num_batched_tokens: 98304
enable_chunked_prefill: true
enable_prefix_caching: true
enforce_eager: false
sampling_stop:
  - "</rerank>"
include_stop_str_in_output: true
```

核心采样参数：

```yaml
rollout_temperature: 0.1
rollout_top_p: 0.6
```

核心 KL 参数：

```yaml
use_kl_loss: true
kl_loss_coef: 0.001
kl_loss_type: low_var_kl
entropy_coeff: 0.0
```

当前 reward 是 stage2 的 agentic rollout reward：

```text
agentic_rag_rollout_reward
```

也就是 reranker 输出排序后，系统会：

1. parser 检查 `<reason>` 和 `<rerank>` 格式。
2. 把 top5 index 映射回真实 doc。
3. 替换当前 search step 的 tool observation。
4. frozen agent 从该位置继续 rollout。
5. continuation 后续 search 仍只用 retriever。
6. 拿 frozen agent 的最终 answer 算 reward。

## 6. 当前资源布局

当前 stage2 消融资源布局：

```text
reranker actor/rollout: NPU 0,1,2,3
frozen agent pool:     NPU 4,5,6
retriever:             NPU 7
```

frozen agent 不是 TP=3，而是 3 个单卡 vLLM 实例：

```text
frozen_agent_0 -> NPU 4 -> port 8141
frozen_agent_1 -> NPU 5 -> port 8142
frozen_agent_2 -> NPU 6 -> port 8143
```

前面挂一个统一 proxy：

```text
http://127.0.0.1:8140
```

proxy 使用：

```text
least_inflight
```

这样 reward 侧只看到一个 agent 地址，内部由 proxy 把 continuation 请求分发到 3 个 frozen agent 实例上。

retriever 当前是单 NPU 实例：

```text
NPU 7 -> backend 8131 -> proxy 8130
```

当前 NPU 观察到的状态大致是：

1. NPU 0-3 是 reranker actor/rollout/ref/update 的主训练侧。
2. NPU 4-6 是 frozen agent vLLM 实例，HBM 占用高，AICore 在 continuation 请求到来时才会动。
3. NPU 7 是 retriever，HBM 占用约 35GB 量级。
4. 当前没有 OOM。

## 7. 当前已经观察到的问题

### 7.1 step 级指标还没出来

当前有效 run 还没有完成一个完整 training step，因此暂时没有这些关键指标：

1. `timing_s/gen`
2. `timing_s/reward`
3. `timing_s/ref`
4. `timing_s/update_actor`
5. `response_length/mean`
6. `response_length/clip_ratio`
7. `reward/mean`
8. format pass rate

也就是说，现在不能直接判断“这轮配置每步到底多少秒”。只能说前半段 rollout 进度比之前 `rollout_n=4` 的失控 step 明显更正常，但还要等完整 step 指标。

### 7.2 仍然存在长尾风险

日志里 64 个 worker 前半段完成很快，但后面 worker 明显拉开差距。这个现象可能来自：

1. 部分 reranker 输出较长，接近 512。
2. 部分 reranker 输出格式不稳定，导致 parser 或 reward 侧路径不一致。
3. 部分 continuation 请求进入 frozen agent 后执行了更多 assistant/tool turns。
4. 单 retriever 在某些 search-heavy continuation 上形成排队。
5. Ray worker / tokenizer 初始化或 warning 噪声影响观测。

后续必须用 rollout 明细和 `timing_s/reward` 拆分确认，不应只根据整体耗时猜。

### 7.3 日志里有 Triton kernel warning

日志中有类似：

```text
Failed to import Triton kernels. No module named 'triton.language.target_info'
```

这目前没有导致训练失败，但它说明当前环境没有使用某些 Triton kernel 路径。这个问题是否影响 NPU 路径性能，需要单独确认，不能直接当成 stage2 慢的唯一根因。

### 7.4 tokenizer warning 很多

日志中有大量 tokenizer warning。它们不一定是主耗时，但会污染日志，也说明 worker 侧可能在反复做 tokenizer 相关初始化或调用。

后续如果发现 `reward` 或 worker 端 CPU 时间异常，要专门看 tokenizer 调用是否在 continuation reward 中被重复放大。

## 8. 消融指标怎么读

每轮 5-step 消融至少要记录这些指标：

1. 是否 OOM。
2. 是否完成 5 step。
3. 每个 step 总耗时。
4. `timing_s/gen`。
5. `timing_s/reward`。
6. `timing_s/ref`。
7. `timing_s/update_actor`。
8. 平均 response length。
9. response clip ratio。
10. format pass rate。
11. reward mean。
12. NPU 0-3 的 AICore 与 HBM。
13. NPU 4-6 的 AICore 与 HBM。
14. NPU 7 的 AICore 与 HBM。
15. frozen agent proxy 是否有排队或 timeout。
16. retriever proxy 是否有排队或 timeout。

判断标准：

1. 如果 `timing_s/gen` 占主导，优先调 reranker rollout vLLM 和生成策略。
2. 如果 `timing_s/reward` 占主导，优先调 frozen agent 并发、continuation turns、retriever 并发。
3. 如果 `timing_s/ref` 占主导，优先调 ref offload/logprob micro batch/token len。
4. 如果 `timing_s/update_actor` 占主导，优先调 actor FSDP、dynamic batch、micro batch、checkpointing/offload。
5. 如果 HBM 没满但 AICore 低，优先查等待、调度、服务瓶颈。
6. 如果 HBM 接近满且 OOM，才回退 offload 或降低局部并发参数，不动 batch/rollout 底线。

## 9. 当前 baseline 消融

### 9.1 baseline-A：无 actor offload + rollout vLLM 0.80

当前正在跑的是 baseline-A：

```text
train_batch_size=64
rollout_n=8
max_response_length=512
actor_activation_offload=false
actor_optimizer_offload=false
actor_param_offload=false
ref_param_offload=true
rollout_gpu_memory_utilization=0.80
max_num_seqs=160
max_num_batched_tokens=98304
frozen_agent_instances=3
retriever_instances=1
```

当前结论只能写到这里：

1. 启动成功。
2. 真实 NPU 训练进程在跑。
3. 没有看到 OOM。
4. step1 尚未完成，所以不能给出完整单步耗时。
5. NPU 0-3 的 HBM 和 AICore 已经较高，说明不是完全没打上计算。
6. 是否真正高效，还要等 step1 完整 timing。

### 9.2 baseline-A 的验收方式

baseline-A 必须至少完成 1 个完整 step 才能判断：

1. 如果 step1 完成且总耗时明显低于之前异常 step，继续跑到 5 step。
2. 如果 step1 卡住超过可接受时间，要先看 worker 明细和服务日志，不要盲目等。
3. 如果 step1 的 `reward` 时间明显高于 `gen`，下一轮优先调 frozen agent/retriever。
4. 如果 step1 的 `gen` 时间明显高于 `reward`，下一轮优先调 reranker rollout。
5. 如果 step1 OOM，下一轮只回退最小必要参数。

## 10. 后续消融计划

下面的计划按优先级排序。每轮仍然固定：

```text
train_batch_size=64
rollout_n=8
max_response_length=512
total_training_steps=5
```

### 10.1 消融 B：提高 rollout vLLM 显存利用率

目标是确认 reranker rollout 侧还能不能继续吃满 0-3 卡。

候选修改：

```yaml
rollout_gpu_memory_utilization: 0.85
max_num_seqs: 192
max_num_batched_tokens: 131072
```

观察重点：

1. 是否 OOM。
2. `timing_s/gen` 是否下降。
3. NPU 0-3 AICore 是否更稳定。
4. response clip ratio 是否没有恶化。

如果 0.85 稳定，再试：

```yaml
rollout_gpu_memory_utilization: 0.90
max_num_seqs: 224
max_num_batched_tokens: 131072
```

但 0.90 有更高 OOM 风险，要优先观察 NPU 0 和 NPU 3 的 HBM 峰值。

### 10.2 消融 C：rollout 调度模式

当前：

```yaml
enforce_eager: false
enable_chunked_prefill: true
enable_prefix_caching: true
```

下一轮可以比较：

```yaml
enforce_eager: true
```

原因是 5-step 短消融里，图捕获或编译带来的前期成本可能不划算。但正式长训不一定应该开 `enforce_eager=true`，要看 5-step 结果和后续长训收益。

观察重点：

1. step1 启动时间是否下降。
2. step2-step5 是否比 baseline 更快。
3. 训练稳定性是否受影响。

### 10.3 消融 D：进一步收紧生成随机性

当前：

```yaml
rollout_temperature: 0.1
rollout_top_p: 0.6
sampling_stop:
  - "</rerank>"
```

如果仍然有大量输出接近 512，下一轮可以试：

```yaml
rollout_temperature: 0.0
rollout_top_p: 1.0
```

这会让生成更确定，可能减少无效长尾，但也可能降低 GRPO group 内差异。因为 stage2 是 reward 训练，不能无脑固定成完全 greedy，必须看 reward 方差和 format pass rate。

如果 `temperature=0.0` 导致 group 内 reward 方差太低，就回退：

```yaml
rollout_temperature: 0.05
rollout_top_p: 0.8
```

### 10.4 消融 E：frozen agent continuation 并发

如果 baseline-A 显示 `timing_s/reward` 是主耗时，下一轮优先调 frozen agent。

当前：

```yaml
frozen_agent_instances: 3
frozen_agent_max_num_seqs: 24
frozen_agent_gpu_memory_utilization: 0.90
AIR_CONTINUATION_BATCH_WORKERS: 64
```

候选修改：

```yaml
frozen_agent_max_num_seqs: 32
frozen_agent_gpu_memory_utilization: 0.94
```

观察重点：

1. NPU 4-6 是否 OOM。
2. frozen agent AICore 是否更高。
3. `timing_s/reward` 是否下降。
4. continuation timeout 是否增加。

如果 3 个 frozen agent 仍然不够，原则上不把 NPU 7 从 retriever 拿回来，先看 retriever 是否真的是瓶颈。如果 retriever 不瓶颈但 frozen agent 明显瓶颈，再讨论是否把 retriever 改 CPU 或混合模式，而不是直接回到 2 retriever 方案。

### 10.5 消融 F：continuation turns 上限

如果 `reward` 慢来自 frozen agent 后续多轮搜索，可以消融 continuation 最大轮数。

当前：

```text
AIR_CONTINUATION_MAX_ASSISTANT_TURNS=6
AIR_CONTINUATION_MAX_USER_TURNS=6
AIR_CONTINUATION_MAX_RESPONSE_LENGTH=1024
AIR_CONTINUATION_MAX_TOOL_RESPONSE_LENGTH=4096
```

候选修改：

```text
AIR_CONTINUATION_MAX_ASSISTANT_TURNS=4
AIR_CONTINUATION_MAX_USER_TURNS=4
```

这个参数会改变 reward 环境的能力上限，所以它不是纯性能参数。只有在确认 reward 阶段严重拖慢，并且大多数 continuation 不需要 6 轮时，才把它作为正式默认。

### 10.6 消融 G：retriever 瓶颈确认

当前 stage2 retriever 是单 NPU 实例：

```text
NPU 7
query_batch_size=32
doc_dtype=float16
```

如果 `reward` 慢，同时 frozen agent AICore 不高，而 retriever NPU 7 有明显高利用率或队列堆积，说明 retriever 可能成为瓶颈。

候选修改：

```yaml
query_batch_size: 64
```

如果单 retriever 仍然不够，再考虑 CPU retriever 或多实例 proxy；但当前不把 2 张 NPU 给 retriever，因为用户已经明确拒绝“2 frozen agent + 2 retriever”的方案。

### 10.7 消融 H：ref 和 logprob 侧

如果 `timing_s/ref` 或 logprob 时间明显高，下一轮看这些参数：

```yaml
ref_param_offload: true
log_prob_micro_batch_size_per_gpu: 1
log_prob_max_token_len_per_gpu: 18432
calculate_rollout_log_probs: true
use_rollout_log_probs: true
```

可能方向：

1. 保持 `calculate_rollout_log_probs=true`，避免 old logprob 二次 forward。
2. 如果 HBM 允许，尝试 `ref_param_offload=false`，看 ref logprob 是否加速。
3. 如果 OOM，再回退 `ref_param_offload=true`。

这轮不优先改，因为当前还没有完整 timing 证明 ref 是瓶颈。

## 11. 结果记录模板

每轮消融结束后按这个格式记录：

```text
run_id:
overlay:
start_time:
end_time:
completed_steps:
oom: yes/no
failed_reason:

train_batch_size:
rollout_n:
max_response_length:
rollout_gpu_memory_utilization:
max_num_seqs:
max_num_batched_tokens:
actor_activation_offload:
ref_param_offload:
frozen_agent_instances:
frozen_agent_max_num_seqs:
retriever_instances:

step_time_mean:
step_time_p50:
step_time_p95:
timing_s_gen_mean:
timing_s_reward_mean:
timing_s_ref_mean:
timing_s_update_actor_mean:
response_length_mean:
response_clip_ratio:
format_pass_rate:
reward_mean:

npu_0_3_hbm_peak:
npu_0_3_aicore_avg:
npu_4_6_hbm_peak:
npu_4_6_aicore_avg:
npu_7_hbm_peak:
npu_7_aicore_avg:

结论:
下一步:
```

## 12. 当前需要等出来的第一批结论

当前最重要的不是再改一轮参数，而是先拿到 baseline-A 的第一个完整 step timing。

拿到 step1 后，按下面逻辑决策：

1. `gen` 慢：优先做消融 B/C/D。
2. `reward` 慢：优先做消融 E/F/G。
3. `ref` 慢：优先做消融 H。
4. `update_actor` 慢：再看 actor micro batch、dynamic token batch、gradient checkpointing。
5. OOM：只回退局部显存参数，不能动 `batch=64` 和 `rollout_n=8`。

如果 baseline-A 5 step 能完整跑完，并且没有 OOM，就把它作为当前 stage2 正式训练候选基线，再基于单步耗时决定是否继续做 B/C/E。

如果 baseline-A 在 step1 长时间没有新进展，要先做故障定位：

1. 看 `rollout_data` 是否有新增样本。
2. 看 frozen agent proxy 是否有 pending 请求。
3. 看 retriever proxy/backend 是否有 pending 请求。
4. 看 Ray worker 是否有异常卡死。
5. 看是否某些 continuation request timeout 后没有返回。

不能在没有定位的情况下直接开下一轮消融，否则会把同一个卡点重复跑出来。

## 13. 和正式入口脚本的关系

当前正式 task 脚本是：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh
```

这个脚本当前默认 overlay 不是 5-step 消融 overlay，而是：

```text
tasks/train_tasks/agenticIterRag/configs/stage2_from_260707144331_data_260707072935_gs40_overlay.yaml
```

如果要跑当前 5-step 消融，需要显式设置：

```bash
AIR_RERANKER_OVERLAY=tasks/train_tasks/agenticIterRag/configs/stage2_ablation_reranker_efficiency_5step_overlay.yaml \
bash tasks/train_tasks/agenticIterRag/run_260703b_AIR_v1_from_dataprod_to_reranker_training.sh
```

这么设计的原因是：正式入口保持干净，消融参数放 overlay 里，不把一堆临时 CLI patch 塞进 task 脚本。

## 14. 文档维护要求

后续每跑完一轮 5-step 消融，都要把结果补到这篇文档或同目录新增结果文档里。

记录时要坚持两点：

1. 已经跑出来的指标写成结论。
2. 没跑完、没日志、没 timing 的内容只能写成判断或待验证，不能写成事实。

所有后续新增或修改的 YAML 配置项，都必须写中文注释，说明这个字段为什么存在、影响哪个阶段、默认值为什么这样设。注释风格继续参考 AIR 现有 YAML 和本次 `stage2_ablation_reranker_efficiency_5step_overlay.yaml`。

## 15. 2026-07-07 更新：后续消融固定 rollout_n=4

用户最新要求覆盖本文前面 `rollout_n=8` 的底线约束：从下一轮 stage2 训练效率消融开始，`rollout_n` 固定为 4。

因此后续固定底线调整为：

```text
train_batch_size = 64
rollout_n = 4
每个 step 的 reranker rollout 数量 = 64 * 4 = 256
max_response_length = 512
total_training_steps = 5
```

旧 n=8 baseline-A 已经完成第 1 个完整 step，关键指标如下：

```text
run_id: 260707-215901-978960-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_5step
completed_steps: 1
timing_s/step: 1204.824
timing_s/gen: 454.296
timing_s/reward: 192.221
timing_s/ref: 145.631
timing_s/update_actor: 408.917
response_length/mean: 287.8125
response_length/clip_ratio: 0.17578125
reward_mean: 0.3410
```

该 run 在 step2 进行中被终止，原因是继续执行 n=8 已不符合新的消融约束。后续不再用 n=8 作为目标配置，只保留它作为“过大 rollout_n 下单步耗时构成”的参考。

当前 5-step 消融 overlay 已改为：

```text
tasks/train_tasks/agenticIterRag/configs/stage2_ablation_reranker_efficiency_5step_overlay.yaml
experiment_name: agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_n4_5step
n_samples_per_prompt: 4
val_n_samples_per_prompt: 4
```

进入实际训练任务之后，人工/agent 观察间隔调整为每 3 分钟一次，不需要按秒级频繁跟踪。

## 16. n=4 baseline-0：保守采样 5-step 结果

本轮作为后续 n=4 消融的稳态基线：

```text
run_id: 260707-223334-353504-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_n4_5step
overlay: tasks/train_tasks/agenticIterRag/configs/stage2_ablation_reranker_efficiency_5step_overlay.yaml
completed_steps: 5
oom: no

train_batch_size: 64
rollout_n: 4
max_response_length: 512
rollout_gpu_memory_utilization: 0.80
max_num_seqs: 160
max_num_batched_tokens: 98304
actor_activation_offload: false
ref_param_offload: true
frozen_agent_instances: 3
retriever_instances: 1

rollout_temperature: 0.1
rollout_top_p: 0.6
enforce_eager: false
```

逐 step 关键指标：

```text
step1: step=766.017s gen=343.197s reward=137.023s ref=73.418s update_actor=210.488s resp_len=283.684 clip=0.140625 reward_mean=0.3860
step2: step=725.459s gen=347.699s reward=64.935s  ref=67.757s update_actor=243.231s resp_len=171.637 clip=0.0078125 reward_mean=-0.0555
step3: step=730.008s gen=344.035s reward=92.741s  ref=70.857s update_actor=220.528s resp_len=189.602 clip=0.0390625 reward_mean=0.4647
step4: step=789.541s gen=402.843s reward=88.561s  ref=68.748s update_actor=227.526s resp_len=212.145 clip=0.08203125 reward_mean=0.3389
step5: step=302.344s gen=39.115s  reward=0.061s   ref=54.866s update_actor=206.512s resp_len=29.063  clip=0.0 reward_mean=-0.5
```

step5 出现全样本 `reward_mean=-0.5`、response 极短、`pg_loss=0` 的退化现象，因此不能把 step5 当作有效效率提升。稳态比较使用 step2-step4：

```text
step_time_mean: 748.336s
timing_s_gen_mean: 364.859s
timing_s_reward_mean: 82.079s
timing_s_ref_mean: 69.121s
timing_s_update_actor_mean: 230.428s
response_length_mean: 191.128
response_clip_ratio: 0.042969
agent_loop_generate_sequences_max_mean: 351.732s
reward_mean: 0.2494
```

结论：

1. `rollout_n=4` 相比旧 n=8 单步压力明显下降，但主瓶颈仍然是 reranker rollout generation 长尾，其次是 actor update。
2. reward 阶段不再是第一瓶颈；本轮不优先改 frozen agent 和 retriever 并发。
3. NPU 0-3 HBM 已经较高，下一轮不直接提高 `rollout_gpu_memory_utilization`、`max_num_seqs` 或 `max_num_batched_tokens`。
4. step5 的退化说明“短输出导致变快”本身不是可接受目标；后续消融必须同时看 reward 分布和 response 长度，不能只看 step time。

### 16.1 temperature 消融已取消

曾启动过一轮确定性生成消融：

```text
run_id: 260707-234424-014136-pipeline-agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_n4_temp0_5step
rollout_temperature: 0.0
rollout_top_p: 1.0
```

该 run 在 step1 rollout 进行中被手动终止，没有产出完整 step metrics。终止原因是：降低 temperature 会显著降低 GRPO group 内采样差异，存在训练无效风险；因此不再继续消融 temperature，也不使用 temp0 结果作为效率结论。

当前 overlay 已恢复为与 CoSearch 训练 rollout 默认采样一致：

```text
experiment_name: agentic_iter_rag_v1_stage2_ablation_reranker_efficiency_n4_cosearch_sampling_5step
rollout_temperature: 1.0
rollout_top_p: 1.0
```

如果后续继续做效率优化，应只考虑非采样语义参数，并且不再启动新的 temperature/top_p 消融。

可选的非 temperature 消融方向，仅作为后续备选，不在当前请求中继续执行：

1. `enforce_eager=true`：验证短 5-step 场景下是否能减少 vLLM graph capture/编译前期开销；风险是稳态 step 可能变慢。
2. `max_num_seqs` / `max_num_batched_tokens`：在 HBM 允许时微调 reranker rollout 并发；风险是 0-3 卡已经接近显存上限，OOM 风险高。
3. `rollout_gpu_memory_utilization`：小幅上调或回退 vLLM KV cache 预算；必须先看 HBM 峰值，不能盲目增加。
4. `ref_param_offload=false`：如果 `timing_s/ref` 成为瓶颈，尝试取消 ref offload；风险是 actor/update 阶段 OOM。
5. actor/ref `*_max_token_len_per_gpu` 和 micro batch：优化 ref/update_actor 动态合批；风险是显存峰值和 step 稳定性。
6. frozen agent `max_num_seqs`：只有当 `timing_s/reward` 成为主瓶颈时才调；当前 n=4 baseline 里 reward 不是第一瓶颈。
7. retriever `query_batch_size`：只有当 reward 慢且 retriever 排队明显时才调；当前没有证据表明 retriever 是主瓶颈。

## 17. 2026-07-08 更新：fixed prompt 与 reward-bound 诊断

### 17.1 prompt 修正与数据重建

已完成的修正：

1. `AgenticIterRag/agentic_iter_rag/llm_reranker/format.py` 中的 CoSearch topM prompt 示例已改成合法 `[1,50]` index。
2. 默认 reranker training config 已改为 `cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example`。
3. stage2 训练 overlay 和 5-step 消融 overlay 都已改为 fixed_example branch manifest。
4. 新增只补产数据的 overlay：

```text
tasks/train_tasks/agenticIterRag/configs/rebuild_branch_260704e_fixed_prompt_overlay.yaml
```

已重新物化的数据集：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/llm_reranker_branch_train_set/260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_first_point_top50_top5_cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example/manifest.json
```

验证结果：

```text
sample_count: 5100
prompt_template_version: cosearch_rerank_topm_v1_plus_no_analyze50_fixed_example
candidate_top_n: 50
visible_top_m: 5
bad old example: not found
fixed example: found
```

### 17.2 reward-bound 诊断设置

诊断脚本：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/reward_bound_diagnosis.py
```

这次先不训练 reranker，只把固定合法 reranker action 接入真实 stage2 continuation reward：

1. `identity`: `[1] > [2] > [3] > [4] > [5]`
2. `random`: 从 top50 中随机取 5 个合法 index
3. `oracle`: 优先把包含 gold answer substring 的 doc 放入 top5

样本过滤：

```text
sample_filter: top50_hit_top5_miss_baseline0
```

这个过滤只看“top50 中有答案、原 top5 没答案、baseline reward=0”的样本，用来回答一个关键问题：如果把正确 doc 放进 agent 可见 top5，frozen agent continuation 是否真的能把最终答案做对。

### 17.3 smoke4 结果

输出目录：

```text
outputs/agenticIterRag/reward_bound_diagnosis/260708_fixed_prompt_top50miss_smoke4
```

结果：

```text
sample_count: 4
identity mean: 0.2500
random mean:   0.2500
oracle mean:   0.7500
exception:     0
```

smoke 说明 reward 链路不是全坏：在这 4 条 baseline0 样本上，oracle 能明显提升 continuation answer reward。

### 17.4 n=100 结果

输出目录：

```text
outputs/agenticIterRag/reward_bound_diagnosis/260708_fixed_prompt_top50miss_n100
```

汇总结果：

```text
sample_count: 100
baseline mean: 0.0000

identity:
  valid_count: 98
  mean: 0.0408
  improved_samples: 4 / 98
  statuses: answered=72, max_turns=26, exception=2

random:
  valid_count: 98
  mean: 0.1503
  improved_samples: 17 / 98
  statuses: answered=76, max_turns=22, exception=2

oracle:
  valid_count: 98
  mean: 0.3545
  improved_samples: 37 / 98
  statuses: answered=77, max_turns=21, exception=2

oracle_vs_identity:
  paired_count: 98
  oracle_better: 33
  same: 65
  worse: 0
  mean_diff: 0.3137
```

这里的 2 条 exception 来自诊断脚本 worker 线程并发 lazy import tokenizer 时的瞬时 `AutoTokenizer` 导入异常。已在脚本里增加单线程 tokenizer 预热，后续复跑不应再污染统计。

### 17.5 当前判断

这个诊断给出的结论比较明确：

1. stage2 answer reward 不是完全无效。oracle top5 在目标样本上能把平均 reward 从 0 提到约 0.35。
2. 但 reward 上限明显不是 1。即便 oracle 把包含答案的 doc 放入 top5，仍有大量样本 continuation 答不出来或走到 max turns。
3. random 也能到 0.15，说明部分样本对“任意替换 top5”敏感，不能把所有提升都解释为 reranker 真学会了 answer-bearing doc。
4. identity 几乎不提升，说明旧 top5 在这个过滤集合里确实通常不可用。
5. 如果直接用全量 5100 样本训练，stage2 reward 信号会被三类样本稀释：已 solved 样本、top50 也没有答案的样本、oracle 可见答案但 frozen agent 仍答不出的样本。

因此下一轮训练不应直接继续全量盲训。更合理的训练策略是：

1. 先构造一个 stage2 hard/improvable 子集：`top50_hit_top5_miss_baseline0` 或至少提高这类样本采样权重。
2. 训练/评估时同时记录 oracle-bound 分桶，区分“reranker 选错”和“agent 即使拿到答案 doc 也不会用”。
3. reward 可以考虑从纯 final answer F1 增加一个弱辅助项，例如 top5 answer-hit / evidence-hit，用来降低 continuation 不稳定带来的 credit assignment 噪声；最终仍以 answer reward 为主。
4. 继续训练前先跑 raw Qwen3-4B sampling 和 stage1 checkpoint sampling，对比固定 oracle 上限，定位当前模型是格式/采样问题还是策略选择问题。
