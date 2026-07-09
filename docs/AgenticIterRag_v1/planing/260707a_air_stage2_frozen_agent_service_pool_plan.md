# AIR LLM Reranker Stage2 Frozen Agent 服务池方案

## 1. 背景

这篇文档只讨论 AIR LLM reranker stage2 的 frozen agent 服务方案。

当前 stage2 的训练目标是：reranker 先对某一步 search 的 top50 文档输出 top5 排序，然后把这 5 篇文档作为新的 observation 交给 frozen search agent，让 frozen agent 从原轨迹中间位置继续 rollout，最后用 answer reward 给 reranker 打分。

这意味着 stage2 的 reward 不只是一个简单 parser，也不是普通 reranker label 对齐，而是一个 agentic 环境交互：

```text
reranker 输出
-> parser 检查
-> 构造新的 top5 observation
-> 拼回 agent 历史上下文
-> frozen agent continuation rollout
-> 后续 search 调 retriever
-> frozen agent 产出 answer
-> answer reward
```

当前 stage2 计划使用：

```text
train_batch_size = 64
rollout_n = 4
```

所以每个训练 step 至少会产生：

```text
64 * 4 = 256 条 reranker rollout
```

每条合法 reranker rollout 都可能触发一次 frozen agent continuation。后续如果 `rollout_n` 提到 8 或 16，那么每 step 的 continuation 数会进一步变成 512 或 1024。

因此，stage2 的核心瓶颈不是“小模型能不能放进多卡”，而是 frozen agent continuation 的服务并发能力。

## 2. 核心结论

stage2 的资源策略应该从“模型并行优先”改成“服务并发优先”。

推荐默认资源分配：

```text
NPU 0,1,2,3: LLM reranker actor / rollout / update，TP=4
NPU 4: frozen agent instance 0，TP=1
NPU 5: frozen agent instance 1，TP=1
NPU 6: frozen agent instance 2，TP=1
NPU 7: retriever instance 0，TP=1
```

也就是：

```text
reranker: 4 卡
frozen agent: 3 个单卡实例
retriever: 1 个单卡实例
```

不要做 frozen agent `tensor_parallel_size=3`。由于模型都是小模型，没有必要为了模型并行把 3 张卡绑成一个 agent 服务。stage2 更需要的是 3 个可以并行响应 continuation 请求的 frozen agent 实例。

同时，明确拒绝把资源回退成 2 个 retriever 实例的方案。retriever 保持 1 张 NPU，优先把更多卡让给 frozen agent continuation 并发。

## 3. 目标

1. 让 stage2 frozen agent 从单个多卡服务变成多实例服务池。
2. frozen agent 对外仍然暴露一个统一服务地址，避免 reward 侧感知多实例细节。
3. proxy 使用 `least_inflight` 调度，把请求优先发给当前负载最低的 frozen agent 实例。
4. continuation 请求以并发单请求形式进入 frozen agent 服务，由 vLLM 内部 continuous batching 自动合批。
5. stage2 在 `train_batch_size=64`、`rollout_n=4` 下，不因为 frozen agent 串行等待导致单 step 时间不可接受。
6. 保持 retriever 单 NPU 实例，后续只通过 batch、proxy 和监控优化，不回退到 2 卡 retriever。

## 4. 非目标

1. 不实现 frozen agent `tensor_parallel_size=3`。
2. 不要求 frozen agent 支持外部显式 batch API。
3. 不把 retriever 改回 2 个 NPU 实例。
4. 不改 reranker stage1 的训练逻辑。
5. 不改变 stage2 reward 语义；仍然是 `agentic_rag_rollout_reward`。
6. 不改变 prompt、parser 约束；仍然使用 CoSearch-aligned topM reranker prompt。

## 5. 为什么 frozen agent 不做外部 batch

stage2 每个 reranker rollout 的 continuation 是多轮、变长、带 search 工具调用的 agentic 行为。

不同样本的后续路径可能完全不同：

```text
样本 A: reranker 格式错误，直接 -0.5，不调 frozen agent
样本 B: frozen agent 看完 observation 直接 answer
样本 C: frozen agent search 1 次后 answer
样本 D: frozen agent search 3 次
样本 E: frozen agent 到 max turns
```

这种场景不适合在 proxy 外部把 256 条样本手动拼成一个 batch 请求。正确做法是：

```text
256 条样本
-> 启动多个 async continuation task
-> 每个 task 发单条 chat completion 请求
-> frozen-agent proxy 并发转发
-> 每个 vLLM agent 实例内部做 continuous batching
```

也就是说：

```text
外部是并发单请求
内部由 vLLM continuous batching 自动合批
```

这样既适配多轮变长 agent 行为，也能让 vLLM 自己做最合适的 prefill/decode 调度。

## 6. 完整链路

stage2 运行时结构如下：

```text
VERL / reranker actor
  |
  | 生成 64*4 条 reranker outputs
  v
agentic_rag_rollout_reward
  |
  | 格式错误样本直接给 format penalty
  | 格式正确样本进入 async continuation pool
  v
frozen-agent proxy
  |
  | least_inflight
  |
  +--> frozen agent vLLM #0 on NPU 4
  +--> frozen agent vLLM #1 on NPU 5
  +--> frozen agent vLLM #2 on NPU 6
  |
  | agent 后续 search
  v
retriever proxy
  |
  +--> retriever backend #0 on NPU 7
```

reward 侧只依赖统一地址：

```bash
AIR_CONTINUATION_AGENT_BASE_URL=http://127.0.0.1:<agent_proxy_port>
AIR_RETRIEVAL_URL=http://127.0.0.1:<retriever_proxy_port>/retrieve
```

reward 侧不应该知道后面有几个 frozen agent 实例，也不应该直接访问实例端口。

## 7. 并发模型

以当前 stage2 参数为例：

```text
train_batch_size = 64
rollout_n = 4
continuation samples per step = 256
```

如果 continuation 串行执行，耗时会非常高。假设每条 continuation 平均 8 秒：

```text
串行耗时 = 256 * 8s = 2048s，大约 34 分钟 / step
```

如果平均 15 秒：

```text
串行耗时 = 256 * 15s = 3840s，大约 64 分钟 / step
```

这不可接受。

推荐并发模型：

```text
256 samples per step
-> agent_loop_num_workers = 64
-> frozen-agent proxy
-> 3 frozen agent instances
-> each instance max_num_seqs = 8
```

理论 active generation capacity：

```text
3 instances * 8 max_num_seqs = 24
```

粗略估算：

```text
ceil(256 / 24) * 8s = 88s
ceil(256 / 24) * 15s = 165s
```

真实训练还会叠加多轮 search、尾部样本、PPO update 等耗时，但这和串行的几十分钟每 step 已经不是一个量级。

如果后续 `rollout_n=8`：

```text
64 * 8 = 512 continuation samples
```

则更需要 frozen agent 服务池，否则 stage2 单 step 会被环境交互拖死。

## 8. frozen-agent proxy 设计

frozen-agent proxy 必须是异步代理，不是阻塞式 round-robin 转发。

核心能力：

1. 对外暴露一个 OpenAI-compatible chat completions endpoint，或者暴露当前 continuation 代码使用的等价 endpoint。
2. 后端维护 3 个 frozen agent instance URL。
3. 每个 backend 维护 `inflight_count`。
4. 新请求选择 `inflight_count` 最少的 backend。
5. 转发请求时不阻塞整个 event loop。
6. backend 失败后进入短暂 cooldown。
7. 请求超时后返回明确错误。
8. 支持健康检查。
9. 支持写出 proxy manifest，记录 backend 状态、端口、pid、策略。

调度策略默认：

```text
least_inflight
```

伪逻辑：

```text
on request:
    backend = min(healthy_backends, key=inflight_count)
    backend.inflight += 1
    try:
        response = await forward(request, backend.url)
    finally:
        backend.inflight -= 1
```

不建议使用简单 round-robin。continuation 长尾很明显，round-robin 不知道哪个实例还在跑长样本，很容易把新请求继续打到慢实例上。

后续可以升级：

```text
least_inflight_then_latency
```

也就是：

```text
score = inflight_count + latency_weight * latency_ewma
```

但第一版默认 `least_inflight` 就够。

## 9. reward 侧并发要求

只有 proxy 并发还不够，`agentic_rag_rollout_reward` 侧也必须并发发请求。

stage2 应该确保：

```yaml
agent_loop_num_workers: 64
```

并且 reward 侧逻辑应该是 batch-level async executor：

```text
batch reward executor
  -> 接收 batch samples
  -> 格式错误样本直接给 -0.5
  -> 格式正确样本进入 async continuation pool
  -> concurrency = agent_loop_num_workers
  -> 调 frozen-agent proxy
  -> 收集 answer reward
  -> 返回 reward list
```

如果当前 VERL custom reward 是逐条同步调用，那么即使 frozen-agent proxy 后面有 3 个实例，也可能吃不满资源。实现前需要重点确认：

1. VERL 当前是 batch-level 调 reward，还是 sample-level 调 reward。
2. 如果是 sample-level，Ray/worker 层是否真的并发。
3. 如果并发不足，需要在 AIR 的 reward wrapper 或 reward manager 侧做 batch-level 并发。

格式错误样本不应该进入 frozen agent continuation。这类样本直接返回 format penalty，避免浪费 agent 服务资源。

## 10. 配置草案

配置文件实现时必须补充充足中文注释，参考现有 AIR YAML 的注释方式。

建议在 stage2 overlay 中写成：

```yaml
resource:
  stage_resources:
    train_llm_reranker:
      phase_services:
        stage2_agentic:
          services:
            reranker_actor:
              # stage2 reranker 训练与 rollout 使用 0-3 四张 NPU。
              gpu_ids: [0, 1, 2, 3]
              tensor_parallel_size: 4

            frozen_agent_vllm:
              # frozen agent 使用多实例服务池，不做 TP=3。
              backend_type: multi_instance_proxy

              proxy:
                # 对 reward 侧暴露的统一 frozen agent 地址。
                host: 127.0.0.1
                port: 8140

                # continuation 长尾明显，默认按当前 in-flight 数做负载均衡。
                strategy: least_inflight
                timeout: 300
                failure_cooldown_seconds: 10
                latency_ewma_alpha: 0.2

              instances:
                - name: frozen_agent_0
                  # 第一个 frozen agent 单卡实例。
                  gpu_ids: [4]
                  tensor_parallel_size: 1
                  port: 8141
                  max_num_seqs: 8
                  gpu_memory_utilization: 0.75

                - name: frozen_agent_1
                  # 第二个 frozen agent 单卡实例。
                  gpu_ids: [5]
                  tensor_parallel_size: 1
                  port: 8142
                  max_num_seqs: 8
                  gpu_memory_utilization: 0.75

                - name: frozen_agent_2
                  # 第三个 frozen agent 单卡实例。
                  gpu_ids: [6]
                  tensor_parallel_size: 1
                  port: 8143
                  max_num_seqs: 8
                  gpu_memory_utilization: 0.75

            recall:
              # retriever 保持单 NPU 实例，占用 NPU 7。
              backend_type: npu
              instance_count: 1

              proxy:
                host: 127.0.0.1
                port: 8130
                strategy: least_inflight
                timeout: 180
                failure_cooldown_seconds: 10

              accelerator_backend:
                gpu_ids: [7]
                query_batch_size: 32
                doc_dtype: float16
```

stage2 训练参数建议：

```yaml
reranker_training:
  training_phases:
    stage2_agentic:
      # stage2 主 batch，和 stage1 / CoSearch 训练主 batch 对齐。
      train_batch_size: 64
      ppo_mini_batch_size: 64

      # 当前默认每个 prompt 采样 4 条 rollout。
      n_samples_per_prompt: 4

      # continuation 并发 worker 数，负责把 256 条样本并发推给 frozen-agent proxy。
      agent_loop_num_workers: 64

      # reranker prompt / response 长度预算。
      max_prompt_length: 16384
      max_response_length: 1024
      rollout_max_model_len: 17408

      # reranker rollout 的 vLLM 合批参数。
      max_num_seqs: 128
      max_num_batched_tokens: 65536

      # stage2 使用 CoSearch GRPO 默认 KL 配置。
      use_kl_loss: true
      kl_loss_coef: 0.001
      kl_loss_type: low_var_kl
      entropy_coeff: 0.0
```

## 11. 启动顺序

stage2 启动顺序建议：

```text
1. 启动 retriever backend on NPU 7
2. 启动 retriever proxy
3. 预检 retriever proxy
4. 并行启动 frozen agent instance 0/1/2
5. 等待三个 frozen agent 实例 ready
6. 启动 frozen-agent proxy
7. 预检 frozen-agent proxy
8. 写 service manifest
9. 启动 VERL stage2
10. 训练结束后倒序 cleanup
```

cleanup 顺序：

```text
1. stop VERL
2. stop reporter
3. stop frozen-agent proxy
4. stop frozen agent instances
5. stop retriever proxy
6. stop retriever backend
```

注意：frozen agent 三个实例应该并行启动。它们互不共享 NPU，模型路径相同但进程独立，并行加载能明显减少 stage2 启动时间。

## 12. 监控指标

必须给这套服务池补充监控，否则无法判断瓶颈是在 frozen agent、retriever、proxy 还是 reward 侧。

frozen-agent proxy 指标：

```text
inflight_per_backend
request_count
error_count
timeout_count
latency_p50
latency_p95
latency_p99
backend_cooldown_count
```

frozen agent vLLM 指标：

```text
NPU utilization on 4/5/6
KV cache usage
running requests
waiting requests
tokens/s
```

retriever 指标：

```text
search latency p50/p95
queue length
NPU 7 utilization
query_batch_size 实际合批情况
```

stage2 trainer 指标：

```text
step_time
reward_time
continuation_time
retriever_time
rollout_generation_time
ppo_update_time
format_invalid_ratio
agent_success_ratio
```

调参判断：

```text
如果 NPU 4/5/6 利用率低，且 proxy inflight 低：
  优先提高 agent_loop_num_workers，例如 64 -> 96。

如果 proxy inflight 高，vLLM waiting requests 多，且 NPU 4/5/6 利用率高：
  frozen agent 已经吃满，不要继续加 worker。

如果 retriever p95 很高，且 NPU 7 利用率高：
  先提高 query_batch_size，例如 32 -> 64，不回退到 2 retriever 实例。

如果 frozen agent 单实例 OOM：
  优先降低 frozen agent instances[*].max_num_seqs，不先降低 reranker train_batch_size。
```

## 13. 实现拆解

代码实现计划中必须补充充足中文注释，参考现有代码文件和配置文件的注释方法。

### 13.1 TrainingServiceManager

需要支持：

1. `frozen_agent_vllm.backend_type=multi_instance_proxy`
2. `frozen_agent_vllm.instances`
3. 并行启动多个 frozen agent vLLM 实例
4. 每个实例独立端口、pid、日志、健康检查
5. 启动 frozen-agent proxy
6. stop 时倒序清理 proxy 和所有实例
7. 写出 service manifest

每个 frozen agent 实例的日志建议：

```text
runtime_services/stage2_agentic/frozen_agent/frozen_agent_0/server.log
runtime_services/stage2_agentic/frozen_agent/frozen_agent_1/server.log
runtime_services/stage2_agentic/frozen_agent/frozen_agent_2/server.log
runtime_services/stage2_agentic/frozen_agent/proxy.log
```

### 13.2 frozen-agent proxy

建议新增独立模块，例如：

```text
agentic_iter_rag/reranker_training/frozen_agent_proxy.py
```

职责：

1. 读取 backend 实例列表
2. 提供 health endpoint
3. 提供 chat completion 转发 endpoint
4. 维护 inflight 计数
5. 执行 least-inflight 调度
6. 记录 latency 和错误
7. 支持 backend cooldown
8. 支持 graceful shutdown

### 13.3 agentic_rag_rollout_reward

需要确认当前 reward 调用粒度。

如果是 batch-level：

1. 在 batch 内拆出格式错误样本
2. 格式正确样本进入 async continuation executor
3. 使用 semaphore 控制并发数
4. 收集 reward list

如果是 sample-level：

1. 确认 VERL/Ray 是否会并发调用
2. 如果并发不足，需要增加 AIR wrapper 或 reward manager 层 batch executor
3. 避免 Python 层逐条串行等待 frozen agent

### 13.4 配置校验

compiler / validator 需要校验：

1. reranker actor NPU 和 frozen agent NPU 不重叠
2. frozen agent NPU 和 retriever NPU 不重叠
3. 每个 frozen agent instance 的端口不重复
4. frozen-agent proxy 端口和 retriever proxy 端口不冲突
5. 每个 frozen agent instance `tensor_parallel_size=1`
6. `AIR_CONTINUATION_AGENT_BASE_URL` 指向 proxy，不指向单个实例
7. `AIR_RETRIEVAL_URL` 指向 retriever proxy

### 13.5 dry-run / manifest

dry-run 输出必须能看清楚：

```text
reranker_actor_gpus: [0,1,2,3]
frozen_agent_instances:
  - name: frozen_agent_0, gpu_ids: [4], port: 8141
  - name: frozen_agent_1, gpu_ids: [5], port: 8142
  - name: frozen_agent_2, gpu_ids: [6], port: 8143
frozen_agent_proxy_url: http://127.0.0.1:8140/v1/chat/completions
recall_gpus: [7]
recall_url: http://127.0.0.1:8130/retrieve
agent_loop_num_workers: 64
```

## 14. 测试计划

### 14.1 配置 dry-run

运行 stage2 dry-run，检查：

1. stage1 disabled
2. stage2 enabled
3. reranker actor 使用 NPU 0-3
4. frozen agent 三实例使用 NPU 4/5/6
5. retriever 使用 NPU 7
6. proxy port 不冲突
7. final config 中 `AIR_CONTINUATION_AGENT_BASE_URL` 指向 frozen-agent proxy

### 14.2 frozen-agent proxy 单元测试

用 fake backend 验证：

1. least-inflight 选择空闲 backend
2. backend 超时后 inflight 能正确递减
3. backend 失败后进入 cooldown
4. 所有 backend 不健康时返回明确错误
5. 并发 100 个请求时不会串行阻塞

### 14.3 服务启动 smoke

只启动服务，不跑训练：

1. retriever backend on NPU 7
2. retriever proxy 可检索
3. frozen agent 3 个实例都 ready
4. frozen-agent proxy health ready
5. 通过 proxy 发 10 条 chat completion 请求，确认请求分散到不同 backend

### 14.4 stage2 1-step 真实训练

用 `total_training_steps=1` 跑 stage2：

1. 确认每 step 会产生 `64*4=256` 条 reranker rollout
2. 确认格式错误样本不进入 frozen agent
3. 确认 frozen-agent proxy inflight 有并发
4. 确认 NPU 4/5/6 都有利用率
5. 确认 retriever NPU 7 没有严重排队
6. 确认 step 完成后有 reward、metrics、rollout_data

### 14.5 性能测试

比较：

1. 旧配置：frozen agent TP2 + retriever 2 实例
2. 新配置：frozen agent 3 个 TP1 实例 + retriever 1 实例

重点看：

```text
stage2 step_time
continuation_time
frozen agent latency p95
retriever latency p95
NPU utilization 4/5/6/7
```

验收目标：

1. 新方案 stage2 单 step 时间不能比旧方案更差。
2. frozen agent NPU 4/5/6 需要都有明显利用率。
3. retriever 单 NPU 7 不能成为主瓶颈。
4. 如果 bottleneck 变成 retriever，只允许先调 `query_batch_size`，不回退到 2 retriever 实例。

## 15. 默认决策

默认采用：

```text
reranker: NPU 0,1,2,3，TP=4
frozen agent: NPU 4,5,6，3 个 TP1 实例
retriever: NPU 7，1 个实例
frozen-agent proxy: least_inflight
agent_loop_num_workers: 64
frozen_agent_instance_max_num_seqs: 8
```

stage2 当前训练参数保持：

```text
train_batch_size = 64
rollout_n = 4
max_response_length = 1024
max_prompt_length = 16384
```

后续如果 rollout_n 提高，优先扩展 continuation 并发和 proxy 调度，不优先降低 batch 或回退 retriever 资源。
