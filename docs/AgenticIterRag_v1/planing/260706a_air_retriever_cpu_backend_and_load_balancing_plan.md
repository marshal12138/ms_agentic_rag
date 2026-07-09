# AIR Retriever CPU Backend 与统一负载均衡实现计划

## 1. 背景

现在 AIR 的 retriever 服务在两个地方会占用 NPU/GPU：

```text
generate_traces stage：agent 生产轨迹时，每次 search 都会调用 retriever。
LLM reranker stage2_agentic：continuation rollout 后续 search 会调用 retriever。
```

这有两个问题：

- generate_traces 本来就要给 agent vLLM 留 NPU/GPU，retriever 再占卡会压缩 agent 推理资源。
- stage2 本来就需要 reranker actor 和 frozen agent，NPU/GPU 很紧张。
- retriever 会把大规模 doc embeddings 加载到 NPU/GPU 显存里，资源占用很重。
- 当前多实例 retriever 虽然有 proxy，但 proxy 只是 round-robin，不知道哪个 backend 正在忙。

所以我们要实现一版 CPU retriever backend，并且把负载均衡能力做成 CPU/NPU/GPU 通用能力。

这件事的核心目标不是简单“把 retriever 放到 CPU 上”，而是：

```text
所有 retriever backend 都挂到一个统一 proxy 后面；
agent / reward / continuation 只访问一个稳定 retrieval_service_url；
proxy 根据 backend 忙闲程度做负载均衡。
```

## 2. 当前机器资源确认

当前服务器资源如下：

```text
内存总量: 1.5 TiB
当前 used: 9.2 GiB
当前 available: 1.5 TiB
CPU: 192 cores
NUMA: 8 nodes
```

当前 retriever 资产大小如下：

```text
retrieval 目录: 139G
e5_Flat.index: 64,559,075,373 bytes，约 60.1 GiB
wiki-18.jsonl: 14,393,573,105 bytes，约 13.4 GiB
e5-base-v2 模型: 2.1G
```

如果 8 个 CPU retriever 实例完全各自加载一份 index/corpus/model，粗略上限是：

```text
index: 60G * 8 = 480G
corpus: 14G * 8 = 112G
model: 2G * 8 = 16G
合计粗估: 600G+，再加 Python/FAISS/datasets 开销
```

当前机器 available memory 约 1.5TiB，所以内存足够。

CPU 也足够：

```text
192 cores / 8 instances = 24 cores per instance
```

为了避免 CPU 线程过度竞争，默认线程配置不能直接打满 192 cores。

默认建议：

```text
CPU retriever instance_count: 8
cpu_threads_per_instance: 16
query_batch_size: 8
```

也就是默认线程预算大约：

```text
8 instances * 16 threads = 128 threads
```

192 核机器上 128 线程的理论占用是 66.7%，低于 70% 的总资源占用阈值。后续如果换机器，要重新按机器核心数校准这个默认值。

## 3. 目标

### 3.1 要实现的能力

第一，新增 CPU retriever backend。

```text
backend_type=cpu 时，不占用 NPU/GPU。
```

第二，CPU retriever 默认启动 8 个实例。

```text
proxy: 8130
backend 0: 8131
backend 1: 8132
backend 2: 8133
backend 3: 8134
backend 4: 8135
backend 5: 8136
backend 6: 8137
backend 7: 8138
```

第三，所有 backend 共用一个对外服务地址。

```text
http://127.0.0.1:8130/retrieve
```

第四，负载均衡不仅用于 CPU，也要用于 NPU/GPU retriever。

```text
CPU backend 多实例 -> load-balancing proxy -> 一个统一 URL
NPU backend 多实例 -> load-balancing proxy -> 一个统一 URL
GPU backend 多实例 -> load-balancing proxy -> 一个统一 URL
```

第五，proxy 要根据 backend 忙闲程度转发。

```text
当某个 backend 正在处理请求，或者请求较多时，新请求优先转到其他空闲 backend。
```

第六，配置上要清楚区分 CPU 参数和 accelerator 参数。

```text
backend_type=cpu 时，只读取 cpu_backend。
backend_type=npu/cuda 时，只读取 accelerator_backend。
```

第七，CPU retriever 模式不仅要支持 LLM reranker stage2，也要支持 `generate_traces` stage。

```text
generate_traces:
  agent_vllm 继续使用 NPU/GPU。
  recall 默认可以切到 CPU backend + load-balancing proxy。

stage2_agentic:
  reranker_actor 和 frozen_agent_vllm 使用 NPU/GPU。
  recall 默认可以切到 CPU backend + load-balancing proxy。
```

## 4. 非目标

这版先不做这些事情：

- 不改 agent search tool 的请求协议。
- 不改 continuation reward 的请求协议。
- 不要求 CPU retriever 与 NPU retriever 延迟完全一致。
- 不做跨机器分布式 retriever。
- 不做共享内存 index server。第一版允许每个 CPU backend 独立加载 index/corpus/model。
- 不改变 LLM reranker stage1 的资源策略。
- 不改变 generate_traces 的 agent rollout 语义；只是把 search tool 底层 retriever 的 backend 从 NPU/GPU 可选切换成 CPU。

## 5. 总体架构

目标架构是：

```text
generate_traces
  agent_vllm             -> NPU/GPU，例如 0-3
  recall proxy           -> CPU service, port 8130
    cpu retriever backend -> port 8131
    cpu retriever backend -> port 8132
    cpu retriever backend -> port 8133
    cpu retriever backend -> port 8134
    cpu retriever backend -> port 8135
    cpu retriever backend -> port 8136
    cpu retriever backend -> port 8137
    cpu retriever backend -> port 8138

stage2_agentic
  reranker_actor          -> NPU 0-3
  frozen_agent_vllm       -> NPU 4-7
  recall proxy            -> CPU service, port 8130
    cpu retriever backend -> port 8131
    cpu retriever backend -> port 8132
    cpu retriever backend -> port 8133
    cpu retriever backend -> port 8134
    cpu retriever backend -> port 8135
    cpu retriever backend -> port 8136
    cpu retriever backend -> port 8137
    cpu retriever backend -> port 8138
```

如果切成 NPU/GPU retriever，也还是同一个 proxy 模式：

```text
generate_traces
  agent_vllm             -> NPU/GPU，例如 0-3
  recall proxy           -> port 8130
    npu retriever backend -> NPU 6, port 8131
    npu retriever backend -> NPU 7, port 8132

stage2_agentic
  reranker_actor          -> NPU 0-3
  frozen_agent_vllm       -> NPU 4-5
  recall proxy            -> port 8130
    npu retriever backend -> NPU 6, port 8131
    npu retriever backend -> NPU 7, port 8132
```

注意，上面这个 NPU/GPU retriever 例子是“显式切回 accelerator retriever”时的备选资源布局。当前默认是 CPU retriever，所以 stage2 的 frozen agent 可以使用 4 张 NPU，也就是 4-7；如果 retriever 又切回 NPU/GPU，就不能继续让 frozen agent 占用 6-7，需要重新做资源互斥配置。

对上层来说，不管 CPU/NPU/GPU 后端怎么变，入口都只有：

```text
retrieval_service_url: http://127.0.0.1:8130/retrieve
```

## 6. 配置设计

### 6.1 推荐配置结构

不要把 CPU 参数直接平铺在 `recall` 下面。

推荐结构如下：

```yaml
recall:
  # backend 类型：cpu / npu / cuda。
  backend_type: cpu

  # 启动几个 retriever backend 实例。
  # CPU 默认 8；NPU/GPU 一般由 accelerator_backend.gpu_ids 推导，也可以显式覆盖。
  instance_count: 8

  # 对外统一 proxy 端口。
  port: 8130

  # backend 实例起始端口。
  backend_base_port: 8131

  # 对外统一服务地址；agent / reward / continuation 只访问这个地址。
  retrieval_service_url: http://127.0.0.1:8130/retrieve

  proxy:
    # 所有 backend_type 都使用该负载均衡策略。
    strategy: least_inflight

    # proxy 转发到单个 backend 的超时时间。
    timeout: 180

    # backend 请求失败后进入短暂 cooldown，避免立即继续打到坏实例。
    failure_cooldown_seconds: 10

    # backend latency 指数滑动平均的更新系数。
    latency_ewma_alpha: 0.2

    # 单个请求最多尝试多少个 backend。
    max_retries_per_request: 8

  cpu_backend:
    # 只有 backend_type=cpu 时读取。
    cpu_threads_per_instance: 16

    # CPU encoder 的 query batch size。
    query_batch_size: 8

    # CPU doc embeddings dtype。第一版默认 float32，优先稳定。
    doc_dtype: float32

  accelerator_backend:
    # 只有 backend_type=npu/cuda 时读取。
    gpu_ids: [6, 7]

    # NPU/GPU encoder 的 query batch size。
    query_batch_size: 32

    # NPU/GPU doc embeddings dtype。
    doc_dtype: float16

  # 是否自动启动 recall backend 和 proxy。
  auto_start: true

  # stage 结束后是否自动停止 recall backend 和 proxy。
  auto_stop: true

  # 等待 recall ready 的最长秒数。
  wait_seconds: 600

  # 默认关闭资产预检，避免重复扫描大 index/corpus。
  asset_precheck: false

  # 默认不发真实 query 预检，只用 health/status 判断 ready。
  query_preflight: false
```

### 6.2 CPU 默认配置

因为当前服务器内存足够，所以 CPU 模式默认按 8 个实例启动：

```yaml
recall:
  backend_type: cpu
  instance_count: 8

  port: 8130
  backend_base_port: 8131
  retrieval_service_url: http://127.0.0.1:8130/retrieve

  proxy:
    strategy: least_inflight
    timeout: 180
    failure_cooldown_seconds: 10
    latency_ewma_alpha: 0.2
    max_retries_per_request: 8

  cpu_backend:
    cpu_threads_per_instance: 16
    query_batch_size: 8
    doc_dtype: float32
```

这套配置在 `generate_traces` 和 `stage2_agentic` 中保持同一语义：

```text
port / retrieval_service_url 是对外 proxy 地址。
backend_base_port + instance_index 是内部 CPU backend 地址。
agent search tool 和 continuation reward 都只访问 proxy。
```

### 6.3 NPU/GPU 配置

负载均衡策略也要应用到 NPU/GPU retriever。

NPU 版本示例：

```yaml
recall:
  backend_type: npu
  instance_count: 2

  port: 8130
  backend_base_port: 8131
  retrieval_service_url: http://127.0.0.1:8130/retrieve

  proxy:
    strategy: least_inflight
    timeout: 180
    failure_cooldown_seconds: 10
    latency_ewma_alpha: 0.2
    max_retries_per_request: 2

  accelerator_backend:
    gpu_ids: [6, 7]
    query_batch_size: 32
    doc_dtype: float16
```

启动后：

```text
proxy: 8130
backend 0: http://127.0.0.1:8131/retrieve, NPU 6
backend 1: http://127.0.0.1:8132/retrieve, NPU 7
```

### 6.4 参数生效边界

实现时必须严格区分参数生效范围。

```text
backend_type=cpu:
  只读取 cpu_backend
  不读取 accelerator_backend.gpu_ids

backend_type=npu:
  只读取 accelerator_backend
  不读取 cpu_backend

backend_type=cuda:
  只读取 accelerator_backend
  不读取 cpu_backend
```

建议增加 fail-fast 校验：

```text
如果 backend_type != cpu，但出现旧的平铺 CPU 参数：
  cpu_threads_per_instance
  cpu_query_batch_size
  cpu_doc_dtype

直接报错，提示必须放到 cpu_backend 下，且只在 backend_type=cpu 时生效。
```

同理：

```text
如果 backend_type=cpu，但配置依赖 gpu_ids 才能启动，也要报错。
```

## 7. 代码实现计划

### 7.1 新增 CPU retriever backend

新增文件：

```text
src/retrievers/cpu_dense_retriever_server.py
```

职责：

- 读取 FAISS index。
- 读取 corpus。
- 加载 e5-base-v2 encoder 到 CPU。
- 提供 `/retrieve`。
- 提供 `/health`。
- 提供 `/status`，返回 index shape、device、dtype、线程配置等信息。

协议要和当前 GPU retriever 保持兼容：

```json
{
  "queries": ["..."],
  "topk": 50,
  "return_scores": true
}
```

返回仍然是：

```json
{
  "result": [
    [
      {"document": {...}, "score": 0.123}
    ]
  ]
}
```

CPU backend 第一版可以复用现有 `gpu_dense_retriever_server.py` 的 Encoder、pooling、load_docs 逻辑，但建议独立成新文件，避免 GPU/NPU 分支太多导致误触 accelerator 环境。

所有新增 Python 代码都要补充充足的中文注释，参考现有 AIR 代码文件的注释方式。尤其是：

- 为什么 CPU 版不做 accelerator 可见性检查。
- 为什么 CPU 版默认 float32。
- 为什么每个实例独立加载 index/corpus/model。
- `/health` 和 `/status` 分别用于什么。

### 7.2 新增 CPU retriever launcher

新增文件：

```text
scripts/agenticIterRag_v1/assets/infer_backend/00_start_cpu_dense_retriever_server.sh
```

职责：

- 设置 Python 环境。
- 检查 index/corpus/model 文件存在。
- 设置 CPU 线程变量。
- 启动 `cpu_dense_retriever_server.py`。

示例启动逻辑：

```bash
export OMP_NUM_THREADS="${CPU_THREADS_PER_INSTANCE}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_INSTANCE}"
export TOKENIZERS_PARALLELISM=false

exec "$PY" src/retrievers/cpu_dense_retriever_server.py \
  --index_path "${INDEX_FILE}" \
  --corpus_path "${CORPUS_FILE}" \
  --retriever_model "${RETRIEVER_MODEL}" \
  --topk "${RECALL_FINAL_TOP_N}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --query_batch_size "${QUERY_BATCH_SIZE}" \
  --doc_dtype "${DOC_DTYPE}"
```

所有新增 shell 配置也要补充充足的中文注释，参考现有 AIR shell 文件的注释方法。

### 7.3 新增或升级负载均衡 proxy

当前文件：

```text
src/retrievers/retrieval_round_robin_proxy.py
```

当前只是 round-robin。

建议新增：

```text
src/retrievers/retrieval_load_balancing_proxy.py
```

职责：

- 对所有 backend type 统一工作。
- 支持 `/retrieve`。
- 支持 `/health`。
- 支持 `/stats`，返回每个 backend 的 in-flight、latency、fail count、cooldown 状态。

负载均衡策略：

```text
least_inflight + latency_ewma + failure_backoff
```

每个 backend 维护：

```text
backend_url
in_flight
ewma_latency_s
fail_count
cooldown_until
last_success_at
```

选择规则：

```text
1. 过滤掉处于 cooldown 的 backend。
2. 优先选择 in_flight 最少的 backend。
3. in_flight 相同，选择 ewma_latency_s 更低的 backend。
4. 仍然相同，再 round-robin 打散。
5. 当前 backend 请求失败，标记失败并重试下一个 backend。
6. 全部 backend 失败才返回 502。
```

这个策略适用于 CPU/NPU/GPU，不允许只在 CPU 模式下生效。

所有新增 proxy 代码要补充充足中文注释，尤其解释：

- 为什么不用纯 round-robin。
- `in_flight` 如何反映忙闲程度。
- `ewma_latency_s` 为什么能避免慢 backend 被持续打满。
- backend 失败后为什么要 cooldown。

### 7.4 改 TrainingServiceManager.start_recall

修改文件：

```text
AgenticIterRag/agentic_iter_rag/reranker_training/service_manager.py
```

当前逻辑大概是：

```text
读取 recall.gpu_ids
gpu_ids 为空直接报错
len(gpu_ids)==1 时直连 backend
len(gpu_ids)>1 时启动多个 backend + round-robin proxy
```

要改成：

```text
读取 recall.backend_type
读取 recall.instance_count
根据 backend_type 启动 CPU 或 accelerator backend
无论几个 backend，都启动统一 load-balancing proxy
返回 proxy retrieval_service_url
```

伪逻辑：

```python
backend_type = recall_cfg.get("backend_type", "npu")
instance_count = resolve_instance_count(recall_cfg, backend_type)

if backend_type == "cpu":
    backend_cfg = recall_cfg["cpu_backend"]
    for instance_index in range(instance_count):
        start_cpu_backend(instance_index, backend_port)
elif backend_type in {"npu", "cuda"}:
    backend_cfg = recall_cfg["accelerator_backend"]
    gpu_ids = backend_cfg["gpu_ids"]
    for instance_index, gpu_id in enumerate(gpu_ids[:instance_count]):
        start_accelerator_backend(gpu_id, backend_port)
else:
    raise ValueError

start_load_balancing_proxy(port=recall_cfg["port"], backend_urls=backend_urls)
wait_for_proxy_health(...)
```

注意：

- CPU 模式不要求 `gpu_ids`。
- CPU 模式不设置 `ASCEND_RT_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES`。
- NPU/GPU 模式不读取 `cpu_backend`。
- 所有模式都走同一个 proxy 地址。
- 停止服务时仍然倒序 stop，先停 proxy，再停 backend。

所有新增逻辑都要补充充足中文注释，参考现有 `service_manager.py` 的注释方法。

### 7.5 改 generate_traces 的 recall 启动逻辑

需要改的地方：

```text
scripts/agenticIterRag_v1/assets/run_pipeline.py
scripts/agenticIterRag_v1/assets/infer_backend/02_air_infer_launcher.sh
scripts/agenticIterRag_v1/assets/infer_backend/00_start_dense_retriever_server.sh
```

当前 `generate_traces` 的资源环境变量主要从：

```text
resource.stage_resources.generate_traces.services.recall
```

翻译成：

```text
RECALL_GPU_ID
RECALL_BACKEND_BASE_PORT
RECALL_SERVICE_URL
```

改造后要支持：

```yaml
resource:
  stage_resources:
    generate_traces:
      services:
        recall:
          backend_type: cpu
          instance_count: 8
          port: 8130
          backend_base_port: 8131
          retrieval_service_url: http://127.0.0.1:8130/retrieve
          proxy:
            strategy: least_inflight
            timeout: 180
            failure_cooldown_seconds: 10
            latency_ewma_alpha: 0.2
            max_retries_per_request: 8
          cpu_backend:
            cpu_threads_per_instance: 16
            query_batch_size: 8
            doc_dtype: float32
          accelerator_backend:
            gpu_ids: [6, 7]
            query_batch_size: 32
            doc_dtype: float16
```

`run_pipeline.py` 要把这些字段传给 infer launcher，例如：

```text
RECALL_BACKEND_TYPE=cpu
RECALL_INSTANCE_COUNT=8
RECALL_PROXY_STRATEGY=least_inflight
RECALL_PROXY_TIMEOUT=180
RECALL_PROXY_FAILURE_COOLDOWN_SECONDS=10
RECALL_PROXY_LATENCY_EWMA_ALPHA=0.2
RECALL_PROXY_MAX_RETRIES_PER_REQUEST=8
RECALL_CPU_THREADS_PER_INSTANCE=16
RECALL_CPU_QUERY_BATCH_SIZE=8
RECALL_CPU_DOC_DTYPE=float32
RECALL_ACCELERATOR_GPU_IDS=6,7
RECALL_ACCELERATOR_QUERY_BATCH_SIZE=32
RECALL_ACCELERATOR_DOC_DTYPE=float16
```

`02_air_infer_launcher.sh` 要按 `RECALL_BACKEND_TYPE` 选择启动方式：

```text
backend_type=cpu:
  启动 8 个 CPU retriever backend。
  启动 load-balancing proxy。
  不设置 NPU/GPU visible devices。

backend_type=npu/cuda:
  根据 accelerator_backend.gpu_ids 启动 NPU/GPU retriever backend。
  启动同一个 load-balancing proxy。
```

generate_traces 和 stage2 要共用同一套 backend/proxy 启动语义，避免两条链路行为不一致。

第一版更务实的拆法：

```text
1. service_manager.py 先实现训练 stage2 的 CPU recall。
2. 02_air_infer_launcher.sh 再复用同样的环境变量和 launcher 脚本实现 generate_traces 的 CPU recall。
3. 两条链路都使用同一个 retrieval_load_balancing_proxy.py。
```

所有新增 shell 逻辑都要写充足中文注释，尤其说明：

- generate_traces 里 CPU retriever 为什么要和 stage2 使用同样的 proxy 策略。
- backend_type=cpu 时为什么不读取 accelerator_backend。
- backend_type=npu/cuda 时为什么不读取 cpu_backend。
- 为什么对外只暴露一个 retrieval_service_url。

### 7.6 更新配置文件

需要更新：

```text
AgenticIterRag/config/resource/local_8gpu_0_7.yaml
tasks/train_tasks/agenticIterRag/configs/from_existing_260704e_traj_to_reranker_training_overlay.yaml
tasks/train_tasks/agenticIterRag/configs/dataproduce_overlay.yaml
```

generate_traces 默认也要支持 CPU retriever，例如：

```yaml
generate_traces:
  services:
    agent_vllm:
      # agent vLLM 继续使用 NPU/GPU。
      gpu_ids: [0, 1, 2, 3]
      tensor_parallel_size: 4
      port: 8140
      served_model_name: agentic-iter-rag-agent
      auto_start: true
      auto_stop: true

    recall:
      # generate_traces 默认可以使用 CPU retriever，避免 retriever 占用 NPU/GPU。
      backend_type: cpu

      # 当前服务器内存约 1.5TiB，CPU 192 cores，默认可以启动 8 个 CPU retriever 实例。
      instance_count: 8

      # 对外统一 proxy 端口。
      port: 8130

      # CPU backend 从 8131 开始递增。
      backend_base_port: 8131

      # agent search tool 只访问这个统一地址。
      retrieval_service_url: http://127.0.0.1:8130/retrieve

      proxy:
        strategy: least_inflight
        timeout: 180
        failure_cooldown_seconds: 10
        latency_ewma_alpha: 0.2
        max_retries_per_request: 8

      cpu_backend:
        cpu_threads_per_instance: 16
        query_batch_size: 8
        doc_dtype: float32

      accelerator_backend:
        gpu_ids: [6, 7]
        query_batch_size: 32
        doc_dtype: float16

      auto_start: true
      auto_stop: true
      wait_seconds: 600
      asset_precheck: false
      query_preflight: false
```

stage2 默认也改成 CPU retriever：

```yaml
stage2_agentic:
  services:
    reranker_actor:
      gpu_ids: [0, 1, 2, 3]
      tensor_parallel_size: 4
      port: 8242

    frozen_agent_vllm:
      gpu_ids: [4, 5, 6, 7]
      tensor_parallel_size: 4
      port: 8140
      served_model_name: agentic-iter-rag-frozen-agent
      auto_start: true
      auto_stop: true

    recall:
      # stage2 默认使用 CPU retriever，释放 NPU/GPU 给 reranker 和 frozen agent。
      backend_type: cpu

      # 当前服务器内存约 1.5TiB，CPU 192 cores，默认可以启动 8 个 CPU retriever 实例。
      instance_count: 8

      # 对外统一 proxy 端口。
      port: 8130

      # CPU backend 从 8131 开始递增。
      backend_base_port: 8131

      # continuation reward 和 agent search tool 只访问这个统一地址。
      retrieval_service_url: http://127.0.0.1:8130/retrieve

      proxy:
        strategy: least_inflight
        timeout: 180
        failure_cooldown_seconds: 10
        latency_ewma_alpha: 0.2
        max_retries_per_request: 8

      cpu_backend:
        cpu_threads_per_instance: 16
        query_batch_size: 8
        doc_dtype: float32

      accelerator_backend:
        gpu_ids: [6, 7]
        query_batch_size: 32
        doc_dtype: float16

      auto_start: true
      auto_stop: true
      wait_seconds: 600
      asset_precheck: false
      query_preflight: false
```

所有新增 YAML 字段都要写中文注释，参考现有 AIR YAML 注释风格。

## 8. 错误处理

### 8.1 配置错误

必须 fail-fast：

```text
backend_type 不在 cpu/npu/cuda -> 报错。
backend_type=cpu 且 cpu_backend 缺失 -> 报错。
backend_type=npu/cuda 且 accelerator_backend.gpu_ids 为空 -> 报错。
backend_type!=cpu 但使用旧的平铺 CPU 参数 -> 报错。
instance_count <= 0 -> 报错。
backend 端口和 proxy 端口冲突 -> 报错。
generate_traces 和 stage2 如果在同一条 pipeline 里先后使用 8130/8131 等端口，要保证前一个 stage 已经 cleanup。
```

### 8.2 Backend 启动失败

如果某个 backend 启动失败：

- 第一版建议 fail-fast。
- 不建议静默少启动几个实例继续跑，因为这会让实际吞吐和配置不一致。

### 8.3 Proxy 转发失败

proxy 对单个请求的行为：

```text
当前 backend 失败 -> 标记 fail_count，进入 cooldown -> 尝试下一个 backend。
全部 backend 都失败 -> 返回 502。
```

### 8.4 Health 行为

CPU/NPU/GPU backend 都要有 health/status。

```text
/health: 服务可用即可。
/status: 返回更详细状态，比如 device、index shape、in_flight、latency。
```

proxy：

```text
/health: 至少一个 backend healthy 就返回 ok。
/stats: 返回所有 backend 的状态。
```

## 9. 测试计划

### 9.1 CPU backend 单实例 smoke

启动 1 个 CPU retriever：

```text
backend_type=cpu
instance_count=1
```

检查：

```text
POST /retrieve 能返回 top50。
/health 返回 ok。
/status 显示 device=cpu。
NPU/GPU 没有新增 retriever 进程占用。
```

### 9.2 CPU backend 8 实例 smoke

启动 8 个 CPU retriever：

```text
backend_type=cpu
instance_count=8
```

检查：

```text
8131-8138 backend 都 ready。
8130 proxy ready。
/stats 能看到 8 个 backend。
内存占用没有超过预期。
CPU 线程数符合 cpu_threads_per_instance。
```

### 9.3 负载均衡测试

构造 fake backend：

```text
backend 1 sleep 5s
backend 2 sleep 0.1s
backend 3 sleep 0.1s
backend 4 sleep 0.1s
```

并发请求 20 个。

检查：

```text
慢 backend 的 in_flight 增高后，新请求被分配到其他 backend。
proxy /stats 中 ewma_latency_s 会反映慢 backend。
失败 backend 会进入 cooldown。
```

### 9.4 NPU/GPU backend 负载均衡回归

配置：

```text
backend_type=npu
accelerator_backend.gpu_ids=[6,7]
```

检查：

```text
proxy 仍然启动。
backend 6/7 都 ready。
负载均衡策略仍然是 least_inflight。
cpu_backend 参数没有被读取。
```

### 9.5 Generate Traces dry-run

配置：

```text
pipeline.resume_from_stage=generate_traces
pipeline.stop_after_stage=generate_traces
resource.stage_resources.generate_traces.services.recall.backend_type=cpu
resource.stage_resources.generate_traces.services.recall.instance_count=8
```

检查：

```text
final config 中 generate_traces recall backend_type=cpu。
agent_vllm 仍然使用 NPU/GPU。
recall 不再占用 NPU/GPU。
retrieval_service_url 仍然是 http://127.0.0.1:8130/retrieve。
infer launcher 环境变量包含 RECALL_BACKEND_TYPE=cpu 和 RECALL_INSTANCE_COUNT=8。
```

### 9.6 Generate Traces 小样本真实 smoke

跑 10 到 20 条小样本：

```text
data.trace_max_samples=10
backend_type=cpu
instance_count=8
```

检查：

```text
CPU retriever backend 启动成功。
load-balancing proxy 启动成功。
agent search tool 能通过统一 URL 调 retriever。
trajectory.jsonl / enhanced_trajectory.jsonl 正常产出。
stage 结束后 proxy 和 backend 全部停止。
NPU/GPU 上没有 retriever 进程残留。
```

### 9.7 Stage2 训练 dry-run

开启：

```yaml
stage2_agentic.enabled: true
```

跑 dry-run。

检查：

```text
stage2 reranker_actor 使用 0-3。
stage2 frozen_agent_vllm 使用 4-7。
stage2 recall backend_type=cpu。
stage2 recall 不占用任何 NPU/GPU。
retrieval_service_url 仍然是 http://127.0.0.1:8130/retrieve。
```

### 9.8 Stage2 真实 1-step smoke

跑：

```text
stage1 已有 checkpoint
stage2_agentic.enabled=true
total_training_steps=1
backend_type=cpu
instance_count=8
```

检查：

```text
CPU retriever backend 启动成功。
proxy 启动成功。
frozen agent 启动成功。
agentic_rag_rollout_reward 能通过统一 URL 调 retriever。
训练结束后 proxy/backend/frozen agent 都被自动停止。
```

## 10. 默认决策

默认采用：

```text
backend_type: cpu
instance_count: 8
proxy.strategy: least_inflight
cpu_threads_per_instance: 16
query_batch_size: 8
doc_dtype: float32
retrieval_service_url: http://127.0.0.1:8130/retrieve
```

原因：

- 当前机器内存约 1.5TiB，足够支撑 8 个 CPU retriever 实例。
- 192 CPU cores 足够支撑 8 个实例并发。
- 每实例 16 线程时总线程预算是 128，约等于 192 核的 66.7%，满足 70% 总资源阈值。
- 统一 proxy 地址可以减少上层配置 bug。
- `least_inflight` 比 round-robin 更符合训练时突发并发请求的负载特点。

如果后续换机器或观察到 CPU 利用率异常，可以重新消融：

```text
cpu_threads_per_instance: 8 -> 12 -> 16
query_batch_size: 8 -> 16
```

如果观察到内存压力过高，可以回退：

```text
instance_count: 8 -> 4 -> 2
```

## 11. 和 AIR Pipeline 的关系

这项改造影响两条链路：

```text
generate_traces:
  需要 retriever 作为 agent search tool 的底层 search 服务。
  默认可切到 CPU retriever + load-balancing proxy。
  agent_vllm 继续使用 NPU/GPU。

stage1_format:
  不需要 retriever。
  不受 CPU retriever 改造影响。

stage2_agentic:
  需要 frozen agent continuation rollout。
  continuation 后续 search 会调用 retriever。
  默认改用 CPU retriever + load-balancing proxy。
```

因此实现和验证时要特别保证：

- `generate_traces` 支持 CPU retriever，同时保持 no-ranker agent rollout 语义不变。
- `generate_traces` 结束后能正确 cleanup CPU backend 和 proxy。
- 不影响 `build_reranker_dataset`。
- 不影响 `build_reranker_branch_dataset`。
- 不影响 `stage1_format` 的 8 卡训练。
- 只在 `stage2_agentic` 需要 continuation reward 时启动 recall 服务。

## 12. 260706 实测记录

### 12.1 generate_traces CPU retriever 基准

本次用 `generate_traces` 小样本验证 CPU retriever，参数如下：

```text
trace_max_samples: 8
backend_type: cpu
instance_count: 8
query_batch_size: 8
proxy.strategy: least_inflight
```

8 实例、每实例 8 线程的结果：

```text
总耗时: 456s
backend request 数: 19
backend total_elapsed_s mean: 43.98s
backend faiss_elapsed_s mean: 43.86s
backend encode_elapsed_s mean: 0.12s
错误数: 0
```

8 实例、每实例 16 线程的结果：

```text
总耗时: 380s
backend request 数: 17
backend total_elapsed_s mean: 33.28s
backend faiss_elapsed_s mean: 33.19s
backend encode_elapsed_s mean: 0.08s
错误数: 0
```

所以当前默认采用：

```text
instance_count: 8
cpu_threads_per_instance: 16
query_batch_size: 8
```

这个默认值的理由很直接：

- 8x16 相比 8x8，总耗时从 456s 降到 380s，约快 17%。
- 检索请求平均耗时从约 44s 降到约 33s。
- 8x16 的总线程数是 128，在 192 核机器上约 66.7%，低于 70% 阈值。
- 两组测试都没有 backend/proxy 错误。

### 12.2 当前瓶颈

CPU retriever 的耗时主要不是 encoder，也不是文档加载，而是 FAISS Flat index 的精确搜索：

```text
encode: 约 0.08s - 0.12s
load docs: 毫秒级
faiss Flat search: 数十秒级
```

所以 CPU backend 已经能把 NPU/GPU 释放出来，但如果后续还要求检索本身大幅提速，真正应该做的是：

```text
Flat exact index -> ANN index，例如 IVF / HNSW / PQ
```

单纯继续增加 CPU 线程，很可能会遇到内存带宽和 NUMA 竞争，不一定线性变快。

### 12.3 stage2_agentic 验证记录

本次也验证了 `train_llm_reranker.stage2_agentic` 的真实 NPU 训练 smoke：

```text
reranker_actor: NPU 0-3
frozen_agent_vllm: NPU 4-7, TP=4
recall: CPU retriever 8 instances + least_inflight proxy
stage2 total_training_steps: 1
```

结论：

- frozen agent 使用 4 张 NPU 的配置可启动。
- stage2 的 `agentic_rag_rollout_reward` 可以通过统一 proxy URL 调用 CPU retriever。
- 训练可以完成 1 step。
- stage 结束后 CPU retriever、proxy、frozen agent、Ray/vLLM 相关进程和端口均可清理。
