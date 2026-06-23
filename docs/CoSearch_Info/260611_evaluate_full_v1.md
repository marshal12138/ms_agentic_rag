# CoSearch vLLM 评估脚本全流程说明

本文梳理评估入口：

```bash
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

该脚本是 CoSearch 当前 vLLM-only 评估入口。它负责准备评估目录、启动 retriever 服务、启动 retrieval proxy、启动 agent vLLM server、启动 reranker vLLM server，然后调用 `scripts/cosearch_local/evaluate_cosearch_vllm.py run` 完成逐样本评估、trace 保存、指标汇总和 markdown 报告生成。

## 1. 入口定位

脚本路径：

```text
scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

推荐从仓库根目录运行：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

该脚本开头使用：

```bash
set -euo pipefail
```

含义是：

- 任一命令失败时退出。
- 使用未定义变量时报错。
- 管道中任一阶段失败时整体失败。

因此它适合作为正式评估入口。若写任务脚本串行跑多个策略，也建议任务脚本同样使用 `set -euo pipefail`。

## 2. 总体执行顺序

直接运行脚本后，执行顺序如下：

1. 解析仓库根目录 `ROOT` 和 Python 解释器 `PY`。
2. 加载 `src/logs/report_system/logging_reports.sh`。
3. 设置 group、strategy、模型、数据、GPU、端口、retriever、prompt、采样和超时等默认参数。
4. 调用 `setup_cosearch_eval_artifact_defaults` 自动生成评估任务名、trace 目录、报告路径和运行时日志目录。
5. 如果 `DRY_RUN=1`，打印关键配置后退出，不启动任何服务。
6. 解析 agent/reranker 模型目录。
7. 启动 dense retriever 实例。
8. 启动 retrieval round-robin proxy。
9. 启动 agent vLLM OpenAI-compatible server。
10. 启动 reranker vLLM OpenAI-compatible server。
11. 调用 `evaluate_cosearch_vllm.py run` 执行评估。
12. 写出 `traces.jsonl`、`metrics.jsonl`、`summary.json`、`run_config.json` 和 markdown report。
13. 脚本退出时通过 `trap cleanup EXIT` 清理本次启动的服务进程组。

## 3. 默认配置

默认 Python：

```bash
PY=/data04/envs/ms/ms_cosearch_official/bin/python
```

默认模型和数据：

```bash
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B
RERANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B
DATA_PATH=${ROOT}/data/co_search/local_flashrag/co_search_ablation.eval.parquet
MAX_EVAL_NUM=-1
BATCH_SIZE=32
KEEP_TRACE=partial
```

其中：

- `MAX_EVAL_NUM=-1` 表示评估完整数据集。
- `MAX_EVAL_NUM=100` 表示只取 parquet 前 100 条。
- `KEEP_TRACE=partial` 保存轻量 trace。
- `KEEP_TRACE=full` 额外保存每次 tool call 的 retrieval top-50 chunks。

默认 GPU 和 vLLM 配置：

```bash
AGENT_GPU_IDS=0,1
RERANKER_GPU_IDS=2,3
AGENT_TP_SIZE=2
RERANKER_TP_SIZE=2
AGENT_PORT=8040
RERANKER_PORT=8041
AGENT_SERVED_MODEL=cosearch-agent
RERANKER_SERVED_MODEL=cosearch-reranker
GPU_MEMORY_UTILIZATION=0.60
MAX_NUM_SEQS=32
MAX_MODEL_LEN=12288
```

默认 retriever 配置：

```bash
RETRIEVER_INSTANCES=1
RETRIEVER_PORT_BASE=8020
PROXY_PORT=8030
PROXY_TIMEOUT=180
RETRIEVER_MODE=gpu
RETRIEVER_DEVICE=cuda
RETRIEVER_GPU_ID=5
RETRIEVER_GPU_IDS=5
RETRIEVER_DOC_DTYPE=float16
RETRIEVER_QUERY_BATCH_SIZE=32
RETRIEVER_STARTUP_TIMEOUT=900
```

默认 prompt、tool 和生成限制：

```bash
TOP_N=50
TOP_M=5
MAX_ASSISTANT_TURNS=6
MAX_USER_TURNS=6
MAX_PROMPT_LENGTH=11264
MAX_RESPONSE_LENGTH=1024
MAX_TOOL_RESPONSE_LENGTH=4096
RERANKER_MAX_PROMPT_LENGTH=16384
RERANKER_MAX_RESPONSE_LENGTH=1024
TEMPERATURE=0.0
TOP_P=1.0
RERANKER_TEMPERATURE=0.0
REQUEST_TIMEOUT=180
VLLM_STARTUP_TIMEOUT=1800
```

## 4. 配置覆盖方法

所有配置都通过环境变量覆盖。常见写法：

```bash
STRATEGY_NAME=cosearch_agent_llm_plus_llm-reranker \
AGENT_MODEL=/path/to/agent_or_checkpoint \
RERANKER_MODEL=/path/to/reranker_or_checkpoint \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

只评估少量数据：

```bash
MAX_EVAL_NUM=100 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

调整 GPU：

```bash
AGENT_GPU_IDS=0,1 \
AGENT_TP_SIZE=2 \
RERANKER_GPU_IDS=2,3 \
RERANKER_TP_SIZE=2 \
RETRIEVER_GPU_ID=5 \
RETRIEVER_GPU_IDS=5 \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

使用 CPU/Search-R1 native FAISS retriever：

```bash
RETRIEVER_MODE=cpu \
RETRIEVER_DEVICE=cpu \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

指定固定任务名，避免自动命名：

```bash
TASK_NAME=my_eval_run_v1 \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

指定完整输出目录：

```bash
TRACE_DIR=/path/to/eval_trace \
REPORT_PATH=/path/to/report.md \
RUNTIME_LOG_DIR=/path/to/runtime_logs \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

## 5. 任务自动命名机制

任务自动命名由 `src/logs/report_system/logging_reports.sh` 中的 `setup_cosearch_eval_artifact_defaults` 完成。

默认 group：

```bash
GROUP_NAME=cosearch
GROUP_SLUG=cosearch
```

默认 strategy：

```bash
STRATEGY_NAME=default
```

脚本会先生成：

```bash
STRATEGY_SLUG=$(slugify_cosearch_name "${STRATEGY_NAME}")
```

注意这里最大长度是 10。因此：

```text
original_agent_llm_plus_llm-reranker -> original_a
cosearch_agent_llm_plus_llm-reranker -> cosearch_a
```

然后生成：

```bash
TASK_NAME=$(date +%y%m%d-%H%M)-${STRATEGY_SLUG}
```

例如：

```text
260611-1958-cosearch_a
```

最终默认路径为：

```bash
EVAL_LOG_ROOT=${ROOT}/log/eval_res/${GROUP_SLUG}
EVAL_REPORT_ROOT=${ROOT}/reports/eval/${GROUP_SLUG}
TRACE_DIR=${EVAL_LOG_ROOT}/${TASK_NAME}
REPORT_PATH=${EVAL_REPORT_ROOT}/${TASK_NAME}.report.md
RUNTIME_LOG_DIR=${TRACE_DIR}/runtime_logs
```

也就是：

```text
log/eval_res/cosearch/<TASK_NAME>/
reports/eval/cosearch/<TASK_NAME>.report.md
log/eval_res/cosearch/<TASK_NAME>/runtime_logs/
```

注意事项：

- 自动命名精度是分钟级。
- `STRATEGY_SLUG` 默认保留完整安全化后的策略名，不再截断为前 10 个字符。
- 如果同一分钟内用相同策略名重复启动，可能写入同一个 `TASK_NAME`。
- 不同策略名只要安全化后的完整 slug 不同，就不会因为前缀相同而冲突。
- 需要完全避免冲突时，显式设置 `TASK_NAME`。

## 6. Dry-run 检查

运行：

```bash
DRY_RUN=1 bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

脚本会打印但不启动服务：

```text
TASK_NAME=...
TRACE_DIR=...
RUNTIME_LOG_DIR=...
REPORT_PATH=...
AGENT_MODEL=...
RERANKER_MODEL=...
AGENT_GPU_IDS=...
AGENT_TP_SIZE=...
RERANKER_GPU_IDS=...
RERANKER_TP_SIZE=...
RETRIEVER_MODE=...
RETRIEVER_DEVICE=...
RETRIEVER_GPU_ID=...
RETRIEVER_GPU_IDS=...
RETRIEVER_LAUNCHER=...
RETRIEVAL_PROXY=...
```

建议每次正式跑大评估前先 dry-run，确认：

- 模型路径正确。
- agent/reranker/retriever GPU 没有冲突。
- `TASK_NAME` 和输出路径符合预期。
- 当前工作目录是仓库根目录，或脚本路径可以正确解析。

## 7. 模型路径解析

在真正启动服务前，脚本先调用：

```bash
python scripts/cosearch_local/evaluate_cosearch_vllm.py resolve-model --path "${AGENT_MODEL}" --role agent
python scripts/cosearch_local/evaluate_cosearch_vllm.py resolve-model --path "${RERANKER_MODEL}" --role reranker
```

如果传入的是可加载 HuggingFace 模型目录，会直接使用该目录。

如果传入的是 VERL checkpoint step 或更上层目录，解析逻辑会自动搜索可加载 HF safetensors：

agent 优先：

```text
hf_safetensors/actor
actor/hf_safetensors
actor
```

reranker 优先：

```text
hf_safetensors/reranker_actor_rollout
reranker_actor_rollout/hf_safetensors
reranker_actor_rollout
```

例如传入：

```text
.../global_step_79/hf_safetensors
```

对于 agent，会解析到：

```text
.../global_step_79/hf_safetensors/actor
```

对于 reranker，会解析到：

```text
.../global_step_79/hf_safetensors/reranker_actor_rollout
```

解析结果会打印到控制台：

```text
resolved agent model: ...
resolved reranker model: ...
```

## 8. 服务启动过程

### 8.1 进程生命周期

脚本维护四类进程：

- dense retriever 进程组。
- retrieval proxy 进程组。
- agent vLLM server 进程组。
- reranker vLLM server 进程组。

每个服务通过 `setsid` 启动为独立进程组。脚本设置了：

```bash
trap cleanup EXIT
```

正常结束、失败退出或被中断时，会向本脚本启动的进程组发送 `TERM`：

```bash
kill -TERM -<pgid>
```

如果服务在启动前已经存在且健康，脚本会复用它，但不会把它加入本次 `cleanup` 列表。

### 8.2 启动 dense retriever

函数：

```bash
start_retrievers
```

默认：

```bash
RETRIEVER_INSTANCES=1
RETRIEVER_PORT_BASE=8020
```

所以默认只启动：

```text
http://127.0.0.1:8020/retrieve
```

如果设置：

```bash
RETRIEVER_INSTANCES=4
```

则启动：

```text
8020, 8021, 8022, 8023
```

每个 retriever 启动前先用 `check_retriever_url` 发送一个 POST 请求做健康检查：

```json
{
  "queries": ["who got the first nobel prize in physics?"],
  "topk": 1,
  "return_scores": true
}
```

如果端口已有健康服务，会打印：

```text
using existing retriever: http://127.0.0.1:<port>/retrieve
```

否则启动：

```bash
bash "${ROOT}/src/retrievers/start_dense_retriever_server.sh"
```

传入环境变量包括：

```bash
PY
PORT
MODE
DEVICE
GPU_ID
RETRIEVER_GPU_IDS
DOC_DTYPE
QUERY_BATCH_SIZE
FAISS_GPU=0
OMP_NUM_THREADS
MKL_NUM_THREADS
```

运行日志写到：

```text
${RUNTIME_LOG_DIR}/retriever_<port>.log
```

启动后等待健康检查通过。超时时间：

```bash
RETRIEVER_STARTUP_TIMEOUT=900
```

如果 retriever 提前退出或超时，脚本会打印日志尾部并退出。

### 8.3 retriever launcher 内部逻辑

launcher 路径：

```text
src/retrievers/start_dense_retriever_server.sh
```

当前评估脚本默认传入：

```bash
MODE=gpu
DEVICE=cuda
GPU_ID=5
RETRIEVER_GPU_IDS=5
DOC_DTYPE=float16
QUERY_BATCH_SIZE=32
```

launcher 会检查检索资源：

```text
data/retrieval/wiki-18/e5_Flat.index
data/retrieval/wiki-18/wiki-18.jsonl
```

默认 retriever 模型：

```text
/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
```

GPU 模式使用本仓库实现：

```text
src/retrievers/gpu_dense_retriever_server.py
```

CPU 模式使用 Search-R1 原生 server：

```text
CoSearch/Search-R1/search_r1/search/retrieval_server.py
```

GPU 模式下，`gpu_dense_retriever_server.py` 会：

- 读取 FAISS flat index。
- 将 doc embeddings 加载到 GPU。
- 使用 e5 encoder 编码 query。
- 用 torch 矩阵乘法计算 query-doc 相似度。
- 提供 `/retrieve` HTTP 接口。

### 8.4 启动 retrieval proxy

函数：

```bash
start_proxy
```

默认端口：

```bash
PROXY_PORT=8030
```

proxy 路径：

```text
src/retrievers/retrieval_round_robin_proxy.py
```

它会把所有 retriever 后端注册为：

```bash
--backend http://127.0.0.1:8020/retrieve
--backend http://127.0.0.1:8021/retrieve
...
```

请求入口：

```text
http://127.0.0.1:8030/retrieve
```

proxy 行为：

- 对 `/retrieve` 请求做 round-robin 分发。
- 某个 backend 失败时尝试下一个 backend。
- 所有 backend 都失败时返回 502。
- 返回 JSON 中会附加 `_proxy_backend` 和 `_proxy_elapsed_s`。

proxy 日志：

```text
${RUNTIME_LOG_DIR}/retrieval_proxy.log
```

### 8.5 启动 agent vLLM server

函数：

```bash
start_vllm_server "agent" ...
```

默认：

```bash
AGENT_GPU_IDS=0,1
AGENT_TP_SIZE=2
AGENT_PORT=8040
AGENT_SERVED_MODEL=cosearch-agent
```

启动命令核心是：

```bash
CUDA_VISIBLE_DEVICES="${AGENT_GPU_IDS}" \
python -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 \
  --port 8040 \
  --model "${AGENT_MODEL_RESOLVED}" \
  --served-model-name cosearch-agent \
  --tensor-parallel-size 2 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --dtype bfloat16 \
  --enforce-eager
```

健康检查：

```text
http://127.0.0.1:8040/v1/models
```

日志：

```text
${RUNTIME_LOG_DIR}/agent_vllm_8040.log
```

### 8.6 启动 reranker vLLM server

函数：

```bash
start_vllm_server "reranker" ...
```

默认：

```bash
RERANKER_GPU_IDS=2,3
RERANKER_TP_SIZE=2
RERANKER_PORT=8041
RERANKER_SERVED_MODEL=cosearch-reranker
```

健康检查：

```text
http://127.0.0.1:8041/v1/models
```

日志：

```text
${RUNTIME_LOG_DIR}/reranker_vllm_8041.log
```

## 9. 评估主流程

服务全部 ready 后，脚本调用：

```bash
python scripts/cosearch_local/evaluate_cosearch_vllm.py run ...
```

关键参数：

```bash
--agent-model "${AGENT_MODEL_RESOLVED}"
--reranker-model "${RERANKER_MODEL_RESOLVED}"
--data-path "${DATA_PATH}"
--max-eval-num "${MAX_EVAL_NUM}"
--batch-size "${BATCH_SIZE}"
--keep-trace "${KEEP_TRACE}"
--trace-dir "${TRACE_DIR}"
--report-path "${REPORT_PATH}"
--strategy-name "${STRATEGY_NAME}"
--retrieval-url "http://127.0.0.1:${PROXY_PORT}/retrieve"
--agent-base-url "http://127.0.0.1:${AGENT_PORT}"
--reranker-base-url "http://127.0.0.1:${RERANKER_PORT}"
--top-n "${TOP_N}"
--top-m "${TOP_M}"
```

`evaluate_cosearch_vllm.py` 内部流程：

1. 加载 agent tokenizer 和 reranker tokenizer。
2. 读取 parquet 数据。
3. 如果 `MAX_EVAL_NUM >= 0`，截取前 N 条。
4. 按 `BATCH_SIZE` 控制 asyncio 并发。
5. 每条样本执行 agent rollout。
6. agent 生成 tool call 后，请求 retrieval proxy 获取 top-N 文档。
7. 调用 reranker vLLM 对 top-N 文档重排，取 top-M。
8. 将 top-M 文档格式化成 Qwen chat-template tool message。
9. agent 继续生成，直到得到 `<answer>...</answer>`、没有有效答案或达到最大 turn。
10. 计算 EM/F1、工具调用数、各阶段耗时。
11. 写 trace、metrics、summary 和 report。

## 10. Prompt 和 tool response 规则

评估脚本有意保持与训练 prompt 语义一致。

关键点：

- 初始 prompt 来自数据集 `prompt` 字段。
- 使用 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)` 渲染。
- 检索结果不手工拼接裸字符串，而是构造 `role="tool"` message。
- tool response 使用训练路径中的 formatter：
  - `format_tool_response(...)`
  - `format_tool_response_with_docid_map(...)`
- 插入 tool message 后，保留新的 user/tool block 和下一轮 assistant generation prefix。

这保证最终 answer-stage prompt 结构类似：

```text
assistant
<reason>...</reason>
<tool_call>...</tool_call>
user
<tool_response>
...
</tool_response>
assistant
```

## 11. 失败样本处理

当前 evaluator 对单条样本有异常兜底。

如果某条样本失败，例如 vLLM 返回：

```text
maximum context length ... request has ... input tokens
```

不会中断整批评估。该样本会被记录为：

```json
{
  "status": "failed",
  "error_type": "context_length_exceeded",
  "em": 0.0,
  "f1": 0.0,
  "tool_calls": 0,
  "agent_turns": 0
}
```

如果是其它异常，`error_type` 默认为异常类名，例如：

```text
RuntimeError
TimeoutError
JSONDecodeError
```

失败样本仍会进入：

```text
traces.jsonl
metrics.jsonl
summary.json
report.md
```

并且 EM/F1 按 0 进入总体平均，不会被过滤掉。

summary 中会额外写：

```json
{
  "status_counts": {
    "answered": 100,
    "failed": 3
  },
  "success_count": 100,
  "failure_count": 3
}
```

报告中也会显示：

```text
Success count
Failure count
Status counts
```

## 12. 日志系统

本评估脚本的日志分两类。

### 12.1 控制台日志

控制台会打印关键阶段：

```text
resolved agent model: ...
resolved reranker model: ...
trace dir: ...
runtime logs: ...
report: ...
starting dense retriever ...
retriever ready: ...
starting retrieval proxy ...
vLLM ready: ...
evaluation complete
report: ...
trace: ...
runtime logs: ...
```

这些日志适合快速判断当前卡在哪个阶段。

### 12.2 runtime logs

所有后台服务 stdout/stderr 会重定向到：

```text
${RUNTIME_LOG_DIR}/
```

默认结构：

```text
log/eval_res/cosearch/<TASK_NAME>/runtime_logs/
  retriever_8020.log
  retrieval_proxy.log
  agent_vllm_8040.log
  reranker_vllm_8041.log
```

如果 `RETRIEVER_INSTANCES=4`，还会有：

```text
retriever_8021.log
retriever_8022.log
retriever_8023.log
```

排查建议：

- retriever 启动失败，看 `retriever_<port>.log`。
- proxy 返回 502，看 `retrieval_proxy.log` 和各 retriever 日志。
- vLLM server 起不来，看 `agent_vllm_8040.log` 或 `reranker_vllm_8041.log`。
- context length、OOM、模型加载失败，大多会在对应 vLLM log 中出现。

## 13. 报告系统

评估报告由 `evaluate_cosearch_vllm.py` 的 `write_report` 写出，不依赖训练报告生成器。

默认报告路径：

```text
reports/eval/cosearch/<TASK_NAME>.report.md
```

报告包含：

- Strategy。
- Dataset。
- Examples。
- Success count。
- Failure count。
- Agent model。
- Reranker model。
- Trace dir。
- Wall time。
- Status counts。
- Effect Metrics。
- Effect Metrics By Dataset。
- Performance Metrics。
- Performance Metrics By Dataset。

Effect metrics 表包含：

```text
N
EM
F1
```

Performance metrics 表包含：

```text
Tool Calls
Agent Turn Avg s
Agent Total Avg s
Retrieve Total Avg s
Reranker Total Avg s
Recall Call Avg s
Recall Total Avg s
Total Avg s
```

报告中的 micro-average 是所有样本整体平均。macro-average 是先按 `data_source` 聚合，再对各数据源平均。

## 14. 产物目录和文件

一次默认评估会生成：

```text
log/eval_res/cosearch/<TASK_NAME>/
  traces.jsonl
  metrics.jsonl
  summary.json
  run_config.json
  runtime_logs/
    retriever_8020.log
    retrieval_proxy.log
    agent_vllm_8040.log
    reranker_vllm_8041.log

reports/eval/cosearch/<TASK_NAME>.report.md
```

### 14.1 traces.jsonl

逐样本 trace，每行一个 JSON。

核心字段：

```text
index
data_source
prompt
sub_queries
reranked_top5_chunks
final_answer
ground_truth_answer
status
metrics
stage_records
```

当 `KEEP_TRACE=full` 时，还会包含：

```text
retrieved_top50_chunks
```

### 14.2 metrics.jsonl

逐样本指标，每行一个 JSON。

核心字段：

```text
index
data_source
em
f1
tool_calls
agent_turns
agent_decision_total_s
agent_decision_avg_s
retrieve_total_s
reranker_total_s
recall_total_s
recall_avg_s
total_s
status
```

失败样本额外有：

```text
error_type
error_message
```

### 14.3 summary.json

聚合指标。

核心结构：

```json
{
  "micro": {},
  "macro": {},
  "by_data_source": {},
  "status_counts": {},
  "success_count": 0,
  "failure_count": 0
}
```

### 14.4 run_config.json

本次评估参数快照，由 `EvalArgs` 序列化得到。

包含：

```text
agent_model
reranker_model
data_path
max_eval_num
batch_size
keep_trace
trace_dir
report_path
strategy_name
retrieval_url
agent_base_url
reranker_base_url
top_n
top_m
max_prompt_length
max_response_length
reranker_max_prompt_length
reranker_max_response_length
temperature
request_timeout
llm_io_jsonl
```

## 15. LLM IO trace

默认：

```bash
LLM_IO_JSONL=
```

不会额外写 LLM IO 文件。

如果要记录模型输入输出：

```bash
LLM_IO_JSONL=/path/to/eval_llm_io.jsonl \
COSEARCH_LLM_IO_MAX_RECORDS=20 \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

记录中包含：

```text
role
assistant_turn
prompt_text
output_text
prompt_token_count
output_token_count
sampling_params
```

用于排查训练/评估 prompt 是否一致，尤其是 tool response 插入后的第二轮 agent prompt。

## 16. 串行运行多个评估

一个任务脚本中可以串行运行多个策略：

```bash
#!/usr/bin/env bash
set -euo pipefail

STRATEGY_NAME=original_agent_llm_plus_llm-reranker \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
RERANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh

STRATEGY_NAME=cosearch_agent_llm_plus_llm-reranker \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors \
RERANKER_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```

这会先跑第一个评估，完成并清理服务后，再跑第二个评估。

注意：

- 两次评估默认会分别启动和清理服务。
- 如果两个评估在同一分钟内使用相同策略名，建议显式设置不同 `TASK_NAME`。
- 如果希望复用外部已启动服务，需要确保端口健康；脚本会复用健康服务，但本次退出不会清理外部服务。

## 17. 常见问题定位

### 17.1 端口已有服务

脚本会先健康检查。如果健康，会复用：

```text
using existing retriever: ...
using existing retrieval proxy: ...
using existing agent vLLM server: ...
using existing reranker vLLM server: ...
```

如果已有服务不是本次配置对应的模型或 retriever 数据，可能导致结果不可信。正式评估前建议确认端口占用。

### 17.2 retriever 资源缺失

如果缺少：

```text
data/retrieval/wiki-18/e5_Flat.index
data/retrieval/wiki-18/wiki-18.jsonl
```

retriever log 中会报 retrieval index/corpus not found。

### 17.3 vLLM context overflow

单样本 context 超限不会中断整批评估。该样本会记为：

```text
status=failed
error_type=context_length_exceeded
em=0
f1=0
```

可在 `metrics.jsonl` 中 grep：

```bash
grep context_length_exceeded log/eval_res/cosearch/<TASK_NAME>/metrics.jsonl
```

### 17.4 显存不足

检查：

```text
runtime_logs/agent_vllm_8040.log
runtime_logs/reranker_vllm_8041.log
runtime_logs/retriever_8020.log
```

可降低：

```bash
BATCH_SIZE
MAX_NUM_SEQS
GPU_MEMORY_UTILIZATION
RETRIEVER_INSTANCES
```

或重新分配：

```bash
AGENT_GPU_IDS
RERANKER_GPU_IDS
RETRIEVER_GPU_IDS
```

## 18. 推荐正式运行模板

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives

TASK_NAME=260611-cosearch-full-v1 \
STRATEGY_NAME=cosearch_agent_llm_plus_llm-reranker \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors \
RERANKER_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
AGENT_GPU_IDS=0,1 \
AGENT_TP_SIZE=2 \
RERANKER_GPU_IDS=2,3 \
RERANKER_TP_SIZE=2 \
RETRIEVER_MODE=gpu \
RETRIEVER_GPU_ID=5 \
RETRIEVER_GPU_IDS=5 \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
```
