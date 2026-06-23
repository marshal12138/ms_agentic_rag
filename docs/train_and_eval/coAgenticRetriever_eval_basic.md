# CoAgenticRetriever 评估使用说明

## 入口脚本

本地评估入口：

```bash
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

推荐从仓库根目录运行：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
```

`02` 当前是 vLLM-only 评估入口：会启动 recall retriever 和 agent vLLM server，不再通过 VERL resume checkpoint 执行评估。

默认评估数据：

```text
data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet
```

## 三种模式

### ranker-only

只验证 recall top-N 到 dense ranker top-M 的排序链路，不启动 agent LLM。

```bash
RUN_MODE=ranker-only \
STRATEGY_NAME=ranker_base_e5_smoke \
RANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
RECALL_MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
RANK_GPU_ID=4 \
RECALL_GPU_ID=5 \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

默认只跑 `MAX_EVAL_STEPS=1` 条 smoke。要跑更多样本：

```bash
MAX_EVAL_STEPS=100
```

### full

完整 CoAgenticRetriever 评估：

```text
agent LLM -> recall retriever top-N -> dense ranker reorder -> top-M tool response -> agent LLM
```

评估未经训练的 base agent + base e5 ranker：

```bash
RUN_MODE=full \
STRATEGY_NAME=base_agent_base_ranker_full \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
RANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
RECALL_MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
AGENT_GPU_IDS=0,1 \
AGENT_TP_SIZE=2 \
RANK_GPU_ID=4 \
RECALL_GPU_ID=5 \
MAX_EVAL_NUM=100 \
TOP_N=50 \
TOP_M=5 \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

评估训练后的 agent/ranker 时，把模型变量指向 checkpoint/run 目录即可。脚本会自动尝试解析：

```text
agent: hf_safetensors/actor, actor/hf_safetensors, actor
ranker: rank_encoder, ranker/rank_encoder, retriever/rank_encoder
```

示例：

```bash
RUN_MODE=full \
STRATEGY_NAME=trained_agent_trained_ranker_full \
AGENT_MODEL=/path/to/coagentic_run_or_global_step \
RANKER_MODEL=/path/to/coagentic_run_or_global_step \
RANKER_BASE_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
RECALL_MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
MAX_EVAL_NUM=100 \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

### no-ranker

禁用 dense ranker，只使用 agent LLM 和 recall retriever：

```text
agent LLM -> recall retriever top-N -> recall top-M tool response -> agent LLM
```

```bash
RUN_MODE=no-ranker \
STRATEGY_NAME=base_agent_recall_only \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
RECALL_MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
AGENT_GPU_IDS=0,1 \
AGENT_TP_SIZE=2 \
RECALL_GPU_ID=5 \
MAX_EVAL_NUM=100 \
TOP_N=50 \
TOP_M=5 \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

no-ranker 下 agent 下一轮 think 前看到的是 `TOP_M` 条文档。默认就是：

```text
recall top50 -> 不做 ranker -> recall top5 给 agent
```

## Dry-run

正式运行前先检查命名和路径：

```bash
DRY_RUN=1 \
RUN_MODE=no-ranker \
STRATEGY_NAME=base_agent_recall_only \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
RECALL_MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

dry-run 不启动 recall service、不启动 vLLM、不执行推理，但会写出 `.env` 和一份 dry-run report。

## 常用配置

- `RUN_MODE`：`ranker-only`、`full`、`no-ranker`。
- `STRATEGY_NAME`：评估策略名，会影响自动 `TASK_NAME` 和产物目录。
- `TASK_NAME`：显式指定任务名，覆盖自动命名。
- `DATA_PATH` / `VAL_DATA`：评估 parquet；默认是 `data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`。
- `AGENT_MODEL`：agent LLM 路径，可以是 Qwen3-4B base 模型，也可以是训练后 checkpoint/run 目录。
- `MODEL_PATH`：兼容旧变量；未设置 `AGENT_MODEL` 时会作为 agent 模型路径。
- `RANKER_MODEL`：dense ranker 路径，可以是 e5-base-v2，也可以是训练后 checkpoint/run 目录。
- `RANKER_BASE_MODEL`：ranker tokenizer/base 路径；当 `RANKER_MODEL` 是 checkpoint 且缺 tokenizer 时建议显式设为 e5-base-v2。
- `RANKER_ENCODER_PATH`：直接指定 ranker encoder；通常可不填，由脚本自动解析。
- `RECALL_MODEL_PATH`：recall retriever/e5 encoder 路径。
- `MAX_EVAL_NUM`：full/no-ranker 的样本数，`-1` 表示全量。
- `MAX_EVAL_STEPS` / `MAX_RANKER_STEPS`：ranker-only 的样本数；默认 `1`。
- `BATCH_SIZE`：full/no-ranker 并发样本数。
- `ENABLE_THINKING`：agent tokenizer `apply_chat_template` 的 Qwen3 think/no-think 开关，默认 `true`；设为 `false` 时会向 evaluator 传 `--no-enable-thinking`。
- `TOP_N`：recall retriever 返回候选数，默认 `50`。
- `TOP_M`：agent tool response 中可见文档数，默认 `5`。
- `RANKER_TOP_K`：ranker 排序后可进入 agent 的上限之一，默认 `50`；最终可见数是 `min(TOP_M, RANKER_TOP_K, docs)`。
- `AGENT_GPU_IDS` / `AGENT_TP_SIZE`：agent vLLM GPU 和 tensor parallel。
- `RANK_GPU_ID`：本地 E5 ranker 所在 GPU。
- `RECALL_GPU_ID`：recall retriever 服务所在 GPU。
- `LLM_IO_JSONL`：保存 agent 输入输出 trace。
- `TRACE_DIR`、`REPORT_PATH`、`RUNTIME_LOG_DIR`：显式覆盖产物目录。

## 模型路径

未经训练模型对照评估：

```bash
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B
RANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
RECALL_MODEL_PATH=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
```

训练后 checkpoint/run 目录评估：

```bash
AGENT_MODEL=/path/to/run_or_global_step
RANKER_MODEL=/path/to/run_or_global_step
RANKER_BASE_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
```

`CHECKPOINT_DIR` 和 `RESUME_FROM_PATH` 仍作为兼容变量保留：如果没有显式设置 `AGENT_MODEL` / `RANKER_MODEL`，脚本会把它们作为模型源尝试解析。但它们不再触发 VERL resume。

## GPU 配置

recall service 默认：

```bash
RECALL_GPU_ID=5
PROXY_PORT=8030
RETRIEVAL_SERVICE_URL=http://127.0.0.1:8030/retrieve
```

如果 `RETRIEVAL_SERVICE_URL` 已可用，脚本会复用服务；否则 `AUTO_START_RECALL_SERVICE=1` 时自动启动。

full 模式建议把 agent、ranker、recall 分开放：

```bash
AGENT_GPU_IDS=0,1
AGENT_TP_SIZE=2
RANK_GPU_ID=4
RECALL_GPU_ID=5
```

no-ranker 不初始化本地 ranker，因此 `RANK_GPU_ID` 不参与真实执行。

## 命名和产物地址

默认 group：

```bash
GROUP_NAME=coAgenticRetriever
```

默认产物目录：

```text
log/eval_res/coAgenticRetriever/
reports/eval/coAgenticRetriever/
```

`STRATEGY_NAME` 会被 slugify，并参与自动任务命名：

```text
STRATEGY_NAME=base_agent_recall_only
STRATEGY_SLUG=base_agent_recall_only
TASK_NAME=260611-2124-base_agent_recall_only
```

默认路径：

```text
TRACE_DIR=log/eval_res/coAgenticRetriever/<TASK_NAME>
RUNTIME_LOG_DIR=log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs
REPORT_PATH=reports/eval/coAgenticRetriever/<TASK_NAME>.report.md
```

`RUN_NAME` 默认等于 `STRATEGY_NAME`，只影响 runtime log 文件名前缀：

```text
<RUN_NAME>.env
<RUN_NAME>.infer.log
<RUN_NAME>.metrics.jsonl
<RUN_NAME>.search_timing.jsonl
<RUN_NAME>.llm_io.jsonl
```

覆盖优先级：

```text
显式 TRACE_DIR / REPORT_PATH / RUNTIME_LOG_DIR
  > 显式 TASK_NAME
  > 自动 TASK_NAME
  > STRATEGY_NAME
```

## 主要产物

所有模式都会写：

```text
reports/eval/coAgenticRetriever/<TASK_NAME>.report.md
log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs/<RUN_NAME>.env
log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs/<RUN_NAME>.infer.log
log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs/<RUN_NAME>.recall_retriever_server.log
log/eval_res/coAgenticRetriever/<TASK_NAME>/summary.json
log/eval_res/coAgenticRetriever/<TASK_NAME>/run_config.json
log/eval_res/coAgenticRetriever/<TASK_NAME>/metrics.jsonl
log/eval_res/coAgenticRetriever/<TASK_NAME>/traces.jsonl
```

ranker-only 额外产物：

```text
log/eval_res/coAgenticRetriever/<TASK_NAME>/ranker_infer_smoke.jsonl
```

full / no-ranker 额外产物：

```text
log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs/<RUN_NAME>.metrics.jsonl
log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs/<RUN_NAME>.search_timing.jsonl
log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs/<RUN_NAME>.llm_io.jsonl
log/eval_res/coAgenticRetriever/<TASK_NAME>/validation_data/metrics.jsonl
log/eval_res/coAgenticRetriever/<TASK_NAME>/validation_data/traces.jsonl
log/eval_res/coAgenticRetriever/<TASK_NAME>/rollout_data/
```

report 会记录：

- run mode 和 ranker enabled。
- strategy、task、run name。
- agent、recall、ranker 模型路径。
- recall top-N、agent 可见 top-M、ranker top-K。
- 关键 JSONL 产物路径。
- full、no-ranker、ranker-only 的实际评估路径。

## no-ranker 与 full 的差异

full：

```text
recall top-N -> dense ranker 重排 -> top-M 给 agent
```

no-ranker：

```text
recall top-N -> 直接取 recall top-M 给 agent
```

两者保持一致的部分：

- agent LLM vLLM 调用方式。
- prompt/chat template 处理。
- search tool call 解析。
- tool response 格式。
- recall retriever 服务。
- agent 多轮搜索和最终回答流程。

no-ranker 不初始化本地 E5 ranker，也不会要求 ranker 模型路径存在。

## LLM IO Trace

调试 agent 输入输出：

```bash
LLM_IO_JSONL=/path/to/coagentic_eval_llm_io.jsonl \
COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS=20 \
RUN_MODE=no-ranker \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

默认路径：

```text
log/eval_res/coAgenticRetriever/<TASK_NAME>/runtime_logs/<RUN_NAME>.llm_io.jsonl
```

建议重点检查 tool response 之后的 agent turn，确认 agent 看到的是 `<tool_response>...</tool_response>` 中的 top-M passages。
