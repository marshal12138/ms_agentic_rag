# CoAgenticRetriever Eval 配置治理方案

## 1. 目标

eval 链路现在按训练 launcher 的配置治理方式组织：task 入口只选择配置组和 overlay，Python compiler 负责合并、校验和生成运行态文件，Bash launcher 只负责启动服务和执行 evaluator。

新链路只接受结构化 eval runtime、eval budget、resource 配置、task overlay、明确支持的 CLI override 和少量规范运行控制 env。用户面对的评估任务名统一叫 `eval_task_name`，运行报告和审计文件也只输出这套命名。

## 2. 配置入口

统一入口仍然是：

```text
CoAgenticRetriever/config/main_run/coAgenticRetriever_main.yaml
```

其中 `eval_config_groups` 只负责选择 eval 配置组：

```yaml
eval_config_groups:
  eval_runtime: coagentic_retriever_vllm
  eval_budget: coagentic_retriever_aligned_budget
  resource: local_eval_4gpu_0_3
```

`main_run` 不放 `run_mode`、`reranker`、模型路径、tool injection、vLLM 参数等细节。

## 3. YAML 文件

### `CoAgenticRetriever/config/eval_runtime/coagentic_retriever_vllm.yaml`

作用：描述 eval runtime 的稳定默认值，包括 evaluator、run mode、reranker、data、generation、vLLM、retrieval、ranker、LLM judge、tool 和 artifact。

关键字段：

```yaml
identity:
  eval_task_name: default
mode:
  run_mode: full
  reranker: dense_e5
entrypoint:
  evaluator: scripts/coagenticRetriever_v2/evaluate_coagentic_vllm.py
tool:
  tool_config: CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
  no_ranker_tool_config: CoAgenticRetriever/config/coagentic_retriever_tool_config_no_ranker.yaml
  inject_tool_schema: false
```

### `CoAgenticRetriever/config/eval_budget/coagentic_retriever_aligned_budget.yaml`

作用：描述 prompt、response、multi-turn 和 vLLM sequence budget。结构跟训练 Hydra 分支对齐：

```yaml
data:
  max_prompt_length: 12000
  max_response_length: 4096
  apply_chat_template_kwargs:
    enable_thinking: false
actor_rollout_ref:
  rollout:
    max_model_len: 16096
    max_num_seqs: 32
    multi_turn:
      max_assistant_turns: 6
      max_user_turns: 6
      max_tool_response_length: 4096
```

### `CoAgenticRetriever/config/resource/local_eval_4gpu_0_3.yaml`

作用：描述本地 eval 的资源布局和服务开关。

默认布局：

```yaml
AGENT_GPU_IDS: "0,1"
RANK_GPU_ID: "2"
RECALL_GPU_ID: "3"
AUTO_START_RECALL_SERVICE: "1"
AUTO_STOP_RECALL_SERVICE: "1"
```

### `tasks/eval_tasks/coAgenticRetriever/configs/eval_CAR_asy_labl_v0701a_npu_fix_overlay.yaml`

作用：只描述本次任务差异。

当前 no-ranker 验证任务包含：

```yaml
identity:
  eval_task_name: async_label_dpskv4f_v0702_no_ranker
mode:
  run_mode: no-ranker
  reranker: dense_e5
models:
  agent_model: /data01/.../global_step_79
  ranker_model: /data01/.../global_step_79
  ranker_base_model: /data01/.../e5-base-v2
data:
  data_path: /data01/.../co_search_ablation.eval.parquet
tool:
  inject_tool_schema: false
```

## 4. Eval Task 脚本形态

task 入口保持很薄：

```bash
bash "${ROOT}/scripts/coagenticRetriever_v2/02_infer_launcher.sh" \
  --main_run_config=coAgenticRetriever_main \
  --EVAL_RUNTIME_CONFIG=coagentic_retriever_vllm \
  --EVAL_BUDGET_CONFIG=coagentic_retriever_aligned_budget \
  --RESOURCE_CONFIG=local_eval_4gpu_0_3 \
  --OVERLAY_YAML=tasks/eval_tasks/coAgenticRetriever/configs/eval_CAR_asy_labl_v0701a_npu_fix_overlay.yaml \
  "$@"
```

task 脚本不直接写 `RUN_MODE`、`RERANKER`、模型路径、budget、tool injection 或设备布局。

## 5. 优先级

当前合并顺序：

1. `main_run` 选择默认 eval/resource/trainer 配置组。
2. 显式 launcher 参数覆盖 `main_run` 里的配置组选择。
3. 读取 eval runtime base。
4. 读取 eval budget base。
5. 读取 resource config。
6. 读取训练侧 data/model/rollout/ranker_base 配置，作为 eval 派生事实源。
7. 按顺序应用 task overlay。
8. 应用明确支持的 CLI `key=value` override。
9. 应用明确支持的环境变量 override。
10. compiler 派生 artifact 路径、runtime tool config、eval args 和 runtime env。

overlay 使用严格合并：写入 base YAML 中不存在的字段会直接报错。

## 6. 事实源

- `eval_task_name`：来自 `identity.eval_task_name`。
- `run_mode` / `reranker`：来自 `mode.*`。
- agent/ranker/data 路径：来自 `models.*` 和 `data.data_path`。
- prompt/response/multi-turn budget：来自 `eval_budget`。
- recall top-N：来自训练侧 ranker base 的 `recall_retriever.recall_final_top_n`。
- agent-visible top-M：来自 static tool config 的 `searchTool_final_top_m`。
- ranker top-K 和 token lengths：来自训练侧 ranker base 的 `ranker.*`。
- resource：来自 `resource/*.yaml`。
- NPU/CUDA 设备前缀：来自 `compatible_accelerator.sh`。

## 7. 输出文件

compiler 在 runtime log 目录生成：

- `<RUN_NAME>.eval_runtime_env.sh`
- `<RUN_NAME>.env`
- `<RUN_NAME>.eval_args.txt`
- `<RUN_NAME>.eval_overlay_yamls.txt`
- `<RUN_NAME>.eval_passthrough_args.txt`
- `<RUN_NAME>.final_eval_config.yaml`
- `<RUN_NAME>.final_eval_config.json`
- `<RUN_NAME>.tool_config.yaml`

这些文件是本次 eval 的审计事实。

## 8. 规范接口边界

新 eval launcher 的接口边界是配置组、overlay、明确支持的 CLI `key=value` override，以及少量规范运行控制 env。报告和 `.env` 只输出 eval task、resource、budget、retrieval、ranker、tool 和 artifact 的规范字段。

如果需要调整一次评估任务，应优先改 task overlay；临时覆盖时使用 compiler 支持的结构化字段名，不在 task 脚本里追加零散 Bash 变量。

## 9. 验证记录

已用最新 checkpoint 做过 `MAX_EVAL_NUM=1` 的 NPU no-ranker 验证，产物在：

```text
log/eval_res/coAgenticRetriever/260702-1439-async_label_dpskv4f_v0702_no_ranker
reports/eval/coAgenticRetriever/260702-1439-async_label_dpskv4f_v0702_no_ranker.report.md
```
