# Evaluation Script Usage

本文记录 `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh` 的运行模式、默认 checkpoint、GPU 默认值，以及推理产物位置。

## 适用脚本

- `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh`
- `tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622.sh`

其中 `tasks/eval_tasks/...` 是任务入口层，用于固定某组评估实验的模型、checkpoint、GPU、策略名和 budget YAML；底层仍调用 `scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh`。

## 名称规则

这个脚本不强制要求 `EXP_NAME`。默认：

- `RUN_NAME=coagentic_retriever_infer_smoke`
- `EXP_NAME=${RUN_NAME}`
- `GROUP_NAME=coAgenticRetriever`

`setup_cosearch_eval_artifact_defaults` 会基于 `RUN_NAME` 生成：

```text
TASK_NAME=<YYMMDD-HHMM>-<STRATEGY_SLUG>
```

主要产物会落到：

- `log/eval_res/<GROUP_SLUG>/<TASK_NAME>/`
- `reports/eval/<GROUP_SLUG>/<TASK_NAME>.report.md`

其中常用路径是：

- `OUT_DIR=${TRACE_DIR}`
- `LOG_DIR=${RUNTIME_LOG_DIR}`

## 运行模式

### 1. `RUN_MODE=dense-reranker-only`

只跑 dense rank-retriever 推理校验，不启动 VERL。

主要输出：

- `retriever_infer_smoke.jsonl`
- `${RUN_NAME}.infer.log`
- `${RUN_NAME}.env`

控制样本数的主要参数：

- `MAX_EVAL_STEPS`

### 2. `RUN_MODE=full`

现在会真正执行 VERL `val_only` full eval。

它会：

- 自动解析 checkpoint 到具体的 `global_step_*`
- 用 `trainer.resume_mode=resume_path`
- 用 `trainer.val_before_train=True`
- 用 `trainer.val_only=True`
- 仅加载 actor 模型参数：`actor_rollout_ref.actor.checkpoint.load_contents=['model']`

同时 rank retriever 不再回退到基座 e5，而是显式指向 checkpoint 内的：

```text
<resume_from_path>/retriever/rank_encoder
```

控制验证集样本数的主要参数：

- `VAL_MAX_SAMPLES`

## 默认 checkpoint

默认 `CHECKPOINT_DIR`：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coagentic_retriever_contrastive_smoke
```

脚本会按下面顺序解析到真正的 `global_step_*`：

1. 如果传入的就是 `global_step_*` 目录，直接使用
2. 否则优先读取 `latest_checkpointed_iteration.txt`
3. 再否则退化到选择目录下最新的 `global_step_*`

解析后的路径会写到环境文件中的：

- `RESUME_FROM_PATH`

## 默认 GPU

`RUN_MODE=full` 默认使用：

- `AGENT_GPU_IDS=6`
- `RANK_GPU_ID=4`
- `RECALL_GPU_ID=5`

因此 full eval 默认可见卡是：

```text
FULL_EVAL_GPU_IDS=6,4
```

其中：

- agent LLM 在 GPU 6
- rank retriever 在当前进程内使用可见设备索引 `cuda:1`，对应物理 GPU 4
- recall service 单独拉起在物理 GPU 5

## full eval 产物

`RUN_MODE=full` 常见输出：

- `runtime_logs/${RUN_NAME}.env`
- `runtime_logs/${RUN_NAME}.infer.log`
- `runtime_logs/${RUN_NAME}.metrics.jsonl`
- `runtime_logs/${RUN_NAME}.search_timing.jsonl`
- `runtime_logs/${RUN_NAME}.recall_retriever_server.log`（仅自启 recall service 时）
- `rollout_data/`
- `validation_data/`

## 示例

### dense-only

```bash
RUN_MODE=dense-reranker-only \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

### full eval

```bash
RUN_MODE=full \
AGENT_GPU_IDS=6 \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

如果要评估指定 checkpoint，可直接传根目录或具体 `global_step_*`：

```bash
RUN_MODE=full \
CHECKPOINT_DIR=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coagentic_retriever_contrastive_smoke \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only.sh
```

### task 层评估入口

当前 async-ranker-training 主评估任务入口：

```text
tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622.sh
```

该脚本默认固定：

- `RUN_MODE=full`
- `INJECT_TOOL_SCHEMA=false`
- `EVAL_BUDGET_YAML=scripts/coagenticRetriever_local/strategies_yaml/rollout_cosearch_aligned_budget.yaml`
- agent/ranker/retriever GPU 和 checkpoint 路径由脚本内环境变量配置，可在调用时覆盖。

运行：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622.sh
```

如果由 `tasks/experiments/*.sh` 编排脚本调用，推荐在编排命令里设置：

```bash
WAIT_FOR_GPU_RELEASE=0
```

由编排层统一等待 GPU。

## Train + Eval 编排中的评估

跨训练和评估的串行任务建议使用：

```text
src/runtime/task_sequence.sh
```

当前示例：

```text
tasks/experiments/tasks_TrainEval_00_example.sh
```

它会先运行训练任务，再释放训练 GPU，最后运行：

```text
tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622.sh
```

dry-run：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
TASK_SEQUENCE_DRY_RUN=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

真实运行时，如果希望编排层的 GPU 释放动作真正生效：

```bash
TASK_SEQUENCE_RELEASE_GPUS=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```
