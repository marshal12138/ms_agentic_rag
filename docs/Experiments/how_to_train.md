# Training Script Usage

本文记录当前训练入口脚本的命名规则、必要参数，以及日志和 checkpoint 的默认产出位置。

## 适用脚本

当前已接入统一命名和防覆盖规则的训练入口主要包括：

- `tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh`
- `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`
- `scripts/cosearch_local/09_train_qwen3_4b_dense_5step_probe.sh`
- `scripts/cosearch_local/09b_train_qwen3_4b_smallbatch_4retrievers_timing copy.sh`
- `scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh`

其中 `tasks/train_tasks/...` 是任务入口层，负责固定本次实验所需的 GPU、budget YAML、async-ranker-training YAML、tool schema 开关等任务参数；底层仍会调用 `scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh`。

这些脚本都遵循同一套命名思路：

- `GROUP_NAME`：实验分组名，用于目录分层
- `EXP_NAME`：实验语义名，用于表达本次实验内容
- `RUN_NAME`：一次具体运行的唯一实例名

## 三个关键命名参数

### 1. `GROUP_NAME`

`GROUP_NAME` 用于顶层目录分组。

默认值如下：

- `scripts/coagenticRetriever_local/*` 默认 `GROUP_NAME=coAgenticRetriever`
- `scripts/cosearch_local/*` 默认 `GROUP_NAME=cosearch`
- 共享兜底默认值为 `defaultGroup`

`GROUP_NAME` 会直接体现在日志和 checkpoint 路径中。

### 2. `EXP_NAME`

`EXP_NAME` 用于表达实验语义，例如：

- `qwen3_4b_probe_rule_v1`
- `coagentic_ranker_neg15_top5`
- `qwen3_4b_smallbatch_debug`

对于上述训练入口，推荐且默认要求显式传入 `EXP_NAME`。

示例：

```bash
EXP_NAME=qwen3_4b_probe_rule_v1 \
bash scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

### 3. `RUN_NAME`

如果不手工设置 `RUN_NAME`，脚本会自动生成：

```text
RUN_NAME=<YYMMDD-HHMMSS>-<EXP_NAME>
```

例如：

```text
260610-154710-qwen3_4b_probe_rule_v1
```

如果手工设置了 `RUN_NAME`，脚本会直接使用该值；此时 `EXP_NAME` 可以不提供。

但更推荐使用 `EXP_NAME` 自动生成 `RUN_NAME`，因为这样更容易保证唯一性。

## 运行限制与防覆盖规则

默认情况下，训练脚本会拒绝复用已有的非空日志目录或 checkpoint 目录。

也就是说，如果目标目录已经存在且非空，脚本会直接退出，避免覆盖旧实验产物。

只有在以下情况之一时，才允许复用目录：

- `ALLOW_RUN_REUSE=1`
- `ALLOW_DIR_REUSE=1`
- `RESUME_MODE!=disable`

因此，默认行为是：

- 不会悄悄覆盖旧日志
- 不会悄悄覆盖旧 checkpoint
- 需要你显式声明才允许复用

## 默认产出目录规则

### 训练日志目录

当前训练日志统一落在：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/train_logs/<GROUP_NAME>/<RUN_NAME>/
```

例如：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/train_logs/cosearch/260610-154710-qwen3_4b_probe_rule_v1/
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/train_logs/coAgenticRetriever/260610-154710-coagentic_ranker_neg15_top5/
```

该目录下常见文件包括：

- `<RUN_NAME>.train.log`
- `<RUN_NAME>.metrics.jsonl`
- `<RUN_NAME>.search_timing.jsonl`
- `<RUN_NAME>.nvidia_smi.csv`
- `<RUN_NAME>.env`
- `<RUN_NAME>.timing_report.step*.md`

### checkpoint 目录

当前训练 checkpoint 统一落在：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/<GROUP_NAME>/<RUN_NAME>/
```

例如：

```text
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/cosearch/260610-154710-qwen3_4b_probe_rule_v1/
/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260610-154710-coagentic_ranker_neg15_top5/
```

默认保留的是当前 run 对应的 checkpoint 根目录，目录下再按训练 step 组织：

```text
global_step_*/
```

## 四个训练脚本的默认分组

### `01_train_qwen3_4b_ablation_1epoch_timing.sh`

- 默认 `GROUP_NAME=coAgenticRetriever`
- 默认 checkpoint 根目录：

```text
checkpoints/qwen3_4b_probe/coAgenticRetriever/<RUN_NAME>/
```

- 默认日志根目录：

```text
log/train_logs/coAgenticRetriever/<RUN_NAME>/
```

示例：

```bash
EXP_NAME=coagentic_ranker_neg15_top5 \
bash scripts/coagenticRetriever_local/01_train_qwen3_4b_ablation_1epoch_timing.sh
```

### `tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh`

这是当前 CoAgenticRetriever async-ranker-training 训练任务入口。它在 task 层固定了常用实验参数：

- agent GPU：默认 `AGENT_GPU_IDS=0,1,2,3`
- dense ranker GPU：默认 `RANK_GPU_ID=4`
- recall retriever GPU：默认 `RECALL_GPU_ID=5`
- LLM judge GPU：默认 `LLM_JUDGE_GPU_IDS=6,7`
- async-ranker-training 配置：默认 `scripts/coagenticRetriever_local/strategies_yaml/async_ranker_training_deepseek_flash.yaml`
- rollout budget YAML：默认 `scripts/coagenticRetriever_local/strategies_yaml/rollout_cosearch_aligned_budget.yaml`
- tool schema 注入：默认 `INJECT_TOOL_SCHEMA=false`

直接运行：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

用于 smoke 时可以覆盖步数：

```bash
TOTAL_STEPS=2 \
bash tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

如果该训练任务由 `tasks/experiments/*.sh` 编排脚本调用，推荐在编排命令里设置：

```bash
WAIT_FOR_GPU_RELEASE=0
```

由编排层统一等待 GPU，避免重复等待。

### `09_train_qwen3_4b_dense_5step_probe.sh`

- 默认 `GROUP_NAME=cosearch`
- 默认 checkpoint 根目录：

```text
checkpoints/qwen3_4b_probe/cosearch/<RUN_NAME>/
```

- 默认日志根目录：

```text
log/train_logs/cosearch/<RUN_NAME>/
```

### `09b_train_qwen3_4b_smallbatch_4retrievers_timing copy.sh`

- 默认 `GROUP_NAME=cosearch`
- 默认 checkpoint 根目录：

```text
checkpoints/qwen3_4b_probe/cosearch/<RUN_NAME>/
```

- 默认日志根目录：

```text
log/train_logs/cosearch/<RUN_NAME>/
```

### `10_train_qwen3_4b_64batch_8retrievers.sh`

- 默认 `GROUP_NAME=cosearch`
- 默认 checkpoint 根目录：

```text
checkpoints/qwen3_4b_probe/cosearch/<RUN_NAME>/
```

- 默认日志根目录：

```text
log/train_logs/cosearch/<RUN_NAME>/
```

示例：

```bash
EXP_NAME=qwen3_4b_probe_rule_v1 \
bash scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

## 推荐用法

推荐始终显式指定 `EXP_NAME`，而不是手工写死 `RUN_NAME`。

推荐形式：

```bash
EXP_NAME=<你的实验名> \
bash <训练脚本>
```

例如：

```bash
EXP_NAME=qwen3_4b_smallbatch_debug \
bash "scripts/cosearch_local/09b_train_qwen3_4b_smallbatch_4retrievers_timing copy.sh"
```

如果确实需要自定义分组，也可以显式覆盖：

```bash
GROUP_NAME=myCustomGroup \
EXP_NAME=qwen3_4b_probe_rule_v2 \
bash scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

此时产物目录会变成：

```text
log/train_logs/myCustomGroup/<RUN_NAME>/
checkpoints/qwen3_4b_probe/myCustomGroup/<RUN_NAME>/
```

## 不推荐的用法

不推荐直接写死一个长期复用的 `RUN_NAME`，例如：

```bash
RUN_NAME=default \
bash scripts/cosearch_local/10_train_qwen3_4b_64batch_8retrievers.sh
```

虽然现在脚本会先检查并阻止覆盖，但这种用法仍然会让目录命名缺乏实验语义，也不利于后续排查和归档。

更合适的做法仍然是：

```bash
EXP_NAME=<语义清晰的实验名>
```

## Train + Eval 编排

跨训练和评估的串行实验不建议手动反复观察 GPU 再执行。推荐在 `tasks/experiments/` 中使用 `src/runtime/task_sequence.sh` 编排。

当前示例：

```text
tasks/experiments/tasks_TrainEval_00_example.sh
```

该示例会：

1. 等待训练所需 GPU。
2. 运行 `tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh`。
3. 训练后执行一次 GPU 释放兜底。
4. 等待评估所需 GPU。
5. 运行 `tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622.sh`。

先 dry-run 检查命令展开：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
TASK_SEQUENCE_DRY_RUN=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

真实运行：

```bash
TASK_SEQUENCE_RELEASE_GPUS=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

编排日志默认写入：

```text
log/task_sequences/<stamp>-<TASK_SEQUENCE_NAME>/
```
