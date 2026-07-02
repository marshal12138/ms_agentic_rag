# `tasks/` 目录说明

`tasks/` 用于存放面向具体实验/评估/训练的数据入口脚本。它的定位是“任务入口层”，用于选择模型、checkpoint、GPU、策略和底层 launcher；通用逻辑应放在 `src/`，底层复用 launcher 仍放在 `scripts/`。

这样做的收益是：

- 避免 `scripts/` 目录继续膨胀为大量一次性实验入口。
- 让常用实验可以用稳定、可读的 task 脚本复现。
- 让 train/eval 编排脚本可以复用同一套 GPU 等待、释放和日志记录能力。

## 目录约定

- `tasks/train_tasks/`：训练任务入口。
- `tasks/eval_tasks/`：评估任务入口。
- `tasks/experiments/`：跨 train/eval 的串行实验编排脚本。
- `tasks/human_task_record.md`：人工任务记录。
- `tasks/train_tasks/train_tasks_records.md`：训练任务记录。

## Task 脚本职责

Task 脚本只负责“本次任务怎么跑”，例如：

- 选择底层 launcher。
- 设置模型、checkpoint、数据、GPU、策略名。
- 设置预算 YAML、tool schema 开关、ranker/retriever 路径等任务级参数。
- 调用 `src/runtime/wait_for_gpus.sh` 或由编排层统一等待 GPU。

Task 脚本不应承担：

- 通用日志系统实现。
- 通用 checkpoint 转换/清理实现。
- 通用 retriever server 实现。
- 多任务编排通用框架实现。

这些能力应放在 `src/`。

## GPU 等待

单个 task 脚本可以使用：

```bash
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-0,1,2,3}"
WAIT_FOR_GPU_RELEASE="${WAIT_FOR_GPU_RELEASE:-1}"
source "${ROOT}/src/runtime/wait_for_gpus.sh"
wait_for_gpus_if_enabled
```

如果 task 被 `tasks/experiments/*.sh` 编排脚本调用，推荐由编排层统一等待 GPU，并在子任务命令里传：

```bash
WAIT_FOR_GPU_RELEASE=0
```

这样可以避免父脚本和子脚本重复等待。

## 实验编排

跨多个 train/eval 任务的串行编排放在：

```text
tasks/experiments/
```

编排脚本应 source：

```bash
source "${ROOT}/src/runtime/task_sequence.sh"
```

常用函数：

```bash
task_sequence_run "任务标记" "0,1,2,3" bash path/to/task.sh
task_sequence_release_gpus "释放标记" "0,1,2,3"
```

其中任务标记只是日志和 `summary.tsv` 中的记号，不是硬性约束。

编排日志默认写入：

```text
log/task_sequences/<stamp>-<TASK_SEQUENCE_NAME>/
  summary.tsv
  001-<label>.log
  002-<label>.log
```

关键开关：

- `TASK_SEQUENCE_DRY_RUN=1`：只展开命令，不执行任务，也不释放 GPU。
- `TASK_SEQUENCE_WAIT_FOR_GPUS=1`：每个 `task_sequence_run` 前等待指定 GPU 空闲。
- `TASK_SEQUENCE_START_INDEX=N`：从第 N 个任务开始执行，便于失败后续跑。
- `TASK_SEQUENCE_CONTINUE_ON_FAIL=1`：子任务失败后继续后续任务。
- `TASK_SEQUENCE_RELEASE_GPUS=1`：`task_sequence_release_gpus` 真正发送信号释放 GPU。
- `TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY=1`：默认只释放当前用户进程。

## 当前示例

`tasks/experiments/tasks_TrainEval_00_example.sh` 演示了一次 train + eval 串行编排：

1. 运行训练任务：

```text
tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh
```

2. 训练结束后兜底释放训练 GPU。

3. 运行评估任务：

```text
tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622.sh
```

示例脚本可以用 dry-run 检查：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
TASK_SEQUENCE_DRY_RUN=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

真实运行时，如果希望 `task_sequence_release_gpus` 真的释放 GPU，需要设置：

```bash
TASK_SEQUENCE_RELEASE_GPUS=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

注意：GPU 释放是兜底动作，正常任务脚本仍应尽量自己关闭由它启动的服务。
