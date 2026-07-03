# `tasks/experiments/` 目录说明

`tasks/experiments/` 用于更高层的实验编排，不用于存放单个训练或评估任务入口。

## 定位

该目录中的脚本负责串行或组合调用多个已有 task，例如：

- 先运行一个 `tasks/train_tasks/` 训练任务，再运行一个或多个 `tasks/eval_tasks/` 评估任务。
- 等待外部训练任务释放 GPU 后，串行运行多组评估。
- 在多个 train/eval 子任务之间做 GPU 等待、兜底释放和日志汇总。

单个训练任务应放在：

```text
tasks/train_tasks/
```

单个评估任务应放在：

```text
tasks/eval_tasks/
```

因此，单个训练 pipeline 即使内部包含多个 stage，也仍属于 `tasks/train_tasks/`。只有当一个脚本需要编排多个独立 train/eval task 时，才应放入 `tasks/experiments/`。

## 和 train/eval task 的区别

`tasks/train_tasks/` 和 `tasks/eval_tasks/` 表达“一个具体任务怎么跑”，包括选择底层 launcher、配置组和 overlay。

`tasks/experiments/` 表达“多个任务如何排队运行”，通常会 source：

```bash
source "${ROOT}/src/runtime/task_sequence.sh"
```

并使用：

```bash
task_sequence_run "任务标记" "0,1,2,3" bash path/to/task.sh
task_sequence_release_gpus "释放标记" "0,1,2,3"
```

编排日志默认写入：

```text
log/task_sequences/<stamp>-<TASK_SEQUENCE_NAME>/
```

常用开关：

- `TASK_SEQUENCE_DRY_RUN=1`：只展开命令，不执行子任务。
- `TASK_SEQUENCE_WAIT_FOR_GPUS=1`：每个子任务前等待对应 GPU 空闲。
- `TASK_SEQUENCE_START_INDEX=N`：从第 N 个子任务开始执行，便于失败后续跑。
- `TASK_SEQUENCE_CONTINUE_ON_FAIL=1`：子任务失败后继续执行后续任务。
- `TASK_SEQUENCE_RELEASE_GPUS=1`：允许 `task_sequence_release_gpus` 真正释放 GPU 进程。

## 当前示例

`tasks_TrainEval_00_example.sh` 演示一次训练后接一次评估。

`tasks_TrainEval_0622a.sh` 演示等待外部训练结束后串行运行多组评估，并在阶段之间释放 GPU。

示例 dry-run：

```bash
cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
TASK_SEQUENCE_DRY_RUN=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

真实运行并允许编排层释放 GPU：

```bash
TASK_SEQUENCE_RELEASE_GPUS=1 bash tasks/experiments/tasks_TrainEval_00_example.sh
```

## 约束

- 不要把单个训练 pipeline task 放在 `tasks/experiments/`。
- 不要在这里重复实现底层 launcher 逻辑。
- 不要在这里写通用日志、服务启动、checkpoint 转换逻辑。
- 编排层可以统一等待和释放 GPU，子任务内部应通过 `WAIT_FOR_GPU_RELEASE=0` 避免重复等待。
