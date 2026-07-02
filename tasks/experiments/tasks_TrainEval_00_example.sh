#!/usr/bin/env bash
set -euo pipefail

# 示例：串行编排一次训练和一次评估。
# 这个脚本用于展示通用编排方式；默认参数保持真实任务形态，但不要求在示例中实际跑通。

ROOT="${ROOT:-/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives}"

# 通用任务编排器：
# - task_sequence_run: 等待指定 GPU 空闲后运行一个子任务，并记录日志/summary。
# - task_sequence_release_gpus: 释放指定 GPU 上当前用户的残留进程。
source "${ROOT}/src/runtime/task_sequence.sh"

# 本编排任务的日志名称。任务名只是 summary/log 中的标记，不是硬性约束。
TASK_SEQUENCE_NAME="${TASK_SEQUENCE_NAME:-train_eval_00_example}"

# 编排层统一控制等待 GPU，子任务自身的 WAIT_FOR_GPU_RELEASE 默认关闭，避免重复等待。
TASK_SEQUENCE_WAIT_FOR_GPUS="${TASK_SEQUENCE_WAIT_FOR_GPUS:-1}"
WAIT_FOR_GPU_INTERVAL_SECONDS="${WAIT_FOR_GPU_INTERVAL_SECONDS:-30}"
WAIT_FOR_GPU_TIMEOUT_SECONDS="${WAIT_FOR_GPU_TIMEOUT_SECONDS:-0}"

# 释放 GPU 的默认策略：
# - 默认开启释放，用于训练结束后确保后续评估可以接管 GPU。
# - 默认只释放当前用户进程，降低误杀其他用户任务的风险。
TASK_SEQUENCE_RELEASE_GPUS="${TASK_SEQUENCE_RELEASE_GPUS:-1}"
TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY="${TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY:-1}"
TASK_SEQUENCE_RELEASE_GRACE_SECONDS="${TASK_SEQUENCE_RELEASE_GRACE_SECONDS:-20}"

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix.sh}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${ROOT}/tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0622.sh}"

# 训练默认占用全套 async-ranker-training 资源：
# agent: 0,1,2,3；ranker: 4；retriever: 5；judge: 6,7。
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"

# 当前评估脚本默认使用 agent: 0,1；ranker: 2；retriever: 3。
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"

# 第 1 步：训练。
task_sequence_run "train-async-ranker-training" "${TRAIN_GPUS}" \
  env \
    WAIT_FOR_GPU_RELEASE=0 \
    RELEASE_GPUS_ON_EXIT="${RELEASE_GPUS_ON_EXIT:-1}" \
    bash "${TRAIN_SCRIPT}"

# 第 2 步：兜底释放训练占用的 GPU。
# 即使训练脚本未来自带 RELEASE_GPUS_ON_EXIT，这里也保留一个编排层兜底。
task_sequence_release_gpus "release-after-train" "${TRAIN_GPUS}"

# 第 3 步：评估。
task_sequence_run "eval-async-ranker-training" "${EVAL_GPUS}" \
  env \
    WAIT_FOR_GPU_RELEASE=0 \
    bash "${EVAL_SCRIPT}"
