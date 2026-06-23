#!/usr/bin/env bash
set -euo pipefail

ROOT="/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives"

# 通用串行任务编排器：
# - task_sequence_run 会在任务启动前等待指定 GPU 空闲，然后执行命令。
# - task_sequence_release_gpus 用于显式释放指定 GPU 上的残留进程。
# - 每个任务的 stdout/stderr 会写入 log/task_sequences/<时间戳>-<任务组名>/。
source "${ROOT}/src/runtime/task_sequence.sh"

# 本次编排任务组名称，只用于日志目录命名；可用 TASK_SEQUENCE_NAME=... 覆盖。
TASK_SEQUENCE_NAME="${TASK_SEQUENCE_NAME:-eval_0622a}"

# 第 1 个评估任务：使用 3,4 作为 agent GPU，5 作为 recall GPU，6,7 作为 LLM judge GPU。
# 第一个参数 "dpskv4f-rank-b" 只是日志标记，不影响子任务逻辑。
# 这里由编排器统一等待 GPU，所以传 WAIT_FOR_GPU_RELEASE=0，避免子任务内部重复等待。
task_sequence_run "dpskv4f-rank-b" "3,4,5,6,7" \
  env WAIT_FOR_GPU_RELEASE=0 \
  bash "${ROOT}/tasks/eval_tasks/coAgenticRetriever/eval_CAR_testing_dpskv4f_as_rank_0622b.sh"

# 显式释放 LLM judge 使用的 GPU 6,7。
# 默认只列出候选进程，不会真正 kill；需要真正释放时显式设置：
#   TASK_SEQUENCE_RELEASE_GPUS=1 bash tasks/eval_tasks/coAgenticRetriever/tasks_0622a.sh
task_sequence_release_gpus "release-judge-6-7" "6,7"

# 第 2 个评估任务：同样等待 3,4,5,6,7 空闲后再启动。
task_sequence_run "dpskv4f-rank-origin" "3,4,5,6,7" \
  env WAIT_FOR_GPU_RELEASE=0 \
  bash "${ROOT}/tasks/eval_tasks/coAgenticRetriever/eval_CAR_testing_dpskv4f_as_rank_0622.sh"
