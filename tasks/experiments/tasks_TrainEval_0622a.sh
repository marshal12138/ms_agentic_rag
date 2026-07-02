#!/usr/bin/env bash
set -euo pipefail

# 0622a 编排：当前训练任务已经在外部启动。
# 本脚本只负责等待训练占用的 GPU 释放，然后串行运行三组评估。

ROOT="${ROOT:-/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives}"
source "${ROOT}/src/runtime/task_sequence.sh"

# 任务名只是 log/task_sequences 下的标记，不是硬性约束。
TASK_SEQUENCE_NAME="${TASK_SEQUENCE_NAME:-train_eval_0622a}"

# 当前正在进行的训练默认占用 0-7。先等待它结束，再执行后续评估。
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"

# 评估 1：当前训练产物 + dense E5 ranker / no-ranker 两组。
EVAL_ASYNC_LABEL_GPUS="${EVAL_ASYNC_LABEL_GPUS:-0,1,2,3}"
EVAL_ASYNC_LABEL_SCRIPT="${EVAL_ASYNC_LABEL_SCRIPT:-${ROOT}/tasks/eval_tasks/coAgenticRetriever/eval_CAR_async_label_dpskv4f_v0623a.sh}"

# 评估 2：原始 Qwen3 agent + DeepSeek-V4-Flash judge reranker。
EVAL_ORI_AGENT_JUDGE_GPUS="${EVAL_ORI_AGENT_JUDGE_GPUS:-3,4,5,6,7}"
EVAL_ORI_AGENT_JUDGE_SCRIPT="${EVAL_ORI_AGENT_JUDGE_SCRIPT:-${ROOT}/tasks/eval_tasks/coAgenticRetriever/eval_CAR_testing_dpskv4f_as_rank_0622.sh}"

# 评估 3：当前训练产物 agent + DeepSeek-V4-Flash judge reranker。
# 06_infer_qwen3_4b_coagentic.sh 默认 AUTO_STOP_LLM_JUDGE=0。
# 因此评估 2 启动的 judge vLLM 可能继续占用 6,7，并被评估 3 复用。
# 评估 3 启动前只等待 agent/retriever GPU 3,4,5，避免因为复用中的 judge 服务而阻塞。
EVAL_TRAINED_AGENT_JUDGE_WAIT_GPUS="${EVAL_TRAINED_AGENT_JUDGE_WAIT_GPUS:-3,4,5}"
EVAL_JUDGE_RELEASE_GPUS="${EVAL_JUDGE_RELEASE_GPUS:-3,4,5,6,7}"
EVAL_TRAINED_AGENT_JUDGE_SCRIPT="${EVAL_TRAINED_AGENT_JUDGE_SCRIPT:-${ROOT}/tasks/eval_tasks/coAgenticRetriever/eval_CAR_testing_dpskv4f_as_rank_0622b.sh}"

# 编排层统一等待 GPU；子任务内部等待关闭，避免重复等待。
TASK_SEQUENCE_WAIT_FOR_GPUS="${TASK_SEQUENCE_WAIT_FOR_GPUS:-1}"
WAIT_FOR_GPU_INTERVAL_SECONDS="${WAIT_FOR_GPU_INTERVAL_SECONDS:-300}"
WAIT_FOR_GPU_TIMEOUT_SECONDS="${WAIT_FOR_GPU_TIMEOUT_SECONDS:-0}"

# 默认真正释放 GPU。只释放当前用户进程，降低误杀风险。
TASK_SEQUENCE_RELEASE_GPUS="${TASK_SEQUENCE_RELEASE_GPUS:-1}"
TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY="${TASK_SEQUENCE_RELEASE_CURRENT_USER_ONLY:-1}"
TASK_SEQUENCE_RELEASE_GRACE_SECONDS="${TASK_SEQUENCE_RELEASE_GRACE_SECONDS:-20}"

# 第 1 步：等待当前外部训练任务结束。
# 使用空命令作为屏障：task_sequence_run 会先等待 TRAIN_GPUS 空闲，然后执行 bash -lc true。
task_sequence_run "wait-current-train-finished" "${TRAIN_GPUS}" \
  bash -lc "true"

# 第 2 步：训练结束后的残留 GPU 释放兜底。
task_sequence_release_gpus "release-after-current-train" "${TRAIN_GPUS}"

# 第 3 步：评估当前训练产物的 dense-ranker full/no-ranker 两组。
task_sequence_run "eval-async-label-dense-and-no-ranker" "${EVAL_ASYNC_LABEL_GPUS}" \
  env WAIT_FOR_GPU_RELEASE=0 \
  bash "${EVAL_ASYNC_LABEL_SCRIPT}"

# 第 4 步：dense/no-ranker 评估结束后的残留释放。
task_sequence_release_gpus "release-after-async-label-eval" "${EVAL_ASYNC_LABEL_GPUS}"

# 第 5 步：原始 agent LLM + DeepSeek judge ranker。
task_sequence_run "eval-origin-agent-llm-judge-ranker" "${EVAL_ORI_AGENT_JUDGE_GPUS}" \
  env WAIT_FOR_GPU_RELEASE=0 \
  bash "${EVAL_ORI_AGENT_JUDGE_SCRIPT}"

# 第 6 步：trained agent LLM + DeepSeek judge ranker。
# 这里只等待 3,4,5；6,7 上的 judge vLLM 可以由上一组评估保留并复用。
task_sequence_run "eval-trained-agent-llm-judge-ranker" "${EVAL_TRAINED_AGENT_JUDGE_WAIT_GPUS}" \
  env WAIT_FOR_GPU_RELEASE=0 \
  bash "${EVAL_TRAINED_AGENT_JUDGE_SCRIPT}"

# 第 7 步：释放全部 GPU。
task_sequence_release_gpus "release-after-all-evals" "0,1,2,3,4,5,6,7"

# 第 8 步：训练新策略。
task_sequence_run "train-async_ranker_training_ds_flash_mix_signal-with-larger-ranker-train-data" "0,1,2,3,4,5,6,7" \
  env WAIT_FOR_GPU_RELEASE=0 \
  bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/coAgenticRetriever/train_CAR_async_ranker_training_ds_flash_mix_signal_fix_exp02.sh

# 第 9 步：释放全部 GPU。
task_sequence_release_gpus "release-after-all-evals" "0,1,2,3,4,5,6,7"

