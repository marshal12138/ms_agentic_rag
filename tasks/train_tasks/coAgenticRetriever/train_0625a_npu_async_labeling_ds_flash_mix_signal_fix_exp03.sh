#!/usr/bin/env bash
set -euo pipefail

# 环境参数 + 任务参数
export RUN_MODE="${RUN_MODE:-no-ranker}"

EXP_NAME="CAR_async_npu_smaller_bs_per_gpu" \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh \
  ranker_training.async_labeling.sample_builder.num_groups_per_step=96 \
  ranker_training.async_labeling.sample_builder_request_batch=3 \
  "$@" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.actor.use_torch_compile=false \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
