# 环境参数 + 任务参数
EXP_NAME="CAR_async_labeling_ds_flash_larger_ranker_tdata" \
bash /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/tasks/train_tasks/coAgenticRetriever/train_CAR_async_labeling_ds_flash_mix_signal_fix_v1.sh \
  ranker_training.async_labeling.sample_builder.num_groups_per_step=96 \
  ranker_training.async_labeling.sample_builder_request_batch=3