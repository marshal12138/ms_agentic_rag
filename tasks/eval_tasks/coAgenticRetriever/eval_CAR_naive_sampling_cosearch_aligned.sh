#!/usr/bin/env bash

cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/
STRATEGY_NAME=coageticretriever_origin_full \
AGENT_GPU_IDS=0,1 \
AGENT_TP_SIZE=2 \
RANK_GPU_ID=2 \
RECALL_GPU_ID=3 \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
RANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
RANKER_BASE_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
DATA_PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet \
RUN_MODE=full \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only_cosearch_aligned.sh



cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
STRATEGY_NAME=coageticretriever_origin_no_ranker \
AGENT_GPU_IDS=0,1 \
AGENT_TP_SIZE=2 \
RANK_GPU_ID=2 \
RECALL_GPU_ID=3 \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
RANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
RUN_MODE=no-ranker \
DATA_PATH=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet \
bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only_cosearch_aligned.sh



# cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
# STRATEGY_NAME=coagtret_naive_sampling_full \
# AGENT_GPU_IDS=0,1 \
# RANK_GPU_ID=2 \
# RECALL_GPU_ID=3 \
# AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260612-003302-CAR_mem_speed_v1/global_step_70 \
# RANKER_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260612-003302-CAR_mem_speed_v1/global_step_70 \
# RANKER_BASE_MODEL=/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2 \
# RUN_MODE=full \
# bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only_cosearch_aligned.sh



# cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
# STRATEGY_NAME=coagtret_naive_sampling_no_ranker \
# AGENT_GPU_IDS=0,1 \
# RANK_GPU_ID=2 \
# RECALL_GPU_ID=3 \
# AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260612-003302-CAR_mem_speed_v1/global_step_70 \
# RANKER_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260612-003302-CAR_mem_speed_v1/global_step_70 \
# RUN_MODE=no-ranker \
# bash scripts/coagenticRetriever_local/02_infer_qwen3_4b_ablation_val_only_cosearch_aligned.sh
