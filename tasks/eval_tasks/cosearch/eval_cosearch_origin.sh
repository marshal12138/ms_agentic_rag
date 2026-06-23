#!/usr/bin/env bash

cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
STRATEGY_NAME=original_agent_llm_plus_llm_ranker \
RUN_MODE=full \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
RERANKER_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh

cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
STRATEGY_NAME=original_agent_llm_no_ranker \
RUN_MODE=no-ranker \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh

cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
STRATEGY_NAME=cosearch_agent_llm_plus_llm_ranker \
RUN_MODE=full \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors \
RERANKER_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh

cd /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives
STRATEGY_NAME=cosearch_agent_llm_no_ranker \
RUN_MODE=no-ranker \
AGENT_MODEL=/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors \
MAX_EVAL_NUM=-1 \
KEEP_TRACE=full \
bash scripts/cosearch_local/11_evaluate_cosearch_base.sh
