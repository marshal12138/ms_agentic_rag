# CoAgenticRetriever vLLM Evaluation Report

## Run

- Status: dry-run
- Group: coAgenticRetriever
- Group slug: coAgenticRetriever
- Task: 260702-final-check-async_label_dpskv4f_v0702_no_ranker
- Strategy: async_label_dpskv4f_v0702_no_ranker
- Run name: async_label_dpskv4f_v0702_no_ranker
- Run mode: no-ranker
- Reranker: dense_e5
- Dataset: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet
- Trace dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker
- Runtime logs: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs

## Models

- Agent model: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260702-010936-CAR_async_ranker_training_ds_flash_mix_signal_b3_v1_select_all/global_step_79
- Recall model: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
- Ranker enabled: false
- Ranker model: not used
- Ranker base model: not used
- Ranker encoder path: not used
- LLM judge endpoint: http://127.0.0.1:8067/v1/chat/completions
- LLM judge model: DeepSeek-V4-Flash

## Artifacts

- Config env: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.env
- Infer log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.infer.log
- Recall service log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.recall_retriever_server.log
- Metrics JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.metrics.jsonl (0 rows)
- Search timing JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.search_timing.jsonl (0 rows)
- LLM IO JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.llm_io.jsonl (0 rows)
- Ranker output JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/ranker_infer_smoke.jsonl (0 rows)
- Validation data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/validation_data
- Rollout data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/rollout_data
- Tool config: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-final-check-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.tool_config.yaml
- Eval budget YAML: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/eval_budget/coagentic_retriever_aligned_budget.yaml

## Key Config

- RECALL_FINAL_TOP_N: 50
- SEARCH_TOOL_FINAL_TOP_M: 5
- RANKER_FINAL_TOP_K: 50
- Runtime alias TOP_N: 50
- Runtime alias TOP_M: 5
- Runtime alias RANKER_TOP_K: 50
- MAX_EVAL_NUM: 1
- EVAL_BATCH_SIZE: 32
- ENABLE_THINKING: false
- MAX_MODEL_LEN: 16096
- STOP_SEQUENCES: none
- COSEARCH_ACCELERATOR: npu
- ASCEND_RT_VISIBLE_DEVICES: 0,1
- AGENT_GPU_IDS: 0,1
- RANK_GPU_ID: 2
- RANKER_CUDA_VISIBLE_DEVICES: 2
- RANKER_DEVICE: npu:0
- LLM_JUDGE_ENDPOINT: http://127.0.0.1:8067/v1/chat/completions
- LLM_JUDGE_MODEL: DeepSeek-V4-Flash
- RECALL_GPU_ID: 3
- RETRIEVAL_SERVICE_URL: http://127.0.0.1:8030/retrieve
