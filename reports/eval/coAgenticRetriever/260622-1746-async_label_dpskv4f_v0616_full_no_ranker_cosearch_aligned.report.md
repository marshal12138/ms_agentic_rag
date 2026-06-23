# CoAgenticRetriever vLLM Evaluation Report

## Run

- Status: dry-run
- Group: coAgenticRetriever
- Group slug: coAgenticRetriever
- Task: 260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned
- Strategy: async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned
- Run name: async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned
- Run mode: no-ranker
- Reranker: dense_e5
- Dataset: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet
- Trace dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned
- Runtime logs: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/runtime_logs

## Models

- Agent model: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260619-153011-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79
- Recall model: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
- Ranker enabled: false
- Ranker model: not used
- Ranker base model: not used
- Ranker encoder path: not used
- LLM judge endpoint: http://127.0.0.1:8067/v1/chat/completions
- LLM judge model: DeepSeek-V4-Flash

## Artifacts

- Config env: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned.env
- Infer log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned.infer.log
- Recall service log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned.recall_retriever_server.log
- Metrics JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned.metrics.jsonl (0 rows)
- Search timing JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned.search_timing.jsonl (0 rows)
- LLM IO JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned.llm_io.jsonl (0 rows)
- Ranker output JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/ranker_infer_smoke.jsonl (0 rows)
- Validation data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/validation_data
- Rollout data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-1746-async_label_dpskv4f_v0616_full_no_ranker_cosearch_aligned/rollout_data
- Tool config: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml

## Key Config

- TOP_N: 50
- TOP_M: 5
- RANKER_TOP_K: 50
- MAX_EVAL_NUM: -1
- EVAL_BATCH_SIZE: 32
- ENABLE_THINKING: false
- MAX_MODEL_LEN: 12288
- STOP_SEQUENCES: none
- AGENT_GPU_IDS: 0,1
- RANK_GPU_ID: 2
- RANKER_CUDA_VISIBLE_DEVICES: 2
- RANKER_DEVICE: cuda:4
- LLM_JUDGE_ENDPOINT: http://127.0.0.1:8067/v1/chat/completions
- LLM_JUDGE_MODEL: DeepSeek-V4-Flash
- RECALL_GPU_ID: 3
- RETRIEVAL_SERVICE_URL: http://127.0.0.1:8035/retrieve
