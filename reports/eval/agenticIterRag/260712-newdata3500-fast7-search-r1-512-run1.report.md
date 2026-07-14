# AgenticIterRag v1 Infer Report

## Run

- Status: exit_143
- Group: agenticIterRag
- Group slug: agenticIterRag
- Task: 260712-newdata3500-fast7-search-r1-512-run1
- Infer task: spad_agent_search_eval
- Infer task slug: spad_agent_search_eval
- Run name: spad_agent_search_eval
- Run mode: no-ranker
- Reranker: dense_e5
- Dataset: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet
- Trace dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/trace
- Runtime logs: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs

## Models

- Agent model: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8
- Recall model: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
- Ranker enabled: false
- Ranker model: not used
- Ranker base model: not used
- Ranker encoder path: not used
- LLM judge endpoint: http://127.0.0.1:8067/v1/chat/completions
- LLM judge model: DeepSeek-V4-Flash

## Artifacts

- Config env: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/spad_agent_search_eval.env
- Infer log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/spad_agent_search_eval.infer.log
- Recall service log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/spad_agent_search_eval.recall_retriever_server.log
- Metrics JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/trace/metrics.jsonl (500 rows)
- Agent timing JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/agent_timing.jsonl (2036 rows)
- Search timing JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/search_timing.jsonl (1356 rows)
- LLM IO JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/llm_io.jsonl (20 rows)
- LLM IO max records: 20
- Ranker output JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/trace/ranker_infer_smoke.jsonl (0 rows)
- Validation data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/trace/validation_data
- Rollout data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/trace/rollout_data
- Tool config: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/AgenticIterRag/config/agentic_iter_rag_tool_config.yaml
- Infer budget config: unknown

## Key Config

- RECALL_FINAL_TOP_N: 50
- SEARCH_TOOL_FINAL_TOP_M: 5
- RANKER_FINAL_TOP_K: 50
- MAX_INFER_NUM: 3500
- INFER_BATCH_SIZE: 336
- FLUSH_EVERY_N: 10
- ENABLE_THINKING: false
- MAX_MODEL_LEN: 12288
- STOP_SEQUENCES: none
- AIR_ACCELERATOR: npu
- ASCEND_RT_VISIBLE_DEVICES: 0,1,2,3,4,5,6
- AGENT_GPU_IDS: 0,1,2,3,4,5,6
- AGENT_TP_SIZE: 1
- AGENT_INSTANCE_COUNT: 7
- AGENT_BACKEND_BASE_PORT: 8241
- AGENT_PROXY_LOG: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/spad_agent_search_eval.agent_proxy.log
- AGENT_PROXY_STRATEGY: least_inflight
- RANK_GPU_ID: 4
- RANKER_CUDA_VISIBLE_DEVICES: 4
- RANKER_DEVICE: npu:0
- LLM_JUDGE_ENDPOINT: http://127.0.0.1:8067/v1/chat/completions
- LLM_JUDGE_MODEL: DeepSeek-V4-Flash
- RECALL_GPU_ID: 7
- RECALL_BACKEND_BASE_PORT: 8231
- RECALL_PROXY_LOG: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fast7-search-r1-512-run1/runtime_logs/spad_agent_search_eval.recall_retriever_server.proxy.log
- RETRIEVAL_SERVICE_URL: http://127.0.0.1:8230/retrieve
- AGENT_MAX_RETRIES: 3
- AGENT_RETRY_DELAY: 1.0
- AGENT_RETRY_BACKOFF: 2.0
- AGENT_HTTP_FORCE_CLOSE: true
- FAIL_ON_INFER_ERROR: false
- RETRIEVAL_MAX_RETRIES: 1
- RETRIEVAL_RETRY_DELAY: 0.5
- RETRIEVAL_RETRY_BACKOFF: 1.0
