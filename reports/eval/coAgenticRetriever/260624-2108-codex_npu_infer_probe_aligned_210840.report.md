# CoAgenticRetriever vLLM Evaluation Report

## Run

- Status: dry-run
- Group: coAgenticRetriever
- Group slug: coAgenticRetriever
- Task: 260624-2108-codex_npu_infer_probe_aligned_210840
- Strategy: codex_npu_infer_probe_aligned_210840
- Run name: codex_npu_infer_probe_aligned_210840
- Run mode: full
- Reranker: dense_e5
- Dataset: data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet
- Trace dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840
- Runtime logs: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/runtime_logs

## Models

- Agent model: /data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B
- Recall model: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
- Ranker enabled: true
- Ranker model: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
- Ranker base model: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
- Ranker encoder path: /data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2
- LLM judge endpoint: http://127.0.0.1:8067/v1/chat/completions
- LLM judge model: DeepSeek-V4-Flash

## Artifacts

- Config env: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/runtime_logs/codex_npu_infer_probe_aligned_210840.env
- Infer log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/runtime_logs/codex_npu_infer_probe_aligned_210840.infer.log
- Recall service log: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/runtime_logs/codex_npu_infer_probe_aligned_210840.recall_retriever_server.log
- Metrics JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/runtime_logs/codex_npu_infer_probe_aligned_210840.metrics.jsonl (0 rows)
- Search timing JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/runtime_logs/codex_npu_infer_probe_aligned_210840.search_timing.jsonl (0 rows)
- LLM IO JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/runtime_logs/codex_npu_infer_probe_aligned_210840.llm_io.jsonl (0 rows)
- Ranker output JSONL: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/ranker_infer_smoke.jsonl (0 rows)
- Validation data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/validation_data
- Rollout data dir: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260624-2108-codex_npu_infer_probe_aligned_210840/rollout_data
- Tool config: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/CoAgenticRetriever/config/coagentic_retriever_tool_config.yaml
- Eval budget YAML: /data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/scripts/coagenticRetriever_local/strategies_yaml/rollout_cosearch_aligned_budget.yaml

## Key Config

- TOP_N: 50
- TOP_M: 5
- RANKER_TOP_K: 50
- MAX_EVAL_NUM: -1
- EVAL_BATCH_SIZE: 32
- ENABLE_THINKING: false
- MAX_MODEL_LEN: 16096
- STOP_SEQUENCES: none
- AGENT_GPU_IDS: 6
- RANK_GPU_ID: 4
- RANKER_CUDA_VISIBLE_DEVICES: 4
- RANKER_DEVICE: npu:4
- LLM_JUDGE_ENDPOINT: http://127.0.0.1:8067/v1/chat/completions
- LLM_JUDGE_MODEL: DeepSeek-V4-Flash
- RECALL_GPU_ID: 5
- RETRIEVAL_SERVICE_URL: http://127.0.0.1:8030/retrieve
