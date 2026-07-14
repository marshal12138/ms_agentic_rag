# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/source/co_search_ablation.infer.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260710-113003-543853-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run1_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run1_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run1_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run1_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run1_search_eval_350/trace/validation_data`
- Wall time: `178.9298s`
- Status counts: `{'answered': 208, 'max_turns': 130, 'no_valid_answer': 12}`

## Retrieval Cutoffs

- RECALL_FINAL_TOP_N: `50`
- SEARCH_TOOL_FINAL_TOP_M: `5`
- RANKER_FINAL_TOP_K: `50`

## Infer Path

- Search path: `agent LLM -> recall retriever recall_final_top_n=50 -> searchTool_final_top_m=5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.1314 | 0.1979 |
| macro-average | 7 | 0.1314 | 0.1979 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0200 | 0.0774 |
| bamboogle | 50 | 0.0800 | 0.1296 |
| hotpotqa | 50 | 0.0600 | 0.1029 |
| musique | 50 | 0.0800 | 0.1319 |
| nq | 50 | 0.1600 | 0.2978 |
| popqa | 50 | 0.1800 | 0.2118 |
| triviaqa | 50 | 0.3400 | 0.4339 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.7629 | 22.2594 | 9.7162 | 0.0000 | 9.7162 | 31.9986 | 5.0000 |
| macro-average | 7 | 2.7629 | 22.2594 | 9.7162 | 0.0000 | 9.7162 | 31.9986 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 3.3200 | 17.7933 | 4.0610 | 0.0000 | 4.0610 | 21.8795 | 5.0000 |
| bamboogle | 50 | 2.5000 | 31.2100 | 11.2799 | 0.0000 | 11.2799 | 42.5074 | 5.0000 |
| hotpotqa | 50 | 3.0600 | 23.7348 | 12.4690 | 0.0000 | 12.4691 | 36.2260 | 5.0000 |
| musique | 50 | 3.4600 | 34.0073 | 23.7278 | 0.0000 | 23.7278 | 57.7607 | 5.0000 |
| nq | 50 | 2.0600 | 19.2562 | 9.5840 | 0.0000 | 9.5840 | 28.8544 | 5.0000 |
| popqa | 50 | 3.0400 | 16.5451 | 3.4351 | 0.0000 | 3.4351 | 20.0210 | 5.0000 |
| triviaqa | 50 | 1.9000 | 13.2692 | 3.4568 | 0.0000 | 3.4568 | 16.7411 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.