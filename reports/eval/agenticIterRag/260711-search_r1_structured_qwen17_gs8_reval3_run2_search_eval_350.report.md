# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/search_r1_structured.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-010148-047274-pipeline-agentic_iter_rag_v1_search_r1_structured_qwen3_1_7b_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run2_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run2_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run2_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run2_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run2_search_eval_350/trace/validation_data`
- Wall time: `139.9113s`
- Status counts: `{'answered': 220, 'no_valid_answer': 50, 'max_turns': 78, 'multiple_tool_calls': 2}`

## Retrieval Cutoffs

- RECALL_FINAL_TOP_N: `50`
- SEARCH_TOOL_FINAL_TOP_M: `5`
- RANKER_FINAL_TOP_K: `50`

## Infer Path

- Search path: `agent LLM -> recall retriever recall_final_top_n=50 -> searchTool_final_top_m=5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.1429 | 0.2114 | 342 | 0.1462 | 0.2093 | 0.1494 |
| macro-average | 7 | 0.1429 | 0.2114 | 48 | 0.1470 | 0.2108 | 0.1506 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1400 | 0.1835 | 50 | 0.1400 | 0.1835 | 0.1400 |
| bamboogle | 50 | 0.1200 | 0.1810 | 50 | 0.1200 | 0.1810 | 0.1200 |
| hotpotqa | 50 | 0.0600 | 0.0992 | 50 | 0.0600 | 0.0992 | 0.0600 |
| musique | 50 | 0.0800 | 0.1210 | 50 | 0.0800 | 0.1210 | 0.0800 |
| nq | 50 | 0.1200 | 0.2436 | 43 | 0.1628 | 0.2523 | 0.1880 |
| popqa | 50 | 0.1600 | 0.2135 | 50 | 0.1600 | 0.2135 | 0.1600 |
| triviaqa | 50 | 0.3200 | 0.4380 | 49 | 0.3061 | 0.4252 | 0.3061 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.2571 | 18.0568 | 6.6048 | 0.0000 | 6.6048 | 24.6832 | 4.9714 |
| macro-average | 7 | 2.2571 | 18.0568 | 6.6048 | 0.0000 | 6.6048 | 24.6832 | 4.9714 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.1600 | 14.2660 | 1.9639 | 0.0000 | 1.9639 | 16.2458 | 5.0000 |
| bamboogle | 50 | 2.0800 | 22.1366 | 12.5696 | 0.0000 | 12.5696 | 34.7210 | 5.0000 |
| hotpotqa | 50 | 2.7000 | 22.3483 | 6.6273 | 0.0000 | 6.6273 | 28.9968 | 4.8000 |
| musique | 50 | 2.7000 | 26.7794 | 14.7718 | 0.0000 | 14.7718 | 41.5713 | 5.0000 |
| nq | 50 | 2.0000 | 17.1031 | 7.2212 | 0.0000 | 7.2212 | 24.3482 | 5.0000 |
| popqa | 50 | 2.6000 | 14.2079 | 1.9814 | 0.0000 | 1.9814 | 16.2337 | 5.0000 |
| triviaqa | 50 | 1.5600 | 9.5560 | 1.0984 | 0.0000 | 1.0984 | 10.6654 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.