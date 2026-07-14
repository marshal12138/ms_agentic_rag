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
- Agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-1.7B`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run3_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run3_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run3_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run3_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run3_search_eval_350/trace/validation_data`
- Wall time: `135.2529s`
- Status counts: `{'answered': 214, 'no_valid_answer': 78, 'multiple_tool_calls': 3, 'max_turns': 55}`

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
| micro-average | 350 | 0.0971 | 0.1771 |
| macro-average | 7 | 0.0971 | 0.1771 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.1139 |
| bamboogle | 50 | 0.1200 | 0.2135 |
| hotpotqa | 50 | 0.0400 | 0.1047 |
| musique | 50 | 0.1000 | 0.1253 |
| nq | 50 | 0.0800 | 0.1912 |
| popqa | 50 | 0.1400 | 0.2026 |
| triviaqa | 50 | 0.1600 | 0.2884 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.1371 | 17.5813 | 6.6005 | 0.0000 | 6.6005 | 24.2010 | 4.9571 |
| macro-average | 7 | 2.1371 | 17.5813 | 6.6005 | 0.0000 | 6.6005 | 24.2010 | 4.9571 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.0800 | 14.3745 | 2.0574 | 0.0000 | 2.0574 | 16.4469 | 4.9000 |
| bamboogle | 50 | 2.1200 | 22.5459 | 14.0841 | 0.0000 | 14.0841 | 36.6454 | 5.0000 |
| hotpotqa | 50 | 2.2000 | 17.0755 | 5.0834 | 0.0000 | 5.0834 | 22.1746 | 4.8000 |
| musique | 50 | 2.8200 | 28.8461 | 13.7748 | 0.0000 | 13.7748 | 42.6416 | 5.0000 |
| nq | 50 | 1.8400 | 17.1205 | 7.2879 | 0.0000 | 7.2879 | 24.4297 | 5.0000 |
| popqa | 50 | 2.3400 | 12.6414 | 2.1376 | 0.0000 | 2.1376 | 14.8144 | 5.0000 |
| triviaqa | 50 | 1.5600 | 10.4652 | 1.7784 | 0.0000 | 1.7784 | 12.2546 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.