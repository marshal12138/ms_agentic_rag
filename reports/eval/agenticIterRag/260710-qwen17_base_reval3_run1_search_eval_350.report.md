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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run1_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run1_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run1_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run1_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run1_search_eval_350/trace/validation_data`
- Wall time: `138.3179s`
- Status counts: `{'answered': 213, 'no_valid_answer': 72, 'multiple_tool_calls': 3, 'max_turns': 62}`

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
| micro-average | 350 | 0.1143 | 0.1835 |
| macro-average | 7 | 0.1143 | 0.1835 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0800 | 0.1421 |
| bamboogle | 50 | 0.1200 | 0.2183 |
| hotpotqa | 50 | 0.0400 | 0.1050 |
| musique | 50 | 0.1000 | 0.1203 |
| nq | 50 | 0.0800 | 0.1699 |
| popqa | 50 | 0.1200 | 0.1915 |
| triviaqa | 50 | 0.2600 | 0.3373 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.1914 | 18.3272 | 6.9059 | 0.0000 | 6.9059 | 25.2534 | 4.9571 |
| macro-average | 7 | 2.1914 | 18.3272 | 6.9059 | 0.0000 | 6.9059 | 25.2534 | 4.9571 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.1200 | 14.4857 | 1.8579 | 0.0000 | 1.8579 | 16.3814 | 4.9000 |
| bamboogle | 50 | 1.9600 | 21.7799 | 14.0755 | 0.0000 | 14.0755 | 35.8698 | 5.0000 |
| hotpotqa | 50 | 2.2000 | 18.4293 | 4.8004 | 0.0000 | 4.8004 | 23.2455 | 4.8000 |
| musique | 50 | 2.7200 | 28.2860 | 14.0937 | 0.0000 | 14.0937 | 42.3999 | 5.0000 |
| nq | 50 | 2.2200 | 22.0358 | 10.0018 | 0.0000 | 10.0018 | 32.0622 | 5.0000 |
| popqa | 50 | 2.4600 | 13.7219 | 1.7493 | 0.0000 | 1.7493 | 15.4888 | 5.0000 |
| triviaqa | 50 | 1.6600 | 9.5517 | 1.7626 | 0.0000 | 1.7626 | 11.3260 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.