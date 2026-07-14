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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run2_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run2_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run2_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run2_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_reval3_run2_search_eval_350/trace/validation_data`
- Wall time: `135.2488s`
- Status counts: `{'answered': 207, 'no_valid_answer': 81, 'multiple_tool_calls': 3, 'max_turns': 59}`

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
| micro-average | 350 | 0.1000 | 0.1770 |
| macro-average | 7 | 0.1000 | 0.1770 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1000 | 0.1675 |
| bamboogle | 50 | 0.1000 | 0.1816 |
| hotpotqa | 50 | 0.0200 | 0.1029 |
| musique | 50 | 0.0800 | 0.1035 |
| nq | 50 | 0.0600 | 0.1640 |
| popqa | 50 | 0.1200 | 0.1915 |
| triviaqa | 50 | 0.2200 | 0.3283 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.1514 | 17.4852 | 7.1810 | 0.0000 | 7.1810 | 24.6859 | 4.9571 |
| macro-average | 7 | 2.1514 | 17.4852 | 7.1810 | 0.0000 | 7.1810 | 24.6859 | 4.9571 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.1600 | 15.1281 | 1.7306 | 0.0000 | 1.7306 | 16.8948 | 4.9000 |
| bamboogle | 50 | 2.0400 | 21.5693 | 15.0905 | 0.0000 | 15.0905 | 36.6749 | 5.0000 |
| hotpotqa | 50 | 2.2000 | 16.5927 | 5.2833 | 0.0000 | 5.2833 | 21.8925 | 4.8000 |
| musique | 50 | 2.6600 | 26.4224 | 15.8785 | 0.0000 | 15.8785 | 42.3212 | 5.0000 |
| nq | 50 | 1.9000 | 18.2420 | 9.0540 | 0.0000 | 9.0540 | 27.3183 | 5.0000 |
| popqa | 50 | 2.5400 | 14.3126 | 1.8062 | 0.0000 | 1.8062 | 16.1359 | 5.0000 |
| triviaqa | 50 | 1.5600 | 10.1292 | 1.4236 | 0.0000 | 1.4236 | 11.5639 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.