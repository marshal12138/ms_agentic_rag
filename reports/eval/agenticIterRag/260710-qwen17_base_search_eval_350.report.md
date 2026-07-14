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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-qwen17_base_search_eval_350/trace/validation_data`
- Wall time: `81.6426s`
- Status counts: `{'answered': 335, 'no_valid_answer': 14, 'max_turns': 1}`

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
| micro-average | 350 | 0.1543 | 0.2512 |
| macro-average | 7 | 0.1543 | 0.2512 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1400 | 0.2060 |
| bamboogle | 50 | 0.1000 | 0.1474 |
| hotpotqa | 50 | 0.0800 | 0.1973 |
| musique | 50 | 0.0400 | 0.1184 |
| nq | 50 | 0.2000 | 0.3193 |
| popqa | 50 | 0.1800 | 0.2726 |
| triviaqa | 50 | 0.3400 | 0.4975 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0171 | 5.7766 | 0.8797 | 0.0000 | 0.8797 | 6.6665 | 5.0000 |
| macro-average | 7 | 1.0171 | 5.7766 | 0.8797 | 0.0000 | 0.8797 | 6.6665 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0800 | 5.1659 | 0.1061 | 0.0000 | 0.1061 | 5.2786 | 5.0000 |
| bamboogle | 50 | 1.0000 | 6.9882 | 2.6882 | 0.0000 | 2.6882 | 9.6830 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 6.1415 | 0.3039 | 0.0000 | 0.3039 | 6.4520 | 5.0000 |
| musique | 50 | 1.0200 | 8.4389 | 0.4273 | 0.0000 | 0.4273 | 8.8733 | 5.0000 |
| nq | 50 | 1.0000 | 5.7688 | 1.2357 | 0.0000 | 1.2357 | 7.0108 | 5.0000 |
| popqa | 50 | 1.0200 | 3.7099 | 1.1765 | 0.0000 | 1.1765 | 4.9182 | 5.0000 |
| triviaqa | 50 | 1.0000 | 4.2233 | 0.2204 | 0.0000 | 0.2204 | 4.4496 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.