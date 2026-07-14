# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/350e/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-1.7B`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run1/trace/validation_data`
- Wall time: `153.8594s`
- Status counts: `{'answered': 200, 'multiple_tool_calls': 2, 'no_valid_answer': 63, 'direct_answer_before_search': 3, 'max_turns': 82}`

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
| micro-average | 350 | 0.0829 | 0.1563 | 350 | 0.0829 | 0.1563 | 0.0829 |
| macro-average | 7 | 0.0829 | 0.1563 | 50 | 0.0829 | 0.1563 | 0.0829 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0200 | 0.1117 | 50 | 0.0200 | 0.1117 | 0.0200 |
| bamboogle | 50 | 0.0600 | 0.1360 | 50 | 0.0600 | 0.1360 | 0.0600 |
| hotpotqa | 50 | 0.1800 | 0.2144 | 50 | 0.1800 | 0.2144 | 0.1800 |
| musique | 50 | 0.0200 | 0.0702 | 50 | 0.0200 | 0.0702 | 0.0200 |
| nq | 50 | 0.1200 | 0.2054 | 50 | 0.1200 | 0.2054 | 0.1200 |
| popqa | 50 | 0.1200 | 0.1610 | 50 | 0.1200 | 0.1610 | 0.1200 |
| triviaqa | 50 | 0.0600 | 0.1953 | 50 | 0.0600 | 0.1953 | 0.0600 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.4200 | 19.4447 | 9.1154 | 0.0000 | 9.1154 | 28.5819 | 4.9286 |
| macro-average | 7 | 2.4200 | 19.4447 | 9.1154 | 0.0000 | 9.1154 | 28.5819 | 4.9286 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.1800 | 14.1713 | 1.8931 | 0.0000 | 1.8932 | 16.0817 | 4.7000 |
| bamboogle | 50 | 2.2800 | 25.6887 | 16.7895 | 0.0000 | 16.7895 | 42.4952 | 4.9000 |
| hotpotqa | 50 | 2.3400 | 17.8535 | 6.8593 | 0.0000 | 6.8593 | 24.7299 | 4.9000 |
| musique | 50 | 2.8000 | 30.2841 | 19.7202 | 0.0000 | 19.7202 | 50.0258 | 5.0000 |
| nq | 50 | 2.2600 | 20.5643 | 14.4261 | 0.0000 | 14.4261 | 35.0066 | 5.0000 |
| popqa | 50 | 2.9800 | 14.3922 | 1.8275 | 0.0000 | 1.8275 | 16.2688 | 5.0000 |
| triviaqa | 50 | 2.1000 | 13.1584 | 2.2919 | 0.0000 | 2.2919 | 15.4655 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.