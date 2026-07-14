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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run2/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run2/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run2/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run2/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run2/trace/validation_data`
- Wall time: `147.0507s`
- Status counts: `{'answered': 207, 'multiple_tool_calls': 2, 'no_valid_answer': 62, 'direct_answer_before_search': 2, 'max_turns': 77}`

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
| micro-average | 350 | 0.0714 | 0.1477 | 350 | 0.0714 | 0.1477 | 0.0714 |
| macro-average | 7 | 0.0714 | 0.1477 | 50 | 0.0714 | 0.1477 | 0.0714 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0200 | 0.0941 | 50 | 0.0200 | 0.0941 | 0.0200 |
| bamboogle | 50 | 0.0600 | 0.1271 | 50 | 0.0600 | 0.1271 | 0.0600 |
| hotpotqa | 50 | 0.1400 | 0.1806 | 50 | 0.1400 | 0.1806 | 0.1400 |
| musique | 50 | 0.0200 | 0.0638 | 50 | 0.0200 | 0.0638 | 0.0200 |
| nq | 50 | 0.0800 | 0.1979 | 50 | 0.0800 | 0.1979 | 0.0800 |
| popqa | 50 | 0.1200 | 0.1610 | 50 | 0.1200 | 0.1610 | 0.1200 |
| triviaqa | 50 | 0.0600 | 0.2090 | 50 | 0.0600 | 0.2090 | 0.0600 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.3886 | 18.9147 | 7.9739 | 0.0000 | 7.9739 | 26.9086 | 4.9429 |
| macro-average | 7 | 2.3886 | 18.9147 | 7.9739 | 0.0000 | 7.9739 | 26.9086 | 4.9429 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.0400 | 13.8635 | 2.0446 | 0.0000 | 2.0446 | 15.9214 | 4.7000 |
| bamboogle | 50 | 2.3400 | 26.9060 | 17.0947 | 0.0000 | 17.0947 | 44.0171 | 4.9000 |
| hotpotqa | 50 | 2.4600 | 17.7820 | 7.9197 | 0.0000 | 7.9197 | 25.7189 | 5.0000 |
| musique | 50 | 2.6400 | 28.9441 | 13.3992 | 0.0000 | 13.3992 | 42.3623 | 5.0000 |
| nq | 50 | 2.3200 | 18.7885 | 11.2122 | 0.0000 | 11.2122 | 30.0166 | 5.0000 |
| popqa | 50 | 2.9000 | 13.8460 | 1.8941 | 0.0000 | 1.8941 | 15.7851 | 5.0000 |
| triviaqa | 50 | 2.0200 | 12.2729 | 2.2529 | 0.0000 | 2.2529 | 14.5389 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.