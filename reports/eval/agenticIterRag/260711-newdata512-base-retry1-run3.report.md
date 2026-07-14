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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run3/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run3/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run3/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run3/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-base-retry1-run3/trace/validation_data`
- Wall time: `148.4273s`
- Status counts: `{'multiple_tool_calls': 2, 'answered': 207, 'no_valid_answer': 63, 'direct_answer_before_search': 1, 'max_turns': 77}`

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
| micro-average | 350 | 0.0886 | 0.1660 | 350 | 0.0886 | 0.1660 | 0.0886 |
| macro-average | 7 | 0.0886 | 0.1660 | 50 | 0.0886 | 0.1660 | 0.0886 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.1230 | 50 | 0.0400 | 0.1230 | 0.0400 |
| bamboogle | 50 | 0.1000 | 0.1755 | 50 | 0.1000 | 0.1755 | 0.1000 |
| hotpotqa | 50 | 0.1600 | 0.2055 | 50 | 0.1600 | 0.2055 | 0.1600 |
| musique | 50 | 0.0200 | 0.0645 | 50 | 0.0200 | 0.0645 | 0.0200 |
| nq | 50 | 0.1200 | 0.2176 | 50 | 0.1200 | 0.2176 | 0.1200 |
| popqa | 50 | 0.1200 | 0.1610 | 50 | 0.1200 | 0.1610 | 0.1200 |
| triviaqa | 50 | 0.0600 | 0.2151 | 50 | 0.0600 | 0.2151 | 0.0600 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.3743 | 19.3733 | 7.7530 | 0.0000 | 7.7530 | 27.1482 | 4.9571 |
| macro-average | 7 | 2.3743 | 19.3733 | 7.7530 | 0.0000 | 7.7530 | 27.1482 | 4.9571 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.1400 | 14.2236 | 1.9406 | 0.0000 | 1.9406 | 16.1788 | 4.7000 |
| bamboogle | 50 | 2.3800 | 26.6949 | 14.6264 | 0.0000 | 14.6264 | 41.3382 | 5.0000 |
| hotpotqa | 50 | 2.2200 | 17.2008 | 7.7002 | 0.0000 | 7.7002 | 24.9167 | 5.0000 |
| musique | 50 | 2.8800 | 32.1779 | 14.4669 | 0.0000 | 14.4669 | 46.6662 | 5.0000 |
| nq | 50 | 2.2400 | 19.6282 | 10.7853 | 0.0000 | 10.7853 | 30.4389 | 5.0000 |
| popqa | 50 | 2.8200 | 13.8065 | 2.0703 | 0.0000 | 2.0703 | 15.9227 | 5.0000 |
| triviaqa | 50 | 1.9400 | 11.8815 | 2.6813 | 0.0000 | 2.6813 | 14.5760 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.