# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Examples: `3500`
- Success count: `3500`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-512-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-512-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-512-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-512-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-512-run1/trace/validation_data`
- Wall time: `721.7524s`
- Status counts: `{'answered': 2179, 'no_valid_answer': 363, 'max_turns': 899, 'multiple_tool_calls': 43, 'direct_answer_before_search': 16}`

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
| micro-average | 3500 | 0.1180 | 0.1965 | 3500 | 0.1180 | 0.1965 | 0.1180 |
| macro-average | 7 | 0.1147 | 0.1929 | 500 | 0.1147 | 0.1929 | 0.1147 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0284 | 0.0924 | 563 | 0.0284 | 0.0924 | 0.0284 |
| bamboogle | 125 | 0.0880 | 0.1636 | 125 | 0.0880 | 0.1636 | 0.0880 |
| hotpotqa | 562 | 0.1103 | 0.2051 | 562 | 0.1103 | 0.2051 | 0.1103 |
| musique | 562 | 0.0214 | 0.0716 | 562 | 0.0214 | 0.0716 | 0.0214 |
| nq | 562 | 0.1779 | 0.2521 | 562 | 0.1779 | 0.2521 | 0.1779 |
| popqa | 563 | 0.2380 | 0.2787 | 563 | 0.2380 | 0.2787 | 0.2380 |
| triviaqa | 563 | 0.1385 | 0.2865 | 563 | 0.1385 | 0.2865 | 0.1385 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.3489 | 65.9007 | 0.4850 | 0.0000 | 0.4850 | 66.4056 | 4.9157 |
| macro-average | 7 | 2.3271 | 65.0993 | 0.4458 | 0.0000 | 0.4458 | 65.5650 | 4.9251 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.1314 | 57.1608 | 0.4048 | 0.0000 | 0.4048 | 57.5836 | 4.5826 |
| bamboogle | 125 | 2.1520 | 58.5801 | 0.1323 | 0.0000 | 0.1323 | 58.7315 | 5.0000 |
| hotpotqa | 562 | 2.3452 | 70.1138 | 0.5306 | 0.0000 | 0.5307 | 70.6640 | 4.9110 |
| musique | 562 | 3.0053 | 98.0977 | 0.5633 | 0.0000 | 0.5633 | 98.6875 | 4.9822 |
| nq | 562 | 2.1388 | 77.8075 | 0.5332 | 0.0000 | 0.5332 | 78.3600 | 5.0000 |
| popqa | 563 | 2.5933 | 34.9088 | 0.6252 | 0.0000 | 0.6252 | 35.5550 | 5.0000 |
| triviaqa | 563 | 1.9236 | 59.0267 | 0.3314 | 0.0000 | 0.3314 | 59.3735 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.