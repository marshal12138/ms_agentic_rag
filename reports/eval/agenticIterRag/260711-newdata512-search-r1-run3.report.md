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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run3/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run3/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run3/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run3/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run3/trace/validation_data`
- Wall time: `153.9862s`
- Status counts: `{'multiple_tool_calls': 3, 'answered': 216, 'no_valid_answer': 33, 'direct_answer_before_search': 1, 'max_turns': 97}`

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
| micro-average | 350 | 0.1029 | 0.1854 | 350 | 0.1029 | 0.1854 | 0.1029 |
| macro-average | 7 | 0.1029 | 0.1854 | 50 | 0.1029 | 0.1854 | 0.1029 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0600 | 0.1317 | 50 | 0.0600 | 0.1317 | 0.0600 |
| bamboogle | 50 | 0.1200 | 0.2029 | 50 | 0.1200 | 0.2029 | 0.1200 |
| hotpotqa | 50 | 0.1000 | 0.1777 | 50 | 0.1000 | 0.1777 | 0.1000 |
| musique | 50 | 0.0200 | 0.1087 | 50 | 0.0200 | 0.1087 | 0.0200 |
| nq | 50 | 0.1400 | 0.2219 | 50 | 0.1400 | 0.2219 | 0.1400 |
| popqa | 50 | 0.2000 | 0.2304 | 50 | 0.2000 | 0.2304 | 0.2000 |
| triviaqa | 50 | 0.0800 | 0.2244 | 50 | 0.0800 | 0.2244 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.4971 | 19.6886 | 9.1555 | 0.0000 | 9.1555 | 28.8655 | 4.9429 |
| macro-average | 7 | 2.4971 | 19.6886 | 9.1555 | 0.0000 | 9.1555 | 28.8655 | 4.9429 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.5400 | 15.7720 | 3.0601 | 0.0000 | 3.0602 | 18.8497 | 4.7000 |
| bamboogle | 50 | 2.4800 | 27.0528 | 14.0568 | 0.0000 | 14.0568 | 41.1277 | 5.0000 |
| hotpotqa | 50 | 2.5600 | 17.8096 | 8.8637 | 0.0000 | 8.8637 | 26.6914 | 4.9000 |
| musique | 50 | 2.9200 | 29.6831 | 19.5520 | 0.0000 | 19.5520 | 49.2564 | 5.0000 |
| nq | 50 | 2.3400 | 20.4549 | 13.8683 | 0.0000 | 13.8683 | 34.3399 | 5.0000 |
| popqa | 50 | 2.6600 | 13.8707 | 2.2204 | 0.0000 | 2.2204 | 16.1352 | 5.0000 |
| triviaqa | 50 | 1.9800 | 13.1771 | 2.4671 | 0.0000 | 2.4671 | 15.6580 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.