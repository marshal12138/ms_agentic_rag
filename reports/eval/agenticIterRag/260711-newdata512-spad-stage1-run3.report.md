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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run3/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run3/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run3/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run3/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run3/trace/validation_data`
- Wall time: `157.8273s`
- Status counts: `{'multiple_tool_calls': 4, 'answered': 243, 'direct_answer_before_search': 4, 'no_valid_answer': 8, 'max_turns': 91}`

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
| micro-average | 350 | 0.1314 | 0.2324 | 350 | 0.1314 | 0.2324 | 0.1314 |
| macro-average | 7 | 0.1314 | 0.2324 | 50 | 0.1314 | 0.2324 | 0.1314 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.1401 | 50 | 0.0400 | 0.1401 | 0.0400 |
| bamboogle | 50 | 0.1000 | 0.2091 | 50 | 0.1000 | 0.2091 | 0.1000 |
| hotpotqa | 50 | 0.1600 | 0.2275 | 50 | 0.1600 | 0.2275 | 0.1600 |
| musique | 50 | 0.1000 | 0.1791 | 50 | 0.1000 | 0.1791 | 0.1000 |
| nq | 50 | 0.2400 | 0.3241 | 50 | 0.2400 | 0.3241 | 0.2400 |
| popqa | 50 | 0.2200 | 0.2702 | 50 | 0.2200 | 0.2702 | 0.2200 |
| triviaqa | 50 | 0.0600 | 0.2765 | 50 | 0.0600 | 0.2765 | 0.0600 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.4543 | 20.7715 | 8.5356 | 0.0000 | 8.5356 | 29.3299 | 4.8857 |
| macro-average | 7 | 2.4543 | 20.7715 | 8.5356 | 0.0000 | 8.5356 | 29.3299 | 4.8857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.1400 | 16.4964 | 2.0575 | 0.0000 | 2.0575 | 18.5689 | 4.5000 |
| bamboogle | 50 | 2.3800 | 25.6375 | 14.5594 | 0.0000 | 14.5594 | 40.2144 | 4.9000 |
| hotpotqa | 50 | 2.0400 | 17.3955 | 6.9995 | 0.0000 | 6.9995 | 24.4101 | 4.9000 |
| musique | 50 | 3.1000 | 32.4527 | 18.7790 | 0.0000 | 18.7790 | 51.2548 | 5.0000 |
| nq | 50 | 2.5000 | 21.9411 | 12.8723 | 0.0000 | 12.8723 | 34.8399 | 4.9000 |
| popqa | 50 | 3.0400 | 18.0797 | 2.1654 | 0.0000 | 2.1654 | 20.2941 | 5.0000 |
| triviaqa | 50 | 1.9800 | 13.3977 | 2.3162 | 0.0000 | 2.3162 | 15.7275 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.