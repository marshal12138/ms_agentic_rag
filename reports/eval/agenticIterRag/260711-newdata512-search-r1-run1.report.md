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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run1/trace/validation_data`
- Wall time: `147.3967s`
- Status counts: `{'multiple_tool_calls': 3, 'answered': 225, 'no_valid_answer': 34, 'direct_answer_before_search': 1, 'max_turns': 87}`

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
| micro-average | 350 | 0.1000 | 0.1838 | 350 | 0.1000 | 0.1838 | 0.1000 |
| macro-average | 7 | 0.1000 | 0.1838 | 50 | 0.1000 | 0.1838 | 0.1000 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0600 | 0.1519 | 50 | 0.0600 | 0.1519 | 0.0600 |
| bamboogle | 50 | 0.1000 | 0.1767 | 50 | 0.1000 | 0.1767 | 0.1000 |
| hotpotqa | 50 | 0.1200 | 0.1881 | 50 | 0.1200 | 0.1881 | 0.1200 |
| musique | 50 | 0.0400 | 0.1117 | 50 | 0.0400 | 0.1117 | 0.0400 |
| nq | 50 | 0.1200 | 0.1996 | 50 | 0.1200 | 0.1996 | 0.1200 |
| popqa | 50 | 0.1600 | 0.2040 | 50 | 0.1600 | 0.2040 | 0.1600 |
| triviaqa | 50 | 0.1000 | 0.2550 | 50 | 0.1000 | 0.2550 | 0.1000 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.3714 | 18.6512 | 8.3197 | 0.0000 | 8.3198 | 26.9928 | 4.9429 |
| macro-average | 7 | 2.3714 | 18.6512 | 8.3197 | 0.0000 | 8.3198 | 26.9928 | 4.9429 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.2800 | 14.6919 | 2.7300 | 0.0000 | 2.7300 | 17.4376 | 4.7000 |
| bamboogle | 50 | 2.3600 | 25.6407 | 13.7525 | 0.0000 | 13.7525 | 39.4100 | 5.0000 |
| hotpotqa | 50 | 2.3800 | 16.3753 | 8.3888 | 0.0000 | 8.3888 | 24.7811 | 4.9000 |
| musique | 50 | 2.8000 | 28.8377 | 17.9313 | 0.0000 | 17.9313 | 46.7904 | 5.0000 |
| nq | 50 | 2.2200 | 18.9983 | 10.7694 | 0.0000 | 10.7694 | 29.7920 | 5.0000 |
| popqa | 50 | 2.7400 | 14.1112 | 2.6540 | 0.0000 | 2.6540 | 16.8101 | 5.0000 |
| triviaqa | 50 | 1.8200 | 11.9035 | 2.0123 | 0.0000 | 2.0123 | 13.9283 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.