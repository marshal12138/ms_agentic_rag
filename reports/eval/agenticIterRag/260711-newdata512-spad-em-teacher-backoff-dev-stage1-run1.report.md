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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-220950-337984-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_em_teacher_backoff_dev/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-em-teacher-backoff-dev-stage1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-em-teacher-backoff-dev-stage1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-em-teacher-backoff-dev-stage1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-em-teacher-backoff-dev-stage1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-em-teacher-backoff-dev-stage1-run1/trace/validation_data`
- Wall time: `147.9548s`
- Status counts: `{'multiple_tool_calls': 2, 'answered': 223, 'no_valid_answer': 48, 'direct_answer_before_search': 3, 'max_turns': 74}`

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
| micro-average | 350 | 0.1200 | 0.1882 | 350 | 0.1200 | 0.1882 | 0.1200 |
| macro-average | 7 | 0.1200 | 0.1882 | 50 | 0.1200 | 0.1882 | 0.1200 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.1963 | 50 | 0.1200 | 0.1963 | 0.1200 |
| bamboogle | 50 | 0.1000 | 0.1551 | 50 | 0.1000 | 0.1551 | 0.1000 |
| hotpotqa | 50 | 0.1600 | 0.2253 | 50 | 0.1600 | 0.2253 | 0.1600 |
| musique | 50 | 0.0000 | 0.0543 | 50 | 0.0000 | 0.0543 | 0.0000 |
| nq | 50 | 0.1800 | 0.2310 | 50 | 0.1800 | 0.2310 | 0.1800 |
| popqa | 50 | 0.2000 | 0.2150 | 50 | 0.2000 | 0.2150 | 0.2000 |
| triviaqa | 50 | 0.0800 | 0.2401 | 50 | 0.0800 | 0.2401 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.3343 | 18.9992 | 9.2016 | 0.0000 | 9.2017 | 28.2211 | 4.9286 |
| macro-average | 7 | 2.3343 | 18.9992 | 9.2016 | 0.0000 | 9.2017 | 28.2211 | 4.9286 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.9400 | 13.5601 | 2.4587 | 0.0000 | 2.4587 | 16.0319 | 4.6000 |
| bamboogle | 50 | 2.2000 | 24.1056 | 17.0131 | 0.0000 | 17.0131 | 41.1343 | 4.9000 |
| hotpotqa | 50 | 2.2600 | 17.0897 | 6.8585 | 0.0000 | 6.8585 | 23.9651 | 5.0000 |
| musique | 50 | 2.8000 | 31.6520 | 19.4279 | 0.0000 | 19.4279 | 51.1002 | 5.0000 |
| nq | 50 | 2.2000 | 19.4701 | 13.0989 | 0.0000 | 13.0989 | 32.5845 | 5.0000 |
| popqa | 50 | 2.8800 | 14.4192 | 2.4062 | 0.0000 | 2.4062 | 16.8709 | 5.0000 |
| triviaqa | 50 | 2.0600 | 12.6982 | 3.1482 | 0.0000 | 3.1483 | 15.8606 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.