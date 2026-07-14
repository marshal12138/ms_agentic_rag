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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run1/trace/validation_data`
- Wall time: `163.7727s`
- Status counts: `{'multiple_tool_calls': 4, 'direct_answer_before_search': 3, 'answered': 244, 'no_valid_answer': 11, 'max_turns': 88}`

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
| micro-average | 350 | 0.1286 | 0.2201 | 350 | 0.1286 | 0.2201 | 0.1286 |
| macro-average | 7 | 0.1286 | 0.2201 | 50 | 0.1286 | 0.2201 | 0.1286 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.1339 | 50 | 0.0400 | 0.1339 | 0.0400 |
| bamboogle | 50 | 0.1600 | 0.2558 | 50 | 0.1600 | 0.2558 | 0.1600 |
| hotpotqa | 50 | 0.1400 | 0.1975 | 50 | 0.1400 | 0.1975 | 0.1400 |
| musique | 50 | 0.0600 | 0.1449 | 50 | 0.0600 | 0.1449 | 0.0600 |
| nq | 50 | 0.2000 | 0.2643 | 50 | 0.2000 | 0.2643 | 0.2000 |
| popqa | 50 | 0.2200 | 0.2676 | 50 | 0.2200 | 0.2676 | 0.2200 |
| triviaqa | 50 | 0.0800 | 0.2765 | 50 | 0.0800 | 0.2765 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.4314 | 20.7044 | 9.0080 | 0.0000 | 9.0081 | 29.7338 | 4.9000 |
| macro-average | 7 | 2.4314 | 20.7044 | 9.0080 | 0.0000 | 9.0081 | 29.7338 | 4.9000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.0800 | 15.6693 | 2.5787 | 0.0000 | 2.5787 | 18.2625 | 4.6000 |
| bamboogle | 50 | 2.4400 | 26.7973 | 15.4899 | 0.0000 | 15.4899 | 42.3059 | 4.9000 |
| hotpotqa | 50 | 2.0800 | 17.5382 | 7.3006 | 0.0000 | 7.3006 | 24.8543 | 4.9000 |
| musique | 50 | 2.9800 | 31.6175 | 18.3880 | 0.0000 | 18.3880 | 50.0274 | 5.0000 |
| nq | 50 | 2.6600 | 23.5704 | 15.1691 | 0.0000 | 15.1691 | 38.7585 | 4.9000 |
| popqa | 50 | 2.9400 | 17.2618 | 2.7037 | 0.0000 | 2.7037 | 20.0122 | 5.0000 |
| triviaqa | 50 | 1.8400 | 12.4761 | 1.4262 | 0.0000 | 1.4262 | 13.9156 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.