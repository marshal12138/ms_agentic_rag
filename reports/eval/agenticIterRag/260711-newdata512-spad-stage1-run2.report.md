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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run2/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run2/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run2/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run2/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage1-run2/trace/validation_data`
- Wall time: `163.9946s`
- Status counts: `{'multiple_tool_calls': 4, 'direct_answer_before_search': 3, 'answered': 239, 'no_valid_answer': 11, 'max_turns': 93}`

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
| micro-average | 350 | 0.1343 | 0.2230 | 350 | 0.1343 | 0.2230 | 0.1343 |
| macro-average | 7 | 0.1343 | 0.2230 | 50 | 0.1343 | 0.2230 | 0.1343 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.1236 | 50 | 0.0400 | 0.1236 | 0.0400 |
| bamboogle | 50 | 0.1400 | 0.2388 | 50 | 0.1400 | 0.2388 | 0.1400 |
| hotpotqa | 50 | 0.2000 | 0.2194 | 50 | 0.2000 | 0.2194 | 0.2000 |
| musique | 50 | 0.0600 | 0.1490 | 50 | 0.0600 | 0.1490 | 0.0600 |
| nq | 50 | 0.2400 | 0.3231 | 50 | 0.2400 | 0.3231 | 0.2400 |
| popqa | 50 | 0.2000 | 0.2410 | 50 | 0.2000 | 0.2410 | 0.2000 |
| triviaqa | 50 | 0.0600 | 0.2658 | 50 | 0.0600 | 0.2658 | 0.0600 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.4657 | 20.6917 | 9.2021 | 0.0000 | 9.2021 | 29.9157 | 4.9000 |
| macro-average | 7 | 2.4657 | 20.6917 | 9.2021 | 0.0000 | 9.2021 | 29.9157 | 4.9000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.0600 | 15.2413 | 2.4604 | 0.0000 | 2.4604 | 17.7165 | 4.6000 |
| bamboogle | 50 | 2.4800 | 27.7725 | 16.3512 | 0.0000 | 16.3512 | 44.1423 | 4.9000 |
| hotpotqa | 50 | 2.2000 | 18.3086 | 7.4005 | 0.0000 | 7.4005 | 25.7257 | 4.9000 |
| musique | 50 | 2.9800 | 29.7712 | 19.4982 | 0.0000 | 19.4982 | 49.2920 | 5.0000 |
| nq | 50 | 2.4600 | 22.0030 | 13.5428 | 0.0000 | 13.5428 | 35.5630 | 4.9000 |
| popqa | 50 | 3.0200 | 17.6167 | 2.5543 | 0.0000 | 2.5543 | 20.2194 | 5.0000 |
| triviaqa | 50 | 2.0600 | 14.1283 | 2.6071 | 0.0000 | 2.6071 | 16.7511 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.