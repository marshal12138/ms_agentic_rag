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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run1/trace/validation_data`
- Wall time: `187.0643s`
- Status counts: `{'multiple_tool_calls': 7, 'direct_answer_before_search': 20, 'answered': 217, 'no_valid_answer': 2, 'max_turns': 104}`

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
| micro-average | 350 | 0.1657 | 0.2350 | 350 | 0.1657 | 0.2350 | 0.1657 |
| macro-average | 7 | 0.1657 | 0.2350 | 50 | 0.1657 | 0.2350 | 0.1657 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1000 | 0.1470 | 50 | 0.1000 | 0.1470 | 0.1000 |
| bamboogle | 50 | 0.1600 | 0.2717 | 50 | 0.1600 | 0.2717 | 0.1600 |
| hotpotqa | 50 | 0.2400 | 0.2881 | 50 | 0.2400 | 0.2881 | 0.2400 |
| musique | 50 | 0.0200 | 0.0594 | 50 | 0.0200 | 0.0594 | 0.0200 |
| nq | 50 | 0.3000 | 0.3630 | 50 | 0.3000 | 0.3630 | 0.3000 |
| popqa | 50 | 0.2400 | 0.2633 | 50 | 0.2400 | 0.2633 | 0.2400 |
| triviaqa | 50 | 0.1000 | 0.2527 | 50 | 0.1000 | 0.2527 | 0.1000 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.8000 | 23.0564 | 12.4363 | 0.0000 | 12.4363 | 35.5187 | 4.6143 |
| macro-average | 7 | 2.8000 | 23.0564 | 12.4363 | 0.0000 | 12.4363 | 35.5187 | 4.6143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.6800 | 17.5432 | 2.5694 | 0.0000 | 2.5694 | 20.1316 | 4.3000 |
| bamboogle | 50 | 2.2400 | 27.6237 | 18.2565 | 0.0000 | 18.2565 | 45.8978 | 4.2000 |
| hotpotqa | 50 | 2.8600 | 23.3370 | 14.5482 | 0.0000 | 14.5482 | 37.9184 | 4.7000 |
| musique | 50 | 3.1600 | 36.4328 | 25.5671 | 0.0000 | 25.5671 | 62.0248 | 4.2000 |
| nq | 50 | 2.8200 | 24.1548 | 19.5070 | 0.0000 | 19.5070 | 43.6819 | 5.0000 |
| popqa | 50 | 3.3200 | 15.9536 | 2.0896 | 0.0000 | 2.0897 | 18.0925 | 5.0000 |
| triviaqa | 50 | 2.5200 | 16.3494 | 4.5163 | 0.0000 | 4.5163 | 20.8841 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.