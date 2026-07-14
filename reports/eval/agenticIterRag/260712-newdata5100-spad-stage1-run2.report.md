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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run2/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run2/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run2/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run2/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run2/trace/validation_data`
- Wall time: `181.9591s`
- Status counts: `{'multiple_tool_calls': 6, 'answered': 227, 'direct_answer_before_search': 19, 'no_valid_answer': 2, 'max_turns': 96}`

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
| micro-average | 350 | 0.1771 | 0.2504 | 350 | 0.1771 | 0.2504 | 0.1771 |
| macro-average | 7 | 0.1771 | 0.2504 | 50 | 0.1771 | 0.2504 | 0.1771 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.1670 | 50 | 0.1200 | 0.1670 | 0.1200 |
| bamboogle | 50 | 0.2000 | 0.3117 | 50 | 0.2000 | 0.3117 | 0.2000 |
| hotpotqa | 50 | 0.2400 | 0.2881 | 50 | 0.2400 | 0.2881 | 0.2400 |
| musique | 50 | 0.0400 | 0.0908 | 50 | 0.0400 | 0.0908 | 0.0400 |
| nq | 50 | 0.3000 | 0.3630 | 50 | 0.3000 | 0.3630 | 0.3000 |
| popqa | 50 | 0.2400 | 0.2633 | 50 | 0.2400 | 0.2633 | 0.2400 |
| triviaqa | 50 | 0.1000 | 0.2685 | 50 | 0.1000 | 0.2685 | 0.1000 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.7486 | 22.7789 | 12.2702 | 0.0000 | 12.2702 | 35.0743 | 4.6429 |
| macro-average | 7 | 2.7486 | 22.7789 | 12.2702 | 0.0000 | 12.2702 | 35.0743 | 4.6429 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.6000 | 16.8662 | 2.3473 | 0.0000 | 2.3473 | 19.2322 | 4.3000 |
| bamboogle | 50 | 2.1800 | 29.1812 | 18.4238 | 0.0000 | 18.4238 | 47.6219 | 4.2000 |
| hotpotqa | 50 | 2.9600 | 24.8205 | 14.8705 | 0.0000 | 14.8705 | 39.7133 | 4.8000 |
| musique | 50 | 2.9800 | 33.2351 | 25.9245 | 0.0000 | 25.9245 | 59.1833 | 4.3000 |
| nq | 50 | 2.8200 | 23.7782 | 18.1340 | 0.0000 | 18.1340 | 41.9323 | 5.0000 |
| popqa | 50 | 3.2400 | 15.7265 | 1.9558 | 0.0000 | 1.9558 | 17.7304 | 5.0000 |
| triviaqa | 50 | 2.4600 | 15.8442 | 4.2355 | 0.0000 | 4.2355 | 20.1068 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.