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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run3/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run3/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run3/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run3/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage1-run3/trace/validation_data`
- Wall time: `185.4565s`
- Status counts: `{'multiple_tool_calls': 6, 'direct_answer_before_search': 21, 'answered': 222, 'no_valid_answer': 2, 'max_turns': 99}`

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
| micro-average | 350 | 0.1686 | 0.2388 | 350 | 0.1686 | 0.2388 | 0.1686 |
| macro-average | 7 | 0.1686 | 0.2388 | 50 | 0.1686 | 0.2388 | 0.1686 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0800 | 0.1248 | 50 | 0.0800 | 0.1248 | 0.0800 |
| bamboogle | 50 | 0.1800 | 0.2917 | 50 | 0.1800 | 0.2917 | 0.1800 |
| hotpotqa | 50 | 0.2600 | 0.3081 | 50 | 0.2600 | 0.3081 | 0.2600 |
| musique | 50 | 0.0200 | 0.0527 | 50 | 0.0200 | 0.0527 | 0.0200 |
| nq | 50 | 0.3000 | 0.3630 | 50 | 0.3000 | 0.3630 | 0.3000 |
| popqa | 50 | 0.2400 | 0.2633 | 50 | 0.2400 | 0.2633 | 0.2400 |
| triviaqa | 50 | 0.1000 | 0.2681 | 50 | 0.1000 | 0.2681 | 0.1000 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.7543 | 23.6805 | 12.4343 | 0.0000 | 12.4343 | 36.1399 | 4.6143 |
| macro-average | 7 | 2.7543 | 23.6805 | 12.4343 | 0.0000 | 12.4343 | 36.1399 | 4.6143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.6400 | 17.6825 | 2.1194 | 0.0000 | 2.1195 | 19.8209 | 4.3000 |
| bamboogle | 50 | 2.2200 | 33.9647 | 17.0701 | 0.0000 | 17.0701 | 51.0510 | 4.2000 |
| hotpotqa | 50 | 2.9800 | 23.1345 | 14.7843 | 0.0000 | 14.7843 | 37.9412 | 4.8000 |
| musique | 50 | 2.8600 | 32.3728 | 27.8837 | 0.0000 | 27.8837 | 60.2794 | 4.1000 |
| nq | 50 | 2.6600 | 24.9042 | 18.1579 | 0.0000 | 18.1579 | 43.0900 | 5.0000 |
| popqa | 50 | 3.3800 | 17.2830 | 3.0095 | 0.0000 | 3.0096 | 20.3417 | 5.0000 |
| triviaqa | 50 | 2.5400 | 16.4220 | 4.0152 | 0.0000 | 4.0152 | 20.4552 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.