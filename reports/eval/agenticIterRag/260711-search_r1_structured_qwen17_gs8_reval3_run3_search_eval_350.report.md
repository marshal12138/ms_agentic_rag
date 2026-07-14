# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/search_r1_structured.eval.parquet`
- Examples: `350`
- Success count: `349`
- Failure count: `1`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-010148-047274-pipeline-agentic_iter_rag_v1_search_r1_structured_qwen3_1_7b_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run3_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run3_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run3_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run3_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_reval3_run3_search_eval_350/trace/validation_data`
- Wall time: `140.3743s`
- Status counts: `{'answered': 222, 'no_valid_answer': 49, 'max_turns': 76, 'failed': 1, 'multiple_tool_calls': 2}`

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
| micro-average | 350 | 0.1400 | 0.2067 | 342 | 0.1404 | 0.2025 | 0.1435 |
| macro-average | 7 | 0.1400 | 0.2067 | 48 | 0.1414 | 0.2041 | 0.1450 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0600 | 0.1067 | 50 | 0.0600 | 0.1067 | 0.0600 |
| bamboogle | 50 | 0.1400 | 0.2204 | 50 | 0.1400 | 0.2204 | 0.1400 |
| hotpotqa | 50 | 0.0400 | 0.0641 | 50 | 0.0400 | 0.0641 | 0.0400 |
| musique | 50 | 0.0800 | 0.1120 | 50 | 0.0800 | 0.1120 | 0.0800 |
| nq | 50 | 0.1400 | 0.2466 | 43 | 0.1628 | 0.2458 | 0.1880 |
| popqa | 50 | 0.1600 | 0.2135 | 50 | 0.1600 | 0.2135 | 0.1600 |
| triviaqa | 50 | 0.3600 | 0.4834 | 49 | 0.3469 | 0.4664 | 0.3469 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.2514 | 18.8246 | 6.6969 | 0.0000 | 6.6969 | 25.5965 | 4.9571 |
| macro-average | 7 | 2.2514 | 18.8246 | 6.6969 | 0.0000 | 6.6969 | 25.5965 | 4.9571 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.3000 | 15.7283 | 2.2116 | 0.0000 | 2.2116 | 18.3403 | 4.9000 |
| bamboogle | 50 | 1.9400 | 23.4790 | 11.8697 | 0.0000 | 11.8697 | 35.3628 | 5.0000 |
| hotpotqa | 50 | 2.5200 | 20.7843 | 6.4180 | 0.0000 | 6.4180 | 27.2218 | 4.8000 |
| musique | 50 | 2.8400 | 29.5967 | 14.7441 | 0.0000 | 14.7441 | 44.3622 | 5.0000 |
| nq | 50 | 2.0400 | 18.3666 | 8.3844 | 0.0000 | 8.3844 | 26.7656 | 5.0000 |
| popqa | 50 | 2.5400 | 14.7348 | 2.0114 | 0.0000 | 2.0114 | 16.7898 | 5.0000 |
| triviaqa | 50 | 1.5800 | 9.0828 | 1.2389 | 0.0000 | 1.2389 | 10.3331 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.