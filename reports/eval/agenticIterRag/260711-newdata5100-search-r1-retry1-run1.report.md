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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run1/trace/validation_data`
- Wall time: `128.1072s`
- Status counts: `{'no_valid_answer': 42, 'answered': 251, 'max_turns': 57}`

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
| micro-average | 350 | 0.1657 | 0.2341 | 350 | 0.1657 | 0.2341 | 0.1657 |
| macro-average | 7 | 0.1657 | 0.2341 | 50 | 0.1657 | 0.2341 | 0.1657 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.1674 | 50 | 0.1200 | 0.1674 | 0.1200 |
| bamboogle | 50 | 0.0800 | 0.1950 | 50 | 0.0800 | 0.1950 | 0.0800 |
| hotpotqa | 50 | 0.1800 | 0.2217 | 50 | 0.1800 | 0.2217 | 0.1800 |
| musique | 50 | 0.0200 | 0.0701 | 50 | 0.0200 | 0.0701 | 0.0200 |
| nq | 50 | 0.3200 | 0.3357 | 50 | 0.3200 | 0.3357 | 0.3200 |
| popqa | 50 | 0.3200 | 0.3660 | 50 | 0.3200 | 0.3660 | 0.3200 |
| triviaqa | 50 | 0.1200 | 0.2830 | 50 | 0.1200 | 0.2830 | 0.1200 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.7714 | 13.9273 | 4.6459 | 0.0000 | 4.6459 | 18.5896 | 5.0000 |
| macro-average | 7 | 1.7714 | 13.9273 | 4.6459 | 0.0000 | 4.6459 | 18.5896 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.9400 | 11.9557 | 3.6022 | 0.0000 | 3.6022 | 15.5710 | 5.0000 |
| bamboogle | 50 | 1.4200 | 17.3487 | 5.2782 | 0.0000 | 5.2782 | 22.6369 | 5.0000 |
| hotpotqa | 50 | 1.6600 | 12.1066 | 5.0178 | 0.0000 | 5.0178 | 17.1361 | 5.0000 |
| musique | 50 | 2.3400 | 22.6619 | 8.9842 | 0.0000 | 8.9843 | 31.6639 | 5.0000 |
| nq | 50 | 1.3600 | 10.7913 | 4.0502 | 0.0000 | 4.0502 | 14.8510 | 5.0000 |
| popqa | 50 | 2.3200 | 15.3353 | 4.0087 | 0.0000 | 4.0087 | 19.3874 | 5.0000 |
| triviaqa | 50 | 1.3600 | 7.2915 | 1.5796 | 0.0000 | 1.5796 | 8.8806 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.