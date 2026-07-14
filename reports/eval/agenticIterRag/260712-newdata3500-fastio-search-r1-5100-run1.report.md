# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Examples: `3500`
- Success count: `3500`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-5100-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-5100-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-5100-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-5100-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-search-r1-5100-run1/trace/validation_data`
- Wall time: `494.8141s`
- Status counts: `{'answered': 2561, 'no_valid_answer': 397, 'max_turns': 542}`

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
| micro-average | 3500 | 0.1800 | 0.2509 | 3500 | 0.1800 | 0.2509 | 0.1800 |
| macro-average | 7 | 0.1742 | 0.2460 | 500 | 0.1742 | 0.2460 | 0.1742 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.1172 | 0.1623 | 563 | 0.1172 | 0.1623 | 0.1172 |
| bamboogle | 125 | 0.1280 | 0.2065 | 125 | 0.1280 | 0.2065 | 0.1280 |
| hotpotqa | 562 | 0.1851 | 0.2633 | 562 | 0.1851 | 0.2633 | 0.1851 |
| musique | 562 | 0.0356 | 0.0818 | 562 | 0.0356 | 0.0818 | 0.0356 |
| nq | 562 | 0.2651 | 0.3342 | 562 | 0.2651 | 0.3342 | 0.2651 |
| popqa | 563 | 0.3375 | 0.3826 | 563 | 0.3375 | 0.3826 | 0.3375 |
| triviaqa | 563 | 0.1510 | 0.2910 | 563 | 0.1510 | 0.2910 | 0.1510 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 1.7291 | 43.7071 | 0.4110 | 0.0000 | 0.4110 | 44.1323 | 5.0000 |
| macro-average | 7 | 1.6917 | 43.2297 | 0.3759 | 0.0000 | 0.3759 | 43.6195 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.8117 | 41.0989 | 0.4121 | 0.0000 | 0.4121 | 41.5249 | 5.0000 |
| bamboogle | 125 | 1.3920 | 39.3543 | 0.0951 | 0.0000 | 0.0951 | 39.4611 | 5.0000 |
| hotpotqa | 562 | 1.5053 | 44.0195 | 0.3804 | 0.0000 | 0.3804 | 44.4129 | 5.0000 |
| musique | 562 | 2.3612 | 66.3323 | 0.4568 | 0.0000 | 0.4569 | 66.8095 | 5.0000 |
| nq | 562 | 1.3043 | 45.7994 | 0.3424 | 0.0000 | 0.3424 | 46.1518 | 5.0000 |
| popqa | 563 | 2.0071 | 26.0574 | 0.6589 | 0.0000 | 0.6589 | 26.7328 | 5.0000 |
| triviaqa | 563 | 1.4600 | 39.9462 | 0.2857 | 0.0000 | 0.2857 | 40.2435 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.