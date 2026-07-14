# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Examples: `3500`
- Success count: `3500`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260712-143738-025140-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_stage1_inflight2_ablation/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-inflight2-ablation-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-inflight2-ablation-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-inflight2-ablation-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-inflight2-ablation-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-inflight2-ablation-run1/trace/validation_data`
- Wall time: `753.3641s`
- Status counts: `{'answered': 2036, 'no_valid_answer': 537, 'max_turns': 827, 'multiple_tool_calls': 71, 'direct_answer_before_search': 29}`

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
| micro-average | 3500 | 0.1054 | 0.1798 | 3500 | 0.1054 | 0.1798 | 0.1054 |
| macro-average | 7 | 0.1044 | 0.1792 | 500 | 0.1044 | 0.1792 | 0.1044 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0355 | 0.0905 | 563 | 0.0355 | 0.0905 | 0.0355 |
| bamboogle | 125 | 0.0960 | 0.1742 | 125 | 0.0960 | 0.1742 | 0.0960 |
| hotpotqa | 562 | 0.1032 | 0.1879 | 562 | 0.1032 | 0.1879 | 0.1032 |
| musique | 562 | 0.0178 | 0.0656 | 562 | 0.0178 | 0.0656 | 0.0178 |
| nq | 562 | 0.1459 | 0.2179 | 562 | 0.1459 | 0.2179 | 0.1459 |
| popqa | 563 | 0.2114 | 0.2587 | 563 | 0.2114 | 0.2587 | 0.2114 |
| triviaqa | 563 | 0.1208 | 0.2593 | 563 | 0.1208 | 0.2593 | 0.1208 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.3566 | 67.9525 | 0.4556 | 0.0000 | 0.4557 | 68.4272 | 4.8557 |
| macro-average | 7 | 2.3588 | 67.1073 | 0.4223 | 0.0000 | 0.4223 | 67.5488 | 4.8674 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.0568 | 57.7965 | 0.5903 | 0.0000 | 0.5903 | 58.4026 | 4.3428 |
| bamboogle | 125 | 2.3760 | 60.2377 | 0.1565 | 0.0000 | 0.1565 | 60.4145 | 4.9600 |
| hotpotqa | 562 | 2.3096 | 73.6198 | 0.3465 | 0.0000 | 0.3465 | 73.9846 | 4.8399 |
| musique | 562 | 2.8630 | 99.4923 | 0.3807 | 0.0000 | 0.3807 | 99.8979 | 4.9377 |
| nq | 562 | 2.1085 | 78.9225 | 0.3784 | 0.0000 | 0.3784 | 79.3183 | 4.9911 |
| popqa | 563 | 2.6714 | 36.0625 | 0.7734 | 0.0000 | 0.7734 | 36.8573 | 5.0000 |
| triviaqa | 563 | 2.1261 | 63.6197 | 0.3306 | 0.0000 | 0.3306 | 63.9668 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.