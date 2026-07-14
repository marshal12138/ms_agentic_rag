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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260712-131305-696244-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_stage1_repeat/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-repeat-inflight1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-repeat-inflight1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-repeat-inflight1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-repeat-inflight1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-stable-repeat-inflight1-run1/trace/validation_data`
- Wall time: `784.8581s`
- Status counts: `{'answered': 1868, 'no_valid_answer': 692, 'max_turns': 846, 'multiple_tool_calls': 61, 'direct_answer_before_search': 33}`

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
| micro-average | 3500 | 0.1051 | 0.1737 | 3500 | 0.1051 | 0.1737 | 0.1051 |
| macro-average | 7 | 0.1068 | 0.1763 | 500 | 0.1068 | 0.1763 | 0.1068 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0409 | 0.0997 | 563 | 0.0409 | 0.0997 | 0.0409 |
| bamboogle | 125 | 0.1200 | 0.1967 | 125 | 0.1200 | 0.1967 | 0.1200 |
| hotpotqa | 562 | 0.1050 | 0.1793 | 562 | 0.1050 | 0.1793 | 0.1050 |
| musique | 562 | 0.0249 | 0.0668 | 562 | 0.0249 | 0.0668 | 0.0249 |
| nq | 562 | 0.1477 | 0.2137 | 562 | 0.1477 | 0.2137 | 0.1477 |
| popqa | 563 | 0.1954 | 0.2407 | 563 | 0.1954 | 0.2407 | 0.1954 |
| triviaqa | 563 | 0.1137 | 0.2370 | 563 | 0.1137 | 0.2370 | 0.1137 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.5257 | 72.7065 | 0.5504 | 0.0000 | 0.5504 | 73.2788 | 4.8629 |
| macro-average | 7 | 2.5056 | 71.8533 | 0.5092 | 0.0000 | 0.5092 | 72.3841 | 4.8648 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.2131 | 61.0260 | 0.4255 | 0.0000 | 0.4255 | 61.4711 | 4.4227 |
| bamboogle | 125 | 2.3440 | 64.9116 | 0.1796 | 0.0000 | 0.1796 | 65.1105 | 4.8800 |
| hotpotqa | 562 | 2.4181 | 78.4524 | 0.5818 | 0.0000 | 0.5818 | 79.0560 | 4.8221 |
| musique | 562 | 3.0427 | 106.3122 | 0.5432 | 0.0000 | 0.5432 | 106.8823 | 4.9555 |
| nq | 562 | 2.3381 | 84.9478 | 0.7374 | 0.0000 | 0.7375 | 85.7044 | 4.9911 |
| popqa | 563 | 2.8348 | 38.0373 | 0.7151 | 0.0000 | 0.7151 | 38.7766 | 5.0000 |
| triviaqa | 563 | 2.3481 | 69.2855 | 0.3818 | 0.0000 | 0.3818 | 69.6881 | 4.9822 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.