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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-5100-stage1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-5100-stage1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-5100-stage1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-5100-stage1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-5100-stage1-run1/trace/validation_data`
- Wall time: `832.6246s`
- Status counts: `{'answered': 2428, 'max_turns': 855, 'no_valid_answer': 6, 'multiple_tool_calls': 112, 'direct_answer_before_search': 99}`

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
| micro-average | 3500 | 0.1923 | 0.2700 | 3500 | 0.1923 | 0.2700 | 0.1923 |
| macro-average | 7 | 0.1905 | 0.2700 | 500 | 0.1905 | 0.2700 | 0.1905 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.1083 | 0.1497 | 563 | 0.1083 | 0.1497 | 0.1083 |
| bamboogle | 125 | 0.1760 | 0.2701 | 125 | 0.1760 | 0.2701 | 0.1760 |
| hotpotqa | 562 | 0.2456 | 0.3265 | 562 | 0.2456 | 0.3265 | 0.2456 |
| musique | 562 | 0.0463 | 0.0886 | 562 | 0.0463 | 0.0886 | 0.0463 |
| nq | 562 | 0.2865 | 0.3637 | 562 | 0.2865 | 0.3637 | 0.2865 |
| popqa | 563 | 0.2824 | 0.3262 | 563 | 0.2824 | 0.3262 | 0.2824 |
| triviaqa | 563 | 0.1883 | 0.3654 | 563 | 0.1883 | 0.3654 | 0.1883 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.6557 | 76.8529 | 0.4967 | 0.0000 | 0.4967 | 77.3726 | 4.6986 |
| macro-average | 7 | 2.6043 | 76.0458 | 0.4561 | 0.0000 | 0.4561 | 76.5245 | 4.6476 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.2593 | 63.2364 | 0.4049 | 0.0000 | 0.4050 | 63.6605 | 4.1119 |
| bamboogle | 125 | 2.1920 | 69.4607 | 0.1318 | 0.0000 | 0.1318 | 69.6119 | 4.2400 |
| hotpotqa | 562 | 2.5498 | 82.0445 | 0.4008 | 0.0000 | 0.4008 | 82.4667 | 4.7509 |
| musique | 562 | 3.0071 | 114.2903 | 0.6110 | 0.0000 | 0.6110 | 114.9315 | 4.4662 |
| nq | 562 | 2.6993 | 91.1380 | 0.3812 | 0.0000 | 0.3812 | 91.5431 | 5.0000 |
| popqa | 563 | 3.0018 | 40.9419 | 0.7511 | 0.0000 | 0.7511 | 41.7174 | 5.0000 |
| triviaqa | 563 | 2.5204 | 71.2086 | 0.5119 | 0.0000 | 0.5120 | 71.7404 | 4.9645 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.