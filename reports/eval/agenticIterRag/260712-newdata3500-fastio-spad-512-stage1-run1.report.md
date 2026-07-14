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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-103304-616277-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata3500-fastio-spad-512-stage1-run1/trace/validation_data`
- Wall time: `775.8274s`
- Status counts: `{'answered': 2437, 'max_turns': 849, 'no_valid_answer': 105, 'multiple_tool_calls': 100, 'direct_answer_before_search': 9}`

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
| micro-average | 3500 | 0.1360 | 0.2265 | 3500 | 0.1360 | 0.2265 | 0.1360 |
| macro-average | 7 | 0.1378 | 0.2289 | 500 | 0.1378 | 0.2289 | 0.1378 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0480 | 0.1107 | 563 | 0.0480 | 0.1107 | 0.0480 |
| bamboogle | 125 | 0.1520 | 0.2485 | 125 | 0.1520 | 0.2485 | 0.1520 |
| hotpotqa | 562 | 0.1441 | 0.2443 | 562 | 0.1441 | 0.2443 | 0.1441 |
| musique | 562 | 0.0320 | 0.0967 | 562 | 0.0320 | 0.0967 | 0.0320 |
| nq | 562 | 0.2011 | 0.2966 | 562 | 0.2011 | 0.2966 | 0.2011 |
| popqa | 563 | 0.2327 | 0.2831 | 563 | 0.2327 | 0.2831 | 0.2327 |
| triviaqa | 563 | 0.1545 | 0.3227 | 563 | 0.1545 | 0.3227 | 0.1545 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.3391 | 71.7492 | 0.4396 | 0.0000 | 0.4397 | 72.2087 | 4.8443 |
| macro-average | 7 | 2.3397 | 71.2449 | 0.4064 | 0.0000 | 0.4064 | 71.6712 | 4.8572 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.9254 | 61.3392 | 0.3430 | 0.0000 | 0.3430 | 61.6977 | 4.2984 |
| bamboogle | 125 | 2.3440 | 67.1057 | 0.1401 | 0.0000 | 0.1401 | 67.2668 | 4.9600 |
| hotpotqa | 562 | 2.1584 | 73.9186 | 0.5788 | 0.0000 | 0.5788 | 74.5164 | 4.7865 |
| musique | 562 | 2.8452 | 104.5960 | 0.3819 | 0.0000 | 0.3819 | 105.0033 | 4.9644 |
| nq | 562 | 2.2278 | 82.9908 | 0.4684 | 0.0000 | 0.4684 | 83.4780 | 4.9911 |
| popqa | 563 | 2.8544 | 42.9159 | 0.6442 | 0.0000 | 0.6443 | 43.5834 | 5.0000 |
| triviaqa | 563 | 2.0231 | 65.8479 | 0.2882 | 0.0000 | 0.2882 | 66.1528 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.