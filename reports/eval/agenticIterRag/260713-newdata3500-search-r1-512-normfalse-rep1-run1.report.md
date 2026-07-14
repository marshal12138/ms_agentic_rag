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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-103539-495712-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512_normfalse_rep1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep1-run1/trace/validation_data`
- Wall time: `742.8613s`
- Status counts: `{'answered': 2312, 'no_valid_answer': 242, 'max_turns': 871, 'multiple_tool_calls': 70, 'direct_answer_before_search': 5}`

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
| micro-average | 3500 | 0.1271 | 0.2108 | 3500 | 0.1271 | 0.2108 | 0.1271 |
| macro-average | 7 | 0.1263 | 0.2140 | 500 | 0.1263 | 0.2140 | 0.1263 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0480 | 0.1086 | 563 | 0.0480 | 0.1086 | 0.0480 |
| bamboogle | 125 | 0.1200 | 0.2404 | 125 | 0.1200 | 0.2404 | 0.1200 |
| hotpotqa | 562 | 0.1068 | 0.2022 | 562 | 0.1068 | 0.2022 | 0.1068 |
| musique | 562 | 0.0391 | 0.0925 | 562 | 0.0391 | 0.0925 | 0.0391 |
| nq | 562 | 0.1993 | 0.2788 | 562 | 0.1993 | 0.2788 | 0.1993 |
| popqa | 563 | 0.2274 | 0.2789 | 563 | 0.2274 | 0.2789 | 0.2274 |
| triviaqa | 563 | 0.1439 | 0.2968 | 563 | 0.1439 | 0.2968 | 0.1439 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.2660 | 67.9156 | 0.3574 | 0.0000 | 0.3574 | 68.2922 | 4.8914 |
| macro-average | 7 | 2.2499 | 67.3781 | 0.3320 | 0.0000 | 0.3320 | 67.7292 | 4.9035 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.9698 | 58.8848 | 0.3573 | 0.0000 | 0.3573 | 59.2576 | 4.4849 |
| bamboogle | 125 | 2.1200 | 62.9700 | 0.1287 | 0.0000 | 0.1287 | 63.1167 | 5.0000 |
| hotpotqa | 562 | 2.1744 | 70.1386 | 0.3120 | 0.0000 | 0.3120 | 70.4694 | 4.8932 |
| musique | 562 | 2.9342 | 102.7396 | 0.3698 | 0.0000 | 0.3698 | 103.1371 | 4.9644 |
| nq | 562 | 2.0730 | 78.9247 | 0.3470 | 0.0000 | 0.3470 | 79.2891 | 5.0000 |
| popqa | 563 | 2.6039 | 37.6702 | 0.5600 | 0.0000 | 0.5600 | 38.2516 | 5.0000 |
| triviaqa | 563 | 1.8739 | 60.3189 | 0.2489 | 0.0000 | 0.2489 | 60.5827 | 4.9822 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.