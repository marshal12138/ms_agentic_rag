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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-110639-534549-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512_normfalse_rep2/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep2-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep2-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep2-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep2-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep2-run1/trace/validation_data`
- Wall time: `836.7722s`
- Status counts: `{'answered': 1940, 'no_valid_answer': 234, 'max_turns': 1324, 'multiple_tool_calls': 2}`

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
| micro-average | 3500 | 0.1017 | 0.1771 | 3500 | 0.1017 | 0.1771 | 0.1017 |
| macro-average | 7 | 0.0975 | 0.1714 | 500 | 0.0975 | 0.1714 | 0.0975 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0302 | 0.1011 | 563 | 0.0302 | 0.1011 | 0.0302 |
| bamboogle | 125 | 0.0640 | 0.1263 | 125 | 0.0640 | 0.1263 | 0.0640 |
| hotpotqa | 562 | 0.0907 | 0.1757 | 562 | 0.0907 | 0.1757 | 0.0907 |
| musique | 562 | 0.0142 | 0.0508 | 562 | 0.0142 | 0.0508 | 0.0142 |
| nq | 562 | 0.1584 | 0.2424 | 562 | 0.1584 | 0.2424 | 0.1584 |
| popqa | 563 | 0.1989 | 0.2438 | 563 | 0.1989 | 0.2438 | 0.1989 |
| triviaqa | 563 | 0.1261 | 0.2597 | 563 | 0.1261 | 0.2597 | 0.1261 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.7989 | 76.4533 | 0.4932 | 0.0000 | 0.4932 | 76.9700 | 4.9971 |
| macro-average | 7 | 2.7919 | 75.0301 | 0.4580 | 0.0000 | 0.4580 | 75.5116 | 4.9975 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 3.0515 | 70.0414 | 0.4837 | 0.0000 | 0.4837 | 70.5490 | 4.9822 |
| bamboogle | 125 | 2.7360 | 63.5261 | 0.1763 | 0.0000 | 0.1763 | 63.7270 | 5.0000 |
| hotpotqa | 562 | 2.7064 | 78.0573 | 0.4467 | 0.0000 | 0.4467 | 78.5260 | 5.0000 |
| musique | 562 | 3.5872 | 115.5498 | 0.5312 | 0.0000 | 0.5312 | 116.1122 | 5.0000 |
| nq | 562 | 2.3256 | 88.6883 | 0.4769 | 0.0000 | 0.4769 | 89.1848 | 5.0000 |
| popqa | 563 | 2.9680 | 40.5958 | 0.7362 | 0.0000 | 0.7362 | 41.3571 | 5.0000 |
| triviaqa | 563 | 2.1687 | 68.7517 | 0.3549 | 0.0000 | 0.3549 | 69.1250 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.