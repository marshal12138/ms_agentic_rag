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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-185433-916978-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512_normfalse_rep3/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep3-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep3-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep3-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep3-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-search-r1-512-normfalse-rep3-run1/trace/validation_data`
- Wall time: `799.3249s`
- Status counts: `{'answered': 2062, 'no_valid_answer': 274, 'max_turns': 1161, 'multiple_tool_calls': 3}`

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
| micro-average | 3500 | 0.1106 | 0.1896 | 3500 | 0.1106 | 0.1896 | 0.1106 |
| macro-average | 7 | 0.1063 | 0.1846 | 500 | 0.1063 | 0.1846 | 0.1063 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0249 | 0.0924 | 563 | 0.0249 | 0.0924 | 0.0249 |
| bamboogle | 125 | 0.0720 | 0.1446 | 125 | 0.0720 | 0.1446 | 0.0720 |
| hotpotqa | 562 | 0.1050 | 0.1955 | 562 | 0.1050 | 0.1955 | 0.1050 |
| musique | 562 | 0.0142 | 0.0556 | 562 | 0.0142 | 0.0556 | 0.0142 |
| nq | 562 | 0.1797 | 0.2609 | 562 | 0.1797 | 0.2609 | 0.1797 |
| popqa | 563 | 0.2096 | 0.2590 | 563 | 0.2096 | 0.2590 | 0.2096 |
| triviaqa | 563 | 0.1385 | 0.2844 | 563 | 0.1385 | 0.2844 | 0.1385 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.5929 | 72.8856 | 0.5093 | 0.0000 | 0.5093 | 73.4164 | 4.9957 |
| macro-average | 7 | 2.5706 | 71.9595 | 0.4701 | 0.0000 | 0.4702 | 72.4510 | 4.9962 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.8099 | 66.4978 | 0.4846 | 0.0000 | 0.4846 | 67.0041 | 4.9734 |
| bamboogle | 125 | 2.3920 | 64.4405 | 0.1565 | 0.0000 | 0.1566 | 64.6176 | 5.0000 |
| hotpotqa | 562 | 2.4484 | 74.7822 | 0.4653 | 0.0000 | 0.4653 | 75.2676 | 5.0000 |
| musique | 562 | 3.4235 | 109.6996 | 0.7811 | 0.0000 | 0.7811 | 110.5134 | 5.0000 |
| nq | 562 | 2.0907 | 83.2660 | 0.4456 | 0.0000 | 0.4456 | 83.7288 | 5.0000 |
| popqa | 563 | 2.8313 | 39.7362 | 0.6201 | 0.0000 | 0.6201 | 40.3784 | 5.0000 |
| triviaqa | 563 | 1.9982 | 65.2941 | 0.3377 | 0.0000 | 0.3377 | 65.6473 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.