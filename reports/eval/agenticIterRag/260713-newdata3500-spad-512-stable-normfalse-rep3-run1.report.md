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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-192527-981329-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_normfalse_rep3/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep3-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep3-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep3-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep3-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep3-run1/trace/validation_data`
- Wall time: `716.5180s`
- Status counts: `{'answered': 2275, 'no_valid_answer': 366, 'max_turns': 756, 'multiple_tool_calls': 62, 'direct_answer_before_search': 41}`

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
| micro-average | 3500 | 0.1231 | 0.2086 | 3500 | 0.1231 | 0.2086 | 0.1231 |
| macro-average | 7 | 0.1219 | 0.2097 | 500 | 0.1219 | 0.2097 | 0.1219 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0409 | 0.1026 | 563 | 0.0409 | 0.1026 | 0.0409 |
| bamboogle | 125 | 0.1120 | 0.2189 | 125 | 0.1120 | 0.2189 | 0.1120 |
| hotpotqa | 562 | 0.1192 | 0.2153 | 562 | 0.1192 | 0.2153 | 0.1192 |
| musique | 562 | 0.0285 | 0.0859 | 562 | 0.0285 | 0.0859 | 0.0285 |
| nq | 562 | 0.1815 | 0.2663 | 562 | 0.1815 | 0.2663 | 0.1815 |
| popqa | 563 | 0.2256 | 0.2783 | 563 | 0.2256 | 0.2783 | 0.2256 |
| triviaqa | 563 | 0.1456 | 0.3007 | 563 | 0.1456 | 0.3007 | 0.1456 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.1923 | 65.4934 | 0.4088 | 0.0000 | 0.4088 | 65.9213 | 4.8514 |
| macro-average | 7 | 2.1852 | 64.8994 | 0.3780 | 0.0000 | 0.3780 | 65.2965 | 4.8591 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.8863 | 56.5346 | 0.3591 | 0.0000 | 0.3591 | 56.9088 | 4.3339 |
| bamboogle | 125 | 2.1280 | 60.0495 | 0.1311 | 0.0000 | 0.1311 | 60.1994 | 4.9200 |
| hotpotqa | 562 | 2.0267 | 68.5475 | 0.2853 | 0.0000 | 0.2853 | 68.8510 | 4.8665 |
| musique | 562 | 2.7082 | 95.8145 | 0.5580 | 0.0000 | 0.5580 | 96.3978 | 4.9199 |
| nq | 562 | 1.9893 | 75.6476 | 0.4401 | 0.0000 | 0.4401 | 76.1051 | 4.9822 |
| popqa | 563 | 2.6714 | 37.6796 | 0.5466 | 0.0000 | 0.5466 | 38.2488 | 5.0000 |
| triviaqa | 563 | 1.8863 | 60.0228 | 0.3257 | 0.0000 | 0.3257 | 60.3645 | 4.9911 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.