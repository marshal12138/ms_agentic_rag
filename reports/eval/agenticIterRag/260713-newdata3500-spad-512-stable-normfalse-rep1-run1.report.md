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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-113910-342014-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_normfalse_rep1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep1-run1/trace/validation_data`
- Wall time: `702.5525s`
- Status counts: `{'answered': 2163, 'no_valid_answer': 512, 'max_turns': 717, 'multiple_tool_calls': 69, 'direct_answer_before_search': 39}`

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
| micro-average | 3500 | 0.1129 | 0.1904 | 3500 | 0.1129 | 0.1904 | 0.1129 |
| macro-average | 7 | 0.1101 | 0.1893 | 500 | 0.1101 | 0.1893 | 0.1101 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0302 | 0.0869 | 563 | 0.0302 | 0.0869 | 0.0302 |
| bamboogle | 125 | 0.0880 | 0.1806 | 125 | 0.0880 | 0.1806 | 0.0880 |
| hotpotqa | 562 | 0.1157 | 0.2011 | 562 | 0.1157 | 0.2011 | 0.1157 |
| musique | 562 | 0.0249 | 0.0729 | 562 | 0.0249 | 0.0729 | 0.0249 |
| nq | 562 | 0.1548 | 0.2345 | 562 | 0.1548 | 0.2345 | 0.1548 |
| popqa | 563 | 0.2291 | 0.2772 | 563 | 0.2291 | 0.2772 | 0.2291 |
| triviaqa | 563 | 0.1279 | 0.2717 | 563 | 0.1279 | 0.2717 | 0.1279 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.2183 | 64.6198 | 0.4531 | 0.0000 | 0.4532 | 65.0912 | 4.8443 |
| macro-average | 7 | 2.2083 | 63.7919 | 0.4193 | 0.0000 | 0.4193 | 64.2295 | 4.8483 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.9112 | 56.6045 | 0.3712 | 0.0000 | 0.3712 | 56.9915 | 4.3162 |
| bamboogle | 125 | 2.1280 | 57.0717 | 0.1487 | 0.0000 | 0.1487 | 57.2384 | 4.8800 |
| hotpotqa | 562 | 2.1477 | 69.6049 | 0.5743 | 0.0000 | 0.5743 | 70.1968 | 4.8043 |
| musique | 562 | 2.7082 | 94.2862 | 0.4744 | 0.0000 | 0.4744 | 94.7842 | 4.9466 |
| nq | 562 | 1.9395 | 73.4209 | 0.3639 | 0.0000 | 0.3639 | 73.8006 | 4.9911 |
| popqa | 563 | 2.6519 | 35.8486 | 0.7281 | 0.0000 | 0.7281 | 36.5981 | 5.0000 |
| triviaqa | 563 | 1.9716 | 59.7066 | 0.2746 | 0.0000 | 0.2747 | 59.9969 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.