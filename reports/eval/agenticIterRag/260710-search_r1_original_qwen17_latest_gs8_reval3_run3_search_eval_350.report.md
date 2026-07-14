# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/source/co_search_ablation.infer.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260710-113003-543853-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run3_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run3_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run3_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run3_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run3_search_eval_350/trace/validation_data`
- Wall time: `178.2647s`
- Status counts: `{'answered': 195, 'max_turns': 134, 'no_valid_answer': 21}`

## Retrieval Cutoffs

- RECALL_FINAL_TOP_N: `50`
- SEARCH_TOOL_FINAL_TOP_M: `5`
- RANKER_FINAL_TOP_K: `50`

## Infer Path

- Search path: `agent LLM -> recall retriever recall_final_top_n=50 -> searchTool_final_top_m=5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.1371 | 0.1969 |
| macro-average | 7 | 0.1371 | 0.1969 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.0984 |
| bamboogle | 50 | 0.1000 | 0.1532 |
| hotpotqa | 50 | 0.0600 | 0.1123 |
| musique | 50 | 0.0400 | 0.0927 |
| nq | 50 | 0.1800 | 0.2817 |
| popqa | 50 | 0.1800 | 0.2038 |
| triviaqa | 50 | 0.3600 | 0.4361 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.8057 | 22.5012 | 10.2492 | 0.0000 | 10.2492 | 32.7730 | 5.0000 |
| macro-average | 7 | 2.8057 | 22.5012 | 10.2492 | 0.0000 | 10.2492 | 32.7730 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 3.2600 | 17.0083 | 3.8825 | 0.0000 | 3.8825 | 20.9124 | 5.0000 |
| bamboogle | 50 | 2.4400 | 31.5523 | 12.3003 | 0.0000 | 12.3003 | 43.8708 | 5.0000 |
| hotpotqa | 50 | 2.8400 | 22.0388 | 10.1576 | 0.0000 | 10.1576 | 32.2173 | 5.0000 |
| musique | 50 | 3.6200 | 36.4998 | 26.6949 | 0.0000 | 26.6949 | 63.2215 | 5.0000 |
| nq | 50 | 2.3400 | 20.8960 | 11.2166 | 0.0000 | 11.2166 | 32.1299 | 5.0000 |
| popqa | 50 | 3.1400 | 16.3008 | 3.5335 | 0.0000 | 3.5335 | 19.8741 | 5.0000 |
| triviaqa | 50 | 2.0000 | 13.2122 | 3.9593 | 0.0000 | 3.9593 | 17.1852 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.