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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260710-021433-474200-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_formal_500_offlinebatch_260710/stages/train_agent/spad_rag/answer_distillation/dpo/dpo_checkpoint_verl`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_500_final_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_500_final_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_500_final_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_500_final_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_500_final_search_eval_350/trace/validation_data`
- Wall time: `74.3506s`
- Status counts: `{'answered': 350}`

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
| micro-average | 350 | 0.1543 | 0.2519 |
| macro-average | 7 | 0.1543 | 0.2519 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1600 | 0.2441 |
| bamboogle | 50 | 0.0600 | 0.1433 |
| hotpotqa | 50 | 0.0600 | 0.1652 |
| musique | 50 | 0.0600 | 0.1302 |
| nq | 50 | 0.1800 | 0.2886 |
| popqa | 50 | 0.2200 | 0.3018 |
| triviaqa | 50 | 0.3400 | 0.4900 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0000 | 5.0521 | 1.0107 | 0.0000 | 1.0107 | 6.0730 | 5.0000 |
| macro-average | 7 | 1.0000 | 5.0521 | 1.0107 | 0.0000 | 1.0107 | 6.0730 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 4.0246 | 0.1307 | 0.0000 | 0.1307 | 4.1614 | 5.0000 |
| bamboogle | 50 | 1.0000 | 7.5326 | 1.2204 | 0.0000 | 1.2204 | 8.7593 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 4.7998 | 0.5329 | 0.0000 | 0.5329 | 5.3402 | 5.0000 |
| musique | 50 | 1.0000 | 6.2771 | 1.8404 | 0.0000 | 1.8404 | 8.1240 | 5.0000 |
| nq | 50 | 1.0000 | 5.7665 | 1.6803 | 0.0000 | 1.6803 | 7.4532 | 5.0000 |
| popqa | 50 | 1.0000 | 2.9856 | 1.1913 | 0.0000 | 1.1913 | 4.2085 | 5.0000 |
| triviaqa | 50 | 1.0000 | 3.9787 | 0.4790 | 0.0000 | 0.4790 | 4.4641 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.