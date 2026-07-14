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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260710-032319-715663-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_7`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_gs7_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_gs7_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_gs7_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_gs7_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_gs7_search_eval_350/trace/validation_data`
- Wall time: `77.2137s`
- Status counts: `{'answered': 344, 'no_valid_answer': 6}`

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
| micro-average | 350 | 0.1714 | 0.2665 |
| macro-average | 7 | 0.1714 | 0.2665 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.1911 |
| bamboogle | 50 | 0.0800 | 0.1574 |
| hotpotqa | 50 | 0.1000 | 0.2192 |
| musique | 50 | 0.1000 | 0.1537 |
| nq | 50 | 0.2200 | 0.3337 |
| popqa | 50 | 0.2200 | 0.3011 |
| triviaqa | 50 | 0.3600 | 0.5091 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0086 | 5.2869 | 0.8690 | 0.0000 | 0.8690 | 6.1658 | 5.0000 |
| macro-average | 7 | 1.0086 | 5.2869 | 0.8690 | 0.0000 | 0.8690 | 6.1658 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0400 | 4.9031 | 0.1753 | 0.0000 | 0.1753 | 5.0847 | 5.0000 |
| bamboogle | 50 | 1.0200 | 6.4285 | 1.9779 | 0.0000 | 1.9779 | 8.4125 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 5.7421 | 0.2594 | 0.0000 | 0.2594 | 6.0078 | 5.0000 |
| musique | 50 | 1.0000 | 7.6098 | 0.8883 | 0.0000 | 0.8883 | 8.5046 | 5.0000 |
| nq | 50 | 1.0000 | 5.2318 | 1.1682 | 0.0000 | 1.1682 | 6.4062 | 5.0000 |
| popqa | 50 | 1.0000 | 3.1600 | 1.2483 | 0.0000 | 1.2483 | 4.4404 | 5.0000 |
| triviaqa | 50 | 1.0000 | 3.9331 | 0.3654 | 0.0000 | 0.3654 | 4.3047 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.