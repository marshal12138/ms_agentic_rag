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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260710-021433-474200-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_formal_500_offlinebatch_260710/stages/train_agent/spad_rag/answer_refresh_data/actor_model_hf/global_step_7`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_stage1_gs7_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_stage1_gs7_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_stage1_gs7_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_stage1_gs7_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-spad_qwen17_stage1_gs7_search_eval_350/trace/validation_data`
- Wall time: `77.3588s`
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
| micro-average | 350 | 0.1486 | 0.2460 |
| macro-average | 7 | 0.1486 | 0.2460 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.2219 |
| bamboogle | 50 | 0.0600 | 0.1279 |
| hotpotqa | 50 | 0.0800 | 0.1785 |
| musique | 50 | 0.0600 | 0.1317 |
| nq | 50 | 0.1800 | 0.2853 |
| popqa | 50 | 0.2000 | 0.2839 |
| triviaqa | 50 | 0.3400 | 0.4929 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0000 | 5.1323 | 0.9770 | 0.0000 | 0.9770 | 6.1195 | 5.0000 |
| macro-average | 7 | 1.0000 | 5.1323 | 0.9770 | 0.0000 | 0.9770 | 6.1195 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 4.1676 | 0.1565 | 0.0000 | 0.1565 | 4.3306 | 5.0000 |
| bamboogle | 50 | 1.0000 | 7.0215 | 2.0561 | 0.0000 | 2.0561 | 9.0840 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 4.9410 | 0.6993 | 0.0000 | 0.6993 | 5.6467 | 5.0000 |
| musique | 50 | 1.0000 | 7.1568 | 1.1615 | 0.0000 | 1.1615 | 8.3247 | 5.0000 |
| nq | 50 | 1.0000 | 5.3477 | 1.3087 | 0.0000 | 1.3087 | 6.6626 | 5.0000 |
| popqa | 50 | 1.0000 | 3.1063 | 1.1641 | 0.0000 | 1.1641 | 4.3031 | 5.0000 |
| triviaqa | 50 | 1.0000 | 4.1849 | 0.2931 | 0.0000 | 0.2931 | 4.4847 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.