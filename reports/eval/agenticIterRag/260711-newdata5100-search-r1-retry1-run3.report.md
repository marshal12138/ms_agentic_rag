# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/350e/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-144201-720888-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run3/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run3/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run3/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run3/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run3/trace/validation_data`
- Wall time: `105.3987s`
- Status counts: `{'answered': 251, 'no_valid_answer': 38, 'max_turns': 61}`

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
| micro-average | 350 | 0.1600 | 0.2360 | 350 | 0.1600 | 0.2360 | 0.1600 |
| macro-average | 7 | 0.1600 | 0.2360 | 50 | 0.1600 | 0.2360 | 0.1600 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.1854 | 50 | 0.1200 | 0.1854 | 0.1200 |
| bamboogle | 50 | 0.0800 | 0.1950 | 50 | 0.0800 | 0.1950 | 0.0800 |
| hotpotqa | 50 | 0.1600 | 0.2150 | 50 | 0.1600 | 0.2150 | 0.1600 |
| musique | 50 | 0.0200 | 0.0756 | 50 | 0.0200 | 0.0756 | 0.0200 |
| nq | 50 | 0.3200 | 0.3357 | 50 | 0.3200 | 0.3357 | 0.3200 |
| popqa | 50 | 0.3000 | 0.3360 | 50 | 0.3000 | 0.3360 | 0.3000 |
| triviaqa | 50 | 0.1200 | 0.3091 | 50 | 0.1200 | 0.3091 | 0.1200 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.8114 | 14.0028 | 4.1940 | 0.0000 | 4.1941 | 18.2168 | 5.0000 |
| macro-average | 7 | 1.8114 | 14.0028 | 4.1940 | 0.0000 | 4.1941 | 18.2168 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.9200 | 13.5227 | 3.7971 | 0.0000 | 3.7971 | 17.3350 | 5.0000 |
| bamboogle | 50 | 1.6400 | 16.4720 | 4.9708 | 0.0000 | 4.9708 | 21.4552 | 5.0000 |
| hotpotqa | 50 | 1.6600 | 11.1027 | 3.9198 | 0.0000 | 3.9198 | 15.0354 | 5.0000 |
| musique | 50 | 2.3200 | 22.3701 | 7.5997 | 0.0000 | 7.5998 | 30.0031 | 5.0000 |
| nq | 50 | 1.3600 | 10.7885 | 3.4240 | 0.0000 | 3.4240 | 14.2224 | 5.0000 |
| popqa | 50 | 2.4400 | 16.4136 | 4.1721 | 0.0000 | 4.1721 | 20.6315 | 5.0000 |
| triviaqa | 50 | 1.3400 | 7.3499 | 1.4748 | 0.0000 | 1.4748 | 8.8349 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.