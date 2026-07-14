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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run2/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run2/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run2/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run2/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata5100-search-r1-retry1-run2/trace/validation_data`
- Wall time: `113.2344s`
- Status counts: `{'no_valid_answer': 40, 'answered': 251, 'max_turns': 59}`

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
| micro-average | 350 | 0.1600 | 0.2337 | 350 | 0.1600 | 0.2337 | 0.1600 |
| macro-average | 7 | 0.1600 | 0.2337 | 50 | 0.1600 | 0.2337 | 0.1600 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1600 | 0.2254 | 50 | 0.1600 | 0.2254 | 0.1600 |
| bamboogle | 50 | 0.0600 | 0.1750 | 50 | 0.0600 | 0.1750 | 0.0600 |
| hotpotqa | 50 | 0.1600 | 0.2150 | 50 | 0.1600 | 0.2150 | 0.1600 |
| musique | 50 | 0.0200 | 0.0701 | 50 | 0.0200 | 0.0701 | 0.0200 |
| nq | 50 | 0.3200 | 0.3390 | 50 | 0.3200 | 0.3390 | 0.3200 |
| popqa | 50 | 0.2800 | 0.3260 | 50 | 0.2800 | 0.3260 | 0.2800 |
| triviaqa | 50 | 0.1200 | 0.2850 | 50 | 0.1200 | 0.2850 | 0.1200 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.7914 | 12.8824 | 4.1930 | 0.0000 | 4.1930 | 17.0922 | 5.0000 |
| macro-average | 7 | 1.7914 | 12.8824 | 4.1930 | 0.0000 | 4.1930 | 17.0922 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.7800 | 10.5468 | 2.9191 | 0.0000 | 2.9191 | 13.4771 | 5.0000 |
| bamboogle | 50 | 1.5600 | 14.1147 | 6.0268 | 0.0000 | 6.0268 | 20.1516 | 5.0000 |
| hotpotqa | 50 | 1.6800 | 10.8712 | 4.2966 | 0.0000 | 4.2967 | 15.1792 | 5.0000 |
| musique | 50 | 2.3200 | 22.4373 | 6.5660 | 0.0000 | 6.5661 | 29.0283 | 5.0000 |
| nq | 50 | 1.3600 | 9.4406 | 3.9204 | 0.0000 | 3.9204 | 13.3698 | 5.0000 |
| popqa | 50 | 2.4200 | 15.5937 | 4.0407 | 0.0000 | 4.0407 | 19.6761 | 5.0000 |
| triviaqa | 50 | 1.4200 | 7.1727 | 1.5814 | 0.0000 | 1.5814 | 8.7632 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.