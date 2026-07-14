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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-120236-859684-pipeline-agentic_iter_rag_v1_search_r1_qwen3_1_7b_newdata_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run2/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run2/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run2/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run2/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-search-r1-run2/trace/validation_data`
- Wall time: `149.6308s`
- Status counts: `{'multiple_tool_calls': 3, 'answered': 223, 'direct_answer_before_search': 1, 'no_valid_answer': 32, 'max_turns': 91}`

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
| micro-average | 350 | 0.0914 | 0.1761 | 350 | 0.0914 | 0.1761 | 0.0914 |
| macro-average | 7 | 0.0914 | 0.1761 | 50 | 0.0914 | 0.1761 | 0.0914 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.1215 | 50 | 0.0400 | 0.1215 | 0.0400 |
| bamboogle | 50 | 0.0600 | 0.1647 | 50 | 0.0600 | 0.1647 | 0.0600 |
| hotpotqa | 50 | 0.1200 | 0.2006 | 50 | 0.1200 | 0.2006 | 0.1200 |
| musique | 50 | 0.0400 | 0.1213 | 50 | 0.0400 | 0.1213 | 0.0400 |
| nq | 50 | 0.1200 | 0.1920 | 50 | 0.1200 | 0.1920 | 0.1200 |
| popqa | 50 | 0.1800 | 0.2149 | 50 | 0.1800 | 0.2149 | 0.1800 |
| triviaqa | 50 | 0.0800 | 0.2177 | 50 | 0.0800 | 0.2177 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.4314 | 18.8416 | 8.1688 | 0.0000 | 8.1688 | 27.0333 | 4.9429 |
| macro-average | 7 | 2.4314 | 18.8416 | 8.1688 | 0.0000 | 8.1688 | 27.0333 | 4.9429 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.6200 | 16.5724 | 2.9900 | 0.0000 | 2.9900 | 19.5813 | 4.7000 |
| bamboogle | 50 | 2.5200 | 25.9459 | 16.0287 | 0.0000 | 16.0287 | 41.9935 | 5.0000 |
| hotpotqa | 50 | 2.3000 | 15.6986 | 7.0719 | 0.0000 | 7.0719 | 22.7959 | 4.9000 |
| musique | 50 | 2.7600 | 28.5038 | 15.5260 | 0.0000 | 15.5260 | 44.0505 | 5.0000 |
| nq | 50 | 2.2200 | 19.3398 | 11.1519 | 0.0001 | 11.1520 | 30.5076 | 5.0000 |
| popqa | 50 | 2.5800 | 13.0526 | 2.0743 | 0.0000 | 2.0744 | 15.1729 | 5.0000 |
| triviaqa | 50 | 2.0200 | 12.7782 | 2.3385 | 0.0000 | 2.3385 | 15.1311 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.