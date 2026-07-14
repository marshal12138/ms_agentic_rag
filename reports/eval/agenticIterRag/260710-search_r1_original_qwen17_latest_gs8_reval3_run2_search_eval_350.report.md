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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run2_search_eval_350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run2_search_eval_350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run2_search_eval_350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run2_search_eval_350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260710-search_r1_original_qwen17_latest_gs8_reval3_run2_search_eval_350/trace/validation_data`
- Wall time: `172.9898s`
- Status counts: `{'answered': 206, 'no_valid_answer': 18, 'max_turns': 126}`

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
| micro-average | 350 | 0.1400 | 0.2038 |
| macro-average | 7 | 0.1400 | 0.2038 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0400 | 0.0952 |
| bamboogle | 50 | 0.1200 | 0.1862 |
| hotpotqa | 50 | 0.0600 | 0.0960 |
| musique | 50 | 0.0200 | 0.0727 |
| nq | 50 | 0.2200 | 0.3353 |
| popqa | 50 | 0.2000 | 0.2338 |
| triviaqa | 50 | 0.3200 | 0.4075 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.7029 | 22.1418 | 9.7342 | 0.0000 | 9.7342 | 31.8976 | 5.0000 |
| macro-average | 7 | 2.7029 | 22.1418 | 9.7342 | 0.0000 | 9.7342 | 31.8976 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 3.1800 | 17.0798 | 3.4823 | 0.0000 | 3.4824 | 20.5835 | 5.0000 |
| bamboogle | 50 | 2.2600 | 28.1240 | 11.0102 | 0.0000 | 11.0102 | 39.1504 | 5.0000 |
| hotpotqa | 50 | 3.0800 | 23.9764 | 13.5441 | 0.0000 | 13.5441 | 37.5430 | 5.0000 |
| musique | 50 | 3.4400 | 37.2685 | 23.3342 | 0.0000 | 23.3342 | 60.6276 | 5.0000 |
| nq | 50 | 1.8800 | 19.4583 | 8.9941 | 0.0000 | 8.9941 | 28.4657 | 5.0000 |
| popqa | 50 | 3.0600 | 15.2532 | 2.9430 | 0.0000 | 2.9430 | 18.2353 | 5.0000 |
| triviaqa | 50 | 2.0200 | 13.8322 | 4.8316 | 0.0000 | 4.8316 | 18.6774 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.