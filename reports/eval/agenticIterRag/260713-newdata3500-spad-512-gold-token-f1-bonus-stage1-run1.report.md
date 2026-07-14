# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Examples: `3500`
- Success count: `3500`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-011350-061908-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_bonus_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-bonus-stage1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-bonus-stage1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-bonus-stage1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-bonus-stage1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-bonus-stage1-run1/trace/validation_data`
- Wall time: `813.2204s`
- Status counts: `{'answered': 2144, 'no_valid_answer': 380, 'max_turns': 879, 'multiple_tool_calls': 64, 'direct_answer_before_search': 33}`

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
| micro-average | 3500 | 0.1231 | 0.2046 | 3500 | 0.1231 | 0.2046 | 0.1231 |
| macro-average | 7 | 0.1263 | 0.2106 | 500 | 0.1263 | 0.2106 | 0.1263 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0409 | 0.1043 | 563 | 0.0409 | 0.1043 | 0.0409 |
| bamboogle | 125 | 0.1520 | 0.2595 | 125 | 0.1520 | 0.2595 | 0.1520 |
| hotpotqa | 562 | 0.1157 | 0.2021 | 562 | 0.1157 | 0.2021 | 0.1157 |
| musique | 562 | 0.0391 | 0.0884 | 562 | 0.0391 | 0.0884 | 0.0391 |
| nq | 562 | 0.1833 | 0.2696 | 562 | 0.1833 | 0.2696 | 0.1833 |
| popqa | 563 | 0.2220 | 0.2694 | 563 | 0.2220 | 0.2694 | 0.2220 |
| triviaqa | 563 | 0.1314 | 0.2813 | 563 | 0.1314 | 0.2813 | 0.1314 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.4654 | 75.3625 | 0.4363 | 0.0000 | 0.4363 | 75.8199 | 4.8600 |
| macro-average | 7 | 2.4475 | 74.3924 | 0.4067 | 0.0000 | 0.4067 | 74.8202 | 4.8712 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.1492 | 63.4418 | 0.5058 | 0.0000 | 0.5058 | 63.9652 | 4.4050 |
| bamboogle | 125 | 2.3040 | 66.5177 | 0.1708 | 0.0000 | 0.1708 | 66.7094 | 4.9600 |
| hotpotqa | 562 | 2.3060 | 79.7362 | 0.4608 | 0.0000 | 0.4608 | 80.2179 | 4.8043 |
| musique | 562 | 2.9911 | 110.2833 | 0.3922 | 0.0000 | 0.3922 | 110.7028 | 4.9555 |
| nq | 562 | 2.2313 | 87.0440 | 0.2827 | 0.0000 | 0.2827 | 87.3455 | 4.9822 |
| popqa | 563 | 2.8774 | 42.4246 | 0.6718 | 0.0000 | 0.6718 | 43.1200 | 5.0000 |
| triviaqa | 563 | 2.2735 | 71.2992 | 0.3629 | 0.0000 | 0.3629 | 71.6809 | 4.9911 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.