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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-022724-631051-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_bonus_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-5100-gold-token-f1-bonus-stage1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-5100-gold-token-f1-bonus-stage1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-5100-gold-token-f1-bonus-stage1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-5100-gold-token-f1-bonus-stage1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-5100-gold-token-f1-bonus-stage1-run1/trace/validation_data`
- Wall time: `922.8344s`
- Status counts: `{'answered': 2216, 'no_valid_answer': 18, 'max_turns': 1256, 'multiple_tool_calls': 9, 'direct_answer_before_search': 1}`

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
| micro-average | 3500 | 0.1837 | 0.2576 | 3500 | 0.1837 | 0.2576 | 0.1837 |
| macro-average | 7 | 0.1873 | 0.2601 | 500 | 0.1873 | 0.2601 | 0.1873 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.1261 | 0.1731 | 563 | 0.1261 | 0.1731 | 0.1261 |
| bamboogle | 125 | 0.2160 | 0.2807 | 125 | 0.2160 | 0.2807 | 0.2160 |
| hotpotqa | 562 | 0.2313 | 0.3187 | 562 | 0.2313 | 0.3187 | 0.2313 |
| musique | 562 | 0.0463 | 0.0962 | 562 | 0.0463 | 0.0962 | 0.0463 |
| nq | 562 | 0.2527 | 0.3287 | 562 | 0.2527 | 0.3287 | 0.2527 |
| popqa | 563 | 0.2735 | 0.3088 | 563 | 0.2735 | 0.3088 | 0.2735 |
| triviaqa | 563 | 0.1652 | 0.3147 | 563 | 0.1652 | 0.3147 | 0.1652 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 3.0071 | 85.4143 | 0.5361 | 0.0000 | 0.5362 | 85.9763 | 4.9857 |
| macro-average | 7 | 3.0037 | 84.3676 | 0.4976 | 0.0000 | 0.4976 | 84.8912 | 4.9829 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 3.0480 | 74.6432 | 0.4962 | 0.0000 | 0.4962 | 75.1639 | 4.9645 |
| bamboogle | 125 | 2.9760 | 75.8564 | 0.1890 | 0.0000 | 0.1891 | 76.0726 | 4.9600 |
| hotpotqa | 562 | 2.9591 | 92.8672 | 0.4036 | 0.0000 | 0.4036 | 93.2968 | 4.9822 |
| musique | 562 | 3.5552 | 126.7775 | 0.6133 | 0.0000 | 0.6133 | 127.4248 | 4.9733 |
| nq | 562 | 2.7473 | 98.0138 | 0.6394 | 0.0000 | 0.6394 | 98.6778 | 5.0000 |
| popqa | 563 | 2.9503 | 42.0894 | 0.6649 | 0.0000 | 0.6649 | 42.7774 | 5.0000 |
| triviaqa | 563 | 2.7904 | 80.3258 | 0.4767 | 0.0000 | 0.4767 | 80.8252 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.