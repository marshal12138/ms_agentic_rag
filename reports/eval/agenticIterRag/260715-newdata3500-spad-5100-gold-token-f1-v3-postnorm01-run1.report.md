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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260714-175600-957643-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-run1/trace/validation_data`
- Wall time: `516.8171s`
- Status counts: `{'answered': 2917, 'no_valid_answer': 94, 'max_turns': 479, 'multiple_tool_calls': 8, 'direct_answer_before_search': 2}`

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
| micro-average | 3500 | 0.1994 | 0.2787 | 3500 | 0.1994 | 0.2787 | 0.1994 |
| macro-average | 7 | 0.1861 | 0.2668 | 500 | 0.1861 | 0.2668 | 0.1861 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.1545 | 0.1948 | 563 | 0.1545 | 0.1948 | 0.1545 |
| bamboogle | 125 | 0.0800 | 0.1721 | 125 | 0.0800 | 0.1721 | 0.0800 |
| hotpotqa | 562 | 0.2189 | 0.3062 | 562 | 0.2189 | 0.3062 | 0.2189 |
| musique | 562 | 0.0356 | 0.0807 | 562 | 0.0356 | 0.0807 | 0.0356 |
| nq | 562 | 0.3060 | 0.3870 | 562 | 0.3060 | 0.3870 | 0.3060 |
| popqa | 563 | 0.3357 | 0.3722 | 563 | 0.3357 | 0.3722 | 0.3357 |
| triviaqa | 563 | 0.1723 | 0.3547 | 563 | 0.1723 | 0.3547 | 0.1723 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 1.6969 | 44.5151 | 0.4094 | 0.0000 | 0.4094 | 44.9399 | 4.9857 |
| macro-average | 7 | 1.6665 | 43.9813 | 0.3757 | 0.0000 | 0.3757 | 44.3721 | 4.9873 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.7691 | 42.2383 | 0.3168 | 0.0000 | 0.3168 | 42.5697 | 4.9467 |
| bamboogle | 125 | 1.4240 | 39.6545 | 0.1065 | 0.0000 | 0.1065 | 39.7732 | 5.0000 |
| hotpotqa | 562 | 1.4324 | 44.6106 | 0.4020 | 0.0000 | 0.4020 | 45.0271 | 4.9644 |
| musique | 562 | 2.2740 | 67.2308 | 0.3783 | 0.0000 | 0.3783 | 67.6303 | 5.0000 |
| nq | 562 | 1.3594 | 47.1103 | 0.4252 | 0.0000 | 0.4252 | 47.5465 | 5.0000 |
| popqa | 563 | 1.9858 | 26.3196 | 0.6693 | 0.0000 | 0.6693 | 27.0069 | 5.0000 |
| triviaqa | 563 | 1.4210 | 40.7053 | 0.3318 | 0.0000 | 0.3318 | 41.0510 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.