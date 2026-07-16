# AgenticIterRag v1 Infer Report

- Infer task: `agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Examples: `3500`
- Success count: `3500`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260716-005244-008472-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm01_hardgatev2_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260716-newdata3500-spad-5100-gold-token-f1-v3-postnorm01-hardgatev2-run1/trace/validation_data`
- Wall time: `472.0412s`
- Status counts: `{'answered': 3014, 'no_valid_answer': 111, 'max_turns': 375}`

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
| micro-average | 3500 | 0.2069 | 0.2911 | 3500 | 0.2069 | 0.2911 | 0.2069 |
| macro-average | 7 | 0.1999 | 0.2836 | 500 | 0.1999 | 0.2836 | 0.1999 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.1350 | 0.1814 | 563 | 0.1350 | 0.1814 | 0.1350 |
| bamboogle | 125 | 0.1440 | 0.2244 | 125 | 0.1440 | 0.2244 | 0.1440 |
| hotpotqa | 562 | 0.1993 | 0.2895 | 562 | 0.1993 | 0.2895 | 0.1993 |
| musique | 562 | 0.0409 | 0.0948 | 562 | 0.0409 | 0.0948 | 0.0409 |
| nq | 562 | 0.3114 | 0.3987 | 562 | 0.3114 | 0.3987 | 0.3114 |
| popqa | 563 | 0.3908 | 0.4382 | 563 | 0.3908 | 0.4382 | 0.3908 |
| triviaqa | 563 | 0.1776 | 0.3584 | 563 | 0.1776 | 0.3584 | 0.1776 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 1.5934 | 41.1439 | 0.3830 | 0.0000 | 0.3830 | 41.5393 | 5.0000 |
| macro-average | 7 | 1.5568 | 40.5373 | 0.3520 | 0.0000 | 0.3520 | 40.9013 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.6252 | 38.7022 | 0.3703 | 0.0000 | 0.3703 | 39.0846 | 5.0000 |
| bamboogle | 125 | 1.2640 | 35.6283 | 0.1045 | 0.0000 | 0.1045 | 35.7424 | 5.0000 |
| hotpotqa | 562 | 1.4235 | 42.1048 | 0.3845 | 0.0000 | 0.3845 | 42.5004 | 5.0000 |
| musique | 562 | 2.1263 | 62.1396 | 0.3584 | 0.0000 | 0.3584 | 62.5152 | 5.0000 |
| nq | 562 | 1.2829 | 44.1655 | 0.2891 | 0.0000 | 0.2891 | 44.4643 | 5.0000 |
| popqa | 563 | 1.8792 | 23.6182 | 0.6578 | 0.0000 | 0.6578 | 24.2910 | 5.0000 |
| triviaqa | 563 | 1.2966 | 37.4022 | 0.2996 | 0.0000 | 0.2996 | 37.7114 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.