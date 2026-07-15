# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Examples: `3500`
- Success count: `3500`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260714-091019-055405-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v2_normfalse_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_79`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260714-newdata3500-spad-5100-gold-token-f1-v2-normfalse-run1/trace/validation_data`
- Wall time: `534.8970s`
- Status counts: `{'answered': 2767, 'no_valid_answer': 81, 'max_turns': 652}`

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
| micro-average | 3500 | 0.1831 | 0.2673 | 3500 | 0.1831 | 0.2673 | 0.1831 |
| macro-average | 7 | 0.1726 | 0.2568 | 500 | 0.1726 | 0.2568 | 0.1726 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.1279 | 0.1792 | 563 | 0.1279 | 0.1792 | 0.1279 |
| bamboogle | 125 | 0.0880 | 0.1732 | 125 | 0.0880 | 0.1732 | 0.0880 |
| hotpotqa | 562 | 0.1886 | 0.2839 | 562 | 0.1886 | 0.2839 | 0.1886 |
| musique | 562 | 0.0320 | 0.0825 | 562 | 0.0320 | 0.0825 | 0.0320 |
| nq | 562 | 0.3025 | 0.3958 | 562 | 0.3025 | 0.3958 | 0.3025 |
| popqa | 563 | 0.3055 | 0.3527 | 563 | 0.3055 | 0.3527 | 0.3055 |
| triviaqa | 563 | 0.1634 | 0.3305 | 563 | 0.1634 | 0.3305 | 0.1634 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 1.8889 | 47.7924 | 0.4684 | 0.0000 | 0.4684 | 48.2769 | 5.0000 |
| macro-average | 7 | 1.8603 | 47.2030 | 0.4289 | 0.0000 | 0.4289 | 47.6483 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.0533 | 44.5716 | 0.5105 | 0.0000 | 0.5105 | 45.0991 | 5.0000 |
| bamboogle | 125 | 1.6320 | 42.4214 | 0.1130 | 0.0000 | 0.1130 | 42.5538 | 5.0000 |
| hotpotqa | 562 | 1.6512 | 49.1349 | 0.3882 | 0.0000 | 0.3882 | 49.5377 | 5.0000 |
| musique | 562 | 2.6281 | 73.7451 | 0.6948 | 0.0000 | 0.6948 | 74.4635 | 5.0000 |
| nq | 562 | 1.4004 | 49.9741 | 0.2733 | 0.0000 | 0.2733 | 50.2580 | 5.0000 |
| popqa | 563 | 2.1030 | 26.4915 | 0.6758 | 0.0000 | 0.6758 | 27.1857 | 5.0000 |
| triviaqa | 563 | 1.5542 | 44.0822 | 0.3465 | 0.0000 | 0.3465 | 44.4404 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.