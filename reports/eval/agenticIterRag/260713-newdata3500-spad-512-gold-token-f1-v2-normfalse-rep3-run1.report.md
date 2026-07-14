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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-201539-129092-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_v2_normfalse_rep3/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep3-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep3-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep3-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep3-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep3-run1/trace/validation_data`
- Wall time: `726.0013s`
- Status counts: `{'answered': 2186, 'no_valid_answer': 471, 'max_turns': 738, 'direct_answer_before_search': 40, 'multiple_tool_calls': 65}`

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
| micro-average | 3500 | 0.1157 | 0.1944 | 3500 | 0.1157 | 0.1944 | 0.1157 |
| macro-average | 7 | 0.1144 | 0.1964 | 500 | 0.1144 | 0.1964 | 0.1144 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0426 | 0.1053 | 563 | 0.0426 | 0.1053 | 0.0426 |
| bamboogle | 125 | 0.1040 | 0.2122 | 125 | 0.1040 | 0.2122 | 0.1040 |
| hotpotqa | 562 | 0.1192 | 0.2096 | 562 | 0.1192 | 0.2096 | 0.1192 |
| musique | 562 | 0.0249 | 0.0760 | 562 | 0.0249 | 0.0760 | 0.0249 |
| nq | 562 | 0.1655 | 0.2422 | 562 | 0.1655 | 0.2422 | 0.1655 |
| popqa | 563 | 0.2185 | 0.2614 | 563 | 0.2185 | 0.2614 | 0.2185 |
| triviaqa | 563 | 0.1261 | 0.2683 | 563 | 0.1261 | 0.2683 | 0.1261 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.3397 | 66.7393 | 0.4444 | 0.0000 | 0.4444 | 67.2030 | 4.8486 |
| macro-average | 7 | 2.3162 | 65.5329 | 0.4098 | 0.0000 | 0.4099 | 65.9620 | 4.8566 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 1.9858 | 56.5599 | 0.3644 | 0.0000 | 0.3644 | 56.9397 | 4.2984 |
| bamboogle | 125 | 2.1280 | 55.7804 | 0.1335 | 0.0000 | 0.1335 | 55.9326 | 4.9200 |
| hotpotqa | 562 | 2.1922 | 71.3299 | 0.4710 | 0.0000 | 0.4710 | 71.8199 | 4.8310 |
| musique | 562 | 2.7722 | 97.4199 | 0.4730 | 0.0000 | 0.4730 | 97.9174 | 4.9555 |
| nq | 562 | 2.1335 | 77.2540 | 0.2970 | 0.0000 | 0.2970 | 77.5680 | 4.9911 |
| popqa | 563 | 2.7442 | 36.6437 | 0.7119 | 0.0000 | 0.7119 | 37.3778 | 5.0000 |
| triviaqa | 563 | 2.2575 | 63.7428 | 0.4181 | 0.0000 | 0.4182 | 64.1784 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.