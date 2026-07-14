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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-132127-010666-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_v2_normfalse_rep1/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep1-run1/trace/validation_data`
- Wall time: `782.6196s`
- Status counts: `{'answered': 2255, 'no_valid_answer': 321, 'max_turns': 867, 'multiple_tool_calls': 42, 'direct_answer_before_search': 15}`

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
| micro-average | 3500 | 0.1174 | 0.2032 | 3500 | 0.1174 | 0.2032 | 0.1174 |
| macro-average | 7 | 0.1159 | 0.2038 | 500 | 0.1159 | 0.2038 | 0.1159 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0373 | 0.1057 | 563 | 0.0373 | 0.1057 | 0.0373 |
| bamboogle | 125 | 0.1040 | 0.2087 | 125 | 0.1040 | 0.2087 | 0.1040 |
| hotpotqa | 562 | 0.0979 | 0.1951 | 562 | 0.0979 | 0.1951 | 0.0979 |
| musique | 562 | 0.0214 | 0.0739 | 562 | 0.0214 | 0.0739 | 0.0214 |
| nq | 562 | 0.1922 | 0.2695 | 562 | 0.1922 | 0.2695 | 0.1922 |
| popqa | 563 | 0.2256 | 0.2821 | 563 | 0.2256 | 0.2821 | 0.2256 |
| triviaqa | 563 | 0.1332 | 0.2915 | 563 | 0.1332 | 0.2915 | 0.1332 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.3043 | 68.7584 | 0.4344 | 0.0000 | 0.4344 | 69.2128 | 4.9171 |
| macro-average | 7 | 2.2884 | 68.1992 | 0.4008 | 0.0000 | 0.4008 | 68.6199 | 4.9264 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.1066 | 59.6194 | 0.4605 | 0.0000 | 0.4605 | 60.0975 | 4.6359 |
| bamboogle | 125 | 2.1600 | 63.6138 | 0.1321 | 0.0000 | 0.1322 | 63.7655 | 5.0000 |
| hotpotqa | 562 | 2.2278 | 70.6000 | 0.3462 | 0.0000 | 0.3462 | 70.9657 | 4.8932 |
| musique | 562 | 2.9786 | 104.7320 | 0.6896 | 0.0000 | 0.6896 | 105.4502 | 4.9644 |
| nq | 562 | 2.1335 | 80.4671 | 0.3390 | 0.0000 | 0.3390 | 80.8253 | 5.0000 |
| popqa | 563 | 2.5897 | 37.8691 | 0.5933 | 0.0000 | 0.5933 | 38.4829 | 5.0000 |
| triviaqa | 563 | 1.8224 | 60.4930 | 0.2452 | 0.0000 | 0.2452 | 60.7524 | 4.9911 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.