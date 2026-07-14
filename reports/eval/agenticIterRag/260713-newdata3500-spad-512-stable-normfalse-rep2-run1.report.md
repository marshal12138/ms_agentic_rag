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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-122915-051755-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stable_normfalse_rep2/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep2-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep2-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep2-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep2-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-stable-normfalse-rep2-run1/trace/validation_data`
- Wall time: `743.8857s`
- Status counts: `{'answered': 2402, 'no_valid_answer': 228, 'max_turns': 806, 'multiple_tool_calls': 47, 'direct_answer_before_search': 17}`

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
| micro-average | 3500 | 0.1297 | 0.2174 | 3500 | 0.1297 | 0.2174 | 0.1297 |
| macro-average | 7 | 0.1331 | 0.2214 | 500 | 0.1331 | 0.2214 | 0.1331 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0409 | 0.1031 | 563 | 0.0409 | 0.1031 | 0.0409 |
| bamboogle | 125 | 0.1600 | 0.2530 | 125 | 0.1600 | 0.2530 | 0.1600 |
| hotpotqa | 562 | 0.1103 | 0.2163 | 562 | 0.1103 | 0.2163 | 0.1103 |
| musique | 562 | 0.0249 | 0.0808 | 562 | 0.0249 | 0.0808 | 0.0249 |
| nq | 562 | 0.2028 | 0.2929 | 562 | 0.2028 | 0.2929 | 0.2028 |
| popqa | 563 | 0.2362 | 0.2861 | 563 | 0.2362 | 0.2861 | 0.2362 |
| triviaqa | 563 | 0.1563 | 0.3172 | 563 | 0.1563 | 0.3172 | 0.1563 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.2757 | 67.6437 | 0.4565 | 0.0000 | 0.4565 | 68.1188 | 4.9086 |
| macro-average | 7 | 2.2691 | 66.9018 | 0.4208 | 0.0000 | 0.4208 | 67.3410 | 4.9143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.0249 | 59.2519 | 0.4658 | 0.0000 | 0.4659 | 59.7338 | 4.5737 |
| bamboogle | 125 | 2.2160 | 60.8676 | 0.1353 | 0.0000 | 0.1353 | 61.0201 | 4.9600 |
| hotpotqa | 562 | 2.1904 | 72.5894 | 0.4288 | 0.0000 | 0.4288 | 73.0358 | 4.9021 |
| musique | 562 | 2.7064 | 96.4669 | 0.6367 | 0.0000 | 0.6367 | 97.1265 | 4.9822 |
| nq | 562 | 2.0427 | 77.7176 | 0.3379 | 0.0000 | 0.3379 | 78.0716 | 4.9911 |
| popqa | 563 | 2.7922 | 40.0000 | 0.6523 | 0.0000 | 0.6523 | 40.6764 | 5.0000 |
| triviaqa | 563 | 1.9112 | 61.4189 | 0.2887 | 0.0000 | 0.2888 | 61.7225 | 4.9911 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.