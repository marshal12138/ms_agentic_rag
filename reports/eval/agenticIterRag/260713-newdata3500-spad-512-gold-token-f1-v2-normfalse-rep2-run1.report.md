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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260713-141055-402010-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_gold_token_f1_v2_normfalse_rep2/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep2-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep2-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep2-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep2-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260713-newdata3500-spad-512-gold-token-f1-v2-normfalse-rep2-run1/trace/validation_data`
- Wall time: `756.6277s`
- Status counts: `{'answered': 2287, 'no_valid_answer': 247, 'max_turns': 896, 'direct_answer_before_search': 11, 'multiple_tool_calls': 59}`

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
| micro-average | 3500 | 0.1286 | 0.2110 | 3500 | 0.1286 | 0.2110 | 0.1286 |
| macro-average | 7 | 0.1294 | 0.2145 | 500 | 0.1294 | 0.2145 | 0.1294 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.0462 | 0.1050 | 563 | 0.0462 | 0.1050 | 0.0462 |
| bamboogle | 125 | 0.1360 | 0.2426 | 125 | 0.1360 | 0.2426 | 0.1360 |
| hotpotqa | 562 | 0.1192 | 0.2149 | 562 | 0.1192 | 0.2149 | 0.1192 |
| musique | 562 | 0.0409 | 0.0943 | 562 | 0.0409 | 0.0943 | 0.0409 |
| nq | 562 | 0.1815 | 0.2627 | 562 | 0.1815 | 0.2627 | 0.1815 |
| popqa | 563 | 0.2327 | 0.2866 | 563 | 0.2327 | 0.2866 | 0.2327 |
| triviaqa | 563 | 0.1492 | 0.2954 | 563 | 0.1492 | 0.2954 | 0.1492 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.3423 | 68.3685 | 0.4223 | 0.0000 | 0.4224 | 68.8113 | 4.9000 |
| macro-average | 7 | 2.3435 | 67.6085 | 0.3931 | 0.0000 | 0.3931 | 68.0221 | 4.9112 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.0284 | 59.3922 | 0.4002 | 0.0000 | 0.4002 | 59.8086 | 4.4938 |
| bamboogle | 125 | 2.3520 | 61.4206 | 0.1592 | 0.0000 | 0.1592 | 61.6009 | 5.0000 |
| hotpotqa | 562 | 2.2438 | 70.3158 | 0.3554 | 0.0000 | 0.3554 | 70.6913 | 4.8843 |
| musique | 562 | 2.9733 | 103.9872 | 0.5085 | 0.0000 | 0.5085 | 104.5240 | 5.0000 |
| nq | 562 | 2.1495 | 78.5550 | 0.4753 | 0.0000 | 0.4753 | 79.0482 | 5.0000 |
| popqa | 563 | 2.7691 | 39.1376 | 0.6074 | 0.0000 | 0.6074 | 39.7699 | 5.0000 |
| triviaqa | 563 | 1.8881 | 60.4509 | 0.2458 | 0.0000 | 0.2458 | 60.7118 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.