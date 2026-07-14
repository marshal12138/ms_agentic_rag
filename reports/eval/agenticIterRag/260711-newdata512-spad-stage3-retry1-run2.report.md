# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/350e/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-115144-826023-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_512_stage3_resume/stages/train_agent/spad_rag/answer_distillation/grpo/grpo_checkpoint_verl/actor_model_hf/global_step_3`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run2/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run2/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run2/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run2/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run2/trace/validation_data`
- Wall time: `168.1554s`
- Status counts: `{'multiple_tool_calls': 3, 'answered': 231, 'no_valid_answer': 14, 'direct_answer_before_search': 3, 'max_turns': 99}`

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
| micro-average | 350 | 0.1400 | 0.2300 | 350 | 0.1400 | 0.2300 | 0.1400 |
| macro-average | 7 | 0.1400 | 0.2300 | 50 | 0.1400 | 0.2300 | 0.1400 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0600 | 0.1562 | 50 | 0.0600 | 0.1562 | 0.0600 |
| bamboogle | 50 | 0.1200 | 0.2361 | 50 | 0.1200 | 0.2361 | 0.1200 |
| hotpotqa | 50 | 0.1400 | 0.2057 | 50 | 0.1400 | 0.2057 | 0.1400 |
| musique | 50 | 0.0800 | 0.1436 | 50 | 0.0800 | 0.1436 | 0.0800 |
| nq | 50 | 0.2800 | 0.3362 | 50 | 0.2800 | 0.3362 | 0.2800 |
| popqa | 50 | 0.2200 | 0.2624 | 50 | 0.2200 | 0.2624 | 0.2200 |
| triviaqa | 50 | 0.0800 | 0.2702 | 50 | 0.0800 | 0.2702 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.5400 | 21.6311 | 8.8977 | 0.0000 | 8.8977 | 30.5508 | 4.9143 |
| macro-average | 7 | 2.5400 | 21.6311 | 8.8977 | 0.0000 | 8.8977 | 30.5508 | 4.9143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.2600 | 16.3197 | 2.6186 | 0.0000 | 2.6186 | 18.9541 | 4.7000 |
| bamboogle | 50 | 2.6400 | 30.0449 | 18.2628 | 0.0000 | 18.2628 | 48.3272 | 4.9000 |
| hotpotqa | 50 | 2.1800 | 18.2429 | 6.3071 | 0.0000 | 6.3072 | 24.5661 | 4.9000 |
| musique | 50 | 3.2800 | 34.9687 | 16.6550 | 0.0000 | 16.6550 | 51.6478 | 5.0000 |
| nq | 50 | 2.4800 | 21.0265 | 13.7987 | 0.0000 | 13.7987 | 34.8423 | 4.9000 |
| popqa | 50 | 2.9200 | 17.1476 | 2.2656 | 0.0000 | 2.2657 | 19.4606 | 5.0000 |
| triviaqa | 50 | 2.0200 | 13.6674 | 2.3757 | 0.0000 | 2.3757 | 16.0573 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.