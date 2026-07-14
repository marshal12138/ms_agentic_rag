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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run3/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run3/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run3/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run3/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run3/trace/validation_data`
- Wall time: `160.9563s`
- Status counts: `{'multiple_tool_calls': 3, 'answered': 229, 'no_valid_answer': 13, 'direct_answer_before_search': 3, 'max_turns': 102}`

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
| micro-average | 350 | 0.1343 | 0.2261 | 350 | 0.1343 | 0.2261 | 0.1343 |
| macro-average | 7 | 0.1343 | 0.2261 | 50 | 0.1343 | 0.2261 | 0.1343 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0600 | 0.1585 | 50 | 0.0600 | 0.1585 | 0.0600 |
| bamboogle | 50 | 0.1000 | 0.2197 | 50 | 0.1000 | 0.2197 | 0.1000 |
| hotpotqa | 50 | 0.1600 | 0.1956 | 50 | 0.1600 | 0.1956 | 0.1600 |
| musique | 50 | 0.0600 | 0.1346 | 50 | 0.0600 | 0.1346 | 0.0600 |
| nq | 50 | 0.2400 | 0.3156 | 50 | 0.2400 | 0.3156 | 0.2400 |
| popqa | 50 | 0.2600 | 0.3010 | 50 | 0.2600 | 0.3010 | 0.2600 |
| triviaqa | 50 | 0.0600 | 0.2578 | 50 | 0.0600 | 0.2578 | 0.0600 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.5143 | 20.6469 | 8.9505 | 0.0000 | 8.9506 | 29.6204 | 4.9143 |
| macro-average | 7 | 2.5143 | 20.6469 | 8.9505 | 0.0000 | 8.9506 | 29.6204 | 4.9143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.1600 | 15.5248 | 2.2931 | 0.0000 | 2.2931 | 17.8332 | 4.7000 |
| bamboogle | 50 | 2.5800 | 25.3040 | 16.4980 | 0.0000 | 16.4980 | 41.8209 | 4.9000 |
| hotpotqa | 50 | 2.1200 | 17.9449 | 5.4401 | 0.0000 | 5.4402 | 23.4009 | 4.9000 |
| musique | 50 | 3.2000 | 32.3866 | 20.7411 | 0.0000 | 20.7411 | 53.1517 | 5.0000 |
| nq | 50 | 2.5400 | 22.0713 | 12.7924 | 0.0000 | 12.7924 | 34.8812 | 4.9000 |
| popqa | 50 | 2.8600 | 16.6187 | 2.1308 | 0.0000 | 2.1308 | 18.7949 | 5.0000 |
| triviaqa | 50 | 2.1400 | 14.6778 | 2.7583 | 0.0000 | 2.7583 | 17.4601 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.