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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-newdata512-spad-stage3-retry1-run1/trace/validation_data`
- Wall time: `162.7770s`
- Status counts: `{'multiple_tool_calls': 3, 'answered': 225, 'no_valid_answer': 19, 'direct_answer_before_search': 3, 'max_turns': 100}`

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
| micro-average | 350 | 0.1343 | 0.2257 | 350 | 0.1343 | 0.2257 | 0.1343 |
| macro-average | 7 | 0.1343 | 0.2257 | 50 | 0.1343 | 0.2257 | 0.1343 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.0800 | 0.1867 | 50 | 0.0800 | 0.1867 | 0.0800 |
| bamboogle | 50 | 0.0800 | 0.2120 | 50 | 0.0800 | 0.2120 | 0.0800 |
| hotpotqa | 50 | 0.1600 | 0.1987 | 50 | 0.1600 | 0.1987 | 0.1600 |
| musique | 50 | 0.0800 | 0.1347 | 50 | 0.0800 | 0.1347 | 0.0800 |
| nq | 50 | 0.2400 | 0.3111 | 50 | 0.2400 | 0.3111 | 0.2400 |
| popqa | 50 | 0.2400 | 0.2899 | 50 | 0.2400 | 0.2899 | 0.2400 |
| triviaqa | 50 | 0.0600 | 0.2466 | 50 | 0.0600 | 0.2466 | 0.0600 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.5314 | 20.8190 | 9.5328 | 0.0000 | 9.5328 | 30.3754 | 4.9143 |
| macro-average | 7 | 2.5314 | 20.8190 | 9.5328 | 0.0000 | 9.5328 | 30.3754 | 4.9143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.2400 | 16.0743 | 2.7036 | 0.0000 | 2.7036 | 18.7943 | 4.7000 |
| bamboogle | 50 | 2.6200 | 28.0887 | 18.4144 | 0.0000 | 18.4144 | 46.5224 | 4.9000 |
| hotpotqa | 50 | 2.2800 | 18.2619 | 8.4069 | 0.0000 | 8.4069 | 26.6952 | 4.9000 |
| musique | 50 | 3.0200 | 31.0117 | 17.8927 | 0.0000 | 17.8927 | 48.9269 | 5.0000 |
| nq | 50 | 2.5000 | 21.3058 | 13.3398 | 0.0000 | 13.3398 | 34.6637 | 4.9000 |
| popqa | 50 | 2.8800 | 16.4181 | 2.5224 | 0.0000 | 2.5224 | 18.9871 | 5.0000 |
| triviaqa | 50 | 2.1800 | 14.5722 | 3.4496 | 0.0000 | 3.4496 | 18.0382 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.