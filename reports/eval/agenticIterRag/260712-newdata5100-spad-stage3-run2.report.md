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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-235953-727858-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100/stages/train_agent/spad_rag/answer_distillation/grpo/grpo_checkpoint_verl/actor_model_hf/global_step_38`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run2/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run2/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run2/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run2/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run2/trace/validation_data`
- Wall time: `50.0733s`
- Status counts: `{'no_valid_answer': 93, 'direct_answer_before_search': 93, 'answered': 161, 'multiple_tool_calls': 1, 'max_turns': 2}`

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
| micro-average | 350 | 0.1343 | 0.2088 | 350 | 0.1343 | 0.2088 | 0.1343 |
| macro-average | 7 | 0.1343 | 0.2088 | 50 | 0.1343 | 0.2088 | 0.1343 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1000 | 0.1707 | 50 | 0.1000 | 0.1707 | 0.1000 |
| bamboogle | 50 | 0.1200 | 0.1983 | 50 | 0.1200 | 0.1983 | 0.1200 |
| hotpotqa | 50 | 0.1800 | 0.2097 | 50 | 0.1800 | 0.2097 | 0.1800 |
| musique | 50 | 0.0200 | 0.0812 | 50 | 0.0200 | 0.0812 | 0.0200 |
| nq | 50 | 0.2400 | 0.3154 | 50 | 0.2400 | 0.3154 | 0.2400 |
| popqa | 50 | 0.2000 | 0.2233 | 50 | 0.2000 | 0.2233 | 0.2000 |
| triviaqa | 50 | 0.0800 | 0.2628 | 50 | 0.0800 | 0.2628 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.5686 | 8.0230 | 0.3929 | 0.0000 | 0.3929 | 8.4243 | 2.3286 |
| macro-average | 7 | 0.5686 | 8.0230 | 0.3929 | 0.0000 | 0.3929 | 8.4243 | 2.3286 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.6000 | 9.4206 | 0.2410 | 0.0000 | 0.2410 | 9.6661 | 2.2000 |
| bamboogle | 50 | 0.4800 | 9.1629 | 0.7880 | 0.0000 | 0.7880 | 9.9551 | 2.2000 |
| hotpotqa | 50 | 0.5800 | 7.9671 | 0.2722 | 0.0000 | 0.2722 | 8.2439 | 2.4000 |
| musique | 50 | 0.3600 | 7.6976 | 0.2753 | 0.0000 | 0.2753 | 7.9763 | 1.3000 |
| nq | 50 | 0.6400 | 7.4900 | 0.7395 | 0.0000 | 0.7395 | 8.2345 | 2.8000 |
| popqa | 50 | 0.7400 | 6.8251 | 0.3561 | 0.0000 | 0.3561 | 7.2134 | 2.6000 |
| triviaqa | 50 | 0.5800 | 7.5980 | 0.0783 | 0.0000 | 0.0783 | 7.6805 | 2.8000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.