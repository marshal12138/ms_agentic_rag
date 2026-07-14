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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run1/trace/validation_data`
- Wall time: `47.4492s`
- Status counts: `{'no_valid_answer': 98, 'direct_answer_before_search': 89, 'answered': 161, 'multiple_tool_calls': 1, 'max_turns': 1}`

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
| micro-average | 350 | 0.1400 | 0.2139 | 350 | 0.1400 | 0.2139 | 0.1400 |
| macro-average | 7 | 0.1400 | 0.2139 | 50 | 0.1400 | 0.2139 | 0.1400 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1000 | 0.1683 | 50 | 0.1000 | 0.1683 | 0.1000 |
| bamboogle | 50 | 0.1400 | 0.2117 | 50 | 0.1400 | 0.2117 | 0.1400 |
| hotpotqa | 50 | 0.1800 | 0.2097 | 50 | 0.1800 | 0.2097 | 0.1800 |
| musique | 50 | 0.0200 | 0.0916 | 50 | 0.0200 | 0.0916 | 0.0200 |
| nq | 50 | 0.2600 | 0.3354 | 50 | 0.2600 | 0.3354 | 0.2600 |
| popqa | 50 | 0.2000 | 0.2233 | 50 | 0.2000 | 0.2233 | 0.2000 |
| triviaqa | 50 | 0.0800 | 0.2574 | 50 | 0.0800 | 0.2574 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.5657 | 7.6318 | 0.3565 | 0.0000 | 0.3565 | 7.9966 | 2.3143 |
| macro-average | 7 | 0.5657 | 7.6318 | 0.3565 | 0.0000 | 0.3565 | 7.9966 | 2.3143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.4800 | 8.4247 | 0.1203 | 0.0000 | 0.1203 | 8.5488 | 2.2000 |
| bamboogle | 50 | 0.5600 | 8.9135 | 0.6345 | 0.0000 | 0.6345 | 9.5528 | 2.3000 |
| hotpotqa | 50 | 0.5600 | 7.5799 | 0.3022 | 0.0000 | 0.3022 | 7.8865 | 2.3000 |
| musique | 50 | 0.4000 | 7.4633 | 0.2531 | 0.0000 | 0.2531 | 7.7200 | 1.4000 |
| nq | 50 | 0.6600 | 7.5577 | 0.7377 | 0.0000 | 0.7377 | 8.3004 | 2.8000 |
| popqa | 50 | 0.7600 | 6.5738 | 0.3355 | 0.0000 | 0.3355 | 6.9416 | 2.6000 |
| triviaqa | 50 | 0.5400 | 6.9097 | 0.1121 | 0.0000 | 0.1121 | 7.0261 | 2.6000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.