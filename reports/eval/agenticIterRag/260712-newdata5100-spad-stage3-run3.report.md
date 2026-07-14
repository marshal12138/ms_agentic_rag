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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run3/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run3/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run3/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run3/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260712-newdata5100-spad-stage3-run3/trace/validation_data`
- Wall time: `46.3300s`
- Status counts: `{'no_valid_answer': 101, 'direct_answer_before_search': 86, 'answered': 160, 'multiple_tool_calls': 1, 'max_turns': 2}`

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
| micro-average | 350 | 0.1371 | 0.2120 | 350 | 0.1371 | 0.2120 | 0.1371 |
| macro-average | 7 | 0.1371 | 0.2120 | 50 | 0.1371 | 0.2120 | 0.1371 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.1826 | 50 | 0.1200 | 0.1826 | 0.1200 |
| bamboogle | 50 | 0.1000 | 0.1717 | 50 | 0.1000 | 0.1717 | 0.1000 |
| hotpotqa | 50 | 0.2000 | 0.2431 | 50 | 0.2000 | 0.2431 | 0.2000 |
| musique | 50 | 0.0200 | 0.0887 | 50 | 0.0200 | 0.0887 | 0.0200 |
| nq | 50 | 0.2400 | 0.3154 | 50 | 0.2400 | 0.3154 | 0.2400 |
| popqa | 50 | 0.2000 | 0.2233 | 50 | 0.2000 | 0.2233 | 0.2000 |
| triviaqa | 50 | 0.0800 | 0.2589 | 50 | 0.0800 | 0.2589 | 0.0800 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.5714 | 7.7796 | 0.3840 | 0.0000 | 0.3840 | 8.1719 | 2.3143 |
| macro-average | 7 | 0.5714 | 7.7796 | 0.3840 | 0.0000 | 0.3840 | 8.1719 | 2.3143 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.6000 | 9.2999 | 0.2277 | 0.0000 | 0.2278 | 9.5323 | 2.1000 |
| bamboogle | 50 | 0.5200 | 8.6892 | 0.5601 | 0.0000 | 0.5601 | 9.2535 | 2.3000 |
| hotpotqa | 50 | 0.5600 | 7.5548 | 0.3147 | 0.0000 | 0.3147 | 7.8740 | 2.3000 |
| musique | 50 | 0.3800 | 7.6346 | 0.4901 | 0.0000 | 0.4901 | 8.1282 | 1.3000 |
| nq | 50 | 0.6200 | 7.2801 | 0.6667 | 0.0000 | 0.6667 | 7.9512 | 2.8000 |
| popqa | 50 | 0.7400 | 6.7275 | 0.3570 | 0.0000 | 0.3570 | 7.1174 | 2.6000 |
| triviaqa | 50 | 0.5800 | 7.2710 | 0.0715 | 0.0000 | 0.0715 | 7.3467 | 2.8000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.