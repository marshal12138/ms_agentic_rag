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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm03_stage1/stages/train_agent/spad_rag/search_policy_rl/actor_model_verl/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1/runtime_logs/search_timing.jsonl`
- Flush every N: `500`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260715-newdata3500-spad-5100-gold-token-f1-v3-postnorm03-run1/trace/validation_data`
- Wall time: `798.4587s`
- Status counts: `{'answered': 2483, 'no_valid_answer': 41, 'max_turns': 950, 'multiple_tool_calls': 24, 'direct_answer_before_search': 2}`

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
| micro-average | 3500 | 0.1929 | 0.2734 | 3500 | 0.1929 | 0.2734 | 0.1929 |
| macro-average | 7 | 0.1954 | 0.2767 | 500 | 0.1954 | 0.2767 | 0.1954 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 0.1119 | 0.1618 | 563 | 0.1119 | 0.1618 | 0.1119 |
| bamboogle | 125 | 0.2160 | 0.3031 | 125 | 0.2160 | 0.3031 | 0.2160 |
| hotpotqa | 562 | 0.2242 | 0.3089 | 562 | 0.2242 | 0.3089 | 0.2242 |
| musique | 562 | 0.0569 | 0.1124 | 562 | 0.0569 | 0.1124 | 0.0569 |
| nq | 562 | 0.2954 | 0.3825 | 562 | 0.2954 | 0.3825 | 0.2954 |
| popqa | 563 | 0.2860 | 0.3202 | 563 | 0.2860 | 0.3202 | 0.2860 |
| triviaqa | 563 | 0.1776 | 0.3478 | 563 | 0.1776 | 0.3478 | 0.1776 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 3500 | 2.6883 | 72.7382 | 0.5982 | 0.0000 | 0.5982 | 73.3609 | 4.9629 |
| macro-average | 7 | 2.6607 | 71.5009 | 0.5514 | 0.0000 | 0.5514 | 72.0769 | 4.9626 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 563 | 2.7815 | 64.3083 | 0.5258 | 0.0000 | 0.5258 | 64.8578 | 4.8313 |
| bamboogle | 125 | 2.4400 | 61.4885 | 0.1765 | 0.0000 | 0.1765 | 61.6904 | 4.9600 |
| hotpotqa | 562 | 2.6032 | 78.5021 | 0.8391 | 0.0000 | 0.8392 | 79.3665 | 4.9555 |
| musique | 562 | 3.1779 | 108.6868 | 0.5642 | 0.0000 | 0.5642 | 109.2859 | 4.9911 |
| nq | 562 | 2.3541 | 82.2170 | 0.4151 | 0.0000 | 0.4151 | 82.6517 | 5.0000 |
| popqa | 563 | 2.7957 | 36.3586 | 0.8152 | 0.0000 | 0.8152 | 37.1971 | 5.0000 |
| triviaqa | 563 | 2.4725 | 68.9448 | 0.5236 | 0.0000 | 0.5236 | 69.4886 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.