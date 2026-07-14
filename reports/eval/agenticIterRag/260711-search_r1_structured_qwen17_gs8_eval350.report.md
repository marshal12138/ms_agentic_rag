# AgenticIterRag v1 Infer Report

- Infer task: `spad_agent_search_eval`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/search_r1_structured.eval.parquet`
- Examples: `350`
- Success count: `349`
- Failure count: `1`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/AIR/260711-010148-047274-pipeline-agentic_iter_rag_v1_search_r1_structured_qwen3_1_7b_512/stages/train_agent/spad_rag/search_policy_rl/actor_model_hf/global_step_8`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8230/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_eval350/trace`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_eval350/trace/metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_eval350/runtime_logs/search_timing.jsonl`
- Flush every N: `10`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_eval350/runtime_logs/llm_io.jsonl`
- LLM IO max records: `20`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval/agenticIterRag/260711-search_r1_structured_qwen17_gs8_eval350/trace/validation_data`
- Wall time: `140.0171s`
- Status counts: `{'answered': 219, 'no_valid_answer': 50, 'max_turns': 79, 'failed': 1, 'multiple_tool_calls': 1}`

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
| micro-average | 350 | 0.1343 | 0.2021 | 342 | 0.1374 | 0.2009 | 0.1406 |
| macro-average | 7 | 0.1343 | 0.2021 | 48 | 0.1385 | 0.2027 | 0.1421 |

## Effect Metrics By Dataset

| Scope | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.1200 | 0.1684 | 50 | 0.1200 | 0.1684 | 0.1200 |
| bamboogle | 50 | 0.1000 | 0.1670 | 50 | 0.1000 | 0.1670 | 0.1000 |
| hotpotqa | 50 | 0.0400 | 0.0778 | 50 | 0.0400 | 0.0778 | 0.0400 |
| musique | 50 | 0.0600 | 0.0943 | 50 | 0.0600 | 0.0943 | 0.0600 |
| nq | 50 | 0.1200 | 0.2387 | 43 | 0.1628 | 0.2550 | 0.1880 |
| popqa | 50 | 0.1400 | 0.1935 | 50 | 0.1400 | 0.1935 | 0.1400 |
| triviaqa | 50 | 0.3600 | 0.4751 | 49 | 0.3469 | 0.4631 | 0.3469 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.2829 | 18.5501 | 6.8174 | 0.0000 | 6.8174 | 25.4420 | 4.9714 |
| macro-average | 7 | 2.2829 | 18.5501 | 6.8174 | 0.0000 | 6.8174 | 25.4420 | 4.9714 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.2200 | 14.6041 | 2.0554 | 0.0000 | 2.0554 | 17.0556 | 4.9000 |
| bamboogle | 50 | 2.1600 | 22.7950 | 13.2087 | 0.0000 | 13.2087 | 36.0197 | 5.0000 |
| hotpotqa | 50 | 2.6000 | 20.6370 | 7.9002 | 0.0000 | 7.9002 | 28.5568 | 4.9000 |
| musique | 50 | 2.7000 | 27.5612 | 13.0912 | 0.0000 | 13.0912 | 40.6723 | 5.0000 |
| nq | 50 | 2.0600 | 19.6803 | 7.9349 | 0.0000 | 7.9349 | 27.6296 | 5.0000 |
| popqa | 50 | 2.6000 | 14.5195 | 2.3473 | 0.0000 | 2.3473 | 16.9106 | 5.0000 |
| triviaqa | 50 | 1.6400 | 10.0540 | 1.1839 | 0.0000 | 1.1839 | 11.2493 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored infer metrics/traces for compatibility with previous full infer artifacts.