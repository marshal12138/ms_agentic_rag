# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `coageticretriever_origin_full-inject`
- Run mode: `full`
- Reranker: `dense_e5`
- Enable thinking: `false`
- Ranker enabled: `true`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- Ranker tokenizer/base model: `/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
- Ranker encoder: `/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2157-coageticretriever_origin_full-inject`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2157-coageticretriever_origin_full-inject/runtime_logs/coageticretriever_origin_full-inject.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2157-coageticretriever_origin_full-inject/runtime_logs/coageticretriever_origin_full-inject.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2157-coageticretriever_origin_full-inject/runtime_logs/coageticretriever_origin_full-inject.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2157-coageticretriever_origin_full-inject/validation_data`
- Wall time: `69.4899s`
- Status counts: `{'answered': 341, 'no_valid_answer': 8, 'max_turns': 1}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.2771 | 0.3640 |
| macro-average | 7 | 0.2771 | 0.3640 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2200 | 0.2847 |
| bamboogle | 50 | 0.2200 | 0.3100 |
| hotpotqa | 50 | 0.2000 | 0.2943 |
| musique | 50 | 0.1200 | 0.2151 |
| nq | 50 | 0.3600 | 0.4446 |
| popqa | 50 | 0.3000 | 0.3669 |
| triviaqa | 50 | 0.5200 | 0.6322 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0857 | 4.3248 | 1.3230 | 0.1681 | 1.4911 | 5.8205 | 5.0000 |
| macro-average | 7 | 1.0857 | 4.3248 | 1.3230 | 0.1681 | 1.4911 | 5.8205 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1400 | 4.8856 | 1.6255 | 0.1939 | 1.8194 | 6.7088 | 5.0000 |
| bamboogle | 50 | 1.0600 | 3.9425 | 1.1083 | 0.1515 | 1.2598 | 5.2059 | 5.0000 |
| hotpotqa | 50 | 1.0800 | 4.3942 | 1.3564 | 0.1717 | 1.5281 | 5.9261 | 5.0000 |
| musique | 50 | 1.2800 | 5.1817 | 1.4609 | 0.2049 | 1.6657 | 6.8520 | 5.0000 |
| nq | 50 | 1.0400 | 3.9250 | 1.0079 | 0.1558 | 1.1637 | 5.0925 | 5.0000 |
| popqa | 50 | 1.0000 | 3.6245 | 1.6942 | 0.1360 | 1.8301 | 5.4640 | 5.0000 |
| triviaqa | 50 | 1.0000 | 4.3199 | 1.0078 | 0.1631 | 1.1710 | 5.4942 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.