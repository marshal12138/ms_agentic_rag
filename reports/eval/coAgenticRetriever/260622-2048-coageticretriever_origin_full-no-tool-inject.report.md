# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `coageticretriever_origin_full-no-tool-inject`
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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2048-coageticretriever_origin_full-no-tool-inject`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2048-coageticretriever_origin_full-no-tool-inject/runtime_logs/coageticretriever_origin_full-no-tool-inject.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2048-coageticretriever_origin_full-no-tool-inject/runtime_logs/coageticretriever_origin_full-no-tool-inject.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2048-coageticretriever_origin_full-no-tool-inject/runtime_logs/coageticretriever_origin_full-no-tool-inject.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2048-coageticretriever_origin_full-no-tool-inject/validation_data`
- Wall time: `71.6902s`
- Status counts: `{'answered': 347, 'no_valid_answer': 3}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3457 | 0.4464 |
| macro-average | 7 | 0.3457 | 0.4464 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2800 | 0.3334 |
| bamboogle | 50 | 0.3600 | 0.4882 |
| hotpotqa | 50 | 0.3600 | 0.4594 |
| musique | 50 | 0.1400 | 0.2438 |
| nq | 50 | 0.3800 | 0.4730 |
| popqa | 50 | 0.3400 | 0.4410 |
| triviaqa | 50 | 0.5600 | 0.6858 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.1114 | 4.7073 | 1.2586 | 0.1781 | 1.4367 | 6.1486 | 5.0000 |
| macro-average | 7 | 1.1114 | 4.7073 | 1.2586 | 0.1781 | 1.4367 | 6.1486 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1600 | 5.3124 | 1.2201 | 0.2014 | 1.4215 | 6.7376 | 5.0000 |
| bamboogle | 50 | 1.1000 | 3.9980 | 1.0356 | 0.1559 | 1.1915 | 5.1931 | 5.0000 |
| hotpotqa | 50 | 1.1000 | 5.0626 | 1.1570 | 0.1866 | 1.3436 | 6.4107 | 5.0000 |
| musique | 50 | 1.3000 | 5.5739 | 1.3279 | 0.2099 | 1.5377 | 7.1162 | 5.0000 |
| nq | 50 | 1.0800 | 4.3459 | 1.0290 | 0.1677 | 1.1967 | 5.5462 | 5.0000 |
| popqa | 50 | 1.0000 | 4.0037 | 1.9049 | 0.1548 | 2.0597 | 6.0726 | 5.0000 |
| triviaqa | 50 | 1.0400 | 4.6543 | 1.1359 | 0.1701 | 1.3060 | 5.9638 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.