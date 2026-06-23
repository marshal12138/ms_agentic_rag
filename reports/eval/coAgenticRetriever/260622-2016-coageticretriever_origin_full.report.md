# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `coageticretriever_origin_full`
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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2016-coageticretriever_origin_full`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2016-coageticretriever_origin_full/runtime_logs/coageticretriever_origin_full.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2016-coageticretriever_origin_full/runtime_logs/coageticretriever_origin_full.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2016-coageticretriever_origin_full/runtime_logs/coageticretriever_origin_full.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2016-coageticretriever_origin_full/validation_data`
- Wall time: `67.7719s`
- Status counts: `{'answered': 336, 'no_valid_answer': 14}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3229 | 0.4223 |
| macro-average | 7 | 0.3229 | 0.4223 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2000 | 0.2504 |
| bamboogle | 50 | 0.3000 | 0.4256 |
| hotpotqa | 50 | 0.3800 | 0.4856 |
| musique | 50 | 0.1400 | 0.2419 |
| nq | 50 | 0.3600 | 0.4508 |
| popqa | 50 | 0.3400 | 0.4410 |
| triviaqa | 50 | 0.5400 | 0.6607 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.1343 | 4.3193 | 0.4093 | 0.1103 | 0.5195 | 4.8434 | 5.0000 |
| macro-average | 7 | 1.1343 | 4.3193 | 0.4093 | 0.1103 | 0.5195 | 4.8434 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.2600 | 4.7995 | 0.6298 | 0.1281 | 0.7579 | 5.5614 | 5.0000 |
| bamboogle | 50 | 1.1400 | 3.3595 | 0.1427 | 0.1010 | 0.2437 | 3.6068 | 5.0000 |
| hotpotqa | 50 | 1.1200 | 4.6351 | 0.2097 | 0.1108 | 0.3205 | 4.9592 | 5.0000 |
| musique | 50 | 1.3000 | 6.1709 | 0.1141 | 0.1232 | 0.2373 | 6.4128 | 5.0000 |
| nq | 50 | 1.0600 | 4.1466 | 0.2202 | 0.0971 | 0.3173 | 4.4673 | 5.0000 |
| popqa | 50 | 1.0200 | 3.7508 | 1.2665 | 0.1077 | 1.3743 | 5.1342 | 5.0000 |
| triviaqa | 50 | 1.0400 | 3.3729 | 0.2819 | 0.1040 | 0.3858 | 3.7620 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.