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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2106-coageticretriever_origin_full-no-tool-inject`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2106-coageticretriever_origin_full-no-tool-inject/runtime_logs/coageticretriever_origin_full-no-tool-inject.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2106-coageticretriever_origin_full-no-tool-inject/runtime_logs/coageticretriever_origin_full-no-tool-inject.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2106-coageticretriever_origin_full-no-tool-inject/runtime_logs/coageticretriever_origin_full-no-tool-inject.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2106-coageticretriever_origin_full-no-tool-inject/validation_data`
- Wall time: `71.0401s`
- Status counts: `{'answered': 344, 'no_valid_answer': 5, 'max_turns': 1}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3400 | 0.4385 |
| macro-average | 7 | 0.3400 | 0.4385 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2200 | 0.2734 |
| bamboogle | 50 | 0.4000 | 0.5182 |
| hotpotqa | 50 | 0.3600 | 0.4554 |
| musique | 50 | 0.1400 | 0.2404 |
| nq | 50 | 0.3800 | 0.4803 |
| popqa | 50 | 0.3200 | 0.4210 |
| triviaqa | 50 | 0.5600 | 0.6805 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.1229 | 4.5940 | 1.3235 | 0.1727 | 1.4963 | 6.0948 | 5.0000 |
| macro-average | 7 | 1.1229 | 4.5940 | 1.3235 | 0.1727 | 1.4963 | 6.0948 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1800 | 4.7027 | 1.8115 | 0.1952 | 2.0067 | 6.7133 | 5.0000 |
| bamboogle | 50 | 1.1400 | 4.0549 | 1.0158 | 0.1532 | 1.1691 | 5.2275 | 5.0000 |
| hotpotqa | 50 | 1.1000 | 4.8846 | 1.1752 | 0.1778 | 1.3530 | 6.2414 | 5.0000 |
| musique | 50 | 1.2400 | 5.1306 | 1.2570 | 0.1943 | 1.4513 | 6.5860 | 5.0000 |
| nq | 50 | 1.1400 | 4.5369 | 1.1285 | 0.1713 | 1.2998 | 5.8405 | 5.0000 |
| popqa | 50 | 1.0000 | 4.2161 | 1.7838 | 0.1449 | 1.9287 | 6.1540 | 5.0000 |
| triviaqa | 50 | 1.0600 | 4.6322 | 1.0928 | 0.1725 | 1.2653 | 5.9011 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.