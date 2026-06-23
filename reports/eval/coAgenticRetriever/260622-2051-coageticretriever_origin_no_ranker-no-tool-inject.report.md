# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `coageticretriever_origin_no_ranker-no-tool-inject`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2051-coageticretriever_origin_no_ranker-no-tool-inject`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2051-coageticretriever_origin_no_ranker-no-tool-inject/runtime_logs/coageticretriever_origin_no_ranker-no-tool-inject.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2051-coageticretriever_origin_no_ranker-no-tool-inject/runtime_logs/coageticretriever_origin_no_ranker-no-tool-inject.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2051-coageticretriever_origin_no_ranker-no-tool-inject/runtime_logs/coageticretriever_origin_no_ranker-no-tool-inject.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2051-coageticretriever_origin_no_ranker-no-tool-inject/validation_data`
- Wall time: `26.8869s`
- Status counts: `{'no_valid_answer': 6, 'answered': 344}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3343 | 0.4337 |
| macro-average | 7 | 0.3343 | 0.4337 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2600 | 0.3068 |
| bamboogle | 50 | 0.3600 | 0.4782 |
| hotpotqa | 50 | 0.3600 | 0.4609 |
| musique | 50 | 0.1400 | 0.2444 |
| nq | 50 | 0.3600 | 0.4619 |
| popqa | 50 | 0.3200 | 0.4095 |
| triviaqa | 50 | 0.5400 | 0.6739 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0857 | 2.2393 | 0.0893 | 0.0000 | 0.0893 | 2.3360 | 5.0000 |
| macro-average | 7 | 1.0857 | 2.2393 | 0.0893 | 0.0000 | 0.0893 | 2.3360 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0800 | 2.4916 | 0.0328 | 0.0000 | 0.0328 | 2.5277 | 5.0000 |
| bamboogle | 50 | 1.1000 | 1.9112 | 0.0418 | 0.0000 | 0.0418 | 1.9563 | 5.0000 |
| hotpotqa | 50 | 1.1000 | 2.3337 | 0.0370 | 0.0000 | 0.0370 | 2.3742 | 5.0000 |
| musique | 50 | 1.2400 | 2.6465 | 0.0526 | 0.0000 | 0.0526 | 2.7030 | 5.0000 |
| nq | 50 | 1.0400 | 1.8309 | 0.0433 | 0.0000 | 0.0433 | 1.8773 | 5.0000 |
| popqa | 50 | 1.0000 | 2.6261 | 0.3784 | 0.0000 | 0.3785 | 3.0362 | 5.0000 |
| triviaqa | 50 | 1.0400 | 1.8348 | 0.0394 | 0.0000 | 0.0394 | 1.8774 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.