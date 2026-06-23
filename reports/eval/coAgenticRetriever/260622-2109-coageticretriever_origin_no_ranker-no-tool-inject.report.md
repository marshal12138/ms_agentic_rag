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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2109-coageticretriever_origin_no_ranker-no-tool-inject`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2109-coageticretriever_origin_no_ranker-no-tool-inject/runtime_logs/coageticretriever_origin_no_ranker-no-tool-inject.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2109-coageticretriever_origin_no_ranker-no-tool-inject/runtime_logs/coageticretriever_origin_no_ranker-no-tool-inject.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2109-coageticretriever_origin_no_ranker-no-tool-inject/runtime_logs/coageticretriever_origin_no_ranker-no-tool-inject.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2109-coageticretriever_origin_no_ranker-no-tool-inject/validation_data`
- Wall time: `28.1372s`
- Status counts: `{'no_valid_answer': 4, 'answered': 346}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3371 | 0.4363 |
| macro-average | 7 | 0.3371 | 0.4363 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2600 | 0.3125 |
| bamboogle | 50 | 0.3600 | 0.4782 |
| hotpotqa | 50 | 0.3800 | 0.4752 |
| musique | 50 | 0.1400 | 0.2429 |
| nq | 50 | 0.3600 | 0.4586 |
| popqa | 50 | 0.3200 | 0.4095 |
| triviaqa | 50 | 0.5400 | 0.6772 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0886 | 2.3359 | 0.0907 | 0.0000 | 0.0907 | 2.4341 | 5.0000 |
| macro-average | 7 | 1.0886 | 2.3359 | 0.0907 | 0.0000 | 0.0907 | 2.4341 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1000 | 2.6227 | 0.0434 | 0.0000 | 0.0434 | 2.6697 | 5.0000 |
| bamboogle | 50 | 1.1200 | 1.9928 | 0.0439 | 0.0000 | 0.0439 | 2.0401 | 5.0000 |
| hotpotqa | 50 | 1.1400 | 2.5408 | 0.0378 | 0.0000 | 0.0378 | 2.5823 | 5.0000 |
| musique | 50 | 1.2200 | 2.7518 | 0.0448 | 0.0000 | 0.0448 | 2.8005 | 5.0000 |
| nq | 50 | 1.0000 | 1.8330 | 0.0373 | 0.0000 | 0.0373 | 1.8733 | 5.0000 |
| popqa | 50 | 1.0000 | 2.6941 | 0.3848 | 0.0000 | 0.3848 | 3.1105 | 5.0000 |
| triviaqa | 50 | 1.0400 | 1.9162 | 0.0430 | 0.0000 | 0.0430 | 1.9624 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.