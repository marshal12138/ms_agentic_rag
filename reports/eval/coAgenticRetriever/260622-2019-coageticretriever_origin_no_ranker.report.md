# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `coageticretriever_origin_no_ranker`
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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2019-coageticretriever_origin_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2019-coageticretriever_origin_no_ranker/runtime_logs/coageticretriever_origin_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2019-coageticretriever_origin_no_ranker/runtime_logs/coageticretriever_origin_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2019-coageticretriever_origin_no_ranker/runtime_logs/coageticretriever_origin_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2019-coageticretriever_origin_no_ranker/validation_data`
- Wall time: `54.1184s`
- Status counts: `{'no_valid_answer': 17, 'answered': 330, 'max_turns': 3}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3200 | 0.4162 |
| macro-average | 7 | 0.3200 | 0.4162 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2400 | 0.2726 |
| bamboogle | 50 | 0.3000 | 0.4282 |
| hotpotqa | 50 | 0.3800 | 0.4856 |
| musique | 50 | 0.1200 | 0.2186 |
| nq | 50 | 0.3600 | 0.4520 |
| popqa | 50 | 0.3000 | 0.3895 |
| triviaqa | 50 | 0.5400 | 0.6668 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.1229 | 3.8494 | 0.0908 | 0.0000 | 0.0908 | 3.9497 | 5.0000 |
| macro-average | 7 | 1.1229 | 3.8494 | 0.0908 | 0.0000 | 0.0908 | 3.9497 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1600 | 4.0235 | 0.0398 | 0.0000 | 0.0398 | 4.0668 | 5.0000 |
| bamboogle | 50 | 1.1600 | 3.3843 | 0.0345 | 0.0000 | 0.0345 | 3.4223 | 5.0000 |
| hotpotqa | 50 | 1.1000 | 4.3789 | 0.0385 | 0.0000 | 0.0385 | 4.4208 | 5.0000 |
| musique | 50 | 1.3200 | 5.9146 | 0.0386 | 0.0000 | 0.0386 | 5.9577 | 5.0000 |
| nq | 50 | 1.0800 | 3.0069 | 0.0359 | 0.0000 | 0.0359 | 3.0460 | 5.0000 |
| popqa | 50 | 1.0000 | 3.5897 | 0.4157 | 0.0000 | 0.4157 | 4.0507 | 5.0000 |
| triviaqa | 50 | 1.0400 | 2.6478 | 0.0326 | 0.0000 | 0.0326 | 2.6834 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.