# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `async_label_dpskv4f_v0623_full_no_ranker`
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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0946-async_label_dpskv4f_v0623_full_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0946-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0946-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0946-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0946-async_label_dpskv4f_v0623_full_no_ranker/validation_data`
- Wall time: `33.6565s`
- Status counts: `{'no_valid_answer': 6, 'answered': 342, 'max_turns': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3314 | 0.4278 |
| macro-average | 7 | 0.3314 | 0.4278 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2400 | 0.2868 |
| bamboogle | 50 | 0.3600 | 0.4782 |
| hotpotqa | 50 | 0.3600 | 0.4575 |
| musique | 50 | 0.1400 | 0.2430 |
| nq | 50 | 0.3600 | 0.4507 |
| popqa | 50 | 0.3200 | 0.4095 |
| triviaqa | 50 | 0.5400 | 0.6686 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.1114 | 2.8349 | 0.0947 | 0.0000 | 0.0947 | 2.9389 | 5.0000 |
| macro-average | 7 | 1.1114 | 2.8349 | 0.0947 | 0.0000 | 0.0947 | 2.9389 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1000 | 3.0544 | 0.0391 | 0.0000 | 0.0391 | 3.0969 | 5.0000 |
| bamboogle | 50 | 1.1200 | 2.4280 | 0.0417 | 0.0000 | 0.0417 | 2.4731 | 5.0000 |
| hotpotqa | 50 | 1.1000 | 2.9009 | 0.0391 | 0.0000 | 0.0391 | 2.9438 | 5.0000 |
| musique | 50 | 1.3000 | 3.4042 | 0.0427 | 0.0000 | 0.0427 | 3.4512 | 5.0000 |
| nq | 50 | 1.0800 | 2.3736 | 0.0450 | 0.0000 | 0.0450 | 2.4219 | 5.0000 |
| popqa | 50 | 1.0200 | 3.3478 | 0.4178 | 0.0000 | 0.4178 | 3.8091 | 5.0000 |
| triviaqa | 50 | 1.0600 | 2.3352 | 0.0376 | 0.0000 | 0.0376 | 2.3762 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.