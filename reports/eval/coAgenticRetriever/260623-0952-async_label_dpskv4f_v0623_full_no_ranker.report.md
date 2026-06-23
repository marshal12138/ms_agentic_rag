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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260623-025839-CAR_async_labeling_ds_flash_larger_ranker_tdata/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0952-async_label_dpskv4f_v0623_full_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0952-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0952-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0952-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0952-async_label_dpskv4f_v0623_full_no_ranker/validation_data`
- Wall time: `27.8583s`
- Status counts: `{'answered': 348, 'no_valid_answer': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3514 | 0.4448 |
| macro-average | 7 | 0.3514 | 0.4448 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.4000 | 0.4452 |
| bamboogle | 50 | 0.2200 | 0.3380 |
| hotpotqa | 50 | 0.3800 | 0.4624 |
| musique | 50 | 0.1400 | 0.1973 |
| nq | 50 | 0.4200 | 0.5501 |
| popqa | 50 | 0.3200 | 0.4210 |
| triviaqa | 50 | 0.5800 | 0.6997 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9971 | 2.3430 | 0.0898 | 0.0000 | 0.0898 | 2.4414 | 4.9857 |
| macro-average | 7 | 0.9971 | 2.3430 | 0.0898 | 0.0000 | 0.0898 | 2.4414 | 4.9857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 2.5114 | 0.0377 | 0.0000 | 0.0377 | 2.5519 | 5.0000 |
| bamboogle | 50 | 1.0000 | 2.0556 | 0.0399 | 0.0000 | 0.0399 | 2.0984 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 2.3704 | 0.0376 | 0.0000 | 0.0376 | 2.4109 | 5.0000 |
| musique | 50 | 1.0000 | 2.3362 | 0.0434 | 0.0000 | 0.0434 | 2.3825 | 5.0000 |
| nq | 50 | 1.0000 | 1.9722 | 0.0370 | 0.0000 | 0.0370 | 2.0121 | 5.0000 |
| popqa | 50 | 1.0000 | 3.1239 | 0.3873 | 0.0000 | 0.3873 | 3.5546 | 5.0000 |
| triviaqa | 50 | 0.9800 | 2.0311 | 0.0457 | 0.0000 | 0.0457 | 2.0795 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.