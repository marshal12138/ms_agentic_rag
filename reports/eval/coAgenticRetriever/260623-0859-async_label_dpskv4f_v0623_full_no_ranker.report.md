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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0859-async_label_dpskv4f_v0623_full_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0859-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0859-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0859-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0859-async_label_dpskv4f_v0623_full_no_ranker/validation_data`
- Wall time: `27.7328s`
- Status counts: `{'answered': 348, 'no_valid_answer': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3514 | 0.4437 |
| macro-average | 7 | 0.3514 | 0.4437 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.3600 | 0.4052 |
| bamboogle | 50 | 0.2200 | 0.3300 |
| hotpotqa | 50 | 0.3800 | 0.4624 |
| musique | 50 | 0.1600 | 0.2173 |
| nq | 50 | 0.4200 | 0.5501 |
| popqa | 50 | 0.3200 | 0.4210 |
| triviaqa | 50 | 0.6000 | 0.7197 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9971 | 2.3126 | 0.1097 | 0.0000 | 0.1097 | 2.4308 | 4.9857 |
| macro-average | 7 | 0.9971 | 2.3126 | 0.1097 | 0.0000 | 0.1097 | 2.4308 | 4.9857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 2.5085 | 0.0448 | 0.0000 | 0.0448 | 2.5561 | 5.0000 |
| bamboogle | 50 | 1.0000 | 2.0422 | 0.0396 | 0.0000 | 0.0396 | 2.0846 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 2.3854 | 0.0437 | 0.0000 | 0.0437 | 2.4319 | 5.0000 |
| musique | 50 | 1.0000 | 2.3077 | 0.0391 | 0.0000 | 0.0391 | 2.3497 | 5.0000 |
| nq | 50 | 1.0000 | 1.9543 | 0.0395 | 0.0000 | 0.0395 | 1.9966 | 5.0000 |
| popqa | 50 | 1.0000 | 2.9664 | 0.5270 | 0.0000 | 0.5270 | 3.5367 | 5.0000 |
| triviaqa | 50 | 0.9800 | 2.0236 | 0.0340 | 0.0000 | 0.0340 | 2.0604 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.