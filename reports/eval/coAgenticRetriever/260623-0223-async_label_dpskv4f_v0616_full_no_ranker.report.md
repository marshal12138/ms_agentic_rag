# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `async_label_dpskv4f_v0616_full_no_ranker`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260622-220205-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0223-async_label_dpskv4f_v0616_full_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0223-async_label_dpskv4f_v0616_full_no_ranker/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0223-async_label_dpskv4f_v0616_full_no_ranker/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0223-async_label_dpskv4f_v0616_full_no_ranker/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0223-async_label_dpskv4f_v0616_full_no_ranker/validation_data`
- Wall time: `22.3106s`
- Status counts: `{'answered': 349, 'no_valid_answer': 1}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3200 | 0.4118 |
| macro-average | 7 | 0.3200 | 0.4118 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2800 | 0.3204 |
| bamboogle | 50 | 0.2200 | 0.3367 |
| hotpotqa | 50 | 0.3000 | 0.3764 |
| musique | 50 | 0.1200 | 0.2361 |
| nq | 50 | 0.4400 | 0.5267 |
| popqa | 50 | 0.2600 | 0.3634 |
| triviaqa | 50 | 0.6200 | 0.7227 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9971 | 1.8421 | 0.0932 | 0.0000 | 0.0932 | 1.9438 | 4.9857 |
| macro-average | 7 | 0.9971 | 1.8421 | 0.0932 | 0.0000 | 0.0932 | 1.9438 | 4.9857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 1.8369 | 0.0494 | 0.0000 | 0.0494 | 1.8890 | 5.0000 |
| bamboogle | 50 | 1.0000 | 1.5916 | 0.0499 | 0.0000 | 0.0499 | 1.6443 | 5.0000 |
| hotpotqa | 50 | 0.9800 | 1.7615 | 0.0427 | 0.0000 | 0.0427 | 1.8070 | 4.9000 |
| musique | 50 | 1.0000 | 1.7029 | 0.0430 | 0.0000 | 0.0430 | 1.7485 | 5.0000 |
| nq | 50 | 1.0000 | 1.6184 | 0.0464 | 0.0000 | 0.0464 | 1.6675 | 5.0000 |
| popqa | 50 | 1.0000 | 2.7326 | 0.3706 | 0.0000 | 0.3706 | 3.1468 | 5.0000 |
| triviaqa | 50 | 1.0000 | 1.6507 | 0.0503 | 0.0000 | 0.0503 | 1.7037 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.