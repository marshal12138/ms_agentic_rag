# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `async_label_dpskv4f_v0616_full`
- Run mode: `full`
- Reranker: `dense_e5`
- Enable thinking: `false`
- Ranker enabled: `true`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260619-153011-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
- Ranker encoder: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260619-153011-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79/ranker/rank_encoder`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2132-async_label_dpskv4f_v0616_full`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2132-async_label_dpskv4f_v0616_full/runtime_logs/async_label_dpskv4f_v0616_full.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2132-async_label_dpskv4f_v0616_full/runtime_logs/async_label_dpskv4f_v0616_full.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2132-async_label_dpskv4f_v0616_full/runtime_logs/async_label_dpskv4f_v0616_full.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2132-async_label_dpskv4f_v0616_full/validation_data`
- Wall time: `43.2848s`
- Status counts: `{'answered': 348, 'no_valid_answer': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3143 | 0.4052 |
| macro-average | 7 | 0.3143 | 0.4052 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2600 | 0.3160 |
| bamboogle | 50 | 0.1400 | 0.2507 |
| hotpotqa | 50 | 0.2800 | 0.3771 |
| musique | 50 | 0.1000 | 0.1974 |
| nq | 50 | 0.4400 | 0.5450 |
| popqa | 50 | 0.4000 | 0.5010 |
| triviaqa | 50 | 0.5800 | 0.6495 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9943 | 2.7090 | 0.7575 | 0.0954 | 0.8529 | 3.5657 | 4.9714 |
| macro-average | 7 | 0.9943 | 2.7090 | 0.7575 | 0.0954 | 0.8529 | 3.5657 | 4.9714 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 2.8929 | 0.7596 | 0.1037 | 0.8633 | 3.7592 | 5.0000 |
| bamboogle | 50 | 1.0000 | 2.4771 | 0.8054 | 0.0891 | 0.8946 | 3.3746 | 5.0000 |
| hotpotqa | 50 | 0.9800 | 2.8205 | 0.6338 | 0.0956 | 0.7294 | 3.5530 | 4.9000 |
| musique | 50 | 1.0000 | 2.8954 | 0.5815 | 0.0950 | 0.6765 | 3.5748 | 5.0000 |
| nq | 50 | 1.0000 | 2.4566 | 0.6959 | 0.0886 | 0.7845 | 3.2441 | 5.0000 |
| popqa | 50 | 1.0000 | 2.7841 | 1.0666 | 0.1026 | 1.1692 | 3.9620 | 5.0000 |
| triviaqa | 50 | 0.9800 | 2.6365 | 0.7596 | 0.0932 | 0.8528 | 3.4921 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.