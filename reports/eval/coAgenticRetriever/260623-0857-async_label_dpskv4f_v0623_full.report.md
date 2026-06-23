# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `async_label_dpskv4f_v0623_full`
- Run mode: `full`
- Reranker: `dense_e5`
- Enable thinking: `false`
- Ranker enabled: `true`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260623-025839-CAR_async_labeling_ds_flash_larger_ranker_tdata/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
- Ranker encoder: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260623-025839-CAR_async_labeling_ds_flash_larger_ranker_tdata/global_step_79/ranker/rank_encoder`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0857-async_label_dpskv4f_v0623_full`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0857-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0857-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0857-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0857-async_label_dpskv4f_v0623_full/validation_data`
- Wall time: `43.2163s`
- Status counts: `{'answered': 348, 'no_valid_answer': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3257 | 0.4327 |
| macro-average | 7 | 0.3257 | 0.4327 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2800 | 0.3167 |
| bamboogle | 50 | 0.2400 | 0.3800 |
| hotpotqa | 50 | 0.3400 | 0.4362 |
| musique | 50 | 0.1200 | 0.2399 |
| nq | 50 | 0.3600 | 0.4952 |
| popqa | 50 | 0.4000 | 0.4876 |
| triviaqa | 50 | 0.5400 | 0.6734 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9971 | 2.8713 | 0.6002 | 0.0954 | 0.6956 | 3.5710 | 4.9857 |
| macro-average | 7 | 0.9971 | 2.8713 | 0.6002 | 0.0954 | 0.6956 | 3.5710 | 4.9857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 3.0712 | 0.8352 | 0.1030 | 0.9383 | 4.0125 | 5.0000 |
| bamboogle | 50 | 1.0000 | 2.5160 | 0.4039 | 0.0893 | 0.4932 | 3.0124 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 2.9802 | 0.4728 | 0.0987 | 0.5715 | 3.5554 | 5.0000 |
| musique | 50 | 1.0000 | 2.9087 | 0.4162 | 0.0943 | 0.5105 | 3.4225 | 5.0000 |
| nq | 50 | 1.0000 | 2.5842 | 0.4828 | 0.0883 | 0.5711 | 3.1584 | 5.0000 |
| popqa | 50 | 1.0000 | 3.2870 | 1.1788 | 0.0995 | 1.2783 | 4.5743 | 5.0000 |
| triviaqa | 50 | 0.9800 | 2.7521 | 0.4114 | 0.0948 | 0.5062 | 3.2612 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.