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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260622-220205-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
- Ranker encoder: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260622-220205-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79/ranker/rank_encoder`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0220-async_label_dpskv4f_v0616_full`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0220-async_label_dpskv4f_v0616_full/runtime_logs/async_label_dpskv4f_v0616_full.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0220-async_label_dpskv4f_v0616_full/runtime_logs/async_label_dpskv4f_v0616_full.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0220-async_label_dpskv4f_v0616_full/runtime_logs/async_label_dpskv4f_v0616_full.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0220-async_label_dpskv4f_v0616_full/validation_data`
- Wall time: `45.8531s`
- Status counts: `{'answered': 349, 'no_valid_answer': 1}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3600 | 0.4388 |
| macro-average | 7 | 0.3600 | 0.4388 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.3000 | 0.3313 |
| bamboogle | 50 | 0.2600 | 0.3493 |
| hotpotqa | 50 | 0.3200 | 0.3973 |
| musique | 50 | 0.1000 | 0.1949 |
| nq | 50 | 0.5000 | 0.6107 |
| popqa | 50 | 0.4400 | 0.5068 |
| triviaqa | 50 | 0.6000 | 0.6813 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9971 | 2.6752 | 0.9213 | 0.0985 | 1.0198 | 3.6990 | 4.9857 |
| macro-average | 7 | 0.9971 | 2.6752 | 0.9213 | 0.0985 | 1.0198 | 3.6990 | 4.9857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 2.7559 | 1.3106 | 0.1037 | 1.4143 | 4.1732 | 5.0000 |
| bamboogle | 50 | 1.0000 | 2.3338 | 0.7003 | 0.0901 | 0.7904 | 3.1273 | 5.0000 |
| hotpotqa | 50 | 0.9800 | 2.6833 | 0.6797 | 0.1013 | 0.7810 | 3.4675 | 4.9000 |
| musique | 50 | 1.0000 | 2.7160 | 0.7592 | 0.0981 | 0.8573 | 3.5765 | 5.0000 |
| nq | 50 | 1.0000 | 2.6876 | 0.7256 | 0.0889 | 0.8144 | 3.5050 | 5.0000 |
| popqa | 50 | 1.0000 | 2.9484 | 1.3208 | 0.1081 | 1.4289 | 4.3866 | 5.0000 |
| triviaqa | 50 | 1.0000 | 2.6013 | 0.9529 | 0.0995 | 1.0525 | 3.6570 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.