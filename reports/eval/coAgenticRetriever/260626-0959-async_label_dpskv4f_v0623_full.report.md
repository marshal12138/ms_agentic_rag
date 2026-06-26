# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `async_label_dpskv4f_v0623_full`
- Run mode: `full`
- Reranker: `dense_e5`
- Enable thinking: `false`
- Ranker enabled: `true`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `333`
- Failure count: `17`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260625-211826-CAR_async_npu_smaller_bs_per_gpu/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2`
- Ranker encoder: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260623-025839-CAR_async_labeling_ds_flash_larger_ranker_tdata/global_step_79/ranker/rank_encoder`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-0959-async_label_dpskv4f_v0623_full`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-0959-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-0959-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-0959-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-0959-async_label_dpskv4f_v0623_full/validation_data`
- Wall time: `274.6387s`
- Status counts: `{'answered': 333, 'failed': 17}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3171 | 0.3889 |
| macro-average | 7 | 0.3171 | 0.3889 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.3400 | 0.3747 |
| bamboogle | 50 | 0.3200 | 0.4035 |
| hotpotqa | 50 | 0.2600 | 0.3432 |
| musique | 50 | 0.1600 | 0.2071 |
| nq | 50 | 0.3400 | 0.4326 |
| popqa | 50 | 0.2200 | 0.2823 |
| triviaqa | 50 | 0.5800 | 0.6789 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.5886 | 21.9626 | 0.4274 | 0.2642 | 0.6916 | 23.2973 | 4.7571 |
| macro-average | 7 | 2.5886 | 21.9626 | 0.4274 | 0.2642 | 0.6916 | 23.2973 | 4.7571 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 3.0200 | 26.7870 | 0.2726 | 0.3130 | 0.5855 | 27.3944 | 5.0000 |
| bamboogle | 50 | 2.4800 | 19.6992 | 0.2379 | 0.2386 | 0.4765 | 20.7621 | 4.9000 |
| hotpotqa | 50 | 2.9000 | 24.6075 | 0.3025 | 0.2958 | 0.5983 | 25.2274 | 5.0000 |
| musique | 50 | 3.1400 | 27.1738 | 0.3656 | 0.3115 | 0.6770 | 27.8761 | 5.0000 |
| nq | 50 | 2.4800 | 21.3257 | 0.2347 | 0.2364 | 0.4710 | 21.8151 | 5.0000 |
| popqa | 50 | 1.7000 | 13.5329 | 1.3256 | 0.2143 | 1.5399 | 18.8841 | 3.4000 |
| triviaqa | 50 | 2.4000 | 20.6120 | 0.2527 | 0.2401 | 0.4929 | 21.1217 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.