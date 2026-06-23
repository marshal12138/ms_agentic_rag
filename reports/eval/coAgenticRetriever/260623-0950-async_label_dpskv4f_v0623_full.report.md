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
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0950-async_label_dpskv4f_v0623_full`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0950-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0950-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0950-async_label_dpskv4f_v0623_full/runtime_logs/async_label_dpskv4f_v0623_full.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0950-async_label_dpskv4f_v0623_full/validation_data`
- Wall time: `44.0940s`
- Status counts: `{'answered': 348, 'no_valid_answer': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> dense_e5 reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `dense_e5`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3257 | 0.4298 |
| macro-average | 7 | 0.3257 | 0.4298 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2600 | 0.2994 |
| bamboogle | 50 | 0.2200 | 0.3600 |
| hotpotqa | 50 | 0.3600 | 0.4606 |
| musique | 50 | 0.1200 | 0.2285 |
| nq | 50 | 0.3800 | 0.4992 |
| popqa | 50 | 0.4000 | 0.4876 |
| triviaqa | 50 | 0.5400 | 0.6734 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9971 | 2.9057 | 0.6326 | 0.0956 | 0.7282 | 3.6379 | 4.9857 |
| macro-average | 7 | 0.9971 | 2.9057 | 0.6326 | 0.0956 | 0.7282 | 3.6379 | 4.9857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 3.0892 | 0.7610 | 0.1043 | 0.8653 | 3.9576 | 5.0000 |
| bamboogle | 50 | 1.0000 | 2.5394 | 0.4729 | 0.0894 | 0.5623 | 3.1049 | 5.0000 |
| hotpotqa | 50 | 1.0000 | 2.9653 | 0.5369 | 0.0985 | 0.6354 | 3.6040 | 5.0000 |
| musique | 50 | 1.0000 | 2.9882 | 0.4700 | 0.0943 | 0.5643 | 3.5557 | 5.0000 |
| nq | 50 | 1.0000 | 2.6232 | 0.5623 | 0.0885 | 0.6508 | 3.2772 | 5.0000 |
| popqa | 50 | 1.0000 | 3.3412 | 1.1526 | 0.0990 | 1.2516 | 4.6020 | 5.0000 |
| triviaqa | 50 | 0.9800 | 2.7934 | 0.4722 | 0.0954 | 0.5675 | 3.3641 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.