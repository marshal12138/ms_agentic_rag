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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260619-153011-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2129-async_label_dpskv4f_v0616_full_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2129-async_label_dpskv4f_v0616_full_no_ranker/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2129-async_label_dpskv4f_v0616_full_no_ranker/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2129-async_label_dpskv4f_v0616_full_no_ranker/runtime_logs/async_label_dpskv4f_v0616_full_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260622-2129-async_label_dpskv4f_v0616_full_no_ranker/validation_data`
- Wall time: `23.5767s`
- Status counts: `{'answered': 348, 'no_valid_answer': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3371 | 0.4196 |
| macro-average | 7 | 0.3371 | 0.4196 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.3000 | 0.3607 |
| bamboogle | 50 | 0.2000 | 0.2667 |
| hotpotqa | 50 | 0.3000 | 0.3971 |
| musique | 50 | 0.1400 | 0.2094 |
| nq | 50 | 0.4400 | 0.5290 |
| popqa | 50 | 0.3800 | 0.4775 |
| triviaqa | 50 | 0.6000 | 0.6965 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9943 | 1.9487 | 0.0951 | 0.0000 | 0.0951 | 2.0506 | 4.9714 |
| macro-average | 7 | 0.9943 | 1.9487 | 0.0951 | 0.0000 | 0.0951 | 2.0506 | 4.9714 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 2.0039 | 0.0412 | 0.0000 | 0.0412 | 2.0479 | 5.0000 |
| bamboogle | 50 | 1.0000 | 1.7565 | 0.0377 | 0.0000 | 0.0377 | 1.7970 | 5.0000 |
| hotpotqa | 50 | 0.9800 | 1.9521 | 0.0386 | 0.0000 | 0.0386 | 1.9935 | 4.9000 |
| musique | 50 | 1.0000 | 1.9725 | 0.0457 | 0.0000 | 0.0457 | 2.0210 | 5.0000 |
| nq | 50 | 1.0000 | 1.7396 | 0.0368 | 0.0000 | 0.0368 | 1.7792 | 5.0000 |
| popqa | 50 | 1.0000 | 2.3613 | 0.4171 | 0.0000 | 0.4171 | 2.8096 | 5.0000 |
| triviaqa | 50 | 0.9800 | 1.8549 | 0.0486 | 0.0000 | 0.0486 | 1.9063 | 4.9000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.