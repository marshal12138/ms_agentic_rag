# CoAgenticRetriever vLLM Evaluation Report

- Eval task: `async_label_dpskv4f_v0702_no_ranker`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260702-010936-CAR_async_ranker_training_ds_flash_mix_signal_b3_v1_select_all/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1509-async_label_dpskv4f_v0702_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1509-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1509-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1509-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1509-async_label_dpskv4f_v0702_no_ranker/validation_data`
- Wall time: `193.1768s`
- Status counts: `{'answered': 348, 'no_valid_answer': 1, 'max_turns': 1}`

## Retrieval Cutoffs

- RECALL_FINAL_TOP_N: `50`
- SEARCH_TOOL_FINAL_TOP_M: `5`
- RANKER_FINAL_TOP_K: `50`

## Eval Path

- Search path: `agent LLM -> recall retriever recall_final_top_n=50 -> searchTool_final_top_m=5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.4086 | 0.4871 |
| macro-average | 7 | 0.4086 | 0.4871 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.3400 | 0.3592 |
| bamboogle | 50 | 0.4200 | 0.5473 |
| hotpotqa | 50 | 0.5000 | 0.6016 |
| musique | 50 | 0.1800 | 0.2500 |
| nq | 50 | 0.4000 | 0.5048 |
| popqa | 50 | 0.3800 | 0.4562 |
| triviaqa | 50 | 0.6400 | 0.6905 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.8400 | 15.8611 | 0.2127 | 0.0000 | 0.2127 | 16.0890 | 5.0000 |
| macro-average | 7 | 1.8400 | 15.8611 | 0.2127 | 0.0000 | 0.2127 | 16.0890 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.0600 | 17.9894 | 0.1255 | 0.0000 | 0.1255 | 18.1270 | 5.0000 |
| bamboogle | 50 | 1.8600 | 14.4452 | 0.1123 | 0.0000 | 0.1123 | 14.5695 | 5.0000 |
| hotpotqa | 50 | 1.7800 | 16.1747 | 0.1037 | 0.0000 | 0.1037 | 16.2900 | 5.0000 |
| musique | 50 | 2.4600 | 20.0228 | 0.1431 | 0.0000 | 0.1431 | 20.1821 | 5.0000 |
| nq | 50 | 1.8400 | 15.9603 | 0.1135 | 0.0000 | 0.1135 | 16.0854 | 5.0000 |
| popqa | 50 | 1.5200 | 13.1504 | 0.8051 | 0.0000 | 0.8051 | 13.9900 | 5.0000 |
| triviaqa | 50 | 1.3600 | 13.2850 | 0.0859 | 0.0000 | 0.0859 | 13.3790 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.