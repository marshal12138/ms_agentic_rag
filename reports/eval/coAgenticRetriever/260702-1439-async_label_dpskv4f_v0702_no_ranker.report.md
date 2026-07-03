# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `async_label_dpskv4f_v0702_no_ranker`
- Run mode: `no-ranker`
- Reranker: `none`
- Enable thinking: `false`
- Ranker enabled: `false`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `1`
- Success count: `1`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260702-010936-CAR_async_ranker_training_ds_flash_mix_signal_b3_v1_select_all/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1439-async_label_dpskv4f_v0702_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1439-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1439-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1439-async_label_dpskv4f_v0702_no_ranker/runtime_logs/async_label_dpskv4f_v0702_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260702-1439-async_label_dpskv4f_v0702_no_ranker/validation_data`
- Wall time: `9.7356s`
- Status counts: `{'answered': 1}`

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
| micro-average | 1 | 0.0000 | 0.5714 |
| macro-average | 1 | 0.0000 | 0.5714 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| popqa | 1 | 0.0000 | 0.5714 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 1 | 1.0000 | 7.5432 | 0.0563 | 0.0000 | 0.0563 | 8.9526 | 5.0000 |
| macro-average | 1 | 1.0000 | 7.5432 | 0.0563 | 0.0000 | 0.0563 | 8.9526 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| popqa | 1 | 1.0000 | 7.5432 | 0.0563 | 0.0000 | 0.0563 | 8.9526 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.