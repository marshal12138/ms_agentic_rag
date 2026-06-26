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
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260625-211826-CAR_async_npu_smaller_bs_per_gpu/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `not used`
- LLM judge model: `not used`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-1407-async_label_dpskv4f_v0623_full_no_ranker`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-1407-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-1407-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-1407-async_label_dpskv4f_v0623_full_no_ranker/runtime_logs/async_label_dpskv4f_v0623_full_no_ranker.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260626-1407-async_label_dpskv4f_v0623_full_no_ranker/validation_data`
- Wall time: `253.9811s`
- Status counts: `{'answered': 350}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> recall top-5 tool response -> agent LLM`
- Dense ranker participation: `disabled`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.4200 | 0.5152 |
| macro-average | 7 | 0.4200 | 0.5152 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.5000 | 0.5262 |
| bamboogle | 50 | 0.4600 | 0.5713 |
| hotpotqa | 50 | 0.5000 | 0.5856 |
| musique | 50 | 0.1800 | 0.2920 |
| nq | 50 | 0.3400 | 0.4458 |
| popqa | 50 | 0.3800 | 0.4943 |
| triviaqa | 50 | 0.5800 | 0.6914 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 2.5029 | 21.4741 | 0.2694 | 0.0000 | 0.2694 | 21.7642 | 5.0000 |
| macro-average | 7 | 2.5029 | 21.4741 | 0.2694 | 0.0000 | 0.2694 | 21.7642 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 2.7800 | 24.4576 | 0.1872 | 0.0000 | 0.1873 | 24.6638 | 5.0000 |
| bamboogle | 50 | 2.5000 | 19.8384 | 0.1601 | 0.0000 | 0.1601 | 20.0158 | 5.0000 |
| hotpotqa | 50 | 2.5200 | 22.2178 | 0.1565 | 0.0000 | 0.1565 | 22.3920 | 5.0000 |
| musique | 50 | 2.8200 | 25.3872 | 0.1699 | 0.0000 | 0.1699 | 25.5771 | 5.0000 |
| nq | 50 | 2.3400 | 19.3304 | 0.1391 | 0.0000 | 0.1391 | 19.4853 | 5.0000 |
| popqa | 50 | 2.3000 | 19.0562 | 0.9333 | 0.0000 | 0.9333 | 20.0301 | 5.0000 |
| triviaqa | 50 | 2.2600 | 20.0313 | 0.1393 | 0.0000 | 0.1393 | 20.1857 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.