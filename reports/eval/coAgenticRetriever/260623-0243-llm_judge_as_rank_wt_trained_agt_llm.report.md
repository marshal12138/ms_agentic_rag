# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `llm_judge_as_rank_wt_trained_agt_llm`
- Run mode: `full`
- Reranker: `llm_as_judge`
- Enable thinking: `false`
- Ranker enabled: `true`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/coAgenticRetriever/260622-220205-CAR_async_labeling_ds_flash_mix_signal_b3_v1_select_all/global_step_79/hf_safetensors/actor`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `http://127.0.0.1:8067/v1/chat/completions`
- LLM judge model: `DeepSeek-V4-Flash`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0243-llm_judge_as_rank_wt_trained_agt_llm`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0243-llm_judge_as_rank_wt_trained_agt_llm/runtime_logs/llm_judge_as_rank_wt_trained_agt_llm.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0243-llm_judge_as_rank_wt_trained_agt_llm/runtime_logs/llm_judge_as_rank_wt_trained_agt_llm.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0243-llm_judge_as_rank_wt_trained_agt_llm/runtime_logs/llm_judge_as_rank_wt_trained_agt_llm.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0243-llm_judge_as_rank_wt_trained_agt_llm/validation_data`
- Wall time: `799.0046s`
- Status counts: `{'answered': 349, 'no_valid_answer': 1}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> llm_as_judge reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `llm_as_judge`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3629 | 0.4525 |
| macro-average | 7 | 0.3629 | 0.4525 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.3400 | 0.3673 |
| bamboogle | 50 | 0.2400 | 0.3760 |
| hotpotqa | 50 | 0.3200 | 0.4231 |
| musique | 50 | 0.1600 | 0.2442 |
| nq | 50 | 0.4600 | 0.5660 |
| popqa | 50 | 0.3800 | 0.4602 |
| triviaqa | 50 | 0.6400 | 0.7307 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9971 | 1.1626 | 0.0543 | 34.3233 | 34.3776 | 35.5477 | 4.9857 |
| macro-average | 7 | 0.9971 | 1.1626 | 0.0543 | 34.3233 | 34.3776 | 35.5477 | 4.9857 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 1.2512 | 0.0562 | 32.9419 | 32.9981 | 34.2526 | 5.0000 |
| bamboogle | 50 | 1.0000 | 1.1033 | 0.0553 | 29.1044 | 29.1597 | 30.2664 | 5.0000 |
| hotpotqa | 50 | 0.9800 | 1.1946 | 0.0405 | 32.7211 | 32.7616 | 33.9595 | 4.9000 |
| musique | 50 | 1.0000 | 1.1769 | 0.0345 | 33.6354 | 33.6698 | 34.8500 | 5.0000 |
| nq | 50 | 1.0000 | 1.1425 | 0.0643 | 35.4855 | 35.5498 | 36.6958 | 5.0000 |
| popqa | 50 | 1.0000 | 1.1574 | 0.0848 | 35.7176 | 35.8024 | 36.9914 | 5.0000 |
| triviaqa | 50 | 1.0000 | 1.1124 | 0.0444 | 40.6576 | 40.7020 | 41.8178 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.