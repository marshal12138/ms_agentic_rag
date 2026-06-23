# CoAgenticRetriever vLLM Evaluation Report

- Strategy: `llm_judge_as_rank_wt_ori_agt_llm`
- Run mode: `full`
- Reranker: `llm_as_judge`
- Enable thinking: `false`
- Ranker enabled: `true`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/coAgenticRetriever/albation_1/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- Ranker tokenizer/base model: `not used`
- Ranker encoder: `not used`
- LLM judge endpoint: `http://127.0.0.1:8067/v1/chat/completions`
- LLM judge model: `DeepSeek-V4-Flash`
- Recall service: `http://127.0.0.1:8030/retrieve`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0227-llm_judge_as_rank_wt_ori_agt_llm`
- Runtime metrics JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0227-llm_judge_as_rank_wt_ori_agt_llm/runtime_logs/llm_judge_as_rank_wt_ori_agt_llm.metrics.jsonl`
- Search timing JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0227-llm_judge_as_rank_wt_ori_agt_llm/runtime_logs/llm_judge_as_rank_wt_ori_agt_llm.search_timing.jsonl`
- LLM IO JSONL: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0227-llm_judge_as_rank_wt_ori_agt_llm/runtime_logs/llm_judge_as_rank_wt_ori_agt_llm.llm_io.jsonl`
- Validation data dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/coAgenticRetriever/260623-0227-llm_judge_as_rank_wt_ori_agt_llm/validation_data`
- Wall time: `875.8026s`
- Status counts: `{'answered': 341, 'no_valid_answer': 7, 'max_turns': 2}`

## Eval Path

- Search path: `agent LLM -> recall retriever top-50 -> llm_as_judge reorder -> top-5 tool response -> agent LLM`
- Ranker participation: `llm_as_judge`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3543 | 0.4475 |
| macro-average | 7 | 0.3543 | 0.4475 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2400 | 0.2787 |
| bamboogle | 50 | 0.3400 | 0.4460 |
| hotpotqa | 50 | 0.4000 | 0.5215 |
| musique | 50 | 0.1400 | 0.2146 |
| nq | 50 | 0.4000 | 0.5200 |
| popqa | 50 | 0.3800 | 0.4676 |
| triviaqa | 50 | 0.5800 | 0.6841 |

## Performance Metrics

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.1171 | 1.9968 | 0.0398 | 37.4231 | 37.4629 | 39.4673 | 5.0000 |
| macro-average | 7 | 1.1171 | 1.9968 | 0.0398 | 37.4231 | 37.4629 | 39.4673 | 5.0000 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1600 | 2.4228 | 0.0346 | 35.4906 | 35.5251 | 37.9516 | 5.0000 |
| bamboogle | 50 | 1.1200 | 1.8292 | 0.0361 | 31.6869 | 31.7230 | 33.5557 | 5.0000 |
| hotpotqa | 50 | 1.1000 | 2.1351 | 0.0319 | 37.9330 | 37.9649 | 40.1037 | 5.0000 |
| musique | 50 | 1.4000 | 2.7681 | 0.0415 | 43.3512 | 43.3927 | 46.1654 | 5.0000 |
| nq | 50 | 1.0400 | 1.6471 | 0.0368 | 35.7375 | 35.7744 | 37.4248 | 5.0000 |
| popqa | 50 | 1.0000 | 1.5991 | 0.0674 | 38.0694 | 38.1368 | 39.7679 | 5.0000 |
| triviaqa | 50 | 1.0000 | 1.5759 | 0.0300 | 39.6931 | 39.7231 | 41.3023 | 5.0000 |

## Artifacts

- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.
- `traces.jsonl`: per-example conversation/search traces.
- `summary.json`: aggregate metrics.
- `run_config.json`: resolved runtime configuration.
- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.