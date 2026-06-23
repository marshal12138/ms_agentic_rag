# CoSearch vLLM Evaluation Report

- Strategy: `cosearch_agent_llm_no_ranker`
- Run mode: `no-ranker`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/local_flashrag/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors/actor`
- Reranker model: `disabled`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/cosearch/260622-1706-cosearch_agent_llm_no_ranker`
- Wall time: `34.5415s`
- Status counts: `{'answered': 349, 'no_valid_answer': 1}`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3514 | 0.4358 |
| macro-average | 7 | 0.3514 | 0.4358 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2600 | 0.2952 |
| bamboogle | 50 | 0.3400 | 0.4108 |
| hotpotqa | 50 | 0.3800 | 0.4971 |
| musique | 50 | 0.1400 | 0.1985 |
| nq | 50 | 0.3800 | 0.4882 |
| popqa | 50 | 0.3600 | 0.4476 |
| triviaqa | 50 | 0.6000 | 0.7134 |

## Performance Metrics

Agent Avg s is the per-query average agent generation time per assistant turn. Recall Avg s is the per-query average retrieve+rerank time per tool call. Total Avg s is the end-to-end per-query latency average.

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0000 | 1.4918 | 2.9837 | 0.0504 | 0.0000 | 0.0504 | 0.0504 | 3.0366 |
| macro-average | 7 | 1.0000 | 1.4918 | 2.9837 | 0.0504 | 0.0000 | 0.0504 | 0.0504 | 3.0366 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 1.7102 | 3.4203 | 0.0387 | 0.0000 | 0.0387 | 0.0387 | 3.4618 |
| bamboogle | 50 | 1.0000 | 1.3895 | 2.7790 | 0.0409 | 0.0000 | 0.0409 | 0.0409 | 2.8225 |
| hotpotqa | 50 | 1.0000 | 1.6027 | 3.2054 | 0.0455 | 0.0000 | 0.0455 | 0.0455 | 3.2537 |
| musique | 50 | 1.0000 | 1.6011 | 3.2021 | 0.0451 | 0.0000 | 0.0451 | 0.0451 | 3.2498 |
| nq | 50 | 1.0000 | 1.3667 | 2.7334 | 0.0387 | 0.0000 | 0.0387 | 0.0387 | 2.7747 |
| popqa | 50 | 1.0000 | 1.3311 | 2.6622 | 0.0989 | 0.0000 | 0.0989 | 0.0989 | 2.7636 |
| triviaqa | 50 | 1.0000 | 1.4416 | 2.8832 | 0.0446 | 0.0000 | 0.0446 | 0.0446 | 2.9303 |
