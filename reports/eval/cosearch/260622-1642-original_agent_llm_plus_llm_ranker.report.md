# CoSearch vLLM Evaluation Report

- Strategy: `original_agent_llm_plus_llm_ranker`
- Run mode: `full`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/local_flashrag/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `348`
- Failure count: `2`
- Agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- Reranker model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/cosearch/260622-1642-original_agent_llm_plus_llm_ranker`
- Wall time: `238.1640s`
- Status counts: `{'answered': 328, 'no_valid_answer': 19, 'max_turns': 1, 'failed': 2}`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3286 | 0.4131 |
| macro-average | 7 | 0.3286 | 0.4131 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2200 | 0.2593 |
| bamboogle | 50 | 0.3600 | 0.4544 |
| hotpotqa | 50 | 0.3600 | 0.4404 |
| musique | 50 | 0.1000 | 0.1529 |
| nq | 50 | 0.3600 | 0.4642 |
| popqa | 50 | 0.3400 | 0.4410 |
| triviaqa | 50 | 0.5600 | 0.6794 |

## Performance Metrics

Agent Avg s is the per-query average agent generation time per assistant turn. Recall Avg s is the per-query average retrieve+rerank time per tool call. Total Avg s is the end-to-end per-query latency average.

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.0886 | 1.5732 | 3.2912 | 0.0656 | 16.9811 | 15.3822 | 17.0467 | 20.4499 |
| macro-average | 7 | 1.0886 | 1.5732 | 3.2912 | 0.0656 | 16.9811 | 15.3822 | 17.0467 | 20.4499 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.2600 | 1.5137 | 3.3676 | 0.0422 | 20.9040 | 16.2147 | 20.9462 | 24.3176 |
| bamboogle | 50 | 1.0400 | 1.5741 | 3.2019 | 0.0309 | 11.3952 | 10.8274 | 11.4261 | 15.3715 |
| hotpotqa | 50 | 1.0200 | 1.5534 | 3.1499 | 0.0332 | 14.7786 | 14.2874 | 14.8118 | 17.9852 |
| musique | 50 | 1.2200 | 2.7232 | 5.9065 | 0.0378 | 16.2990 | 14.0235 | 16.3368 | 22.2474 |
| nq | 50 | 1.0000 | 1.1884 | 2.3768 | 0.0323 | 14.5499 | 14.5822 | 14.5822 | 16.9619 |
| popqa | 50 | 1.0600 | 1.2919 | 2.6711 | 0.2479 | 25.9466 | 23.4484 | 26.1945 | 28.8690 |
| triviaqa | 50 | 1.0200 | 1.1675 | 2.3648 | 0.0350 | 14.9942 | 14.2920 | 15.0291 | 17.3971 |
