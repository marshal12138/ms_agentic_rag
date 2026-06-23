# CoSearch vLLM Evaluation Report

- Strategy: `cosearch_agent_llm_plus_llm_ranker`
- Run mode: `full`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/local_flashrag/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `348`
- Failure count: `2`
- Agent model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors/actor`
- Reranker model: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/checkpoints/qwen3_4b_probe/qwen3_4b_ablation_4retrievers_timing/global_step_79/hf_safetensors/reranker_actor_rollout`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/cosearch/260622-1659-cosearch_agent_llm_plus_llm_ranker`
- Wall time: `229.0939s`
- Status counts: `{'answered': 347, 'no_valid_answer': 1, 'failed': 2}`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3600 | 0.4547 |
| macro-average | 7 | 0.3600 | 0.4547 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2600 | 0.3004 |
| bamboogle | 50 | 0.3200 | 0.4663 |
| hotpotqa | 50 | 0.4200 | 0.5062 |
| musique | 50 | 0.1800 | 0.2694 |
| nq | 50 | 0.4200 | 0.5348 |
| popqa | 50 | 0.3400 | 0.4316 |
| triviaqa | 50 | 0.5800 | 0.6742 |

## Performance Metrics

Agent Avg s is the per-query average agent generation time per assistant turn. Recall Avg s is the per-query average retrieve+rerank time per tool call. Total Avg s is the end-to-end per-query latency average.

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 0.9943 | 0.8954 | 1.7909 | 0.0674 | 18.1970 | 18.2644 | 18.2644 | 20.1153 |
| macro-average | 7 | 0.9943 | 0.8954 | 1.7909 | 0.0674 | 18.1970 | 18.2644 | 18.2644 | 20.1153 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.0000 | 0.9902 | 1.9804 | 0.0322 | 23.0107 | 23.0429 | 23.0429 | 25.0266 |
| bamboogle | 50 | 1.0000 | 0.8486 | 1.6972 | 0.0309 | 14.3365 | 14.3674 | 14.3674 | 16.0679 |
| hotpotqa | 50 | 0.9800 | 0.9561 | 1.9122 | 0.0332 | 17.8584 | 17.8916 | 17.8916 | 19.8255 |
| musique | 50 | 1.0000 | 0.9941 | 1.9881 | 0.0373 | 19.9377 | 19.9751 | 19.9751 | 21.9667 |
| nq | 50 | 1.0000 | 0.8337 | 1.6675 | 0.0406 | 15.8116 | 15.8522 | 15.8522 | 17.5229 |
| popqa | 50 | 1.0000 | 0.7685 | 1.5370 | 0.2653 | 21.1766 | 21.4420 | 21.4420 | 22.9822 |
| triviaqa | 50 | 0.9800 | 0.8769 | 1.7538 | 0.0323 | 15.2471 | 15.2795 | 15.2795 | 17.4153 |
