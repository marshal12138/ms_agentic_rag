# CoSearch vLLM Evaluation Report

- Strategy: `original_agent_llm_no_ranker`
- Run mode: `no-ranker`
- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/local_flashrag/co_search_ablation.eval.parquet`
- Examples: `350`
- Success count: `350`
- Failure count: `0`
- Agent model: `/data01/ms_wksp/agent_up_to_date/models/llm/Qwen3-4B`
- Reranker model: `disabled`
- Trace dir: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/log/eval_res/cosearch/260622-1657-original_agent_llm_no_ranker`
- Wall time: `65.5505s`
- Status counts: `{'no_valid_answer': 13, 'answered': 335, 'max_turns': 2}`

## Effect Metrics

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| micro-average | 350 | 0.3229 | 0.4214 |
| macro-average | 7 | 0.3229 | 0.4214 |

## Effect Metrics By Dataset

| Scope | N | EM | F1 |
|---|---:|---:|---:|
| 2wikimultihopqa | 50 | 0.2600 | 0.2992 |
| bamboogle | 50 | 0.3000 | 0.4182 |
| hotpotqa | 50 | 0.3600 | 0.4706 |
| musique | 50 | 0.1200 | 0.2158 |
| nq | 50 | 0.3600 | 0.4657 |
| popqa | 50 | 0.3200 | 0.4095 |
| triviaqa | 50 | 0.5400 | 0.6706 |

## Performance Metrics

Agent Avg s is the per-query average agent generation time per assistant turn. Recall Avg s is the per-query average retrieve+rerank time per tool call. Total Avg s is the end-to-end per-query latency average.

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| micro-average | 350 | 1.1257 | 2.2842 | 4.8546 | 0.0457 | 0.0000 | 0.0412 | 0.0457 | 4.9034 |
| macro-average | 7 | 1.1257 | 2.2842 | 4.8546 | 0.0457 | 0.0000 | 0.0412 | 0.0457 | 4.9034 |

## Performance Metrics By Dataset

| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2wikimultihopqa | 50 | 1.1800 | 2.4745 | 5.3266 | 0.0433 | 0.0000 | 0.0364 | 0.0433 | 5.3730 |
| bamboogle | 50 | 1.1400 | 2.0664 | 4.3187 | 0.0389 | 0.0000 | 0.0336 | 0.0389 | 4.3609 |
| hotpotqa | 50 | 1.1000 | 2.6653 | 5.7128 | 0.0423 | 0.0000 | 0.0386 | 0.0423 | 5.7583 |
| musique | 50 | 1.3000 | 3.3046 | 7.3729 | 0.0453 | 0.0000 | 0.0349 | 0.0453 | 7.4220 |
| nq | 50 | 1.1000 | 1.9414 | 4.0826 | 0.0390 | 0.0000 | 0.0358 | 0.0390 | 4.1249 |
| popqa | 50 | 1.0200 | 1.7137 | 3.4656 | 0.0712 | 0.0000 | 0.0706 | 0.0712 | 3.5394 |
| triviaqa | 50 | 1.0400 | 1.8233 | 3.7031 | 0.0400 | 0.0000 | 0.0384 | 0.0400 | 3.7457 |
