# New-Data Model Evaluation

- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Dataset SHA256: `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283`
- Each model has 1 isolated inference run(s). Repeats are not pooled as independent examples.
- Paired bootstrap averages repeats per question before resampling questions.

## Overall

| Model | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spad_512_stable_original_inflight1 | 0.1360 +/- 0.0000 | 0.2265 +/- 0.0000 | 0.1360 +/- 0.0000 | 0.2265 +/- 0.0000 | 0.1360 +/- 0.0000 | 0.6989 +/- 0.0000 | 0.9689 +/- 0.0000 | 2.3391 +/- 0.0000 | 0.3340 +/- 0.0000 | 0.2426 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | 0.1051 +/- 0.0000 | 0.1737 +/- 0.0000 | 0.1051 +/- 0.0000 | 0.1737 +/- 0.0000 | 0.1051 +/- 0.0000 | 0.5431 +/- 0.0000 | 0.9726 +/- 0.0000 | 2.5257 +/- 0.0000 | 0.4326 +/- 0.0000 | 0.2417 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | 0.1054 +/- 0.0000 | 0.1798 +/- 0.0000 | 0.1054 +/- 0.0000 | 0.1798 +/- 0.0000 | 0.1054 +/- 0.0000 | 0.5900 +/- 0.0000 | 0.9711 +/- 0.0000 | 2.3566 +/- 0.0000 | 0.3466 +/- 0.0000 | 0.2363 +/- 0.0000 |

## Per Run

| Model | Repeat | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spad_512_stable_original_inflight1 | 1 | 0.1360 | 0.2265 | 0.1360 | 0.2265 | 0.1360 | 0.6989 | 0.9689 | 2.3391 | 0.3340 | 0.2426 |
| spad_512_stable_repeat_inflight1 | 1 | 0.1051 | 0.1737 | 0.1051 | 0.1737 | 0.1051 | 0.5431 | 0.9726 | 2.5257 | 0.4326 | 0.2417 |
| spad_512_stable_ablation_inflight2 | 1 | 0.1054 | 0.1798 | 0.1054 | 0.1798 | 0.1054 | 0.5900 | 0.9711 | 2.3566 | 0.3466 | 0.2363 |

## Search Count Buckets

| Model | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| spad_512_stable_original_inflight1 | 0.0311 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.2703 +/- 0.0000 | 0.0397 +/- 0.0000 | 0.0149 +/- 0.0000 | 0.2440 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | 0.0274 +/- 0.0000 | 0.2223 +/- 0.0000 | 0.4491 +/- 0.0000 | 0.0426 +/- 0.0000 | 0.0154 +/- 0.0000 | 0.2431 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | 0.0289 +/- 0.0000 | 0.3743 +/- 0.0000 | 0.2974 +/- 0.0000 | 0.0489 +/- 0.0000 | 0.0120 +/- 0.0000 | 0.2386 +/- 0.0000 |

## Per Data Source

| Model | Data source | EM | F1 | Structured EM | Group F1 | Group recall |
|---|---|---:|---:|---:|---:|---:|
| spad_512_stable_original_inflight1 | 2wikimultihopqa | 0.0480 +/- 0.0000 | 0.1107 +/- 0.0000 | 0.0480 +/- 0.0000 | 0.1107 +/- 0.0000 | 0.0480 +/- 0.0000 |
| spad_512_stable_original_inflight1 | bamboogle | 0.1520 +/- 0.0000 | 0.2485 +/- 0.0000 | 0.1520 +/- 0.0000 | 0.2485 +/- 0.0000 | 0.1520 +/- 0.0000 |
| spad_512_stable_original_inflight1 | hotpotqa | 0.1441 +/- 0.0000 | 0.2443 +/- 0.0000 | 0.1441 +/- 0.0000 | 0.2443 +/- 0.0000 | 0.1441 +/- 0.0000 |
| spad_512_stable_original_inflight1 | musique | 0.0320 +/- 0.0000 | 0.0967 +/- 0.0000 | 0.0320 +/- 0.0000 | 0.0967 +/- 0.0000 | 0.0320 +/- 0.0000 |
| spad_512_stable_original_inflight1 | nq | 0.2011 +/- 0.0000 | 0.2966 +/- 0.0000 | 0.2011 +/- 0.0000 | 0.2966 +/- 0.0000 | 0.2011 +/- 0.0000 |
| spad_512_stable_original_inflight1 | popqa | 0.2327 +/- 0.0000 | 0.2831 +/- 0.0000 | 0.2327 +/- 0.0000 | 0.2831 +/- 0.0000 | 0.2327 +/- 0.0000 |
| spad_512_stable_original_inflight1 | triviaqa | 0.1545 +/- 0.0000 | 0.3227 +/- 0.0000 | 0.1545 +/- 0.0000 | 0.3227 +/- 0.0000 | 0.1545 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | 2wikimultihopqa | 0.0409 +/- 0.0000 | 0.0997 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.0997 +/- 0.0000 | 0.0409 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | bamboogle | 0.1200 +/- 0.0000 | 0.1967 +/- 0.0000 | 0.1200 +/- 0.0000 | 0.1967 +/- 0.0000 | 0.1200 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | hotpotqa | 0.1050 +/- 0.0000 | 0.1793 +/- 0.0000 | 0.1050 +/- 0.0000 | 0.1793 +/- 0.0000 | 0.1050 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | musique | 0.0249 +/- 0.0000 | 0.0668 +/- 0.0000 | 0.0249 +/- 0.0000 | 0.0668 +/- 0.0000 | 0.0249 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | nq | 0.1477 +/- 0.0000 | 0.2137 +/- 0.0000 | 0.1477 +/- 0.0000 | 0.2137 +/- 0.0000 | 0.1477 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | popqa | 0.1954 +/- 0.0000 | 0.2407 +/- 0.0000 | 0.1954 +/- 0.0000 | 0.2407 +/- 0.0000 | 0.1954 +/- 0.0000 |
| spad_512_stable_repeat_inflight1 | triviaqa | 0.1137 +/- 0.0000 | 0.2370 +/- 0.0000 | 0.1137 +/- 0.0000 | 0.2370 +/- 0.0000 | 0.1137 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | 2wikimultihopqa | 0.0355 +/- 0.0000 | 0.0905 +/- 0.0000 | 0.0355 +/- 0.0000 | 0.0905 +/- 0.0000 | 0.0355 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | bamboogle | 0.0960 +/- 0.0000 | 0.1742 +/- 0.0000 | 0.0960 +/- 0.0000 | 0.1742 +/- 0.0000 | 0.0960 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | hotpotqa | 0.1032 +/- 0.0000 | 0.1879 +/- 0.0000 | 0.1032 +/- 0.0000 | 0.1879 +/- 0.0000 | 0.1032 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | musique | 0.0178 +/- 0.0000 | 0.0656 +/- 0.0000 | 0.0178 +/- 0.0000 | 0.0656 +/- 0.0000 | 0.0178 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | nq | 0.1459 +/- 0.0000 | 0.2179 +/- 0.0000 | 0.1459 +/- 0.0000 | 0.2179 +/- 0.0000 | 0.1459 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | popqa | 0.2114 +/- 0.0000 | 0.2587 +/- 0.0000 | 0.2114 +/- 0.0000 | 0.2587 +/- 0.0000 | 0.2114 +/- 0.0000 |
| spad_512_stable_ablation_inflight2 | triviaqa | 0.1208 +/- 0.0000 | 0.2593 +/- 0.0000 | 0.1208 +/- 0.0000 | 0.2593 +/- 0.0000 | 0.1208 +/- 0.0000 |

## Paired Comparisons

| Comparison | Metric | Delta (right-left) | 95% CI |
|---|---|---:|---:|
| spad_512_stable_original_inflight1 -> spad_512_stable_repeat_inflight1 | em | -0.0309 | [-0.0403, -0.0217] |
| spad_512_stable_original_inflight1 -> spad_512_stable_repeat_inflight1 | f1 | -0.0528 | [-0.0626, -0.0431] |
| spad_512_stable_original_inflight1 -> spad_512_stable_repeat_inflight1 | structured_em | -0.0309 | [-0.0403, -0.0217] |
| spad_512_stable_original_inflight1 -> spad_512_stable_repeat_inflight1 | answer_group_f1 | -0.0528 | [-0.0626, -0.0431] |
| spad_512_stable_original_inflight1 -> spad_512_stable_repeat_inflight1 | answer_group_recall | -0.0309 | [-0.0403, -0.0217] |
| spad_512_stable_original_inflight1 -> spad_512_stable_repeat_inflight1 | valid_complete_answer_rate | -0.1557 | [-0.1734, -0.1383] |
| spad_512_stable_repeat_inflight1 -> spad_512_stable_ablation_inflight2 | em | 0.0003 | [-0.0066, 0.0071] |
| spad_512_stable_repeat_inflight1 -> spad_512_stable_ablation_inflight2 | f1 | 0.0061 | [-0.0007, 0.0131] |
| spad_512_stable_repeat_inflight1 -> spad_512_stable_ablation_inflight2 | structured_em | 0.0003 | [-0.0066, 0.0071] |
| spad_512_stable_repeat_inflight1 -> spad_512_stable_ablation_inflight2 | answer_group_f1 | 0.0061 | [-0.0007, 0.0131] |
| spad_512_stable_repeat_inflight1 -> spad_512_stable_ablation_inflight2 | answer_group_recall | 0.0003 | [-0.0066, 0.0071] |
| spad_512_stable_repeat_inflight1 -> spad_512_stable_ablation_inflight2 | valid_complete_answer_rate | 0.0469 | [0.0334, 0.0603] |
| spad_512_stable_original_inflight1 -> spad_512_stable_ablation_inflight2 | em | -0.0306 | [-0.0394, -0.0217] |
| spad_512_stable_original_inflight1 -> spad_512_stable_ablation_inflight2 | f1 | -0.0467 | [-0.0559, -0.0372] |
| spad_512_stable_original_inflight1 -> spad_512_stable_ablation_inflight2 | structured_em | -0.0306 | [-0.0394, -0.0217] |
| spad_512_stable_original_inflight1 -> spad_512_stable_ablation_inflight2 | answer_group_f1 | -0.0467 | [-0.0559, -0.0372] |
| spad_512_stable_original_inflight1 -> spad_512_stable_ablation_inflight2 | answer_group_recall | -0.0306 | [-0.0394, -0.0217] |
| spad_512_stable_original_inflight1 -> spad_512_stable_ablation_inflight2 | valid_complete_answer_rate | -0.1089 | [-0.1263, -0.0914] |
