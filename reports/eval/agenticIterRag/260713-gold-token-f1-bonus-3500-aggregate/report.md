# New-Data Model Evaluation

- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Dataset SHA256: `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283`
- Each model has 1 isolated inference run(s). Repeats are not pooled as independent examples.
- Paired bootstrap averages repeats per question before resampling questions.

## Overall

| Model | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spad_512_stable_historical_inflight1 | 0.1360 +/- 0.0000 | 0.2265 +/- 0.0000 | 0.1360 +/- 0.0000 | 0.2265 +/- 0.0000 | 0.1360 +/- 0.0000 | 0.6989 +/- 0.0000 | 0.9689 +/- 0.0000 | 2.3391 +/- 0.0000 | 0.3340 +/- 0.0000 | 0.2426 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | 0.1054 +/- 0.0000 | 0.1798 +/- 0.0000 | 0.1054 +/- 0.0000 | 0.1798 +/- 0.0000 | 0.1054 +/- 0.0000 | 0.5900 +/- 0.0000 | 0.9711 +/- 0.0000 | 2.3566 +/- 0.0000 | 0.3466 +/- 0.0000 | 0.2363 +/- 0.0000 |
| spad_5100_stable_inflight2 | 0.1923 +/- 0.0000 | 0.2700 +/- 0.0000 | 0.1923 +/- 0.0000 | 0.2700 +/- 0.0000 | 0.1923 +/- 0.0000 | 0.7220 +/- 0.0000 | 0.9397 +/- 0.0000 | 2.6557 +/- 0.0000 | 0.5906 +/- 0.0000 | 0.2443 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | 0.1231 +/- 0.0000 | 0.2046 +/- 0.0000 | 0.1231 +/- 0.0000 | 0.2046 +/- 0.0000 | 0.1231 +/- 0.0000 | 0.6220 +/- 0.0000 | 0.9720 +/- 0.0000 | 2.4654 +/- 0.0000 | 0.3820 +/- 0.0000 | 0.2511 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | 0.1837 +/- 0.0000 | 0.2576 +/- 0.0000 | 0.1837 +/- 0.0000 | 0.2576 +/- 0.0000 | 0.1837 +/- 0.0000 | 0.6334 +/- 0.0000 | 0.9971 +/- 0.0000 | 3.0071 +/- 0.0000 | 0.5763 +/- 0.0000 | 0.3589 +/- 0.0000 |

## Per Run

| Model | Repeat | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spad_512_stable_historical_inflight1 | 1 | 0.1360 | 0.2265 | 0.1360 | 0.2265 | 0.1360 | 0.6989 | 0.9689 | 2.3391 | 0.3340 | 0.2426 |
| spad_512_stable_repeat_inflight2 | 1 | 0.1054 | 0.1798 | 0.1054 | 0.1798 | 0.1054 | 0.5900 | 0.9711 | 2.3566 | 0.3466 | 0.2363 |
| spad_5100_stable_inflight2 | 1 | 0.1923 | 0.2700 | 0.1923 | 0.2700 | 0.1923 | 0.7220 | 0.9397 | 2.6557 | 0.5906 | 0.2443 |
| spad_512_gold_token_f1_bonus_inflight2 | 1 | 0.1231 | 0.2046 | 0.1231 | 0.2046 | 0.1231 | 0.6220 | 0.9720 | 2.4654 | 0.3820 | 0.2511 |
| spad_5100_gold_token_f1_bonus_inflight2 | 1 | 0.1837 | 0.2576 | 0.1837 | 0.2576 | 0.1837 | 0.6334 | 0.9971 | 3.0071 | 0.5763 | 0.3589 |

## Search Count Buckets

| Model | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| spad_512_stable_historical_inflight1 | 0.0311 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.2703 +/- 0.0000 | 0.0397 +/- 0.0000 | 0.0149 +/- 0.0000 | 0.2440 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | 0.0289 +/- 0.0000 | 0.3743 +/- 0.0000 | 0.2974 +/- 0.0000 | 0.0489 +/- 0.0000 | 0.0120 +/- 0.0000 | 0.2386 +/- 0.0000 |
| spad_5100_stable_inflight2 | 0.0603 +/- 0.0000 | 0.0314 +/- 0.0000 | 0.6071 +/- 0.0000 | 0.0403 +/- 0.0000 | 0.0151 +/- 0.0000 | 0.2457 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | 0.0280 +/- 0.0000 | 0.3074 +/- 0.0000 | 0.3531 +/- 0.0000 | 0.0466 +/- 0.0000 | 0.0123 +/- 0.0000 | 0.2526 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | 0.0029 +/- 0.0000 | 0.1414 +/- 0.0000 | 0.4331 +/- 0.0000 | 0.0511 +/- 0.0000 | 0.0111 +/- 0.0000 | 0.3603 +/- 0.0000 |

## Per Data Source

| Model | Data source | EM | F1 | Structured EM | Group F1 | Group recall |
|---|---|---:|---:|---:|---:|---:|
| spad_512_stable_historical_inflight1 | 2wikimultihopqa | 0.0480 +/- 0.0000 | 0.1107 +/- 0.0000 | 0.0480 +/- 0.0000 | 0.1107 +/- 0.0000 | 0.0480 +/- 0.0000 |
| spad_512_stable_historical_inflight1 | bamboogle | 0.1520 +/- 0.0000 | 0.2485 +/- 0.0000 | 0.1520 +/- 0.0000 | 0.2485 +/- 0.0000 | 0.1520 +/- 0.0000 |
| spad_512_stable_historical_inflight1 | hotpotqa | 0.1441 +/- 0.0000 | 0.2443 +/- 0.0000 | 0.1441 +/- 0.0000 | 0.2443 +/- 0.0000 | 0.1441 +/- 0.0000 |
| spad_512_stable_historical_inflight1 | musique | 0.0320 +/- 0.0000 | 0.0967 +/- 0.0000 | 0.0320 +/- 0.0000 | 0.0967 +/- 0.0000 | 0.0320 +/- 0.0000 |
| spad_512_stable_historical_inflight1 | nq | 0.2011 +/- 0.0000 | 0.2966 +/- 0.0000 | 0.2011 +/- 0.0000 | 0.2966 +/- 0.0000 | 0.2011 +/- 0.0000 |
| spad_512_stable_historical_inflight1 | popqa | 0.2327 +/- 0.0000 | 0.2831 +/- 0.0000 | 0.2327 +/- 0.0000 | 0.2831 +/- 0.0000 | 0.2327 +/- 0.0000 |
| spad_512_stable_historical_inflight1 | triviaqa | 0.1545 +/- 0.0000 | 0.3227 +/- 0.0000 | 0.1545 +/- 0.0000 | 0.3227 +/- 0.0000 | 0.1545 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | 2wikimultihopqa | 0.0355 +/- 0.0000 | 0.0905 +/- 0.0000 | 0.0355 +/- 0.0000 | 0.0905 +/- 0.0000 | 0.0355 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | bamboogle | 0.0960 +/- 0.0000 | 0.1742 +/- 0.0000 | 0.0960 +/- 0.0000 | 0.1742 +/- 0.0000 | 0.0960 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | hotpotqa | 0.1032 +/- 0.0000 | 0.1879 +/- 0.0000 | 0.1032 +/- 0.0000 | 0.1879 +/- 0.0000 | 0.1032 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | musique | 0.0178 +/- 0.0000 | 0.0656 +/- 0.0000 | 0.0178 +/- 0.0000 | 0.0656 +/- 0.0000 | 0.0178 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | nq | 0.1459 +/- 0.0000 | 0.2179 +/- 0.0000 | 0.1459 +/- 0.0000 | 0.2179 +/- 0.0000 | 0.1459 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | popqa | 0.2114 +/- 0.0000 | 0.2587 +/- 0.0000 | 0.2114 +/- 0.0000 | 0.2587 +/- 0.0000 | 0.2114 +/- 0.0000 |
| spad_512_stable_repeat_inflight2 | triviaqa | 0.1208 +/- 0.0000 | 0.2593 +/- 0.0000 | 0.1208 +/- 0.0000 | 0.2593 +/- 0.0000 | 0.1208 +/- 0.0000 |
| spad_5100_stable_inflight2 | 2wikimultihopqa | 0.1083 +/- 0.0000 | 0.1497 +/- 0.0000 | 0.1083 +/- 0.0000 | 0.1497 +/- 0.0000 | 0.1083 +/- 0.0000 |
| spad_5100_stable_inflight2 | bamboogle | 0.1760 +/- 0.0000 | 0.2701 +/- 0.0000 | 0.1760 +/- 0.0000 | 0.2701 +/- 0.0000 | 0.1760 +/- 0.0000 |
| spad_5100_stable_inflight2 | hotpotqa | 0.2456 +/- 0.0000 | 0.3265 +/- 0.0000 | 0.2456 +/- 0.0000 | 0.3265 +/- 0.0000 | 0.2456 +/- 0.0000 |
| spad_5100_stable_inflight2 | musique | 0.0463 +/- 0.0000 | 0.0886 +/- 0.0000 | 0.0463 +/- 0.0000 | 0.0886 +/- 0.0000 | 0.0463 +/- 0.0000 |
| spad_5100_stable_inflight2 | nq | 0.2865 +/- 0.0000 | 0.3637 +/- 0.0000 | 0.2865 +/- 0.0000 | 0.3637 +/- 0.0000 | 0.2865 +/- 0.0000 |
| spad_5100_stable_inflight2 | popqa | 0.2824 +/- 0.0000 | 0.3262 +/- 0.0000 | 0.2824 +/- 0.0000 | 0.3262 +/- 0.0000 | 0.2824 +/- 0.0000 |
| spad_5100_stable_inflight2 | triviaqa | 0.1883 +/- 0.0000 | 0.3654 +/- 0.0000 | 0.1883 +/- 0.0000 | 0.3654 +/- 0.0000 | 0.1883 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | 2wikimultihopqa | 0.0409 +/- 0.0000 | 0.1043 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.1043 +/- 0.0000 | 0.0409 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | bamboogle | 0.1520 +/- 0.0000 | 0.2595 +/- 0.0000 | 0.1520 +/- 0.0000 | 0.2595 +/- 0.0000 | 0.1520 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | hotpotqa | 0.1157 +/- 0.0000 | 0.2021 +/- 0.0000 | 0.1157 +/- 0.0000 | 0.2021 +/- 0.0000 | 0.1157 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | musique | 0.0391 +/- 0.0000 | 0.0884 +/- 0.0000 | 0.0391 +/- 0.0000 | 0.0884 +/- 0.0000 | 0.0391 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | nq | 0.1833 +/- 0.0000 | 0.2696 +/- 0.0000 | 0.1833 +/- 0.0000 | 0.2696 +/- 0.0000 | 0.1833 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | popqa | 0.2220 +/- 0.0000 | 0.2694 +/- 0.0000 | 0.2220 +/- 0.0000 | 0.2694 +/- 0.0000 | 0.2220 +/- 0.0000 |
| spad_512_gold_token_f1_bonus_inflight2 | triviaqa | 0.1314 +/- 0.0000 | 0.2813 +/- 0.0000 | 0.1314 +/- 0.0000 | 0.2813 +/- 0.0000 | 0.1314 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | 2wikimultihopqa | 0.1261 +/- 0.0000 | 0.1731 +/- 0.0000 | 0.1261 +/- 0.0000 | 0.1731 +/- 0.0000 | 0.1261 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | bamboogle | 0.2160 +/- 0.0000 | 0.2807 +/- 0.0000 | 0.2160 +/- 0.0000 | 0.2807 +/- 0.0000 | 0.2160 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | hotpotqa | 0.2313 +/- 0.0000 | 0.3187 +/- 0.0000 | 0.2313 +/- 0.0000 | 0.3187 +/- 0.0000 | 0.2313 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | musique | 0.0463 +/- 0.0000 | 0.0962 +/- 0.0000 | 0.0463 +/- 0.0000 | 0.0962 +/- 0.0000 | 0.0463 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | nq | 0.2527 +/- 0.0000 | 0.3287 +/- 0.0000 | 0.2527 +/- 0.0000 | 0.3287 +/- 0.0000 | 0.2527 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | popqa | 0.2735 +/- 0.0000 | 0.3088 +/- 0.0000 | 0.2735 +/- 0.0000 | 0.3088 +/- 0.0000 | 0.2735 +/- 0.0000 |
| spad_5100_gold_token_f1_bonus_inflight2 | triviaqa | 0.1652 +/- 0.0000 | 0.3147 +/- 0.0000 | 0.1652 +/- 0.0000 | 0.3147 +/- 0.0000 | 0.1652 +/- 0.0000 |

## Paired Comparisons

| Comparison | Metric | Delta (right-left) | 95% CI |
|---|---|---:|---:|
| spad_512_stable_historical_inflight1 -> spad_512_gold_token_f1_bonus_inflight2 | em | -0.0129 | [-0.0203, -0.0051] |
| spad_512_stable_historical_inflight1 -> spad_512_gold_token_f1_bonus_inflight2 | f1 | -0.0220 | [-0.0295, -0.0142] |
| spad_512_stable_historical_inflight1 -> spad_512_gold_token_f1_bonus_inflight2 | structured_em | -0.0129 | [-0.0203, -0.0051] |
| spad_512_stable_historical_inflight1 -> spad_512_gold_token_f1_bonus_inflight2 | answer_group_f1 | -0.0220 | [-0.0295, -0.0142] |
| spad_512_stable_historical_inflight1 -> spad_512_gold_token_f1_bonus_inflight2 | answer_group_recall | -0.0129 | [-0.0203, -0.0051] |
| spad_512_stable_historical_inflight1 -> spad_512_gold_token_f1_bonus_inflight2 | valid_complete_answer_rate | -0.0769 | [-0.0917, -0.0617] |
| spad_512_stable_repeat_inflight2 -> spad_512_gold_token_f1_bonus_inflight2 | em | 0.0177 | [0.0097, 0.0257] |
| spad_512_stable_repeat_inflight2 -> spad_512_gold_token_f1_bonus_inflight2 | f1 | 0.0248 | [0.0163, 0.0330] |
| spad_512_stable_repeat_inflight2 -> spad_512_gold_token_f1_bonus_inflight2 | structured_em | 0.0177 | [0.0097, 0.0257] |
| spad_512_stable_repeat_inflight2 -> spad_512_gold_token_f1_bonus_inflight2 | answer_group_f1 | 0.0248 | [0.0163, 0.0330] |
| spad_512_stable_repeat_inflight2 -> spad_512_gold_token_f1_bonus_inflight2 | answer_group_recall | 0.0177 | [0.0097, 0.0257] |
| spad_512_stable_repeat_inflight2 -> spad_512_gold_token_f1_bonus_inflight2 | valid_complete_answer_rate | 0.0320 | [0.0166, 0.0477] |
| spad_5100_stable_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | em | -0.0086 | [-0.0177, 0.0009] |
| spad_5100_stable_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | f1 | -0.0125 | [-0.0224, -0.0025] |
| spad_5100_stable_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | structured_em | -0.0086 | [-0.0177, 0.0009] |
| spad_5100_stable_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | answer_group_f1 | -0.0125 | [-0.0224, -0.0025] |
| spad_5100_stable_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | answer_group_recall | -0.0086 | [-0.0177, 0.0009] |
| spad_5100_stable_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | valid_complete_answer_rate | -0.0886 | [-0.1046, -0.0726] |
| spad_512_gold_token_f1_bonus_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | em | 0.0606 | [0.0497, 0.0720] |
| spad_512_gold_token_f1_bonus_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | f1 | 0.0530 | [0.0420, 0.0644] |
| spad_512_gold_token_f1_bonus_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | structured_em | 0.0606 | [0.0497, 0.0720] |
| spad_512_gold_token_f1_bonus_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | answer_group_f1 | 0.0530 | [0.0420, 0.0644] |
| spad_512_gold_token_f1_bonus_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | answer_group_recall | 0.0606 | [0.0497, 0.0720] |
| spad_512_gold_token_f1_bonus_inflight2 -> spad_5100_gold_token_f1_bonus_inflight2 | valid_complete_answer_rate | 0.0114 | [-0.0071, 0.0303] |
| spad_512_stable_historical_inflight1 -> spad_5100_stable_inflight2 | em | 0.0563 | [0.0457, 0.0674] |
| spad_512_stable_historical_inflight1 -> spad_5100_stable_inflight2 | f1 | 0.0435 | [0.0330, 0.0547] |
| spad_512_stable_historical_inflight1 -> spad_5100_stable_inflight2 | structured_em | 0.0563 | [0.0457, 0.0674] |
| spad_512_stable_historical_inflight1 -> spad_5100_stable_inflight2 | answer_group_f1 | 0.0435 | [0.0330, 0.0547] |
| spad_512_stable_historical_inflight1 -> spad_5100_stable_inflight2 | answer_group_recall | 0.0563 | [0.0457, 0.0674] |
| spad_512_stable_historical_inflight1 -> spad_5100_stable_inflight2 | valid_complete_answer_rate | 0.0231 | [0.0069, 0.0394] |
