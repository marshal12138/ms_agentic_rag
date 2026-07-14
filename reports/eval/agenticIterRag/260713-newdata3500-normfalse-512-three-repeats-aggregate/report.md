# New-Data Model Evaluation

- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Dataset SHA256: `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283`
- Each model has 1 isolated inference run(s). Repeats are not pooled as independent examples.
- Paired bootstrap averages repeats per question before resampling questions.

## Overall

| Model | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_search_r1_512_normtrue | 0.1180 +/- 0.0000 | 0.1965 +/- 0.0000 | 0.1180 +/- 0.0000 | 0.1965 +/- 0.0000 | 0.1180 +/- 0.0000 | 0.6271 +/- 0.0000 | 0.9831 +/- 0.0000 | 2.3489 +/- 0.0000 | 0.3640 +/- 0.0000 | 0.2569 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | 0.1360 +/- 0.0000 | 0.2265 +/- 0.0000 | 0.1360 +/- 0.0000 | 0.2265 +/- 0.0000 | 0.1360 +/- 0.0000 | 0.6989 +/- 0.0000 | 0.9689 +/- 0.0000 | 2.3391 +/- 0.0000 | 0.3340 +/- 0.0000 | 0.2426 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | 0.1054 +/- 0.0000 | 0.1798 +/- 0.0000 | 0.1054 +/- 0.0000 | 0.1798 +/- 0.0000 | 0.1054 +/- 0.0000 | 0.5900 +/- 0.0000 | 0.9711 +/- 0.0000 | 2.3566 +/- 0.0000 | 0.3466 +/- 0.0000 | 0.2363 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | 0.1231 +/- 0.0000 | 0.2046 +/- 0.0000 | 0.1231 +/- 0.0000 | 0.2046 +/- 0.0000 | 0.1231 +/- 0.0000 | 0.6220 +/- 0.0000 | 0.9720 +/- 0.0000 | 2.4654 +/- 0.0000 | 0.3820 +/- 0.0000 | 0.2511 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | 0.1271 +/- 0.0000 | 0.2108 +/- 0.0000 | 0.1271 +/- 0.0000 | 0.2108 +/- 0.0000 | 0.1271 +/- 0.0000 | 0.6620 +/- 0.0000 | 0.9783 +/- 0.0000 | 2.2660 +/- 0.0000 | 0.3194 +/- 0.0000 | 0.2489 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | 0.1017 +/- 0.0000 | 0.1771 +/- 0.0000 | 0.1017 +/- 0.0000 | 0.1771 +/- 0.0000 | 0.1017 +/- 0.0000 | 0.5543 +/- 0.0000 | 0.9994 +/- 0.0000 | 2.7989 +/- 0.0000 | 0.4994 +/- 0.0000 | 0.3783 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | 0.1106 +/- 0.0000 | 0.1896 +/- 0.0000 | 0.1106 +/- 0.0000 | 0.1896 +/- 0.0000 | 0.1106 +/- 0.0000 | 0.5891 +/- 0.0000 | 0.9991 +/- 0.0000 | 2.5929 +/- 0.0000 | 0.4343 +/- 0.0000 | 0.3317 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | 0.1129 +/- 0.0000 | 0.1904 +/- 0.0000 | 0.1129 +/- 0.0000 | 0.1904 +/- 0.0000 | 0.1129 +/- 0.0000 | 0.6291 +/- 0.0000 | 0.9689 +/- 0.0000 | 2.2183 +/- 0.0000 | 0.3120 +/- 0.0000 | 0.2049 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | 0.1297 +/- 0.0000 | 0.2174 +/- 0.0000 | 0.1297 +/- 0.0000 | 0.2174 +/- 0.0000 | 0.1297 +/- 0.0000 | 0.6911 +/- 0.0000 | 0.9817 +/- 0.0000 | 2.2757 +/- 0.0000 | 0.3177 +/- 0.0000 | 0.2303 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | 0.1231 +/- 0.0000 | 0.2086 +/- 0.0000 | 0.1231 +/- 0.0000 | 0.2086 +/- 0.0000 | 0.1231 +/- 0.0000 | 0.6617 +/- 0.0000 | 0.9703 +/- 0.0000 | 2.1923 +/- 0.0000 | 0.2940 +/- 0.0000 | 0.2160 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | 0.1174 +/- 0.0000 | 0.2032 +/- 0.0000 | 0.1174 +/- 0.0000 | 0.2032 +/- 0.0000 | 0.1174 +/- 0.0000 | 0.6486 +/- 0.0000 | 0.9834 +/- 0.0000 | 2.3043 +/- 0.0000 | 0.3349 +/- 0.0000 | 0.2477 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | 0.1286 +/- 0.0000 | 0.2110 +/- 0.0000 | 0.1286 +/- 0.0000 | 0.2110 +/- 0.0000 | 0.1286 +/- 0.0000 | 0.6566 +/- 0.0000 | 0.9800 +/- 0.0000 | 2.3423 +/- 0.0000 | 0.3494 +/- 0.0000 | 0.2560 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | 0.1157 +/- 0.0000 | 0.1944 +/- 0.0000 | 0.1157 +/- 0.0000 | 0.1944 +/- 0.0000 | 0.1157 +/- 0.0000 | 0.6360 +/- 0.0000 | 0.9697 +/- 0.0000 | 2.3397 +/- 0.0000 | 0.3686 +/- 0.0000 | 0.2109 +/- 0.0000 |

## Per Run

| Model | Repeat | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| historical_search_r1_512_normtrue | 1 | 0.1180 | 0.1965 | 0.1180 | 0.1965 | 0.1180 | 0.6271 | 0.9831 | 2.3489 | 0.3640 | 0.2569 |
| historical_spad_512_stable_inflight1_normtrue | 1 | 0.1360 | 0.2265 | 0.1360 | 0.2265 | 0.1360 | 0.6989 | 0.9689 | 2.3391 | 0.3340 | 0.2426 |
| historical_spad_512_stable_inflight2_normtrue | 1 | 0.1054 | 0.1798 | 0.1054 | 0.1798 | 0.1054 | 0.5900 | 0.9711 | 2.3566 | 0.3466 | 0.2363 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | 1 | 0.1231 | 0.2046 | 0.1231 | 0.2046 | 0.1231 | 0.6220 | 0.9720 | 2.4654 | 0.3820 | 0.2511 |
| search_r1_512_normfalse_rep1 | 1 | 0.1271 | 0.2108 | 0.1271 | 0.2108 | 0.1271 | 0.6620 | 0.9783 | 2.2660 | 0.3194 | 0.2489 |
| search_r1_512_normfalse_rep2 | 1 | 0.1017 | 0.1771 | 0.1017 | 0.1771 | 0.1017 | 0.5543 | 0.9994 | 2.7989 | 0.4994 | 0.3783 |
| search_r1_512_normfalse_rep3 | 1 | 0.1106 | 0.1896 | 0.1106 | 0.1896 | 0.1106 | 0.5891 | 0.9991 | 2.5929 | 0.4343 | 0.3317 |
| spad_512_stable_inflight2_normfalse_rep1 | 1 | 0.1129 | 0.1904 | 0.1129 | 0.1904 | 0.1129 | 0.6291 | 0.9689 | 2.2183 | 0.3120 | 0.2049 |
| spad_512_stable_inflight2_normfalse_rep2 | 1 | 0.1297 | 0.2174 | 0.1297 | 0.2174 | 0.1297 | 0.6911 | 0.9817 | 2.2757 | 0.3177 | 0.2303 |
| spad_512_stable_inflight2_normfalse_rep3 | 1 | 0.1231 | 0.2086 | 0.1231 | 0.2086 | 0.1231 | 0.6617 | 0.9703 | 2.1923 | 0.2940 | 0.2160 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | 1 | 0.1174 | 0.2032 | 0.1174 | 0.2032 | 0.1174 | 0.6486 | 0.9834 | 2.3043 | 0.3349 | 0.2477 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | 1 | 0.1286 | 0.2110 | 0.1286 | 0.2110 | 0.1286 | 0.6566 | 0.9800 | 2.3423 | 0.3494 | 0.2560 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | 1 | 0.1157 | 0.1944 | 0.1157 | 0.1944 | 0.1157 | 0.6360 | 0.9697 | 2.3397 | 0.3686 | 0.2109 |

## Search Count Buckets

| Model | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| historical_search_r1_512_normtrue | 0.0169 +/- 0.0000 | 0.4580 +/- 0.0000 | 0.2137 +/- 0.0000 | 0.0403 +/- 0.0000 | 0.0131 +/- 0.0000 | 0.2580 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | 0.0311 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.2703 +/- 0.0000 | 0.0397 +/- 0.0000 | 0.0149 +/- 0.0000 | 0.2440 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | 0.0289 +/- 0.0000 | 0.3743 +/- 0.0000 | 0.2974 +/- 0.0000 | 0.0489 +/- 0.0000 | 0.0120 +/- 0.0000 | 0.2386 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | 0.0280 +/- 0.0000 | 0.3074 +/- 0.0000 | 0.3531 +/- 0.0000 | 0.0466 +/- 0.0000 | 0.0123 +/- 0.0000 | 0.2526 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | 0.0217 +/- 0.0000 | 0.4969 +/- 0.0000 | 0.1846 +/- 0.0000 | 0.0377 +/- 0.0000 | 0.0089 +/- 0.0000 | 0.2503 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | 0.0006 +/- 0.0000 | 0.4043 +/- 0.0000 | 0.1637 +/- 0.0000 | 0.0371 +/- 0.0000 | 0.0157 +/- 0.0000 | 0.3786 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | 0.0009 +/- 0.0000 | 0.4620 +/- 0.0000 | 0.1563 +/- 0.0000 | 0.0374 +/- 0.0000 | 0.0111 +/- 0.0000 | 0.3323 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | 0.0311 +/- 0.0000 | 0.4040 +/- 0.0000 | 0.3057 +/- 0.0000 | 0.0403 +/- 0.0000 | 0.0123 +/- 0.0000 | 0.2066 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | 0.0183 +/- 0.0000 | 0.4491 +/- 0.0000 | 0.2471 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.0131 +/- 0.0000 | 0.2314 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | 0.0297 +/- 0.0000 | 0.4586 +/- 0.0000 | 0.2463 +/- 0.0000 | 0.0371 +/- 0.0000 | 0.0117 +/- 0.0000 | 0.2166 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | 0.0166 +/- 0.0000 | 0.4831 +/- 0.0000 | 0.1929 +/- 0.0000 | 0.0431 +/- 0.0000 | 0.0154 +/- 0.0000 | 0.2489 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | 0.0200 +/- 0.0000 | 0.4583 +/- 0.0000 | 0.2086 +/- 0.0000 | 0.0429 +/- 0.0000 | 0.0131 +/- 0.0000 | 0.2571 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | 0.0303 +/- 0.0000 | 0.3109 +/- 0.0000 | 0.3869 +/- 0.0000 | 0.0457 +/- 0.0000 | 0.0134 +/- 0.0000 | 0.2129 +/- 0.0000 |

## Per Data Source

| Model | Data source | EM | F1 | Structured EM | Group F1 | Group recall |
|---|---|---:|---:|---:|---:|---:|
| historical_search_r1_512_normtrue | 2wikimultihopqa | 0.0284 +/- 0.0000 | 0.0924 +/- 0.0000 | 0.0284 +/- 0.0000 | 0.0924 +/- 0.0000 | 0.0284 +/- 0.0000 |
| historical_search_r1_512_normtrue | bamboogle | 0.0880 +/- 0.0000 | 0.1636 +/- 0.0000 | 0.0880 +/- 0.0000 | 0.1636 +/- 0.0000 | 0.0880 +/- 0.0000 |
| historical_search_r1_512_normtrue | hotpotqa | 0.1103 +/- 0.0000 | 0.2051 +/- 0.0000 | 0.1103 +/- 0.0000 | 0.2051 +/- 0.0000 | 0.1103 +/- 0.0000 |
| historical_search_r1_512_normtrue | musique | 0.0214 +/- 0.0000 | 0.0716 +/- 0.0000 | 0.0214 +/- 0.0000 | 0.0716 +/- 0.0000 | 0.0214 +/- 0.0000 |
| historical_search_r1_512_normtrue | nq | 0.1779 +/- 0.0000 | 0.2521 +/- 0.0000 | 0.1779 +/- 0.0000 | 0.2521 +/- 0.0000 | 0.1779 +/- 0.0000 |
| historical_search_r1_512_normtrue | popqa | 0.2380 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.2380 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.2380 +/- 0.0000 |
| historical_search_r1_512_normtrue | triviaqa | 0.1385 +/- 0.0000 | 0.2865 +/- 0.0000 | 0.1385 +/- 0.0000 | 0.2865 +/- 0.0000 | 0.1385 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | 2wikimultihopqa | 0.0480 +/- 0.0000 | 0.1107 +/- 0.0000 | 0.0480 +/- 0.0000 | 0.1107 +/- 0.0000 | 0.0480 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | bamboogle | 0.1520 +/- 0.0000 | 0.2485 +/- 0.0000 | 0.1520 +/- 0.0000 | 0.2485 +/- 0.0000 | 0.1520 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | hotpotqa | 0.1441 +/- 0.0000 | 0.2443 +/- 0.0000 | 0.1441 +/- 0.0000 | 0.2443 +/- 0.0000 | 0.1441 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | musique | 0.0320 +/- 0.0000 | 0.0967 +/- 0.0000 | 0.0320 +/- 0.0000 | 0.0967 +/- 0.0000 | 0.0320 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | nq | 0.2011 +/- 0.0000 | 0.2966 +/- 0.0000 | 0.2011 +/- 0.0000 | 0.2966 +/- 0.0000 | 0.2011 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | popqa | 0.2327 +/- 0.0000 | 0.2831 +/- 0.0000 | 0.2327 +/- 0.0000 | 0.2831 +/- 0.0000 | 0.2327 +/- 0.0000 |
| historical_spad_512_stable_inflight1_normtrue | triviaqa | 0.1545 +/- 0.0000 | 0.3227 +/- 0.0000 | 0.1545 +/- 0.0000 | 0.3227 +/- 0.0000 | 0.1545 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | 2wikimultihopqa | 0.0355 +/- 0.0000 | 0.0905 +/- 0.0000 | 0.0355 +/- 0.0000 | 0.0905 +/- 0.0000 | 0.0355 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | bamboogle | 0.0960 +/- 0.0000 | 0.1742 +/- 0.0000 | 0.0960 +/- 0.0000 | 0.1742 +/- 0.0000 | 0.0960 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | hotpotqa | 0.1032 +/- 0.0000 | 0.1879 +/- 0.0000 | 0.1032 +/- 0.0000 | 0.1879 +/- 0.0000 | 0.1032 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | musique | 0.0178 +/- 0.0000 | 0.0656 +/- 0.0000 | 0.0178 +/- 0.0000 | 0.0656 +/- 0.0000 | 0.0178 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | nq | 0.1459 +/- 0.0000 | 0.2179 +/- 0.0000 | 0.1459 +/- 0.0000 | 0.2179 +/- 0.0000 | 0.1459 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | popqa | 0.2114 +/- 0.0000 | 0.2587 +/- 0.0000 | 0.2114 +/- 0.0000 | 0.2587 +/- 0.0000 | 0.2114 +/- 0.0000 |
| historical_spad_512_stable_inflight2_normtrue | triviaqa | 0.1208 +/- 0.0000 | 0.2593 +/- 0.0000 | 0.1208 +/- 0.0000 | 0.2593 +/- 0.0000 | 0.1208 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | 2wikimultihopqa | 0.0409 +/- 0.0000 | 0.1043 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.1043 +/- 0.0000 | 0.0409 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | bamboogle | 0.1520 +/- 0.0000 | 0.2595 +/- 0.0000 | 0.1520 +/- 0.0000 | 0.2595 +/- 0.0000 | 0.1520 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | hotpotqa | 0.1157 +/- 0.0000 | 0.2021 +/- 0.0000 | 0.1157 +/- 0.0000 | 0.2021 +/- 0.0000 | 0.1157 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | musique | 0.0391 +/- 0.0000 | 0.0884 +/- 0.0000 | 0.0391 +/- 0.0000 | 0.0884 +/- 0.0000 | 0.0391 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | nq | 0.1833 +/- 0.0000 | 0.2696 +/- 0.0000 | 0.1833 +/- 0.0000 | 0.2696 +/- 0.0000 | 0.1833 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | popqa | 0.2220 +/- 0.0000 | 0.2694 +/- 0.0000 | 0.2220 +/- 0.0000 | 0.2694 +/- 0.0000 | 0.2220 +/- 0.0000 |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue | triviaqa | 0.1314 +/- 0.0000 | 0.2813 +/- 0.0000 | 0.1314 +/- 0.0000 | 0.2813 +/- 0.0000 | 0.1314 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | 2wikimultihopqa | 0.0480 +/- 0.0000 | 0.1086 +/- 0.0000 | 0.0480 +/- 0.0000 | 0.1086 +/- 0.0000 | 0.0480 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | bamboogle | 0.1200 +/- 0.0000 | 0.2404 +/- 0.0000 | 0.1200 +/- 0.0000 | 0.2404 +/- 0.0000 | 0.1200 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | hotpotqa | 0.1068 +/- 0.0000 | 0.2022 +/- 0.0000 | 0.1068 +/- 0.0000 | 0.2022 +/- 0.0000 | 0.1068 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | musique | 0.0391 +/- 0.0000 | 0.0925 +/- 0.0000 | 0.0391 +/- 0.0000 | 0.0925 +/- 0.0000 | 0.0391 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | nq | 0.1993 +/- 0.0000 | 0.2788 +/- 0.0000 | 0.1993 +/- 0.0000 | 0.2788 +/- 0.0000 | 0.1993 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | popqa | 0.2274 +/- 0.0000 | 0.2789 +/- 0.0000 | 0.2274 +/- 0.0000 | 0.2789 +/- 0.0000 | 0.2274 +/- 0.0000 |
| search_r1_512_normfalse_rep1 | triviaqa | 0.1439 +/- 0.0000 | 0.2968 +/- 0.0000 | 0.1439 +/- 0.0000 | 0.2968 +/- 0.0000 | 0.1439 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | 2wikimultihopqa | 0.0302 +/- 0.0000 | 0.1011 +/- 0.0000 | 0.0302 +/- 0.0000 | 0.1011 +/- 0.0000 | 0.0302 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | bamboogle | 0.0640 +/- 0.0000 | 0.1263 +/- 0.0000 | 0.0640 +/- 0.0000 | 0.1263 +/- 0.0000 | 0.0640 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | hotpotqa | 0.0907 +/- 0.0000 | 0.1757 +/- 0.0000 | 0.0907 +/- 0.0000 | 0.1757 +/- 0.0000 | 0.0907 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | musique | 0.0142 +/- 0.0000 | 0.0508 +/- 0.0000 | 0.0142 +/- 0.0000 | 0.0508 +/- 0.0000 | 0.0142 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | nq | 0.1584 +/- 0.0000 | 0.2424 +/- 0.0000 | 0.1584 +/- 0.0000 | 0.2424 +/- 0.0000 | 0.1584 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | popqa | 0.1989 +/- 0.0000 | 0.2438 +/- 0.0000 | 0.1989 +/- 0.0000 | 0.2438 +/- 0.0000 | 0.1989 +/- 0.0000 |
| search_r1_512_normfalse_rep2 | triviaqa | 0.1261 +/- 0.0000 | 0.2597 +/- 0.0000 | 0.1261 +/- 0.0000 | 0.2597 +/- 0.0000 | 0.1261 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | 2wikimultihopqa | 0.0249 +/- 0.0000 | 0.0924 +/- 0.0000 | 0.0249 +/- 0.0000 | 0.0924 +/- 0.0000 | 0.0249 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | bamboogle | 0.0720 +/- 0.0000 | 0.1446 +/- 0.0000 | 0.0720 +/- 0.0000 | 0.1446 +/- 0.0000 | 0.0720 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | hotpotqa | 0.1050 +/- 0.0000 | 0.1955 +/- 0.0000 | 0.1050 +/- 0.0000 | 0.1955 +/- 0.0000 | 0.1050 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | musique | 0.0142 +/- 0.0000 | 0.0556 +/- 0.0000 | 0.0142 +/- 0.0000 | 0.0556 +/- 0.0000 | 0.0142 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | nq | 0.1797 +/- 0.0000 | 0.2609 +/- 0.0000 | 0.1797 +/- 0.0000 | 0.2609 +/- 0.0000 | 0.1797 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | popqa | 0.2096 +/- 0.0000 | 0.2590 +/- 0.0000 | 0.2096 +/- 0.0000 | 0.2590 +/- 0.0000 | 0.2096 +/- 0.0000 |
| search_r1_512_normfalse_rep3 | triviaqa | 0.1385 +/- 0.0000 | 0.2844 +/- 0.0000 | 0.1385 +/- 0.0000 | 0.2844 +/- 0.0000 | 0.1385 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | 2wikimultihopqa | 0.0302 +/- 0.0000 | 0.0869 +/- 0.0000 | 0.0302 +/- 0.0000 | 0.0869 +/- 0.0000 | 0.0302 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | bamboogle | 0.0880 +/- 0.0000 | 0.1806 +/- 0.0000 | 0.0880 +/- 0.0000 | 0.1806 +/- 0.0000 | 0.0880 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | hotpotqa | 0.1157 +/- 0.0000 | 0.2011 +/- 0.0000 | 0.1157 +/- 0.0000 | 0.2011 +/- 0.0000 | 0.1157 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | musique | 0.0249 +/- 0.0000 | 0.0729 +/- 0.0000 | 0.0249 +/- 0.0000 | 0.0729 +/- 0.0000 | 0.0249 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | nq | 0.1548 +/- 0.0000 | 0.2345 +/- 0.0000 | 0.1548 +/- 0.0000 | 0.2345 +/- 0.0000 | 0.1548 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | popqa | 0.2291 +/- 0.0000 | 0.2772 +/- 0.0000 | 0.2291 +/- 0.0000 | 0.2772 +/- 0.0000 | 0.2291 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep1 | triviaqa | 0.1279 +/- 0.0000 | 0.2717 +/- 0.0000 | 0.1279 +/- 0.0000 | 0.2717 +/- 0.0000 | 0.1279 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | 2wikimultihopqa | 0.0409 +/- 0.0000 | 0.1031 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.1031 +/- 0.0000 | 0.0409 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | bamboogle | 0.1600 +/- 0.0000 | 0.2530 +/- 0.0000 | 0.1600 +/- 0.0000 | 0.2530 +/- 0.0000 | 0.1600 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | hotpotqa | 0.1103 +/- 0.0000 | 0.2163 +/- 0.0000 | 0.1103 +/- 0.0000 | 0.2163 +/- 0.0000 | 0.1103 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | musique | 0.0249 +/- 0.0000 | 0.0808 +/- 0.0000 | 0.0249 +/- 0.0000 | 0.0808 +/- 0.0000 | 0.0249 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | nq | 0.2028 +/- 0.0000 | 0.2929 +/- 0.0000 | 0.2028 +/- 0.0000 | 0.2929 +/- 0.0000 | 0.2028 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | popqa | 0.2362 +/- 0.0000 | 0.2861 +/- 0.0000 | 0.2362 +/- 0.0000 | 0.2861 +/- 0.0000 | 0.2362 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep2 | triviaqa | 0.1563 +/- 0.0000 | 0.3172 +/- 0.0000 | 0.1563 +/- 0.0000 | 0.3172 +/- 0.0000 | 0.1563 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | 2wikimultihopqa | 0.0409 +/- 0.0000 | 0.1026 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.1026 +/- 0.0000 | 0.0409 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | bamboogle | 0.1120 +/- 0.0000 | 0.2189 +/- 0.0000 | 0.1120 +/- 0.0000 | 0.2189 +/- 0.0000 | 0.1120 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | hotpotqa | 0.1192 +/- 0.0000 | 0.2153 +/- 0.0000 | 0.1192 +/- 0.0000 | 0.2153 +/- 0.0000 | 0.1192 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | musique | 0.0285 +/- 0.0000 | 0.0859 +/- 0.0000 | 0.0285 +/- 0.0000 | 0.0859 +/- 0.0000 | 0.0285 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | nq | 0.1815 +/- 0.0000 | 0.2663 +/- 0.0000 | 0.1815 +/- 0.0000 | 0.2663 +/- 0.0000 | 0.1815 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | popqa | 0.2256 +/- 0.0000 | 0.2783 +/- 0.0000 | 0.2256 +/- 0.0000 | 0.2783 +/- 0.0000 | 0.2256 +/- 0.0000 |
| spad_512_stable_inflight2_normfalse_rep3 | triviaqa | 0.1456 +/- 0.0000 | 0.3007 +/- 0.0000 | 0.1456 +/- 0.0000 | 0.3007 +/- 0.0000 | 0.1456 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | 2wikimultihopqa | 0.0373 +/- 0.0000 | 0.1057 +/- 0.0000 | 0.0373 +/- 0.0000 | 0.1057 +/- 0.0000 | 0.0373 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | bamboogle | 0.1040 +/- 0.0000 | 0.2087 +/- 0.0000 | 0.1040 +/- 0.0000 | 0.2087 +/- 0.0000 | 0.1040 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | hotpotqa | 0.0979 +/- 0.0000 | 0.1951 +/- 0.0000 | 0.0979 +/- 0.0000 | 0.1951 +/- 0.0000 | 0.0979 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | musique | 0.0214 +/- 0.0000 | 0.0739 +/- 0.0000 | 0.0214 +/- 0.0000 | 0.0739 +/- 0.0000 | 0.0214 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | nq | 0.1922 +/- 0.0000 | 0.2695 +/- 0.0000 | 0.1922 +/- 0.0000 | 0.2695 +/- 0.0000 | 0.1922 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | popqa | 0.2256 +/- 0.0000 | 0.2821 +/- 0.0000 | 0.2256 +/- 0.0000 | 0.2821 +/- 0.0000 | 0.2256 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 | triviaqa | 0.1332 +/- 0.0000 | 0.2915 +/- 0.0000 | 0.1332 +/- 0.0000 | 0.2915 +/- 0.0000 | 0.1332 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | 2wikimultihopqa | 0.0462 +/- 0.0000 | 0.1050 +/- 0.0000 | 0.0462 +/- 0.0000 | 0.1050 +/- 0.0000 | 0.0462 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | bamboogle | 0.1360 +/- 0.0000 | 0.2426 +/- 0.0000 | 0.1360 +/- 0.0000 | 0.2426 +/- 0.0000 | 0.1360 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | hotpotqa | 0.1192 +/- 0.0000 | 0.2149 +/- 0.0000 | 0.1192 +/- 0.0000 | 0.2149 +/- 0.0000 | 0.1192 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | musique | 0.0409 +/- 0.0000 | 0.0943 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.0943 +/- 0.0000 | 0.0409 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | nq | 0.1815 +/- 0.0000 | 0.2627 +/- 0.0000 | 0.1815 +/- 0.0000 | 0.2627 +/- 0.0000 | 0.1815 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | popqa | 0.2327 +/- 0.0000 | 0.2866 +/- 0.0000 | 0.2327 +/- 0.0000 | 0.2866 +/- 0.0000 | 0.2327 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 | triviaqa | 0.1492 +/- 0.0000 | 0.2954 +/- 0.0000 | 0.1492 +/- 0.0000 | 0.2954 +/- 0.0000 | 0.1492 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | 2wikimultihopqa | 0.0426 +/- 0.0000 | 0.1053 +/- 0.0000 | 0.0426 +/- 0.0000 | 0.1053 +/- 0.0000 | 0.0426 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | bamboogle | 0.1040 +/- 0.0000 | 0.2122 +/- 0.0000 | 0.1040 +/- 0.0000 | 0.2122 +/- 0.0000 | 0.1040 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | hotpotqa | 0.1192 +/- 0.0000 | 0.2096 +/- 0.0000 | 0.1192 +/- 0.0000 | 0.2096 +/- 0.0000 | 0.1192 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | musique | 0.0249 +/- 0.0000 | 0.0760 +/- 0.0000 | 0.0249 +/- 0.0000 | 0.0760 +/- 0.0000 | 0.0249 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | nq | 0.1655 +/- 0.0000 | 0.2422 +/- 0.0000 | 0.1655 +/- 0.0000 | 0.2422 +/- 0.0000 | 0.1655 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | popqa | 0.2185 +/- 0.0000 | 0.2614 +/- 0.0000 | 0.2185 +/- 0.0000 | 0.2614 +/- 0.0000 | 0.2185 +/- 0.0000 |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | triviaqa | 0.1261 +/- 0.0000 | 0.2683 +/- 0.0000 | 0.1261 +/- 0.0000 | 0.2683 +/- 0.0000 | 0.1261 +/- 0.0000 |

## Paired Comparisons

| Comparison | Metric | Delta (right-left) | 95% CI |
|---|---|---:|---:|
| historical_search_r1_512_normtrue -> search_r1_512_normfalse_rep3 | em | -0.0074 | [-0.0149, 0.0000] |
| historical_search_r1_512_normtrue -> search_r1_512_normfalse_rep3 | f1 | -0.0069 | [-0.0150, 0.0012] |
| historical_search_r1_512_normtrue -> search_r1_512_normfalse_rep3 | structured_em | -0.0074 | [-0.0149, 0.0000] |
| historical_search_r1_512_normtrue -> search_r1_512_normfalse_rep3 | answer_group_f1 | -0.0069 | [-0.0150, 0.0012] |
| historical_search_r1_512_normtrue -> search_r1_512_normfalse_rep3 | answer_group_recall | -0.0074 | [-0.0149, 0.0000] |
| historical_search_r1_512_normtrue -> search_r1_512_normfalse_rep3 | valid_complete_answer_rate | -0.0380 | [-0.0546, -0.0217] |
| search_r1_512_normfalse_rep1 -> search_r1_512_normfalse_rep3 | em | -0.0166 | [-0.0243, -0.0091] |
| search_r1_512_normfalse_rep1 -> search_r1_512_normfalse_rep3 | f1 | -0.0211 | [-0.0295, -0.0129] |
| search_r1_512_normfalse_rep1 -> search_r1_512_normfalse_rep3 | structured_em | -0.0166 | [-0.0243, -0.0091] |
| search_r1_512_normfalse_rep1 -> search_r1_512_normfalse_rep3 | answer_group_f1 | -0.0211 | [-0.0295, -0.0129] |
| search_r1_512_normfalse_rep1 -> search_r1_512_normfalse_rep3 | answer_group_recall | -0.0166 | [-0.0243, -0.0091] |
| search_r1_512_normfalse_rep1 -> search_r1_512_normfalse_rep3 | valid_complete_answer_rate | -0.0729 | [-0.0897, -0.0563] |
| search_r1_512_normfalse_rep2 -> search_r1_512_normfalse_rep3 | em | 0.0089 | [0.0029, 0.0149] |
| search_r1_512_normfalse_rep2 -> search_r1_512_normfalse_rep3 | f1 | 0.0126 | [0.0063, 0.0188] |
| search_r1_512_normfalse_rep2 -> search_r1_512_normfalse_rep3 | structured_em | 0.0089 | [0.0029, 0.0149] |
| search_r1_512_normfalse_rep2 -> search_r1_512_normfalse_rep3 | answer_group_f1 | 0.0126 | [0.0063, 0.0188] |
| search_r1_512_normfalse_rep2 -> search_r1_512_normfalse_rep3 | answer_group_recall | 0.0089 | [0.0029, 0.0149] |
| search_r1_512_normfalse_rep2 -> search_r1_512_normfalse_rep3 | valid_complete_answer_rate | 0.0349 | [0.0211, 0.0480] |
| historical_spad_512_stable_inflight2_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | em | 0.0177 | [0.0103, 0.0254] |
| historical_spad_512_stable_inflight2_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | f1 | 0.0288 | [0.0209, 0.0366] |
| historical_spad_512_stable_inflight2_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | structured_em | 0.0177 | [0.0103, 0.0254] |
| historical_spad_512_stable_inflight2_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_f1 | 0.0288 | [0.0209, 0.0366] |
| historical_spad_512_stable_inflight2_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_recall | 0.0177 | [0.0103, 0.0254] |
| historical_spad_512_stable_inflight2_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | valid_complete_answer_rate | 0.0717 | [0.0566, 0.0871] |
| historical_spad_512_stable_inflight1_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | em | -0.0129 | [-0.0209, -0.0046] |
| historical_spad_512_stable_inflight1_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | f1 | -0.0180 | [-0.0263, -0.0094] |
| historical_spad_512_stable_inflight1_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | structured_em | -0.0129 | [-0.0209, -0.0046] |
| historical_spad_512_stable_inflight1_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_f1 | -0.0180 | [-0.0263, -0.0094] |
| historical_spad_512_stable_inflight1_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_recall | -0.0129 | [-0.0209, -0.0046] |
| historical_spad_512_stable_inflight1_normtrue -> spad_512_stable_inflight2_normfalse_rep3 | valid_complete_answer_rate | -0.0371 | [-0.0531, -0.0209] |
| spad_512_stable_inflight2_normfalse_rep1 -> spad_512_stable_inflight2_normfalse_rep3 | em | 0.0103 | [0.0029, 0.0177] |
| spad_512_stable_inflight2_normfalse_rep1 -> spad_512_stable_inflight2_normfalse_rep3 | f1 | 0.0182 | [0.0106, 0.0261] |
| spad_512_stable_inflight2_normfalse_rep1 -> spad_512_stable_inflight2_normfalse_rep3 | structured_em | 0.0103 | [0.0029, 0.0177] |
| spad_512_stable_inflight2_normfalse_rep1 -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_f1 | 0.0182 | [0.0106, 0.0261] |
| spad_512_stable_inflight2_normfalse_rep1 -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_recall | 0.0103 | [0.0029, 0.0177] |
| spad_512_stable_inflight2_normfalse_rep1 -> spad_512_stable_inflight2_normfalse_rep3 | valid_complete_answer_rate | 0.0326 | [0.0180, 0.0471] |
| spad_512_stable_inflight2_normfalse_rep2 -> spad_512_stable_inflight2_normfalse_rep3 | em | -0.0066 | [-0.0137, 0.0009] |
| spad_512_stable_inflight2_normfalse_rep2 -> spad_512_stable_inflight2_normfalse_rep3 | f1 | -0.0089 | [-0.0163, -0.0014] |
| spad_512_stable_inflight2_normfalse_rep2 -> spad_512_stable_inflight2_normfalse_rep3 | structured_em | -0.0066 | [-0.0137, 0.0009] |
| spad_512_stable_inflight2_normfalse_rep2 -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_f1 | -0.0089 | [-0.0163, -0.0014] |
| spad_512_stable_inflight2_normfalse_rep2 -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_recall | -0.0066 | [-0.0137, 0.0009] |
| spad_512_stable_inflight2_normfalse_rep2 -> spad_512_stable_inflight2_normfalse_rep3 | valid_complete_answer_rate | -0.0294 | [-0.0431, -0.0154] |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | em | -0.0074 | [-0.0157, 0.0009] |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | f1 | -0.0101 | [-0.0185, -0.0016] |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | structured_em | -0.0074 | [-0.0157, 0.0009] |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_f1 | -0.0101 | [-0.0185, -0.0016] |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_recall | -0.0074 | [-0.0157, 0.0009] |
| historical_spad_512_gold_token_f1_v1_inflight2_normtrue -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | valid_complete_answer_rate | 0.0140 | [-0.0014, 0.0294] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | em | -0.0017 | [-0.0103, 0.0069] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | f1 | -0.0088 | [-0.0181, 0.0006] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | structured_em | -0.0017 | [-0.0103, 0.0069] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_f1 | -0.0088 | [-0.0181, 0.0006] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_recall | -0.0017 | [-0.0103, 0.0069] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep1 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | valid_complete_answer_rate | -0.0126 | [-0.0303, 0.0051] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | em | -0.0129 | [-0.0214, -0.0046] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | f1 | -0.0166 | [-0.0255, -0.0078] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | structured_em | -0.0129 | [-0.0214, -0.0046] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_f1 | -0.0166 | [-0.0255, -0.0078] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_recall | -0.0129 | [-0.0214, -0.0046] |
| spad_512_gold_token_f1_v2_inflight2_normfalse_rep2 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | valid_complete_answer_rate | -0.0206 | [-0.0380, -0.0029] |
| search_r1_512_normfalse_rep3 -> spad_512_stable_inflight2_normfalse_rep3 | em | 0.0126 | [0.0040, 0.0211] |
| search_r1_512_normfalse_rep3 -> spad_512_stable_inflight2_normfalse_rep3 | f1 | 0.0189 | [0.0095, 0.0283] |
| search_r1_512_normfalse_rep3 -> spad_512_stable_inflight2_normfalse_rep3 | structured_em | 0.0126 | [0.0040, 0.0211] |
| search_r1_512_normfalse_rep3 -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_f1 | 0.0189 | [0.0095, 0.0283] |
| search_r1_512_normfalse_rep3 -> spad_512_stable_inflight2_normfalse_rep3 | answer_group_recall | 0.0126 | [0.0040, 0.0211] |
| search_r1_512_normfalse_rep3 -> spad_512_stable_inflight2_normfalse_rep3 | valid_complete_answer_rate | 0.0726 | [0.0537, 0.0914] |
| search_r1_512_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | em | 0.0051 | [-0.0040, 0.0143] |
| search_r1_512_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | f1 | 0.0048 | [-0.0048, 0.0148] |
| search_r1_512_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | structured_em | 0.0051 | [-0.0040, 0.0143] |
| search_r1_512_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_f1 | 0.0048 | [-0.0048, 0.0148] |
| search_r1_512_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_recall | 0.0051 | [-0.0040, 0.0143] |
| search_r1_512_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | valid_complete_answer_rate | 0.0469 | [0.0274, 0.0669] |
| spad_512_stable_inflight2_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | em | -0.0074 | [-0.0149, 0.0000] |
| spad_512_stable_inflight2_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | f1 | -0.0141 | [-0.0220, -0.0064] |
| spad_512_stable_inflight2_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | structured_em | -0.0074 | [-0.0149, 0.0000] |
| spad_512_stable_inflight2_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_f1 | -0.0141 | [-0.0220, -0.0064] |
| spad_512_stable_inflight2_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | answer_group_recall | -0.0074 | [-0.0149, 0.0000] |
| spad_512_stable_inflight2_normfalse_rep3 -> spad_512_gold_token_f1_v2_inflight2_normfalse_rep3 | valid_complete_answer_rate | -0.0257 | [-0.0406, -0.0106] |
