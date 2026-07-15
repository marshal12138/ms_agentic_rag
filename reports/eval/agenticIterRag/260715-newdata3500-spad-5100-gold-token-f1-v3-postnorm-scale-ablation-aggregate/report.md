# New-Data Model Evaluation

- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Dataset SHA256: `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283`
- Each model has 1 isolated inference run(s). Repeats are not pooled as independent examples.
- Paired bootstrap averages repeats per question before resampling questions.

## Overall

| Model | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| search_r1_512 | 0.1180 +/- 0.0000 | 0.1965 +/- 0.0000 | 0.1180 +/- 0.0000 | 0.1965 +/- 0.0000 | 0.1180 +/- 0.0000 | 0.6271 +/- 0.0000 | 0.9831 +/- 0.0000 | 2.3489 +/- 0.0000 | 0.3640 +/- 0.0000 | 0.2569 +/- 0.0000 |
| search_r1_5100 | 0.1800 +/- 0.0000 | 0.2509 +/- 0.0000 | 0.1800 +/- 0.0000 | 0.2509 +/- 0.0000 | 0.1800 +/- 0.0000 | 0.7317 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.7291 +/- 0.0000 | 0.1786 +/- 0.0000 | 0.1549 +/- 0.0000 |
| spad_5100_stable_normtrue | 0.1923 +/- 0.0000 | 0.2700 +/- 0.0000 | 0.1923 +/- 0.0000 | 0.2700 +/- 0.0000 | 0.1923 +/- 0.0000 | 0.7220 +/- 0.0000 | 0.9397 +/- 0.0000 | 2.6557 +/- 0.0000 | 0.5906 +/- 0.0000 | 0.2443 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | 0.1837 +/- 0.0000 | 0.2576 +/- 0.0000 | 0.1837 +/- 0.0000 | 0.2576 +/- 0.0000 | 0.1837 +/- 0.0000 | 0.6334 +/- 0.0000 | 0.9971 +/- 0.0000 | 3.0071 +/- 0.0000 | 0.5763 +/- 0.0000 | 0.3589 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | 0.1831 +/- 0.0000 | 0.2673 +/- 0.0000 | 0.1831 +/- 0.0000 | 0.2673 +/- 0.0000 | 0.1831 +/- 0.0000 | 0.7906 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.8889 +/- 0.0000 | 0.2154 +/- 0.0000 | 0.1863 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | 0.1994 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.1994 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.1994 +/- 0.0000 | 0.8340 +/- 0.0000 | 0.9971 +/- 0.0000 | 1.6969 +/- 0.0000 | 0.1571 +/- 0.0000 | 0.1369 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | 0.1929 +/- 0.0000 | 0.2734 +/- 0.0000 | 0.1929 +/- 0.0000 | 0.2734 +/- 0.0000 | 0.1929 +/- 0.0000 | 0.7100 +/- 0.0000 | 0.9926 +/- 0.0000 | 2.6883 +/- 0.0000 | 0.5649 +/- 0.0000 | 0.2714 +/- 0.0000 |

## Per Run

| Model | Repeat | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| search_r1_512 | 1 | 0.1180 | 0.1965 | 0.1180 | 0.1965 | 0.1180 | 0.6271 | 0.9831 | 2.3489 | 0.3640 | 0.2569 |
| search_r1_5100 | 1 | 0.1800 | 0.2509 | 0.1800 | 0.2509 | 0.1800 | 0.7317 | 1.0000 | 1.7291 | 0.1786 | 0.1549 |
| spad_5100_stable_normtrue | 1 | 0.1923 | 0.2700 | 0.1923 | 0.2700 | 0.1923 | 0.7220 | 0.9397 | 2.6557 | 0.5906 | 0.2443 |
| spad_5100_gold_token_f1_v1_normtrue | 1 | 0.1837 | 0.2576 | 0.1837 | 0.2576 | 0.1837 | 0.6334 | 0.9971 | 3.0071 | 0.5763 | 0.3589 |
| spad_5100_gold_token_f1_v2_normfalse | 1 | 0.1831 | 0.2673 | 0.1831 | 0.2673 | 0.1831 | 0.7906 | 1.0000 | 1.8889 | 0.2154 | 0.1863 |
| spad_5100_gold_token_f1_v3_postnorm01 | 1 | 0.1994 | 0.2787 | 0.1994 | 0.2787 | 0.1994 | 0.8340 | 0.9971 | 1.6969 | 0.1571 | 0.1369 |
| spad_5100_gold_token_f1_v3_postnorm03 | 1 | 0.1929 | 0.2734 | 0.1929 | 0.2734 | 0.1929 | 0.7100 | 0.9926 | 2.6883 | 0.5649 | 0.2714 |

## Search Count Buckets

| Model | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| search_r1_512 | 0.0169 +/- 0.0000 | 0.4580 +/- 0.0000 | 0.2137 +/- 0.0000 | 0.0403 +/- 0.0000 | 0.0131 +/- 0.0000 | 0.2580 +/- 0.0000 |
| search_r1_5100 | 0.0000 +/- 0.0000 | 0.7469 +/- 0.0000 | 0.0886 +/- 0.0000 | 0.0080 +/- 0.0000 | 0.0017 +/- 0.0000 | 0.1549 +/- 0.0000 |
| spad_5100_stable_normtrue | 0.0603 +/- 0.0000 | 0.0314 +/- 0.0000 | 0.6071 +/- 0.0000 | 0.0403 +/- 0.0000 | 0.0151 +/- 0.0000 | 0.2457 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | 0.0029 +/- 0.0000 | 0.1414 +/- 0.0000 | 0.4331 +/- 0.0000 | 0.0511 +/- 0.0000 | 0.0111 +/- 0.0000 | 0.3603 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | 0.0000 +/- 0.0000 | 0.6849 +/- 0.0000 | 0.1154 +/- 0.0000 | 0.0120 +/- 0.0000 | 0.0014 +/- 0.0000 | 0.1863 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | 0.0029 +/- 0.0000 | 0.7294 +/- 0.0000 | 0.1126 +/- 0.0000 | 0.0154 +/- 0.0000 | 0.0026 +/- 0.0000 | 0.1371 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | 0.0074 +/- 0.0000 | 0.1871 +/- 0.0000 | 0.4763 +/- 0.0000 | 0.0406 +/- 0.0000 | 0.0160 +/- 0.0000 | 0.2726 +/- 0.0000 |

## Per Data Source

| Model | Data source | EM | F1 | Structured EM | Group F1 | Group recall |
|---|---|---:|---:|---:|---:|---:|
| search_r1_512 | 2wikimultihopqa | 0.0284 +/- 0.0000 | 0.0924 +/- 0.0000 | 0.0284 +/- 0.0000 | 0.0924 +/- 0.0000 | 0.0284 +/- 0.0000 |
| search_r1_512 | bamboogle | 0.0880 +/- 0.0000 | 0.1636 +/- 0.0000 | 0.0880 +/- 0.0000 | 0.1636 +/- 0.0000 | 0.0880 +/- 0.0000 |
| search_r1_512 | hotpotqa | 0.1103 +/- 0.0000 | 0.2051 +/- 0.0000 | 0.1103 +/- 0.0000 | 0.2051 +/- 0.0000 | 0.1103 +/- 0.0000 |
| search_r1_512 | musique | 0.0214 +/- 0.0000 | 0.0716 +/- 0.0000 | 0.0214 +/- 0.0000 | 0.0716 +/- 0.0000 | 0.0214 +/- 0.0000 |
| search_r1_512 | nq | 0.1779 +/- 0.0000 | 0.2521 +/- 0.0000 | 0.1779 +/- 0.0000 | 0.2521 +/- 0.0000 | 0.1779 +/- 0.0000 |
| search_r1_512 | popqa | 0.2380 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.2380 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.2380 +/- 0.0000 |
| search_r1_512 | triviaqa | 0.1385 +/- 0.0000 | 0.2865 +/- 0.0000 | 0.1385 +/- 0.0000 | 0.2865 +/- 0.0000 | 0.1385 +/- 0.0000 |
| search_r1_5100 | 2wikimultihopqa | 0.1172 +/- 0.0000 | 0.1623 +/- 0.0000 | 0.1172 +/- 0.0000 | 0.1623 +/- 0.0000 | 0.1172 +/- 0.0000 |
| search_r1_5100 | bamboogle | 0.1280 +/- 0.0000 | 0.2065 +/- 0.0000 | 0.1280 +/- 0.0000 | 0.2065 +/- 0.0000 | 0.1280 +/- 0.0000 |
| search_r1_5100 | hotpotqa | 0.1851 +/- 0.0000 | 0.2633 +/- 0.0000 | 0.1851 +/- 0.0000 | 0.2633 +/- 0.0000 | 0.1851 +/- 0.0000 |
| search_r1_5100 | musique | 0.0356 +/- 0.0000 | 0.0818 +/- 0.0000 | 0.0356 +/- 0.0000 | 0.0818 +/- 0.0000 | 0.0356 +/- 0.0000 |
| search_r1_5100 | nq | 0.2651 +/- 0.0000 | 0.3342 +/- 0.0000 | 0.2651 +/- 0.0000 | 0.3342 +/- 0.0000 | 0.2651 +/- 0.0000 |
| search_r1_5100 | popqa | 0.3375 +/- 0.0000 | 0.3826 +/- 0.0000 | 0.3375 +/- 0.0000 | 0.3826 +/- 0.0000 | 0.3375 +/- 0.0000 |
| search_r1_5100 | triviaqa | 0.1510 +/- 0.0000 | 0.2910 +/- 0.0000 | 0.1510 +/- 0.0000 | 0.2910 +/- 0.0000 | 0.1510 +/- 0.0000 |
| spad_5100_stable_normtrue | 2wikimultihopqa | 0.1083 +/- 0.0000 | 0.1497 +/- 0.0000 | 0.1083 +/- 0.0000 | 0.1497 +/- 0.0000 | 0.1083 +/- 0.0000 |
| spad_5100_stable_normtrue | bamboogle | 0.1760 +/- 0.0000 | 0.2701 +/- 0.0000 | 0.1760 +/- 0.0000 | 0.2701 +/- 0.0000 | 0.1760 +/- 0.0000 |
| spad_5100_stable_normtrue | hotpotqa | 0.2456 +/- 0.0000 | 0.3265 +/- 0.0000 | 0.2456 +/- 0.0000 | 0.3265 +/- 0.0000 | 0.2456 +/- 0.0000 |
| spad_5100_stable_normtrue | musique | 0.0463 +/- 0.0000 | 0.0886 +/- 0.0000 | 0.0463 +/- 0.0000 | 0.0886 +/- 0.0000 | 0.0463 +/- 0.0000 |
| spad_5100_stable_normtrue | nq | 0.2865 +/- 0.0000 | 0.3637 +/- 0.0000 | 0.2865 +/- 0.0000 | 0.3637 +/- 0.0000 | 0.2865 +/- 0.0000 |
| spad_5100_stable_normtrue | popqa | 0.2824 +/- 0.0000 | 0.3262 +/- 0.0000 | 0.2824 +/- 0.0000 | 0.3262 +/- 0.0000 | 0.2824 +/- 0.0000 |
| spad_5100_stable_normtrue | triviaqa | 0.1883 +/- 0.0000 | 0.3654 +/- 0.0000 | 0.1883 +/- 0.0000 | 0.3654 +/- 0.0000 | 0.1883 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | 2wikimultihopqa | 0.1261 +/- 0.0000 | 0.1731 +/- 0.0000 | 0.1261 +/- 0.0000 | 0.1731 +/- 0.0000 | 0.1261 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | bamboogle | 0.2160 +/- 0.0000 | 0.2807 +/- 0.0000 | 0.2160 +/- 0.0000 | 0.2807 +/- 0.0000 | 0.2160 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | hotpotqa | 0.2313 +/- 0.0000 | 0.3187 +/- 0.0000 | 0.2313 +/- 0.0000 | 0.3187 +/- 0.0000 | 0.2313 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | musique | 0.0463 +/- 0.0000 | 0.0962 +/- 0.0000 | 0.0463 +/- 0.0000 | 0.0962 +/- 0.0000 | 0.0463 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | nq | 0.2527 +/- 0.0000 | 0.3287 +/- 0.0000 | 0.2527 +/- 0.0000 | 0.3287 +/- 0.0000 | 0.2527 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | popqa | 0.2735 +/- 0.0000 | 0.3088 +/- 0.0000 | 0.2735 +/- 0.0000 | 0.3088 +/- 0.0000 | 0.2735 +/- 0.0000 |
| spad_5100_gold_token_f1_v1_normtrue | triviaqa | 0.1652 +/- 0.0000 | 0.3147 +/- 0.0000 | 0.1652 +/- 0.0000 | 0.3147 +/- 0.0000 | 0.1652 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | 2wikimultihopqa | 0.1279 +/- 0.0000 | 0.1792 +/- 0.0000 | 0.1279 +/- 0.0000 | 0.1792 +/- 0.0000 | 0.1279 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | bamboogle | 0.0880 +/- 0.0000 | 0.1732 +/- 0.0000 | 0.0880 +/- 0.0000 | 0.1732 +/- 0.0000 | 0.0880 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | hotpotqa | 0.1886 +/- 0.0000 | 0.2839 +/- 0.0000 | 0.1886 +/- 0.0000 | 0.2839 +/- 0.0000 | 0.1886 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | musique | 0.0320 +/- 0.0000 | 0.0825 +/- 0.0000 | 0.0320 +/- 0.0000 | 0.0825 +/- 0.0000 | 0.0320 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | nq | 0.3025 +/- 0.0000 | 0.3958 +/- 0.0000 | 0.3025 +/- 0.0000 | 0.3958 +/- 0.0000 | 0.3025 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | popqa | 0.3055 +/- 0.0000 | 0.3527 +/- 0.0000 | 0.3055 +/- 0.0000 | 0.3527 +/- 0.0000 | 0.3055 +/- 0.0000 |
| spad_5100_gold_token_f1_v2_normfalse | triviaqa | 0.1634 +/- 0.0000 | 0.3305 +/- 0.0000 | 0.1634 +/- 0.0000 | 0.3305 +/- 0.0000 | 0.1634 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | 2wikimultihopqa | 0.1545 +/- 0.0000 | 0.1948 +/- 0.0000 | 0.1545 +/- 0.0000 | 0.1948 +/- 0.0000 | 0.1545 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | bamboogle | 0.0800 +/- 0.0000 | 0.1721 +/- 0.0000 | 0.0800 +/- 0.0000 | 0.1721 +/- 0.0000 | 0.0800 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | hotpotqa | 0.2189 +/- 0.0000 | 0.3062 +/- 0.0000 | 0.2189 +/- 0.0000 | 0.3062 +/- 0.0000 | 0.2189 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | musique | 0.0356 +/- 0.0000 | 0.0807 +/- 0.0000 | 0.0356 +/- 0.0000 | 0.0807 +/- 0.0000 | 0.0356 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | nq | 0.3060 +/- 0.0000 | 0.3870 +/- 0.0000 | 0.3060 +/- 0.0000 | 0.3870 +/- 0.0000 | 0.3060 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | popqa | 0.3357 +/- 0.0000 | 0.3722 +/- 0.0000 | 0.3357 +/- 0.0000 | 0.3722 +/- 0.0000 | 0.3357 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | triviaqa | 0.1723 +/- 0.0000 | 0.3547 +/- 0.0000 | 0.1723 +/- 0.0000 | 0.3547 +/- 0.0000 | 0.1723 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | 2wikimultihopqa | 0.1119 +/- 0.0000 | 0.1618 +/- 0.0000 | 0.1119 +/- 0.0000 | 0.1618 +/- 0.0000 | 0.1119 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | bamboogle | 0.2160 +/- 0.0000 | 0.3031 +/- 0.0000 | 0.2160 +/- 0.0000 | 0.3031 +/- 0.0000 | 0.2160 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | hotpotqa | 0.2242 +/- 0.0000 | 0.3089 +/- 0.0000 | 0.2242 +/- 0.0000 | 0.3089 +/- 0.0000 | 0.2242 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | musique | 0.0569 +/- 0.0000 | 0.1124 +/- 0.0000 | 0.0569 +/- 0.0000 | 0.1124 +/- 0.0000 | 0.0569 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | nq | 0.2954 +/- 0.0000 | 0.3825 +/- 0.0000 | 0.2954 +/- 0.0000 | 0.3825 +/- 0.0000 | 0.2954 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | popqa | 0.2860 +/- 0.0000 | 0.3202 +/- 0.0000 | 0.2860 +/- 0.0000 | 0.3202 +/- 0.0000 | 0.2860 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm03 | triviaqa | 0.1776 +/- 0.0000 | 0.3478 +/- 0.0000 | 0.1776 +/- 0.0000 | 0.3478 +/- 0.0000 | 0.1776 +/- 0.0000 |

## Paired Comparisons

| Comparison | Metric | Delta (right-left) | 95% CI |
|---|---|---:|---:|
| search_r1_512 -> spad_5100_gold_token_f1_v3_postnorm03 | em | 0.0749 | [0.0637, 0.0863] |
| search_r1_512 -> spad_5100_gold_token_f1_v3_postnorm03 | f1 | 0.0768 | [0.0657, 0.0884] |
| search_r1_512 -> spad_5100_gold_token_f1_v3_postnorm03 | structured_em | 0.0749 | [0.0637, 0.0863] |
| search_r1_512 -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_f1 | 0.0768 | [0.0657, 0.0884] |
| search_r1_512 -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_recall | 0.0749 | [0.0637, 0.0863] |
| search_r1_512 -> spad_5100_gold_token_f1_v3_postnorm03 | valid_complete_answer_rate | 0.0829 | [0.0649, 0.1014] |
| search_r1_5100 -> spad_5100_gold_token_f1_v3_postnorm03 | em | 0.0129 | [0.0011, 0.0243] |
| search_r1_5100 -> spad_5100_gold_token_f1_v3_postnorm03 | f1 | 0.0224 | [0.0103, 0.0348] |
| search_r1_5100 -> spad_5100_gold_token_f1_v3_postnorm03 | structured_em | 0.0129 | [0.0011, 0.0243] |
| search_r1_5100 -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_f1 | 0.0224 | [0.0103, 0.0348] |
| search_r1_5100 -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_recall | 0.0129 | [0.0011, 0.0243] |
| search_r1_5100 -> spad_5100_gold_token_f1_v3_postnorm03 | valid_complete_answer_rate | -0.0217 | [-0.0403, -0.0031] |
| spad_5100_stable_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | em | 0.0006 | [-0.0091, 0.0103] |
| spad_5100_stable_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | f1 | 0.0033 | [-0.0068, 0.0136] |
| spad_5100_stable_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | structured_em | 0.0006 | [-0.0091, 0.0103] |
| spad_5100_stable_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_f1 | 0.0033 | [-0.0068, 0.0136] |
| spad_5100_stable_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_recall | 0.0006 | [-0.0091, 0.0103] |
| spad_5100_stable_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | valid_complete_answer_rate | -0.0120 | [-0.0274, 0.0034] |
| spad_5100_gold_token_f1_v1_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | em | 0.0091 | [-0.0009, 0.0189] |
| spad_5100_gold_token_f1_v1_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | f1 | 0.0158 | [0.0053, 0.0262] |
| spad_5100_gold_token_f1_v1_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | structured_em | 0.0091 | [-0.0009, 0.0189] |
| spad_5100_gold_token_f1_v1_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_f1 | 0.0158 | [0.0053, 0.0262] |
| spad_5100_gold_token_f1_v1_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_recall | 0.0091 | [-0.0009, 0.0189] |
| spad_5100_gold_token_f1_v1_normtrue -> spad_5100_gold_token_f1_v3_postnorm03 | valid_complete_answer_rate | 0.0766 | [0.0603, 0.0929] |
| spad_5100_gold_token_f1_v2_normfalse -> spad_5100_gold_token_f1_v3_postnorm03 | em | 0.0097 | [-0.0009, 0.0206] |
| spad_5100_gold_token_f1_v2_normfalse -> spad_5100_gold_token_f1_v3_postnorm03 | f1 | 0.0061 | [-0.0054, 0.0176] |
| spad_5100_gold_token_f1_v2_normfalse -> spad_5100_gold_token_f1_v3_postnorm03 | structured_em | 0.0097 | [-0.0009, 0.0206] |
| spad_5100_gold_token_f1_v2_normfalse -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_f1 | 0.0061 | [-0.0054, 0.0176] |
| spad_5100_gold_token_f1_v2_normfalse -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_recall | 0.0097 | [-0.0009, 0.0206] |
| spad_5100_gold_token_f1_v2_normfalse -> spad_5100_gold_token_f1_v3_postnorm03 | valid_complete_answer_rate | -0.0806 | [-0.0971, -0.0643] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm03 | em | -0.0066 | [-0.0171, 0.0043] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm03 | f1 | -0.0053 | [-0.0167, 0.0062] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm03 | structured_em | -0.0066 | [-0.0171, 0.0043] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_f1 | -0.0053 | [-0.0167, 0.0062] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm03 | answer_group_recall | -0.0066 | [-0.0171, 0.0043] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm03 | valid_complete_answer_rate | -0.1240 | [-0.1403, -0.1080] |
