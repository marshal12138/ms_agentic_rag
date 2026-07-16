# New-Data Model Evaluation

- Dataset: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/global_train_eval_data/3500e/co_search_ablation.eval.parquet`
- Dataset SHA256: `bc628ed38bc3a99d7ba0ee6056a179c25cc78fcfe818b10a9233ead0256f0283`
- Each model has 1 isolated inference run(s). Repeats are not pooled as independent examples.
- Paired bootstrap averages repeats per question before resampling questions.

## Overall

| Model | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spad_5100_gold_token_f1_v3_postnorm01 | 0.1994 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.1994 +/- 0.0000 | 0.2787 +/- 0.0000 | 0.1994 +/- 0.0000 | 0.8340 +/- 0.0000 | 0.9971 +/- 0.0000 | 1.6969 +/- 0.0000 | 0.1571 +/- 0.0000 | 0.1369 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | 0.2069 +/- 0.0000 | 0.2911 +/- 0.0000 | 0.2069 +/- 0.0000 | 0.2911 +/- 0.0000 | 0.2069 +/- 0.0000 | 0.8611 +/- 0.0000 | 1.0000 +/- 0.0000 | 1.5934 +/- 0.0000 | 0.1331 +/- 0.0000 | 0.1071 +/- 0.0000 |

## Per Run

| Model | Repeat | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spad_5100_gold_token_f1_v3_postnorm01 | 1 | 0.1994 | 0.2787 | 0.1994 | 0.2787 | 0.1994 | 0.8340 | 0.9971 | 1.6969 | 0.1571 | 0.1369 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | 1 | 0.2069 | 0.2911 | 0.2069 | 0.2911 | 0.2069 | 0.8611 | 1.0000 | 1.5934 | 0.1331 | 0.1071 |

## Search Count Buckets

| Model | 0 | 1 | 2 | 3 | 4 | 5+ |
|---|---:|---:|---:|---:|---:|---:|
| spad_5100_gold_token_f1_v3_postnorm01 | 0.0029 +/- 0.0000 | 0.7294 +/- 0.0000 | 0.1126 +/- 0.0000 | 0.0154 +/- 0.0000 | 0.0026 +/- 0.0000 | 0.1371 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | 0.0000 +/- 0.0000 | 0.7391 +/- 0.0000 | 0.1429 +/- 0.0000 | 0.0106 +/- 0.0000 | 0.0003 +/- 0.0000 | 0.1071 +/- 0.0000 |

## Per Data Source

| Model | Data source | EM | F1 | Structured EM | Group F1 | Group recall |
|---|---|---:|---:|---:|---:|---:|
| spad_5100_gold_token_f1_v3_postnorm01 | 2wikimultihopqa | 0.1545 +/- 0.0000 | 0.1948 +/- 0.0000 | 0.1545 +/- 0.0000 | 0.1948 +/- 0.0000 | 0.1545 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | bamboogle | 0.0800 +/- 0.0000 | 0.1721 +/- 0.0000 | 0.0800 +/- 0.0000 | 0.1721 +/- 0.0000 | 0.0800 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | hotpotqa | 0.2189 +/- 0.0000 | 0.3062 +/- 0.0000 | 0.2189 +/- 0.0000 | 0.3062 +/- 0.0000 | 0.2189 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | musique | 0.0356 +/- 0.0000 | 0.0807 +/- 0.0000 | 0.0356 +/- 0.0000 | 0.0807 +/- 0.0000 | 0.0356 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | nq | 0.3060 +/- 0.0000 | 0.3870 +/- 0.0000 | 0.3060 +/- 0.0000 | 0.3870 +/- 0.0000 | 0.3060 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | popqa | 0.3357 +/- 0.0000 | 0.3722 +/- 0.0000 | 0.3357 +/- 0.0000 | 0.3722 +/- 0.0000 | 0.3357 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01 | triviaqa | 0.1723 +/- 0.0000 | 0.3547 +/- 0.0000 | 0.1723 +/- 0.0000 | 0.3547 +/- 0.0000 | 0.1723 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | 2wikimultihopqa | 0.1350 +/- 0.0000 | 0.1814 +/- 0.0000 | 0.1350 +/- 0.0000 | 0.1814 +/- 0.0000 | 0.1350 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | bamboogle | 0.1440 +/- 0.0000 | 0.2244 +/- 0.0000 | 0.1440 +/- 0.0000 | 0.2244 +/- 0.0000 | 0.1440 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | hotpotqa | 0.1993 +/- 0.0000 | 0.2895 +/- 0.0000 | 0.1993 +/- 0.0000 | 0.2895 +/- 0.0000 | 0.1993 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | musique | 0.0409 +/- 0.0000 | 0.0948 +/- 0.0000 | 0.0409 +/- 0.0000 | 0.0948 +/- 0.0000 | 0.0409 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | nq | 0.3114 +/- 0.0000 | 0.3987 +/- 0.0000 | 0.3114 +/- 0.0000 | 0.3987 +/- 0.0000 | 0.3114 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | popqa | 0.3908 +/- 0.0000 | 0.4382 +/- 0.0000 | 0.3908 +/- 0.0000 | 0.4382 +/- 0.0000 | 0.3908 +/- 0.0000 |
| spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | triviaqa | 0.1776 +/- 0.0000 | 0.3584 +/- 0.0000 | 0.1776 +/- 0.0000 | 0.3584 +/- 0.0000 | 0.1776 +/- 0.0000 |

## Paired Comparisons

| Comparison | Metric | Delta (right-left) | 95% CI |
|---|---|---:|---:|
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | em | 0.0074 | [-0.0020, 0.0169] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | f1 | 0.0124 | [0.0025, 0.0222] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | structured_em | 0.0074 | [-0.0020, 0.0169] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | answer_group_f1 | 0.0124 | [0.0025, 0.0222] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | answer_group_recall | 0.0074 | [-0.0020, 0.0169] |
| spad_5100_gold_token_f1_v3_postnorm01 -> spad_5100_gold_token_f1_v3_postnorm01_hardgatev2 | valid_complete_answer_rate | 0.0271 | [0.0143, 0.0400] |
