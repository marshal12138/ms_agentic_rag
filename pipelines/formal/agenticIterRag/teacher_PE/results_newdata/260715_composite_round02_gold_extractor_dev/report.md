# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_stage_a_prod_dev`
- Stage-B calls: 200/384 (0.5208)
- Mean elapsed ratio versus Stage A: 1.4146
- Within 2x budget: `true`
# Teacher Prompt Ablation: hard_gate_gold_extractor_v1

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `71ac001078dbc78dfe0970c2d964e0c0308b148722a091c21b725d94953cd8aa`
- Started: `2026-07-15T22:26:09.334885+08:00`
- Finished: `2026-07-15T22:26:36.477923+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8125 | 0.7020 | 1.0000 | 0.8424 | 0.8564 | 0.8493 | 0.8568 | 55 | 17 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7567 | 0.8493 | 0.6641 | 0.6077 | 148/181 | 0.8122 | 0.7092 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 148 | 28 | 5 | 0 |
| I | 21 | 155 | 5 | 0 |
| A | 12 | 1 | 9 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 29
- Missed I as S: 21
- Missed I as A: 5
- Missed I due to format error: 0

## Output Length

- Average reason characters: 244.5
- Average supported-answer characters: 15.6
- Maximum supported-answer characters: 120
- Average completion tokens: 112.2

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9091 | 0.8759 | 0.8922 | 0.4587 | 0.6755 |
| teacher_not_called_control | 163 | 0.6731 | 0.7955 | 0.7292 | 0.7792 | 0.7542 |
| L1_steps_01_20 | 96 | 0.8864 | 0.8478 | 0.8667 | 0.6665 | 0.7666 |
| L2_steps_21_40 | 96 | 0.8723 | 0.9111 | 0.8913 | 0.6709 | 0.7811 |
| L3_steps_41_60 | 96 | 0.8462 | 0.8250 | 0.8354 | 0.7390 | 0.7872 |
| L4_steps_61_79 | 96 | 0.7778 | 0.8400 | 0.8077 | 0.5631 | 0.6854 |
