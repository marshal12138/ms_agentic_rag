# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_stage_a_prod_dev`
- Stage-B calls: 200/384 (0.5208)
- Mean elapsed ratio versus Stage A: 1.3524
- Within 2x budget: `true`
# Teacher Prompt Ablation: hard_gate_r5_v1

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `8f2ae04c6f3ba3003360f2b65a0f34401268b5cf210d4496e75f38f746ee9230`
- Started: `2026-07-15T22:25:30.146877+08:00`
- Finished: `2026-07-15T22:25:47.659277+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8177 | 0.7269 | 1.0000 | 0.8424 | 0.8564 | 0.8493 | 0.8568 | 55 | 15 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7907 | 0.8493 | 0.7321 | 0.6851 | 149/181 | 0.8893 | 0.7036 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 149 | 28 | 4 | 0 |
| I | 23 | 155 | 3 | 0 |
| A | 11 | 1 | 10 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 29
- Missed I as S: 23
- Missed I as A: 3
- Missed I due to format error: 0

## Output Length

- Average reason characters: 222.6
- Average supported-answer characters: 15.5
- Maximum supported-answer characters: 172
- Average completion tokens: 102.7

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9091 | 0.8759 | 0.8922 | 0.5770 | 0.7346 |
| teacher_not_called_control | 163 | 0.6731 | 0.7955 | 0.7292 | 0.8190 | 0.7741 |
| L1_steps_01_20 | 96 | 0.8864 | 0.8478 | 0.8667 | 0.7882 | 0.8274 |
| L2_steps_21_40 | 96 | 0.8723 | 0.9111 | 0.8913 | 0.7190 | 0.8051 |
| L3_steps_41_60 | 96 | 0.8462 | 0.8250 | 0.8354 | 0.7962 | 0.8158 |
| L4_steps_61_79 | 96 | 0.7778 | 0.8400 | 0.8077 | 0.6033 | 0.7055 |
