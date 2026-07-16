# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_stage_a_prod_dev`
- Stage-B calls: 200/384 (0.5208)
- Mean elapsed ratio versus Stage A: 1.5019
- Within 2x budget: `true`
# Teacher Prompt Ablation: hard_gate_gold_draft_selector_v1

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `fce47e990eb543582a06c5ddde02f1319dd3a17de4dfad01840cdae4b08a3ed0`
- Started: `2026-07-15T22:26:45.341593+08:00`
- Finished: `2026-07-15T22:27:11.085083+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8177 | 0.7223 | 1.0000 | 0.8424 | 0.8564 | 0.8493 | 0.8568 | 55 | 15 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7506 | 0.8493 | 0.6519 | 0.5746 | 150/181 | 0.7866 | 0.7325 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 150 | 28 | 3 | 0 |
| I | 24 | 155 | 2 | 0 |
| A | 12 | 1 | 9 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 29
- Missed I as S: 24
- Missed I as A: 2
- Missed I due to format error: 0

## Output Length

- Average reason characters: 333.0
- Average supported-answer characters: 16.3
- Maximum supported-answer characters: 120
- Average completion tokens: 130.8

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9091 | 0.8759 | 0.8922 | 0.3852 | 0.6387 |
| teacher_not_called_control | 163 | 0.6731 | 0.7955 | 0.7292 | 0.8013 | 0.7652 |
| L1_steps_01_20 | 96 | 0.8864 | 0.8478 | 0.8667 | 0.6500 | 0.7583 |
| L2_steps_21_40 | 96 | 0.8723 | 0.9111 | 0.8913 | 0.6772 | 0.7843 |
| L3_steps_41_60 | 96 | 0.8462 | 0.8250 | 0.8354 | 0.7193 | 0.7774 |
| L4_steps_61_79 | 96 | 0.7778 | 0.8400 | 0.8077 | 0.5452 | 0.6765 |
