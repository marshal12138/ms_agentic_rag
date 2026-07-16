# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_holdout_stage_a_rep3`
- Stage-B calls: 68/128 (0.5312)
- Mean elapsed ratio versus Stage A: 1.2254
- Within 2x budget: `true`
# Teacher Prompt Ablation: hard_gate_r5_literal_canonical_v2

- Family: `hard_gate_composite`
- Evaluated split: `holdout`
- Cases: 128
- Prompt SHA256: `e20b489fade128fd28ce54bf341148561d8eb7e157febc064598bf73beace815`
- Started: `2026-07-15T22:52:03.448136+08:00`
- Finished: `2026-07-15T22:52:07.705989+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8281 | 0.6409 | 1.0000 | 0.8500 | 0.8500 | 0.8500 | 0.8594 | 18 | 4 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8750 | 0.8500 | 0.9000 | 0.9000 | 54/60 | 1.0000 | 0.7678 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 54 | 6 | 0 | 0 |
| I | 9 | 51 | 0 | 0 |
| A | 4 | 3 | 1 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 9
- Missed I as S: 9
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 211.6
- Average supported-answer characters: 12.6
- Maximum supported-answer characters: 50
- Average completion tokens: 104.4

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 72 | 0.9333 | 0.8235 | 0.8750 | 0.8667 | 0.8708 |
| teacher_not_called_control | 56 | 0.6000 | 1.0000 | 0.7500 | 0.9111 | 0.8306 |
| L1_steps_01_20 | 32 | 0.9333 | 0.9333 | 0.9333 | 0.9375 | 0.9354 |
| L2_steps_21_40 | 32 | 0.9231 | 0.8000 | 0.8571 | 0.9286 | 0.8929 |
| L3_steps_41_60 | 32 | 0.7857 | 0.7857 | 0.7857 | 0.9375 | 0.8616 |
| L4_steps_61_79 | 32 | 0.7778 | 0.8750 | 0.8235 | 0.7857 | 0.8046 |
