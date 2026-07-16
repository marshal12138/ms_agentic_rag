# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_stage_a_prod_rep2_dev`
- Stage-B calls: 200/384 (0.5208)
- Mean elapsed ratio versus Stage A: 1.3560
- Within 2x budget: `true`
# Teacher Prompt Ablation: hard_gate_r5_literal_canonical_v2

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `e20b489fade128fd28ce54bf341148561d8eb7e157febc064598bf73beace815`
- Started: `2026-07-15T22:50:11.030621+08:00`
- Finished: `2026-07-15T22:50:29.066479+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8333 | 0.6775 | 1.0000 | 0.8587 | 0.8729 | 0.8658 | 0.8724 | 49 | 15 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8428 | 0.8658 | 0.8199 | 0.7956 | 157/181 | 0.9453 | 0.7159 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 157 | 21 | 3 | 0 |
| I | 22 | 158 | 1 | 0 |
| A | 12 | 5 | 5 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 26
- Missed I as S: 22
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 216.8
- Average supported-answer characters: 15.4
- Maximum supported-answer characters: 172
- Average completion tokens: 101.0

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8905 | 0.8905 | 0.8905 | 0.6986 | 0.7946 |
| teacher_not_called_control | 163 | 0.7660 | 0.8182 | 0.7912 | 0.8879 | 0.8396 |
| L1_steps_01_20 | 96 | 0.8913 | 0.8913 | 0.8913 | 0.8468 | 0.8691 |
| L2_steps_21_40 | 96 | 0.8085 | 0.8444 | 0.8261 | 0.8140 | 0.8200 |
| L3_steps_41_60 | 96 | 0.8974 | 0.8750 | 0.8861 | 0.8762 | 0.8811 |
| L4_steps_61_79 | 96 | 0.8462 | 0.8800 | 0.8627 | 0.7268 | 0.7948 |
