# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_stage_a_prod_rep3_dev`
- Stage-B calls: 384/384 (1.0000)
- Mean elapsed ratio versus Stage A: 1.8208
- Within 2x budget: `true`
# Teacher Prompt Ablation: dual_all_r5_gold_f1_08_override_v1

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `fa41902a1946a5508754d89f275486b1bf2310de29779a0a5380e4da94473a93`
- Started: `2026-07-15T22:35:15.240682+08:00`
- Finished: `2026-07-15T22:35:49.937422+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8203 | 0.6491 | 1.0000 | 0.9068 | 0.8066 | 0.8538 | 0.8698 | 50 | 19 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8361 | 0.8538 | 0.8184 | 0.7680 | 165/181 | 0.8978 | 0.7810 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 165 | 13 | 3 | 0 |
| I | 33 | 146 | 2 | 0 |
| A | 16 | 2 | 4 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 15
- Missed I as S: 33
- Missed I as A: 2
- Missed I due to format error: 0

## Output Length

- Average reason characters: 205.2
- Average supported-answer characters: 14.9
- Maximum supported-answer characters: 172
- Average completion tokens: 126.2

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9206 | 0.8467 | 0.8821 | 0.6206 | 0.7514 |
| teacher_not_called_control | 163 | 0.8571 | 0.6818 | 0.7595 | 0.9293 | 0.8444 |
| L1_steps_01_20 | 96 | 0.9231 | 0.7826 | 0.8471 | 0.8201 | 0.8336 |
| L2_steps_21_40 | 96 | 0.9250 | 0.8222 | 0.8706 | 0.8151 | 0.8429 |
| L3_steps_41_60 | 96 | 0.9211 | 0.8750 | 0.8974 | 0.8762 | 0.8868 |
| L4_steps_61_79 | 96 | 0.8636 | 0.7600 | 0.8085 | 0.7496 | 0.7791 |
