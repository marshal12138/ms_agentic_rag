# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_holdout_stage_a_rep2`
- Stage-B calls: 128/128 (1.0000)
- Mean elapsed ratio versus Stage A: 1.6950
- Within 2x budget: `true`
# Teacher Prompt Ablation: dual_all_r5_gold_f1_08_literal_canonical_v2

- Family: `hard_gate_composite`
- Evaluated split: `holdout`
- Cases: 128
- Prompt SHA256: `1ba800b5b2f207614f82891cf5d7e6fa30a111ea8a077882115431fd29af0731`
- Started: `2026-07-15T22:47:33.697541+08:00`
- Finished: `2026-07-15T22:47:43.089562+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8125 | 0.6853 | 1.0000 | 0.9556 | 0.7167 | 0.8190 | 0.8516 | 19 | 5 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8970 | 0.8190 | 0.9750 | 0.9667 | 59/60 | 0.9915 | 0.8511 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 59 | 1 | 0 | 0 |
| I | 17 | 43 | 0 | 0 |
| A | 5 | 1 | 2 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 2
- Missed I as S: 17
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 183.6
- Average supported-answer characters: 13.0
- Maximum supported-answer characters: 66
- Average completion tokens: 127.5

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 72 | 0.9730 | 0.7059 | 0.8182 | 0.9667 | 0.8924 |
| teacher_not_called_control | 56 | 0.8750 | 0.7778 | 0.8235 | 0.9778 | 0.9007 |
| L1_steps_01_20 | 32 | 1.0000 | 0.7333 | 0.8462 | 1.0000 | 0.9231 |
| L2_steps_21_40 | 32 | 1.0000 | 0.7333 | 0.8462 | 0.9643 | 0.9052 |
| L3_steps_41_60 | 32 | 0.8889 | 0.5714 | 0.6957 | 1.0000 | 0.8478 |
| L4_steps_61_79 | 32 | 0.9286 | 0.8125 | 0.8667 | 0.9286 | 0.8976 |
