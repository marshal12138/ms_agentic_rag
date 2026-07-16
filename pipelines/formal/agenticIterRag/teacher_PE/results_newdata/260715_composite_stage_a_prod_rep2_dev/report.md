# Teacher Prompt Ablation: baseline_current_v2

- Family: `instruction_only`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `d3cca6c54a41e93bf30aaf1b6a4946a59c014c805bd7c280777e740efed09a70`
- Started: `2026-07-15T22:32:49.878788+08:00`
- Finished: `2026-07-15T22:33:33.080832+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8333 | 0.7132 | 1.0000 | 0.8587 | 0.8729 | 0.8658 | 0.8724 | 49 | 15 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7523 | 0.8658 | 0.6388 | 0.5525 | 154/181 | 0.7508 | 0.7437 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 154 | 21 | 6 | 0 |
| I | 22 | 158 | 1 | 0 |
| A | 9 | 5 | 8 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 26
- Missed I as S: 22
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 259.9
- Average supported-answer characters: 16.8
- Maximum supported-answer characters: 114
- Average completion tokens: 77.9

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8905 | 0.8905 | 0.8905 | 0.2975 | 0.5940 |
| teacher_not_called_control | 163 | 0.7660 | 0.8182 | 0.7912 | 0.8301 | 0.8107 |
| L1_steps_01_20 | 96 | 0.8913 | 0.8913 | 0.8913 | 0.6467 | 0.7690 |
| L2_steps_21_40 | 96 | 0.8085 | 0.8444 | 0.8261 | 0.6447 | 0.7354 |
| L3_steps_41_60 | 96 | 0.8974 | 0.8750 | 0.8861 | 0.6632 | 0.7746 |
| L4_steps_61_79 | 96 | 0.8462 | 0.8800 | 0.8627 | 0.5940 | 0.7284 |
