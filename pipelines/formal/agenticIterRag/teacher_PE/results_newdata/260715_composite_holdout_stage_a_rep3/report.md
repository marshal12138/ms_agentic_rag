# Teacher Prompt Ablation: baseline_current_v2

- Family: `instruction_only`
- Evaluated split: `holdout`
- Cases: 128
- Prompt SHA256: `d3cca6c54a41e93bf30aaf1b6a4946a59c014c805bd7c280777e740efed09a70`
- Started: `2026-07-15T22:47:52.542827+08:00`
- Finished: `2026-07-15T22:48:08.502442+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8438 | 0.7921 | 1.0000 | 0.8500 | 0.8500 | 0.8500 | 0.8594 | 18 | 2 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7966 | 0.8500 | 0.7431 | 0.6500 | 52/60 | 0.8575 | 0.8214 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 52 | 6 | 2 | 0 |
| I | 9 | 51 | 0 | 0 |
| A | 0 | 3 | 5 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 9
- Missed I as S: 9
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 267.4
- Average supported-answer characters: 13.0
- Maximum supported-answer characters: 40
- Average completion tokens: 80.7

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 72 | 0.9333 | 0.8235 | 0.8750 | 0.3615 | 0.6182 |
| teacher_not_called_control | 56 | 0.6000 | 1.0000 | 0.7500 | 0.8704 | 0.8102 |
| L1_steps_01_20 | 32 | 0.9333 | 0.9333 | 0.9333 | 0.7396 | 0.8365 |
| L2_steps_21_40 | 32 | 0.9231 | 0.8000 | 0.8571 | 0.8571 | 0.8571 |
| L3_steps_41_60 | 32 | 0.7857 | 0.7857 | 0.7857 | 0.7972 | 0.7915 |
| L4_steps_61_79 | 32 | 0.7778 | 0.8750 | 0.8235 | 0.5714 | 0.6975 |
