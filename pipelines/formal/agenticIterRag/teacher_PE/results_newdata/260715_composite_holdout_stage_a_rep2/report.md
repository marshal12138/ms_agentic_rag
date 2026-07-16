# Teacher Prompt Ablation: baseline_current_v2

- Family: `instruction_only`
- Evaluated split: `holdout`
- Cases: 128
- Prompt SHA256: `d3cca6c54a41e93bf30aaf1b6a4946a59c014c805bd7c280777e740efed09a70`
- Started: `2026-07-15T22:47:04.679572+08:00`
- Finished: `2026-07-15T22:47:18.354555+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8672 | 0.8085 | 1.0000 | 0.8947 | 0.8500 | 0.8718 | 0.8828 | 15 | 2 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8311 | 0.8718 | 0.7904 | 0.7000 | 55/60 | 0.8622 | 0.8686 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 55 | 3 | 2 | 0 |
| I | 9 | 51 | 0 | 0 |
| A | 0 | 3 | 5 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 6
- Missed I as S: 9
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 265.2
- Average supported-answer characters: 12.7
- Maximum supported-answer characters: 40
- Average completion tokens: 80.1

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 72 | 0.9767 | 0.8235 | 0.8936 | 0.4615 | 0.6775 |
| teacher_not_called_control | 56 | 0.6429 | 1.0000 | 0.7826 | 0.9000 | 0.8413 |
| L1_steps_01_20 | 32 | 0.9333 | 0.9333 | 0.9333 | 0.7396 | 0.8365 |
| L2_steps_21_40 | 32 | 1.0000 | 0.8000 | 0.8889 | 0.8929 | 0.8909 |
| L3_steps_41_60 | 32 | 0.8462 | 0.7857 | 0.8148 | 0.8806 | 0.8477 |
| L4_steps_61_79 | 32 | 0.8235 | 0.8750 | 0.8485 | 0.6429 | 0.7457 |
