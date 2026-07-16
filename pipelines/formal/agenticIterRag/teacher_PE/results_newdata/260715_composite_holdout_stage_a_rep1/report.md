# Teacher Prompt Ablation: baseline_current_v2

- Family: `instruction_only`
- Evaluated split: `holdout`
- Cases: 128
- Prompt SHA256: `d3cca6c54a41e93bf30aaf1b6a4946a59c014c805bd7c280777e740efed09a70`
- Started: `2026-07-15T22:46:08.852220+08:00`
- Finished: `2026-07-15T22:46:22.696282+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8359 | 0.7580 | 1.0000 | 0.8500 | 0.8500 | 0.8500 | 0.8594 | 18 | 3 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8038 | 0.8500 | 0.7576 | 0.6667 | 52/60 | 0.8741 | 0.8081 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 52 | 6 | 2 | 0 |
| I | 9 | 51 | 0 | 0 |
| A | 1 | 3 | 4 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 9
- Missed I as S: 9
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 254.9
- Average supported-answer characters: 13.0
- Maximum supported-answer characters: 40
- Average completion tokens: 77.7

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 72 | 0.9333 | 0.8235 | 0.8750 | 0.4193 | 0.6471 |
| teacher_not_called_control | 56 | 0.6000 | 1.0000 | 0.7500 | 0.8704 | 0.8102 |
| L1_steps_01_20 | 32 | 0.9286 | 0.8667 | 0.8966 | 0.7396 | 0.8181 |
| L2_steps_21_40 | 32 | 0.9286 | 0.8667 | 0.8966 | 0.8571 | 0.8768 |
| L3_steps_41_60 | 32 | 0.7857 | 0.7857 | 0.7857 | 0.8514 | 0.8186 |
| L4_steps_61_79 | 32 | 0.7778 | 0.8750 | 0.8235 | 0.5714 | 0.6975 |
