# Teacher Prompt Ablation: baseline_current_v2

- Family: `instruction_only`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `d3cca6c54a41e93bf30aaf1b6a4946a59c014c805bd7c280777e740efed09a70`
- Started: `2026-07-15T22:24:17.361160+08:00`
- Finished: `2026-07-15T22:25:05.196031+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8099 | 0.7190 | 1.0000 | 0.8424 | 0.8564 | 0.8493 | 0.8568 | 55 | 18 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7302 | 0.8493 | 0.6111 | 0.5193 | 145/181 | 0.7628 | 0.7058 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 145 | 28 | 8 | 0 |
| I | 23 | 155 | 3 | 0 |
| A | 10 | 1 | 11 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 29
- Missed I as S: 23
- Missed I as A: 3
- Missed I due to format error: 0

## Output Length

- Average reason characters: 266.7
- Average supported-answer characters: 16.8
- Maximum supported-answer characters: 120
- Average completion tokens: 79.6

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9091 | 0.8759 | 0.8922 | 0.3386 | 0.6154 |
| teacher_not_called_control | 163 | 0.6731 | 0.7955 | 0.7292 | 0.7637 | 0.7464 |
| L1_steps_01_20 | 96 | 0.8864 | 0.8478 | 0.8667 | 0.6500 | 0.7583 |
| L2_steps_21_40 | 96 | 0.8723 | 0.9111 | 0.8913 | 0.6090 | 0.7502 |
| L3_steps_41_60 | 96 | 0.8462 | 0.8250 | 0.8354 | 0.6502 | 0.7428 |
| L4_steps_61_79 | 96 | 0.7778 | 0.8400 | 0.8077 | 0.5208 | 0.6643 |
