# Teacher Prompt Ablation: gold_support_check

- Family: `gold_aware`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `ed12218224699c1dffaf5bf5a4f93a40e03ac00c0489adb7000d5c636b983d0c`
- Started: `2026-07-15T18:16:38.114068+08:00`
- Finished: `2026-07-15T18:17:14.412549+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8021 | 0.6381 | 1.0000 | 0.8187 | 0.8232 | 0.8209 | 0.8307 | 65 | 11 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8200 | 0.8209 | 0.8192 | 0.7680 | 155/181 | 0.9566 | 0.7296 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 155 | 24 | 2 | 0 |
| I | 30 | 149 | 2 | 0 |
| A | 9 | 9 | 4 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 33
- Missed I as S: 30
- Missed I as A: 2
- Missed I due to format error: 0

## Output Length

- Average reason characters: 117.3
- Average supported-answer characters: 14.3
- Maximum supported-answer characters: 172
- Average completion tokens: 47.2

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8163 | 0.8759 | 0.8451 | 0.6303 | 0.7377 |
| teacher_not_called_control | 163 | 0.8286 | 0.6591 | 0.7342 | 0.9250 | 0.8296 |
| L1_steps_01_20 | 96 | 0.7959 | 0.8478 | 0.8211 | 0.7776 | 0.7993 |
| L2_steps_21_40 | 96 | 0.8780 | 0.8000 | 0.8372 | 0.8701 | 0.8536 |
| L3_steps_41_60 | 96 | 0.8537 | 0.8750 | 0.8642 | 0.8782 | 0.8712 |
| L4_steps_61_79 | 96 | 0.7647 | 0.7800 | 0.7723 | 0.7415 | 0.7569 |
