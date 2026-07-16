# Teacher Prompt Ablation: question_tail_answer_alignment_v3

- Family: `answer_alignment_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `a1bc6d41ac5d320cdc846c891e4686a5f5ccabfa393616365403c4a4e1b95d63`
- Started: `2026-07-15T18:04:10.471746+08:00`
- Finished: `2026-07-15T18:04:51.048571+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8177 | 0.7311 | 1.0000 | 0.8466 | 0.8840 | 0.8649 | 0.8698 | 50 | 20 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7342 | 0.8649 | 0.6035 | 0.5304 | 142/181 | 0.7693 | 0.6995 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 142 | 28 | 11 | 0 |
| I | 20 | 160 | 1 | 0 |
| A | 9 | 1 | 12 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 29
- Missed I as S: 20
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 267.7
- Average supported-answer characters: 16.1
- Maximum supported-answer characters: 117
- Average completion tokens: 80.1

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8963 | 0.8832 | 0.8897 | 0.2785 | 0.5841 |
| teacher_not_called_control | 163 | 0.7222 | 0.8864 | 0.7959 | 0.7856 | 0.7908 |
| L1_steps_01_20 | 96 | 0.8723 | 0.8913 | 0.8817 | 0.6008 | 0.7413 |
| L2_steps_21_40 | 96 | 0.8636 | 0.8444 | 0.8539 | 0.6191 | 0.7365 |
| L3_steps_41_60 | 96 | 0.9231 | 0.9000 | 0.9114 | 0.6722 | 0.7918 |
| L4_steps_61_79 | 96 | 0.7627 | 0.9000 | 0.8257 | 0.5065 | 0.6661 |
