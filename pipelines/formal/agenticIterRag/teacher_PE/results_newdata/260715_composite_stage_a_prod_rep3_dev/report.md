# Teacher Prompt Ablation: baseline_current_v2

- Family: `instruction_only`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `d3cca6c54a41e93bf30aaf1b6a4946a59c014c805bd7c280777e740efed09a70`
- Started: `2026-07-15T22:34:24.737187+08:00`
- Finished: `2026-07-15T22:35:07.439398+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8307 | 0.7040 | 1.0000 | 0.8548 | 0.8785 | 0.8665 | 0.8724 | 49 | 16 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7483 | 0.8665 | 0.6301 | 0.5414 | 153/181 | 0.7454 | 0.7228 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 153 | 24 | 4 | 0 |
| I | 21 | 159 | 1 | 0 |
| A | 12 | 3 | 7 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 27
- Missed I as S: 21
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 265.5
- Average supported-answer characters: 16.5
- Maximum supported-answer characters: 107
- Average completion tokens: 79.3

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8913 | 0.8978 | 0.8945 | 0.3177 | 0.6061 |
| teacher_not_called_control | 163 | 0.7500 | 0.8182 | 0.7826 | 0.8051 | 0.7938 |
| L1_steps_01_20 | 96 | 0.9091 | 0.8696 | 0.8889 | 0.6865 | 0.7877 |
| L2_steps_21_40 | 96 | 0.8667 | 0.8667 | 0.8667 | 0.6400 | 0.7533 |
| L3_steps_41_60 | 96 | 0.8222 | 0.9250 | 0.8706 | 0.6140 | 0.7423 |
| L4_steps_61_79 | 96 | 0.8269 | 0.8600 | 0.8431 | 0.5745 | 0.7088 |
