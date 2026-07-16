# Teacher Prompt Ablation: gold_binary_support_evidence_only_v3

- Family: `gold_aware_instruction_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `9297986cdb9cfb3b558b934c2023d702fc831efbd16144ee6061010933dc97f6`
- Started: `2026-07-15T18:57:40.145072+08:00`
- Finished: `2026-07-15T18:58:18.119859+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8047 | 0.5522 | 1.0000 | 0.7980 | 0.8729 | 0.8338 | 0.8359 | 63 | 12 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8189 | 0.8338 | 0.8040 | 0.7790 | 151/181 | 0.9637 | 0.7222 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 151 | 30 | 0 | 0 |
| I | 23 | 158 | 0 | 0 |
| A | 12 | 10 | 0 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 40
- Missed I as S: 23
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 139.9
- Average supported-answer characters: 14.5
- Maximum supported-answer characters: 166
- Average completion tokens: 56.2

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.7937 | 0.9270 | 0.8552 | 0.5618 | 0.7085 |
| teacher_not_called_control | 163 | 0.8158 | 0.7045 | 0.7561 | 0.9397 | 0.8479 |
| L1_steps_01_20 | 96 | 0.7451 | 0.8261 | 0.7835 | 0.7393 | 0.7614 |
| L2_steps_21_40 | 96 | 0.8085 | 0.8444 | 0.8261 | 0.8062 | 0.8161 |
| L3_steps_41_60 | 96 | 0.8750 | 0.8750 | 0.8750 | 0.9194 | 0.8972 |
| L4_steps_61_79 | 96 | 0.7833 | 0.9400 | 0.8545 | 0.7350 | 0.7948 |
