# Teacher Prompt Ablation: gold_i_guard_evidence_only_v3

- Family: `gold_aware_instruction_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `e517f22e20afeacb52bc7926d67b5edd9351d0172d946d8bf889687657361bda`
- Started: `2026-07-15T18:32:53.792084+08:00`
- Finished: `2026-07-15T18:33:36.333339+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7188 | 0.5367 | 1.0000 | 0.8473 | 0.6133 | 0.7115 | 0.7656 | 90 | 18 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7719 | 0.7115 | 0.8322 | 0.8066 | 163/181 | 0.9241 | 0.7646 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 163 | 14 | 4 | 0 |
| I | 68 | 111 | 2 | 0 |
| A | 14 | 6 | 2 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 20
- Missed I as S: 68
- Missed I as A: 2
- Missed I due to format error: 0

## Output Length

- Average reason characters: 194.7
- Average supported-answer characters: 15.0
- Maximum supported-answer characters: 166
- Average completion tokens: 66.5

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8571 | 0.6569 | 0.7438 | 0.6636 | 0.7037 |
| teacher_not_called_control | 163 | 0.8077 | 0.4773 | 0.6000 | 0.9267 | 0.7634 |
| L1_steps_01_20 | 96 | 0.8824 | 0.6522 | 0.7500 | 0.7766 | 0.7633 |
| L2_steps_21_40 | 96 | 0.8750 | 0.6222 | 0.7273 | 0.8844 | 0.8058 |
| L3_steps_41_60 | 96 | 0.7857 | 0.5500 | 0.6471 | 0.8994 | 0.7732 |
| L4_steps_61_79 | 96 | 0.8378 | 0.6200 | 0.7126 | 0.7593 | 0.7360 |
