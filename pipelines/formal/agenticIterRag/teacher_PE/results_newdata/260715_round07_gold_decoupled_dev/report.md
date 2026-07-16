# Teacher Prompt Ablation: gold_decoupled_status_answer_v3

- Family: `gold_aware_instruction_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `1e71c7d3ff99ea7437b3c92fe15f84dd737e52eebde4cfe24fc07a0621479ff8`
- Started: `2026-07-15T18:29:57.060084+08:00`
- Finished: `2026-07-15T18:30:39.906706+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7266 | 0.6035 | 0.9974 | 0.8860 | 0.5580 | 0.6847 | 0.7578 | 93 | 12 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7864 | 0.6847 | 0.8880 | 0.8232 | 173/181 | 0.9291 | 0.8043 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 173 | 8 | 0 | 0 |
| I | 77 | 101 | 2 | 1 |
| A | 12 | 5 | 5 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 13
- Missed I as S: 77
- Missed I as A: 2
- Missed I due to format error: 1

## Output Length

- Average reason characters: 122.8
- Average supported-answer characters: 15.3
- Maximum supported-answer characters: 197
- Average completion tokens: 50.0

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8763 | 0.6204 | 0.7265 | 0.7327 | 0.7296 |
| teacher_not_called_control | 163 | 0.9412 | 0.3636 | 0.5246 | 0.9750 | 0.7498 |
| L1_steps_01_20 | 96 | 0.8846 | 0.5000 | 0.6389 | 0.9182 | 0.7786 |
| L2_steps_21_40 | 96 | 0.8710 | 0.6000 | 0.7105 | 0.8461 | 0.7783 |
| L3_steps_41_60 | 96 | 0.9231 | 0.6000 | 0.7273 | 0.9195 | 0.8234 |
| L4_steps_61_79 | 96 | 0.8710 | 0.5400 | 0.6667 | 0.8589 | 0.7628 |
