# Teacher Prompt Ablation: gold_binary_support_evidence_only_v3

- Family: `gold_aware_instruction_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `9297986cdb9cfb3b558b934c2023d702fc831efbd16144ee6061010933dc97f6`
- Started: `2026-07-15T18:36:34.774336+08:00`
- Finished: `2026-07-15T18:37:14.330451+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8151 | 0.5602 | 1.0000 | 0.8000 | 0.8840 | 0.8399 | 0.8411 | 61 | 10 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8296 | 0.8399 | 0.8192 | 0.8011 | 153/181 | 0.9691 | 0.7206 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 153 | 28 | 0 | 0 |
| I | 20 | 160 | 1 | 0 |
| A | 10 | 12 | 0 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 40
- Missed I as S: 20
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 141.6
- Average supported-answer characters: 14.4
- Maximum supported-answer characters: 166
- Average completion tokens: 56.7

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.7914 | 0.9416 | 0.8600 | 0.5889 | 0.7244 |
| teacher_not_called_control | 163 | 0.8378 | 0.7045 | 0.7654 | 0.9483 | 0.8569 |
| L1_steps_01_20 | 96 | 0.7500 | 0.8478 | 0.7959 | 0.7423 | 0.7691 |
| L2_steps_21_40 | 96 | 0.8163 | 0.8889 | 0.8511 | 0.8295 | 0.8403 |
| L3_steps_41_60 | 96 | 0.8333 | 0.8750 | 0.8537 | 0.8944 | 0.8740 |
| L4_steps_61_79 | 96 | 0.8070 | 0.9200 | 0.8598 | 0.8049 | 0.8323 |
