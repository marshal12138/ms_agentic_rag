# Teacher Prompt Ablation: gold_support_evidence_only_v3

- Family: `gold_aware_layout_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `64bc90b958296a033f5832ed922d1edce3e99775f6bb6578639c52415b5ae783`
- Started: `2026-07-15T18:20:45.663930+08:00`
- Finished: `2026-07-15T18:21:20.043766+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8177 | 0.6132 | 1.0000 | 0.8081 | 0.8840 | 0.8443 | 0.8464 | 59 | 11 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8240 | 0.8443 | 0.8037 | 0.7514 | 152/181 | 0.9570 | 0.7019 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 152 | 28 | 1 | 0 |
| I | 21 | 160 | 0 | 0 |
| A | 10 | 10 | 2 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 38
- Missed I as S: 21
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 115.1
- Average supported-answer characters: 14.9
- Maximum supported-answer characters: 172
- Average completion tokens: 46.8

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8182 | 0.9197 | 0.8660 | 0.6411 | 0.7535 |
| teacher_not_called_control | 163 | 0.7727 | 0.7727 | 0.7727 | 0.8948 | 0.8338 |
| L1_steps_01_20 | 96 | 0.7959 | 0.8478 | 0.8211 | 0.8201 | 0.8206 |
| L2_steps_21_40 | 96 | 0.8085 | 0.8444 | 0.8261 | 0.7841 | 0.8051 |
| L3_steps_41_60 | 96 | 0.8444 | 0.9500 | 0.8941 | 0.8562 | 0.8751 |
| L4_steps_61_79 | 96 | 0.7895 | 0.9000 | 0.8411 | 0.7415 | 0.7913 |
