# Teacher Prompt Ablation: gold_support_evidence_only_v3

- Family: `gold_aware_layout_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `64bc90b958296a033f5832ed922d1edce3e99775f6bb6578639c52415b5ae783`
- Started: `2026-07-15T18:45:50.729618+08:00`
- Finished: `2026-07-15T18:46:28.545197+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8229 | 0.6549 | 1.0000 | 0.8298 | 0.8619 | 0.8455 | 0.8516 | 57 | 11 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8323 | 0.8455 | 0.8190 | 0.7680 | 156/181 | 0.9503 | 0.7298 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 156 | 23 | 2 | 0 |
| I | 24 | 156 | 1 | 0 |
| A | 9 | 9 | 4 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 32
- Missed I as S: 24
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 116.5
- Average supported-answer characters: 14.8
- Maximum supported-answer characters: 172
- Average completion tokens: 47.1

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8322 | 0.9051 | 0.8671 | 0.6376 | 0.7524 |
| teacher_not_called_control | 163 | 0.8205 | 0.7273 | 0.7711 | 0.9207 | 0.8459 |
| L1_steps_01_20 | 96 | 0.8261 | 0.8261 | 0.8261 | 0.8244 | 0.8252 |
| L2_steps_21_40 | 96 | 0.8043 | 0.8222 | 0.8132 | 0.7841 | 0.7986 |
| L3_steps_41_60 | 96 | 0.8780 | 0.9000 | 0.8889 | 0.8762 | 0.8825 |
| L4_steps_61_79 | 96 | 0.8182 | 0.9000 | 0.8571 | 0.7798 | 0.8185 |
