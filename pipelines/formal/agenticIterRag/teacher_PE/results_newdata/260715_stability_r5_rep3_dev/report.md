# Teacher Prompt Ablation: gold_support_evidence_only_v3

- Family: `gold_aware_layout_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `64bc90b958296a033f5832ed922d1edce3e99775f6bb6578639c52415b5ae783`
- Started: `2026-07-15T18:49:22.781678+08:00`
- Finished: `2026-07-15T18:49:58.059968+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8125 | 0.6249 | 1.0000 | 0.8352 | 0.8398 | 0.8375 | 0.8464 | 59 | 13 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8303 | 0.8375 | 0.8230 | 0.7680 | 157/181 | 0.9489 | 0.7323 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 157 | 21 | 3 | 0 |
| I | 27 | 152 | 2 | 0 |
| A | 10 | 9 | 3 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 30
- Missed I as S: 27
- Missed I as A: 2
- Missed I due to format error: 0

## Output Length

- Average reason characters: 115.0
- Average supported-answer characters: 14.7
- Maximum supported-answer characters: 172
- Average completion tokens: 46.8

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8356 | 0.8905 | 0.8622 | 0.6411 | 0.7516 |
| teacher_not_called_control | 163 | 0.8333 | 0.6818 | 0.7500 | 0.9250 | 0.8375 |
| L1_steps_01_20 | 96 | 0.8043 | 0.8043 | 0.8043 | 0.8095 | 0.8069 |
| L2_steps_21_40 | 96 | 0.8409 | 0.8222 | 0.8315 | 0.7841 | 0.8078 |
| L3_steps_41_60 | 96 | 0.8750 | 0.8750 | 0.8750 | 0.8962 | 0.8856 |
| L4_steps_61_79 | 96 | 0.8269 | 0.8600 | 0.8431 | 0.7902 | 0.8167 |
