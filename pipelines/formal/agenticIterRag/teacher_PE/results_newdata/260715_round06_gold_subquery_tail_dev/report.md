# Teacher Prompt Ablation: gold_support_subquery_question_tail_v3

- Family: `gold_aware_layout_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `6c75e327383d92233933183151f7451fb91dc99101eb316f8d6590b91463f397`
- Started: `2026-07-15T18:24:08.930597+08:00`
- Finished: `2026-07-15T18:24:45.994943+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7917 | 0.6251 | 1.0000 | 0.8152 | 0.8287 | 0.8219 | 0.8307 | 65 | 15 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8023 | 0.8219 | 0.7828 | 0.7403 | 150/181 | 0.9445 | 0.7177 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 150 | 27 | 4 | 0 |
| I | 28 | 150 | 3 | 0 |
| A | 11 | 7 | 4 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 34
- Missed I as S: 28
- Missed I as A: 3
- Missed I due to format error: 0

## Output Length

- Average reason characters: 112.2
- Average supported-answer characters: 14.1
- Maximum supported-answer characters: 172
- Average completion tokens: 46.1

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8138 | 0.8613 | 0.8369 | 0.5566 | 0.6968 |
| teacher_not_called_control | 163 | 0.8205 | 0.7273 | 0.7711 | 0.9095 | 0.8403 |
| L1_steps_01_20 | 96 | 0.7308 | 0.8261 | 0.7755 | 0.7204 | 0.7479 |
| L2_steps_21_40 | 96 | 0.8605 | 0.8222 | 0.8409 | 0.7701 | 0.8055 |
| L3_steps_41_60 | 96 | 0.8750 | 0.8750 | 0.8750 | 0.8462 | 0.8606 |
| L4_steps_61_79 | 96 | 0.8163 | 0.8000 | 0.8081 | 0.7902 | 0.7992 |
