# Teacher Prompt Ablation: gold_support_question_tail_v3

- Family: `gold_aware_layout_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `40b75b4c9ba375e32fa3acfaac1a9ff722062700f27d5aaa614f6e9373dd15d8`
- Started: `2026-07-15T18:13:25.931545+08:00`
- Finished: `2026-07-15T18:13:57.219031+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7969 | 0.6771 | 1.0000 | 0.7949 | 0.8564 | 0.8245 | 0.8281 | 66 | 12 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7853 | 0.8245 | 0.7461 | 0.6961 | 144/181 | 0.9378 | 0.6804 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 144 | 33 | 4 | 0 |
| I | 24 | 155 | 2 | 0 |
| A | 8 | 7 | 7 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 40
- Missed I as S: 24
- Missed I as A: 2
- Missed I due to format error: 0

## Output Length

- Average reason characters: 110.5
- Average supported-answer characters: 14.0
- Maximum supported-answer characters: 172
- Average completion tokens: 46.0

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8188 | 0.8905 | 0.8531 | 0.5546 | 0.7039 |
| teacher_not_called_control | 163 | 0.7174 | 0.7500 | 0.7333 | 0.8534 | 0.7934 |
| L1_steps_01_20 | 96 | 0.7800 | 0.8478 | 0.8125 | 0.7274 | 0.7700 |
| L2_steps_21_40 | 96 | 0.8333 | 0.8889 | 0.8602 | 0.7539 | 0.8070 |
| L3_steps_41_60 | 96 | 0.8605 | 0.9250 | 0.8916 | 0.8209 | 0.8562 |
| L4_steps_61_79 | 96 | 0.7222 | 0.7800 | 0.7500 | 0.6683 | 0.7091 |
