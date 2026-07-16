# Teacher Prompt Ablation: gold_support_evidence_only_v3

- Family: `gold_aware_layout_ablation`
- Evaluated split: `holdout`
- Cases: 128
- Prompt SHA256: `64bc90b958296a033f5832ed922d1edce3e99775f6bb6578639c52415b5ae783`
- Started: `2026-07-15T19:05:05.799719+08:00`
- Finished: `2026-07-15T19:05:14.785898+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8125 | 0.6216 | 1.0000 | 0.8909 | 0.8167 | 0.8522 | 0.8672 | 17 | 7 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8638 | 0.8522 | 0.8754 | 0.8333 | 54/60 | 0.9727 | 0.8126 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 54 | 4 | 2 | 0 |
| I | 11 | 49 | 0 | 0 |
| A | 5 | 2 | 1 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 6
- Missed I as S: 11
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 116.7
- Average supported-answer characters: 13.3
- Maximum supported-answer characters: 66
- Average completion tokens: 47.6

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 72 | 0.9111 | 0.8039 | 0.8542 | 0.7444 | 0.7993 |
| teacher_not_called_control | 56 | 0.8000 | 0.8889 | 0.8421 | 0.9190 | 0.8806 |
| L1_steps_01_20 | 32 | 1.0000 | 0.8000 | 0.8889 | 0.9375 | 0.9132 |
| L2_steps_21_40 | 32 | 0.8125 | 0.8667 | 0.8387 | 0.8214 | 0.8301 |
| L3_steps_41_60 | 32 | 0.9091 | 0.7143 | 0.8000 | 0.8854 | 0.8427 |
| L4_steps_61_79 | 32 | 0.8750 | 0.8750 | 0.8750 | 0.8469 | 0.8610 |
