# Teacher Prompt Ablation: gold_support_evidence_only_v3

- Family: `gold_aware_layout_ablation`
- Evaluated split: `holdout`
- Cases: 128
- Prompt SHA256: `64bc90b958296a033f5832ed922d1edce3e99775f6bb6578639c52415b5ae783`
- Started: `2026-07-15T19:01:56.834243+08:00`
- Finished: `2026-07-15T19:02:06.592181+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8203 | 0.6746 | 1.0000 | 0.9231 | 0.8000 | 0.8571 | 0.8750 | 16 | 7 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8690 | 0.8571 | 0.8810 | 0.8333 | 55/60 | 0.9610 | 0.8182 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 55 | 3 | 2 | 0 |
| I | 12 | 48 | 0 | 0 |
| A | 5 | 1 | 2 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 4
- Missed I as S: 12
- Missed I as A: 0
- Missed I due to format error: 0

## Output Length

- Average reason characters: 115.5
- Average supported-answer characters: 13.5
- Maximum supported-answer characters: 66
- Average completion tokens: 47.0

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 72 | 0.9318 | 0.8039 | 0.8632 | 0.7444 | 0.8038 |
| teacher_not_called_control | 56 | 0.8750 | 0.7778 | 0.8235 | 0.9265 | 0.8750 |
| L1_steps_01_20 | 32 | 1.0000 | 0.8000 | 0.8889 | 0.9375 | 0.9132 |
| L2_steps_21_40 | 32 | 0.8667 | 0.8667 | 0.8667 | 0.8214 | 0.8440 |
| L3_steps_41_60 | 32 | 0.9091 | 0.7143 | 0.8000 | 0.8854 | 0.8427 |
| L4_steps_61_79 | 32 | 0.9286 | 0.8125 | 0.8667 | 0.8707 | 0.8687 |
