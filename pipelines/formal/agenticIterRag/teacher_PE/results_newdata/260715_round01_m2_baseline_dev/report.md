# Teacher Prompt Ablation: baseline_question_tail_evidence_only_v2

- Family: `layout_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `d27ed640aa7b94e8931763fa5a39053846922973f8c09a9436ea976e8573d311`
- Started: `2026-07-15T18:00:02.313053+08:00`
- Finished: `2026-07-15T18:01:18.054234+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8229 | 0.7231 | 0.9974 | 0.8342 | 0.8895 | 0.8610 | 0.8646 | 52 | 16 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7277 | 0.8610 | 0.5944 | 0.5249 | 144/181 | 0.7471 | 0.6845 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 144 | 28 | 9 | 0 |
| I | 15 | 161 | 4 | 1 |
| A | 7 | 4 | 11 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 32
- Missed I as S: 15
- Missed I as A: 4
- Missed I due to format error: 1

## Output Length

- Average reason characters: 285.2
- Average supported-answer characters: 16.5
- Maximum supported-answer characters: 162
- Average completion tokens: 85.4

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8841 | 0.8905 | 0.8873 | 0.2951 | 0.5912 |
| teacher_not_called_control | 163 | 0.7091 | 0.8864 | 0.7879 | 0.7621 | 0.7750 |
| L1_steps_01_20 | 96 | 0.8511 | 0.8696 | 0.8602 | 0.6020 | 0.7311 |
| L2_steps_21_40 | 96 | 0.8125 | 0.8667 | 0.8387 | 0.6016 | 0.7202 |
| L3_steps_41_60 | 96 | 0.8605 | 0.9250 | 0.8916 | 0.6164 | 0.7540 |
| L4_steps_61_79 | 96 | 0.8182 | 0.9000 | 0.8571 | 0.5512 | 0.7042 |
