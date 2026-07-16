# Teacher Prompt Ablation: gold_compact_balanced_v3

- Family: `gold_aware_instruction_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `58e88907c8567ef1431ee9e2899b488f0600bce226fb9cc48cd4430c35703ace`
- Started: `2026-07-15T18:39:51.369535+08:00`
- Finished: `2026-07-15T18:40:28.327984+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7682 | 0.6235 | 0.9688 | 0.8434 | 0.7735 | 0.8069 | 0.8125 | 67 | 17 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.7828 | 0.8069 | 0.7587 | 0.7182 | 151/181 | 0.9094 | 0.7057 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 151 | 22 | 4 | 4 |
| I | 34 | 140 | 0 | 7 |
| A | 13 | 4 | 4 | 1 |

## I Error Breakdown

- False I (manual S/A -> I): 26
- Missed I as S: 34
- Missed I as A: 0
- Missed I due to format error: 7

## Output Length

- Average reason characters: 97.3
- Average supported-answer characters: 13.7
- Maximum supported-answer characters: 151
- Average completion tokens: 43.7

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8433 | 0.8248 | 0.8339 | 0.5381 | 0.6860 |
| teacher_not_called_control | 163 | 0.8438 | 0.6136 | 0.7105 | 0.8823 | 0.7964 |
| L1_steps_01_20 | 96 | 0.8571 | 0.7826 | 0.8182 | 0.7606 | 0.7894 |
| L2_steps_21_40 | 96 | 0.8095 | 0.7556 | 0.7816 | 0.7488 | 0.7652 |
| L3_steps_41_60 | 96 | 0.9677 | 0.7500 | 0.8451 | 0.7956 | 0.8203 |
| L4_steps_61_79 | 96 | 0.7843 | 0.8000 | 0.7921 | 0.7220 | 0.7570 |
