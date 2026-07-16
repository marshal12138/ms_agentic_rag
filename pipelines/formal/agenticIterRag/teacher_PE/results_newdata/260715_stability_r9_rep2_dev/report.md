# Teacher Prompt Ablation: gold_binary_support_evidence_only_v3

- Family: `gold_aware_instruction_ablation`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `9297986cdb9cfb3b558b934c2023d702fc831efbd16144ee6061010933dc97f6`
- Started: `2026-07-15T18:54:09.173010+08:00`
- Finished: `2026-07-15T18:54:47.495525+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8203 | 0.5638 | 1.0000 | 0.8020 | 0.8950 | 0.8460 | 0.8464 | 59 | 10 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8291 | 0.8460 | 0.8123 | 0.7845 | 153/181 | 0.9609 | 0.7205 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 153 | 28 | 0 | 0 |
| I | 18 | 162 | 1 | 0 |
| A | 10 | 12 | 0 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 40
- Missed I as S: 18
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 141.1
- Average supported-answer characters: 14.6
- Maximum supported-answer characters: 166
- Average completion tokens: 56.5

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.7963 | 0.9416 | 0.8629 | 0.5926 | 0.7277 |
| teacher_not_called_control | 163 | 0.8250 | 0.7500 | 0.7857 | 0.9353 | 0.8605 |
| L1_steps_01_20 | 96 | 0.7547 | 0.8696 | 0.8081 | 0.7606 | 0.7843 |
| L2_steps_21_40 | 96 | 0.8333 | 0.8889 | 0.8602 | 0.8178 | 0.8390 |
| L3_steps_41_60 | 96 | 0.8636 | 0.9500 | 0.9048 | 0.9194 | 0.9121 |
| L4_steps_61_79 | 96 | 0.7719 | 0.8800 | 0.8224 | 0.7350 | 0.7787 |
