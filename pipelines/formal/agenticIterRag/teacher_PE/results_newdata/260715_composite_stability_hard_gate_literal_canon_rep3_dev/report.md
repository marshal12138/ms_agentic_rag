# Composite Budget

- Stage A: `baseline_current_v2` from `results_newdata/260715_composite_stage_a_prod_rep3_dev`
- Stage-B calls: 198/384 (0.5156)
- Mean elapsed ratio versus Stage A: 1.3591
- Within 2x budget: `true`
# Teacher Prompt Ablation: hard_gate_r5_literal_canonical_v2

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `e20b489fade128fd28ce54bf341148561d8eb7e157febc064598bf73beace815`
- Started: `2026-07-15T22:50:45.003102+08:00`
- Finished: `2026-07-15T22:51:02.409419+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8281 | 0.6586 | 1.0000 | 0.8548 | 0.8785 | 0.8665 | 0.8724 | 49 | 17 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8377 | 0.8665 | 0.8089 | 0.7845 | 155/181 | 0.9446 | 0.7127 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 155 | 24 | 2 | 0 |
| I | 21 | 159 | 1 | 0 |
| A | 15 | 3 | 4 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 27
- Missed I as S: 21
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 220.1
- Average supported-answer characters: 15.0
- Maximum supported-answer characters: 172
- Average completion tokens: 102.1

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.8913 | 0.8978 | 0.8945 | 0.6832 | 0.7889 |
| teacher_not_called_control | 163 | 0.7500 | 0.8182 | 0.7826 | 0.8793 | 0.8310 |
| L1_steps_01_20 | 96 | 0.9091 | 0.8696 | 0.8889 | 0.8681 | 0.8785 |
| L2_steps_21_40 | 96 | 0.8667 | 0.8667 | 0.8667 | 0.8140 | 0.8403 |
| L3_steps_41_60 | 96 | 0.8222 | 0.9250 | 0.8706 | 0.8162 | 0.8434 |
| L4_steps_61_79 | 96 | 0.8269 | 0.8600 | 0.8431 | 0.7268 | 0.7850 |
