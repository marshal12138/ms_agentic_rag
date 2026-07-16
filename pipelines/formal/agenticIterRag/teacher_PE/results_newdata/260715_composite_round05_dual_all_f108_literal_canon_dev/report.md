# Deterministic Composite Derivation

- Derived at: `2026-07-15T22:45:54.275099+08:00`
- Source two-stage run: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/teacher_PE/results_newdata/260715_composite_round04_dual_all_r5_f108_dev`
- Additional model requests: `0`
- Evidence-literal canonicalizations: `12`
- Mean elapsed ratio versus Stage A: `1.7586`
- Within 2x budget: `true`
# Teacher Prompt Ablation: dual_all_r5_gold_f1_08_literal_canonical_v2

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `1ba800b5b2f207614f82891cf5d7e6fa30a111ea8a077882115431fd29af0731`
- Started: `2026-07-15T22:30:24.247262+08:00`
- Finished: `2026-07-15T22:31:00.145688+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8281 | 0.7167 | 1.0000 | 0.9231 | 0.7956 | 0.8546 | 0.8724 | 49 | 17 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8562 | 0.8546 | 0.8578 | 0.8287 | 166/181 | 0.9353 | 0.7744 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 166 | 12 | 3 | 0 |
| I | 35 | 144 | 2 | 0 |
| A | 14 | 0 | 8 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 12
- Missed I as S: 35
- Missed I as A: 2
- Missed I due to format error: 0

## Output Length

- Average reason characters: 205.2
- Average supported-answer characters: 14.8
- Maximum supported-answer characters: 172
- Average completion tokens: 126.8

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9417 | 0.8248 | 0.8794 | 0.6964 | 0.7879 |
| teacher_not_called_control | 163 | 0.8611 | 0.7045 | 0.7750 | 0.9483 | 0.8616 |
| L1_steps_01_20 | 96 | 0.9231 | 0.7826 | 0.8471 | 0.8438 | 0.8454 |
| L2_steps_21_40 | 96 | 0.9487 | 0.8222 | 0.8810 | 0.8605 | 0.8707 |
| L3_steps_41_60 | 96 | 0.9412 | 0.8000 | 0.8649 | 0.9162 | 0.8905 |
| L4_steps_61_79 | 96 | 0.8864 | 0.7800 | 0.8298 | 0.8000 | 0.8149 |
