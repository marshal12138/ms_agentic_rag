# Deterministic Composite Derivation

- Derived at: `2026-07-15T22:45:54.324815+08:00`
- Source two-stage run: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/teacher_PE/results_newdata/260715_composite_stability_dual_all_f108_rep3_dev`
- Additional model requests: `0`
- Evidence-literal canonicalizations: `13`
- Mean elapsed ratio versus Stage A: `1.8208`
- Within 2x budget: `true`
# Teacher Prompt Ablation: dual_all_r5_gold_f1_08_literal_canonical_v2

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `1ba800b5b2f207614f82891cf5d7e6fa30a111ea8a077882115431fd29af0731`
- Started: `2026-07-15T22:35:15.240682+08:00`
- Finished: `2026-07-15T22:35:49.937422+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8229 | 0.6553 | 1.0000 | 0.9068 | 0.8066 | 0.8538 | 0.8698 | 50 | 18 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8558 | 0.8538 | 0.8578 | 0.8287 | 166/181 | 0.9353 | 0.7748 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 166 | 13 | 2 | 0 |
| I | 34 | 146 | 1 | 0 |
| A | 16 | 2 | 4 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 15
- Missed I as S: 34
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 205.3
- Average supported-answer characters: 14.7
- Maximum supported-answer characters: 172
- Average completion tokens: 126.2

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9206 | 0.8467 | 0.8821 | 0.7118 | 0.7970 |
| teacher_not_called_control | 163 | 0.8571 | 0.6818 | 0.7595 | 0.9397 | 0.8496 |
| L1_steps_01_20 | 96 | 0.9231 | 0.7826 | 0.8471 | 0.8650 | 0.8561 |
| L2_steps_21_40 | 96 | 0.9250 | 0.8222 | 0.8706 | 0.8837 | 0.8772 |
| L3_steps_41_60 | 96 | 0.9211 | 0.8750 | 0.8974 | 0.8962 | 0.8968 |
| L4_steps_61_79 | 96 | 0.8636 | 0.7600 | 0.8085 | 0.7756 | 0.7921 |
