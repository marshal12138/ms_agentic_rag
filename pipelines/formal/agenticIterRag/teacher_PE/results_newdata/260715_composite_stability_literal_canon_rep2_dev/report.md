# Deterministic Composite Derivation

- Derived at: `2026-07-15T22:45:54.331615+08:00`
- Source two-stage run: `/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/pipelines/formal/agenticIterRag/teacher_PE/results_newdata/260715_composite_stability_dual_all_f108_rep2_dev`
- Additional model requests: `0`
- Evidence-literal canonicalizations: `15`
- Mean elapsed ratio versus Stage A: `1.7918`
- Within 2x budget: `true`
# Teacher Prompt Ablation: dual_all_r5_gold_f1_08_literal_canonical_v2

- Family: `hard_gate_composite`
- Evaluated split: `dev`
- Cases: 384
- Prompt SHA256: `1ba800b5b2f207614f82891cf5d7e6fa30a111ea8a077882115431fd29af0731`
- Started: `2026-07-15T22:33:40.653476+08:00`
- Finished: `2026-07-15T22:34:15.089849+08:00`

## Main Metrics

| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8359 | 0.6822 | 1.0000 | 0.9130 | 0.8122 | 0.8596 | 0.8750 | 48 | 15 |

## Equal-Weight Objective

The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. A manual-S case predicted as non-S or failing parse contributes zero answer score.

| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8725 | 0.8596 | 0.8854 | 0.8564 | 169/181 | 0.9483 | 0.7785 |

## Confusion Matrix

Rows are manual labels; columns are model predictions.

| Manual \ Pred | S | I | A | Format error |
| --- | ---: | ---: | ---: | ---: |
| S | 169 | 10 | 2 | 0 |
| I | 33 | 147 | 1 | 0 |
| A | 13 | 4 | 5 | 0 |

## I Error Breakdown

- False I (manual S/A -> I): 14
- Missed I as S: 33
- Missed I as A: 1
- Missed I due to format error: 0

## Output Length

- Average reason characters: 201.2
- Average supported-answer characters: 14.6
- Maximum supported-answer characters: 172
- Average completion tokens: 125.2

## Operational Slices

The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.

| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_called | 221 | 0.9206 | 0.8467 | 0.8821 | 0.7579 | 0.8200 |
| teacher_not_called_control | 163 | 0.8857 | 0.7045 | 0.7848 | 0.9569 | 0.8709 |
| L1_steps_01_20 | 96 | 0.9024 | 0.8043 | 0.8506 | 0.8863 | 0.8684 |
| L2_steps_21_40 | 96 | 0.8780 | 0.8000 | 0.8372 | 0.9070 | 0.8721 |
| L3_steps_41_60 | 96 | 0.9706 | 0.8250 | 0.8919 | 0.9362 | 0.9140 |
| L4_steps_61_79 | 96 | 0.9111 | 0.8200 | 0.8632 | 0.8000 | 0.8316 |
