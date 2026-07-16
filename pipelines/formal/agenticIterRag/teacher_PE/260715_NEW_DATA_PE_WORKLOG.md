# SPAD Teacher PE New-Data Sampling and Manual Audit Work Log

Document generated at: 2026-07-15 17:00:00 CST (+08:00)

Last updated at: 2026-07-15 23:39:33 CST (+08:00)

## Scope

This log records the new-data Teacher prompt-engineering benchmark construction and manual S/I/A audit. All artifacts remain in `pipelines/formal/agenticIterRag/teacher_PE`.

## Source Run

- Run: `260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm03_stage1`
- Source: 79 rollout JSONL files, 40,448 rows, 5,056 groups, exactly 8 trajectories per group.
- Sampling seed: 42.
- Step strata: 1-20, 21-40, 41-60, and 61-79.
- Quota: 128 groups per stratum, one representative trajectory per selected group.
- Representative rule: prefer a `teacher_called=true` row within each selected group, then use a stable SHA256 ordering; otherwise retain a stable non-called control row.

## Progress

### 2026-07-15 16:47-16:54 CST - Sampling

- Added `sample_newdata_rollouts_512.py`.
- Sequentially scanned and hashed the 7.8 GB rollout source.
- Generated `benchmark_newdata_512.jsonl` and `benchmark_newdata_512.manifest.json`.
- Validated 512 unique groups and exactly 128 cases in each of the four step strata.
- Selected 293 teacher-called representatives and 219 non-called controls.
- Parsed one to five evidence rounds per case with zero evidence parser warnings.
- Corrected the source-run path inference and regenerated the benchmark.
- Final benchmark SHA256: `398eeb25304a39b8cf99597aabeb7d377b91279c5006fc12549c79d89ac4830e`.

### 2026-07-15 17:00 CST - Manual Audit Started

- Added `build_manual_judgments_newdata_512.py` and the annotation JSONL contract.
- Manual labels use only the Original question and accumulated visible evidence. Gold answers, actor answers, and historical teacher outputs are audit references rather than evidence.
- Label policy: `S` for one complete supported answer, `I` for a missing required fact or bridge, and `A` for multiple incompatible complete candidates satisfying the question.
- Status at this checkpoint: manual review in progress.

### 2026-07-15 17:45 CST - Manual Audit Completed and Frozen

- Completed manual review of all 512 sampled cases; there are no missing or invalid labels.
- Final label distribution: `S=241`, `I=241`, and `A=30`.
- Per-stratum distributions:
  - steps 1-20: `S=63`, `I=61`, `A=4`;
  - steps 21-40: `S=57`, `I=60`, `A=11`;
  - steps 41-60: `S=66`, `I=54`, `A=8`;
  - steps 61-79: `S=55`, `I=66`, `A=7`.
- Of 292 cases comparable with a historical teacher decision, 239 labels agree (`81.8493%`). This is a diagnostic only: historical outputs were not used as annotation evidence.
- Frozen artifacts: `manual_annotations_newdata_512.jsonl`, `manual_judgments_newdata_512.tsv`, and `manual_judgments_newdata_512.manifest.json`.
- Canonical benchmark-content SHA256: `398eeb25304a39b8cf99597aabeb7d377b91279c5006fc12549c79d89ac4830e`.
- Canonical annotation-content SHA256: `550ad12fd33a1d88ee99cfeda2e0c74df5b230f8eaa6a9a7da20cb0a448a35e6`.
- Raw-file SHA256 values at freeze time:
  - `benchmark_newdata_512.jsonl`: `ff2026bece9f657c36244f316947376fa26daa0562cdf9dfc3547b453cee3dec`;
  - `manual_annotations_newdata_512.jsonl`: `6ffd0de2bdc017818d609ababb6e4c6f87c854d3fb47b9498b699ab064717760`;
  - `manual_judgments_newdata_512.tsv`: `c602e6038f1a39a2c7742ce4e3cb0c2ef674e294d4d811402ffe85aec0b7844d`.

### 2026-07-15 17:52 CST - PE Split and Dual Objective Frozen

- Added `build_newdata_ablation_benchmark.py` and generated `benchmark_newdata_512_ablation.jsonl` plus its manifest.
- The deterministic split uses seed `260715`: each of the four step layers contributes 96 dev and 32 holdout cases.
- Dev contains 384 cases (`S=181`, `I=181`, `A=22`); frozen holdout contains 128 (`S=60`, `I=60`, `A=8`).
- All 512 normalized Original questions are unique, so no question group crosses the split.
- Ablation benchmark canonical SHA256: `4e208eafb72795ab6445194daecce3f33e247cf0d105d6714592c91e3d06cd2a`.
- Prompt selection uses dev only. Holdout remains sealed until a candidate is selected.
- The equal-weight selection objective is `0.5 * I_F1 + 0.5 * gold_token_F1_coverage_on_manual_S`.
- Gold token-F1 coverage averages over every manual-S case. A non-S prediction or parse failure on a manual-S case contributes zero; conditional token-F1 is reported separately and cannot hide missed answerable cases.
- Gold exact-match coverage and manual-answer token-F1 coverage are retained as diagnostics.
- The actual `teacher_called=true` subset is the primary operational slice. All 512 cases remain in the run so non-called controls and layer drift can be inspected separately.
- On the 241 manual-S cases, the evidence-grounded manual answer itself reaches gold EM `0.7344` and token-F1 `0.8084` (`dev` token-F1 `0.7915`, holdout `0.8594`). Therefore gold alignment is an intentional dataset-reference objective, not a perfect proxy for evidence correctness; both gold and manual-answer scores must be read together.
- Added the `question_tail_answer_alignment_v3` candidate. It preserves the previously successful question-tail evidence-only layout and changes only exact answer-span extraction guidance.
- CPU regression suite: 10/10 tests passed.

### 2026-07-15 17:53-17:59 CST - Teacher Service Loaded

- The four historical containers were absent, so four GLM-4.7-Flash TP=2 replicas were started on ports 8067-8070.
- The launcher polling interval was changed from 15 seconds to 120 seconds for this run.
- Readiness checkpoints were `0/4` at 120 seconds, `0/4` at 240 seconds, and `4/4` at 361 seconds. No off-cycle readiness probes were made.
- The services remain running and will be reused without restart until the user explicitly requests that ablation stop.

### 2026-07-15 18:00 CST - New-Data Ablation Round 1 Started

- Baseline: `baseline_question_tail_evidence_only_v2`, the prior M2/P1/S0 winner.
- Scope: 384 dev cases on all four replicas, no thinking, temperature 0, response cache disabled.
- Output: `results_newdata/260715_round01_m2_baseline_dev`.
- First inference checkpoint is scheduled after three minutes based on the historical 237-case runtime.

### 2026-07-15 18:01 CST - Round 1 Baseline Completed

- Wall time: 75.81 seconds; the process was collected at its scheduled three-minute checkpoint.
- Requests: 384; cache hits: 0; request errors: 0; parse rate: `383/384 = 0.9974`.
- All-dev I precision/recall/F1: `0.8342/0.8895/0.8610`.
- All-dev gold token-F1 coverage: `0.5944`; gold EM coverage: `0.5249`; manual-answer token-F1 coverage: `0.6845`.
- All-dev equal-weight objective: `0.7277`.
- Actual teacher-called slice (221 cases): I F1 `0.8873`, gold token-F1 coverage `0.2951`, objective `0.5912`.
- Non-called control slice (163 cases): I F1 `0.7879`, gold token-F1 coverage `0.7621`, objective `0.7750`.
- Layer objectives L1-L4: `0.7311`, `0.7202`, `0.7540`, and `0.7042`; the late-step layer is the weakest baseline slice.

### 2026-07-15 18:04 CST - Round 2 Answer Alignment Started

- Candidate: `question_tail_answer_alignment_v3`.
- The user layout is byte-identical to Round 1; only the system instruction adds exact answer-type and canonical passage-span guidance.
- Scope and cache policy are unchanged. Based on Round 1 wall time, the inference checkpoint is scheduled after two minutes.
- Output: `results_newdata/260715_round02_answer_alignment_dev`.

### 2026-07-15 18:05 CST - Round 2 Answer Alignment Completed

- Wall time: 40.64 seconds; requests/cache/errors/parse: `384/0/0/384`.
- All-dev I precision/recall/F1: `0.8466/0.8840/0.8649`.
- All-dev gold token-F1 coverage: `0.6035`; manual-answer token-F1 coverage: `0.6995`; equal objective: `0.7342`.
- Versus Round 1, all-dev I F1 improved `+0.0039`, gold coverage `+0.0091`, and objective `+0.0065`.
- Actual teacher-called objective fell from `0.5912` to `0.5841` because gold coverage fell `0.2951 -> 0.2785`, despite I F1 improving `0.8873 -> 0.8897`.
- Control objective improved `0.7750 -> 0.7908`. L3 improved strongly, but L4 objective fell `0.7042 -> 0.6661`.
- Interim decision: this variant leads the all-dev aggregate, but not the primary teacher-called slice; it is not yet a replacement for the baseline.

### 2026-07-15 18:10 CST - Round 3 Gold-Hypothesis Question-Tail Started

- Candidate: `gold_support_question_tail_v3`; prompt SHA256 `40b75b4c9ba375e32fa3acfaac1a9ff722062700f27d5aaa614f6e9373dd15d8`.
- This crosses the historical gold-hypothesis verifier with the successful no-subquery question-tail user layout.
- Output: `results_newdata/260715_round03_gold_qtail_dev`; detached runner log: `results_newdata/260715_round03_gold_qtail_dev.runner.log`.
- PID at launch: 939284. The first inference checkpoint is scheduled after two minutes.
- Teacher-called manual-S cases have a manual-answer-to-gold token-F1 of only `0.4870` versus `0.9622` for controls. Gold and evidence-correctness diagnostics must therefore remain separate even though the requested objective weights gold equally.

### 2026-07-15 18:12-18:13 CST - Round 3 Launcher Correction

- At the scheduled two-minute checkpoint, detached PID 939284 had exited without creating the result directory; its runner log was zero bytes.
- The failed detached launch made no persisted model requests and is excluded from valid experiments.
- Round 3 was restarted in a managed foreground session at 18:13 CST. The vLLM replicas were not restarted or modified.
- Progress output is limited to the final 384-case checkpoint so a single scheduled process read can collect the result without draining intermediate output chunks.

### 2026-07-15 18:14 CST - Round 3 Gold Question-Tail Completed

- Wall time: 31.35 seconds; requests/cache/errors/parse: `384/0/0/384`.
- All-dev I precision/recall/F1: `0.7949/0.8564/0.8245`.
- All-dev gold token-F1 coverage: `0.7461`; manual-answer coverage: `0.6804`; equal objective: `0.7853`.
- Actual teacher-called slice: I F1 `0.8531`, gold coverage `0.5546`, manual-answer coverage `0.4178`, equal objective `0.7039`.
- Relative to the no-gold baseline, gold raises the teacher-called gold coverage `+0.2595` and objective `+0.1127`, while reducing teacher-called I F1 `-0.0342` and manual-answer coverage `-0.1283`.
- The result is the current leader under the user's equal-weight objective, but it is a different production regime because the reference gold is supplied to the teacher. It must remain separately labeled from no-gold candidates.

### 2026-07-15 18:16 CST - Round 4 Gold v2 Layout Started

- Candidate: existing historical `gold_support_check` with the same system instruction as Round 3, but the old v2 user layout retains sub-query strings and has no question tail.
- This is a direct layout cross-check against Round 3.
- Output: `results_newdata/260715_round04_gold_v2_layout_dev`; response cache remains disabled.
- Based on the 31-second Round 3 wall time, the next process checkpoint remains conservatively scheduled after two minutes.

### 2026-07-15 18:17 CST - Round 4 Gold v2 Layout Completed

- Wall time: 36.37 seconds; requests/cache/errors/parse: `384/0/0/384`.
- All-dev I precision/recall/F1: `0.8187/0.8232/0.8209`.
- All-dev gold token-F1 coverage: `0.8192`; manual-answer coverage: `0.7296`; equal objective: `0.8200`.
- Actual teacher-called slice: I P/R/F1 `0.8163/0.8759/0.8451`, gold coverage `0.6303`, manual-answer coverage `0.4405`, equal objective `0.7377`.
- This is the current leader for both the all-dev and teacher-called equal-weight objectives.
- Relative to Round 3 with the same system instruction, the old v2 layout improves the all-dev objective by `+0.0348` and teacher-called objective by `+0.0338`.
- L1-L4 objectives are `0.7993`, `0.8536`, `0.8712`, and `0.7569`; L4 remains the weakest layer.

### 2026-07-15 18:21 CST - Round 5 Gold, No Sub-Query, No Tail Started

- Candidate: `gold_support_evidence_only_v3`.
- This differs from Round 4 only by hiding sub-query strings, isolating the no-subquery factor without adding a question tail.
- Output: `results_newdata/260715_round05_gold_no_subquery_no_tail_dev`; response cache disabled; next checkpoint after two minutes.

### 2026-07-15 18:21 CST - Round 5 Completed

- Wall time: 34.45 seconds; requests/cache/errors/parse: `384/0/0/384`.
- All-dev I precision/recall/F1: `0.8081/0.8840/0.8443`.
- All-dev gold token-F1 coverage: `0.8037`; manual-answer coverage: `0.7019`; equal objective: `0.8240`.
- Actual teacher-called slice: I P/R/F1 `0.8182/0.9197/0.8660`, gold coverage `0.6411`, manual-answer coverage `0.4251`, equal objective `0.7535`.
- This now leads both all-dev and teacher-called objectives. Compared with Round 4, removing sub-query improves all-dev objective `+0.0040` and called objective `+0.0158`.
- L1-L4 objectives: `0.8206`, `0.8051`, `0.8751`, and `0.7913`; the L4 weakness narrows substantially.

### 2026-07-15 18:24 CST - Round 6 Gold, Sub-Query, Question Tail Started

- Candidate: `gold_support_subquery_question_tail_v3`, the fourth corner of the gold-layout 2x2.
- It retains sub-query strings and repeats the Original question at the tail.
- Output: `results_newdata/260715_round06_gold_subquery_tail_dev`; response cache disabled; next checkpoint after two minutes.

### 2026-07-15 18:25 CST - Round 6 Completed and Layout Factorial Closed

- Wall time: 37.13 seconds; requests/cache/errors/parse: `384/0/0/384`.
- All-dev I F1 `0.8219`, gold coverage `0.7828`, equal objective `0.8023`.
- Actual teacher-called I F1 `0.8369`, gold coverage `0.5566`, equal objective `0.6968`.
- It underperforms Round 5 by `-0.0217` all-dev objective and `-0.0567` teacher-called objective.
- Gold-aware 2x2 conclusion: hiding sub-query is beneficial; a tail question is harmful. The winning layout is question + gold + full evidence, with no sub-query and no repeated tail question.
- An offline two-call hybrid using a no-gold status and gold-aware answer peaks at only `0.7872` all-dev / `0.7155` teacher-called, below the Round 5 single-call leader; no multi-call run is justified.

### 2026-07-15 18:30 CST - Round 7 Decoupled Gold Status/Answer Started

- Candidate: `gold_decoupled_status_answer_v3`; it reuses the Round 5 layout byte-for-byte.
- Stage 1 instructs the teacher to judge S/I/A as if gold were hidden. Stage 2 uses an evidence-supported gold alias only to normalize an S answer.
- Output: `results_newdata/260715_round07_gold_decoupled_dev`; response cache disabled; next checkpoint after two minutes.
- Regression suite after registration: 13/13 tests passed.

### 2026-07-15 18:30 CST - Round 7 Completed and Rejected

- Wall time: 42.91 seconds; cache/errors/parse: `0/0/383 of 384`.
- All-dev I P/R/F1 `0.8860/0.5580/0.6847`, gold coverage `0.8880`, equal objective `0.7864`.
- Actual teacher-called I F1 `0.7265`, gold coverage `0.7327`, equal objective `0.7296`.
- The model interpreted the decoupled Stage 1 as permission to over-answer: 77 manual-I cases became S. The gold gain cannot compensate for the I-recall collapse.
- Decision: reject the explicit status/answer decoupling prompt. It is inferior to Round 5 on all-dev and teacher-called objectives and has one format error.

### 2026-07-15 18:33 CST - Round 8 Strict Gold I-Guard Started

- Candidate: `gold_i_guard_evidence_only_v3`, crossing the historical strict gold audit instruction with the Round 5 layout.
- Output: `results_newdata/260715_round08_gold_i_guard_dev`; response cache disabled; next checkpoint after two minutes.
- Registry regression suite after adding Rounds 8/9 candidates: 14/14 tests passed.

### 2026-07-15 18:34 CST - Round 8 Completed and Rejected

- Wall time: 42.61 seconds; cache/errors/parse: `0/0/384 of 384`.
- All-dev I P/R/F1 `0.8473/0.6133/0.7115`, gold coverage `0.8322`, equal objective `0.7719`.
- Actual teacher-called I F1 `0.7438`, gold coverage `0.6636`, equal objective `0.7037`.
- Despite its name, the strict audit over-accepted complete candidates and missed 70 manual-I cases. It is worse and slower than Round 5.
- Decision: reject; do not pursue longer missing-bridge audits on this model.

### 2026-07-15 18:37 CST - Round 9 Gold Binary Support Started

- Candidate: `gold_binary_support_evidence_only_v3`, using the Round 5 layout with the historical gold-relation binary gate.
- Output: `results_newdata/260715_round09_gold_binary_support_dev`; response cache disabled; next checkpoint after two minutes.

### 2026-07-15 18:37 CST - Round 9 Completed

- Wall time: 39.62 seconds; requests/cache/errors/parse: `384/0/0/384`.
- All-dev I P/R/F1 `0.8000/0.8840/0.8399`, gold coverage `0.8192`, manual-answer coverage `0.7206`, equal objective `0.8296`.
- This is the best all-dev equal objective so far, `+0.0055` over Round 5.
- Actual teacher-called I F1 `0.8600`, gold coverage `0.5889`, equal objective `0.7244`, which remains below Round 5's operational `0.7535`.
- It predicts no manual-A case correctly (`A recall=0`), a secondary but material diagnostic even though the requested main objective tolerates S/A confusion.
- Interim leaders therefore differ: Round 9 for the all-dev aggregate; Round 5 for the actual teacher-called operational slice.

### 2026-07-15 18:40 CST - Round 10 Compact Balanced Gold Started

- Candidate: `gold_compact_balanced_v3`, 1206 system-prompt characters versus Round 5's 1633.
- It retains candidate count, gold-is-not-evidence, and supported-non-gold-answer rules on the Round 5 layout.
- Output: `results_newdata/260715_round10_gold_compact_balanced_dev`; response cache disabled; next checkpoint after two minutes.

### 2026-07-15 18:40 CST - Round 10 Completed and Rejected

- Wall time: 37.02 seconds; cache/request errors: `0/0`.
- All-dev I F1 `0.8069`, gold coverage `0.7587`, equal objective `0.7828`.
- Parse rate fell to `372/384 = 0.9688`; all 12 failures were missing reason/status tags.
- Actual teacher-called equal objective is `0.6860` with parse `0.9729`.
- Decision: reject. Shortening the verifier from 1633 to 1206 characters reproduced the historical compact-prompt format and recall regression.

## Ten-Run Review - 2026-07-15 18:45 CST

- Ten valid, cache-free dev ablations are complete. The failed empty detached launch is excluded.
- Best no-gold aggregate: Round 2 answer alignment, objective `0.7342`; best no-gold teacher-called: Round 1 baseline, `0.5912`.
- Best gold-aware aggregate: Round 9 gold binary support, objective `0.8296`.
- Best actual teacher-called operational result: Round 5 gold support on the no-subquery/no-tail layout, objective `0.7535`.
- Round 5 is the current operational candidate because actual teacher calls are the primary slice. Round 9 remains an aggregate/control-heavy Pareto candidate and cannot replace it from one run.
- The historical no-gold winner transferred only partially: hiding sub-query still helps, but repeating the question at the tail hurts once gold is supplied.
- Gold-aware prompts materially increase gold hit but sometimes return the dataset gold over the evidence-grounded manual answer. This is visible in the separate manual-answer coverage and must not be hidden by the equal objective.
- Longer explicit decoupling and strict audit instructions over-answer and collapse I recall. The compact prompt introduces format errors. Multi-call status/answer hybrids do not beat the single-call leader.
- All 10 runs remain dev-only. The 128-case holdout is still sealed.
- Next action: two additional cache-free dev repeats each for Round 5 and Round 9. Select by repeated means, with teacher-called objective primary and all-dev objective secondary; only then run the selected candidate on holdout.

### 2026-07-15 18:46 CST - Stability Repeats Started

- First repeat: Round 5 candidate `gold_support_evidence_only_v3`, output `results_newdata/260715_stability_r5_rep2_dev`.
- Settings match the original run and response cache remains disabled. Checkpoint interval: two minutes.

### 2026-07-15 18:47 CST - R5 Repeat 2 Completed

- Wall time 37.88 seconds; cache/errors/parse `0/0/384`.
- All-dev I F1 `0.8455`, gold coverage `0.8190`, equal objective `0.8323`.
- Teacher-called I F1 `0.8671`, gold coverage `0.6376`, equal objective `0.7524`.
- Although the all-dev objective moved by `+0.0083` from the first run, the operational objective stayed close (`0.7535 -> 0.7524`).
- R5 Repeat 3 started at 18:50 CST with output `results_newdata/260715_stability_r5_rep3_dev`; cache disabled and checkpoint after two minutes.

### 2026-07-15 18:51 CST - R5 Repeat 3 and Three-Run Summary

- Repeat 3 wall time 35.34 seconds; cache/errors/parse `0/0/384`.
- Repeat 3 all-dev I F1/gold/objective: `0.8375/0.8230/0.8303`; teacher-called: `0.8622/0.6411/0.7516`.
- Three-run R5 all-dev means: I P/R/F1 `0.8243/0.8619/0.8424`, gold coverage `0.8153`, objective `0.8288`; objective range `[0.8240, 0.8323]`.
- Three-run R5 teacher-called means: I P/R/F1 `0.8287/0.9051/0.8651`, gold coverage `0.6399`, objective `0.7525`; objective range `[0.7516, 0.7535]`.
- Parse is 1.0 in all three runs. The operational objective is highly stable despite case-level status agreement of 95.31% between the first two runs.

### 2026-07-15 18:54 CST - R9 Stability Repeats Started

- R9 Repeat 2 output: `results_newdata/260715_stability_r9_rep2_dev`; response cache disabled; checkpoint after two minutes.

### 2026-07-15 18:55 CST - R9 Repeat 2 Completed

- Wall time 38.38 seconds; cache/errors/parse `0/0/384`.
- All-dev I F1/gold/objective `0.8460/0.8123/0.8291`.
- Teacher-called I F1/gold/objective `0.8629/0.5926/0.7277`.
- R9 Repeat 3 started at 18:58 CST with output `results_newdata/260715_stability_r9_rep3_dev`; cache disabled and checkpoint after two minutes.

### 2026-07-15 18:58 CST - R9 Repeat 3 and Candidate Selection

- Repeat 3 wall time 38.04 seconds; cache/errors/parse `0/0/384`.
- Repeat 3 all-dev I F1/gold/objective `0.8338/0.8040/0.8189`; teacher-called `0.8552/0.5618/0.7085`.
- Three-run R9 all-dev means: I F1 `0.8399`, gold `0.8118`, objective `0.8258`.
- Three-run R9 teacher-called means: I F1 `0.8594`, gold `0.5811`, objective `0.7202`.
- R5 wins the repeated all-dev objective (`0.8288 > 0.8258`) and the primary teacher-called objective (`0.7525 > 0.7202`), with lower mean wall time (`35.89s < 38.68s`). Both have parse 1.0.
- Dev selection is frozen to `gold_support_evidence_only_v3` (R5). R9's single-run aggregate lead did not survive repeated means.

### 2026-07-15 19:02 CST - Frozen Holdout Evaluation Started

- The 128-case holdout was unsealed only after the dev candidate and selection rule were frozen.
- Candidate: `gold_support_evidence_only_v3`; output `results_newdata/260715_holdout_r5_rep1`.
- No other prompt will be tuned from holdout results. Cache is disabled; expected short runtime gives a one-minute checkpoint.

### 2026-07-15 19:03-19:05 CST - Frozen Holdout Three-Run Evaluation Completed

- All three runs contain 128 requests, zero cache hits, zero request errors, and parse rate 1.0.
- All-holdout three-run means [min, max]:
  - I precision `0.9184 [0.8909, 0.9412]`;
  - I recall `0.8056 [0.8000, 0.8167]`;
  - I F1 `0.8581 [0.8522, 0.8649]`;
  - gold token-F1 coverage `0.8902 [0.8754, 0.9143]`;
  - equal objective `0.8741 [0.8638, 0.8896]`.
- Actual teacher-called holdout means [min, max]:
  - I precision `0.9321 [0.9111, 0.9535]`;
  - I recall `0.8039 [0.8039, 0.8039]`;
  - I F1 `0.8632 [0.8542, 0.8723]`;
  - gold token-F1 coverage `0.7667 [0.7444, 0.8111]`;
  - equal objective `0.8149 [0.7993, 0.8417]`.
- Mean holdout wall time is 9.14 seconds. The first run was collected at its one-minute checkpoint; repeats 2/3 completed inside their initial command windows, so no extra polling was needed.
- Holdout confirms the frozen R5 candidate without prompting another selection cycle.
- Final artifacts: `NEW_DATA_RESULTS_INDEX.md` contains every persisted run; `NEW_DATA_STABILITY.md` contains the repeated candidate and holdout summaries.

### 2026-07-15 19:07 CST - Final Consistency and Service Check

- Audited 17 complete result directories containing 5,760 persisted predictions.
- Every run has the expected prediction count; aggregate response-cache hits and request errors are both zero.
- Fourteen of 17 runs have parse rate 1.0. The minimum is the intentionally rejected compact Round 10 at 0.96875.
- CPU regression suite: 14/14 tests passed; `git diff --check` passed.
- All four GLM-4.7-Flash replicas on ports 8067-8070 are `running=true` and `ready=true`.
- The vLLM services remain running for continued ablation and will not be stopped until the user explicitly requests it.

## Combination-Strategy Ablation - 2026-07-15 22:24-22:53 CST

### 2026-07-15 22:24-22:27 CST - Shared Gate and Three Hard-Gate Designs

- The user authorized up to `2x` the single-prompt Teacher inference time. Formal training remains single-replica; the four replicas are only the PE execution topology.
- Generated a fresh, cache-free production no-gold Stage-A baseline on all 384 dev cases. Its teacher-called I P/R/F1 was `0.9091/0.8759/0.8922`, gold coverage `0.3386`, and equal objective `0.6154`.
- Tested three Stage-B designs behind a binding Stage-A I gate: R5 verifier, dedicated gold extractor, and draft/gold selector.
- The R5 hard gate led: teacher-called I F1 stayed exactly `0.8922`, gold coverage rose to `0.5770`, objective to `0.7346`, and mean elapsed ratio was `1.3524x`.
- The dedicated extractor and selector were rejected. Their teacher-called objectives were `0.6755` and `0.6387`; both generated weaker gold-aligned answers than R5.

### 2026-07-15 22:30-22:45 CST - Dual-All Override and Deterministic Answer Policy

- Tested R5 on every case and allowed it to override Stage-A I only when it returned S with gold token-F1 at least `0.8`.
- Three independent complete dev chains stayed within budget at `1.7586x-1.8208x`. Before deterministic answer postprocessing, teacher-called means were I F1 `0.8812`, gold coverage `0.6283`, and objective `0.7547`.
- Added a deterministic policy: preserve the sole S answer across stages; when both stages return S, select the higher-gold-F1 answer; then replace it with a reference answer only when that normalized literal occurs in Search evidence. No manual label is used by this policy.
- The three derived v2 results reuse three independent Stage-A/Stage-B inference outputs and record `model_requests_this_derivation=0`; they are not represented as fresh model repeats.
- Dual-all v2 dev teacher-called means [min, max]: I F1 `0.8812 [0.8794, 0.8821]`, gold coverage `0.7220 [0.6964, 0.7579]`, manual-answer coverage `0.5358 [0.5277, 0.5479]`, objective `0.8016 [0.7879, 0.8200]`.

### 2026-07-15 22:46-22:48 CST - Dual-All Frozen Holdout Failure Mode

- Ran three fresh full-chain repeats on the 128-case holdout with no cache hits, request errors, or parse errors.
- Teacher-called means: I P/R/F1 `0.9473/0.6993/0.8045`, gold coverage `0.9222`, and objective `0.8634`; elapsed ratio `1.7540x`.
- The high gold score hid a material I-recall regression. Gold-aware Stage B overrode Stage-A I when a gold literal appeared even though the queried relation remained unsupported.
- Decision: reject dual-all v2 for production. A gold-overlap threshold is not a reliable reason-completeness gate.
- This result consumed the originally frozen holdout for combination-strategy selection. Later uses of the same 128 cases are explicitly diagnostic and are not untouched final estimates.

### 2026-07-15 22:50-22:51 CST - Conservative Hard-Gate v2 Stability

- Registered `hard_gate_r5_literal_canonical_v2`: Stage-A I is binding and cannot be overridden; R5 runs only for Stage-A S/A; answer selection and evidence-literal canonicalization match dual-all v2.
- Three cache-free dev chains produced teacher-called means [min, max]: I P `0.8970 [0.8905, 0.9091]`, I R `0.8881 [0.8759, 0.8978]`, I F1 `0.8924 [0.8905, 0.8945]`, gold coverage `0.6825 [0.6656, 0.6986]`, manual-answer coverage `0.4833 [0.4764, 0.4907]`, objective `0.7874 [0.7789, 0.7946]`.
- Relative to single-prompt R5 dev means, hard-gate v2 improved teacher-called I F1 by `+0.0273`, gold coverage by `+0.0426`, and objective by `+0.0349`.
- Stage-B was called for only `37.6%-40.3%` of teacher-called cases. Mean elapsed ratio was `1.3558x`; mean teacher-called inference time was `9.28s` in the four-replica no-queue PE measurements.

### 2026-07-15 22:51-22:52 CST - Reused-Holdout Diagnostic

- Three cache-free diagnostic repeats of hard-gate v2 had teacher-called means [min, max]: I P `0.9478 [0.9333, 0.9767]`, I R `0.8235 [0.8235, 0.8235]`, I F1 `0.8812 [0.8750, 0.8936]`, gold coverage `0.9000 [0.8667, 0.9667]`, objective `0.8906 [0.8708, 0.9301]`.
- Relative to the original frozen single-prompt R5 holdout means, I F1 improved `+0.0180`, gold coverage `+0.1333`, and objective `+0.0757`.
- Mean elapsed ratio was `1.2307x`; Stage B ran on `38.4%` of actual teacher calls. This is below the dev ratio because more holdout cases were gated as I.
- Manual-answer coverage was `0.5267`, below R5 holdout's `0.6156`. Evidence-literal canonicalization optimizes dataset-gold wording and can move away from the separately annotated evidence answer; both metrics remain visible.
- Current decision: hard-gate v2 is the leading combination strategy under the requested equal I/gold objective and the `2x` budget. Before calling it an unbiased production winner, validate it on a new untouched rollout sample; do not tune it repeatedly on the 3500 evaluation set.

### 2026-07-15 22:53 CST - Service and Artifact State

- Combination code: `composite_prompt_variants.py`, `run_composite_ablation.py`, and `derive_composite_policy.py`.
- CPU regression suite: 19/19 tests passed after adding evidence-presence and absent-gold injection guards.
- All outputs remain under this pipeline directory. The four vLLM services remain running and were never restarted during combination ablation.
- Final audit at 22:55 CST covered 41 complete result directories and 12,672 predictions. Prediction-count mismatches, aggregate cache hits, request errors, and composite runs above the `2x` budget were all zero.
- `NEW_DATA_RESULTS_INDEX.md` and `NEW_DATA_STABILITY.md` were regenerated at 22:55 CST. All four services on ports 8067-8070 reported `running=true` and `ready=true`; they remain active.

### 2026-07-15 23:04 CST - Training Production Prompt Added to the Comparison

- The source training config for run `260715-005906-987696` records `prompt_version: spad_teacher_evidence_status_answer_v2`.
- Verified programmatically that PE variant `baseline_current_v2` produces exactly equal system and user messages to the production `build_teacher_messages(..., prompt_version="spad_teacher_evidence_status_answer_v2")` implementation on the frozen benchmark.
- Historical outputs persisted by the training run cover 293 teacher-called samples, including one unparsable/missing status. On all 293, I P/R/F1 was `0.9121/0.8830/0.8973`, gold coverage `0.3086`, and equal objective `0.6030`. On the 221-case dev slice, the corresponding values were `0.8971/0.8905/0.8938`, `0.2831`, and `0.5884`.
- Three fresh cache-free production-prompt dev repeats gave teacher-called I P/R/F1 `0.8970/0.8881/0.8924`, gold coverage `0.3180`, manual-answer coverage `0.6133`, and objective `0.6052`. This closely reproduces the historical I behavior while avoiding reliance on one stored generation.
- Hard-gate v2 preserves those fresh production I metrics exactly (`0.8970/0.8881/0.8924`) because Stage-A I is binding, while increasing gold coverage `0.3180 -> 0.6825` and objective `0.6052 -> 0.7874` at `1.3558x` mean elapsed.
- Fresh holdout diagnostics show the same structural result: production prompt `I F1/gold/objective = 0.8812/0.4141/0.6476`; hard-gate v2 `0.8812/0.9000/0.8906`. The combination does not change I decisions and improves only the non-I answer path.
- At 23:39 CST, added `NEW_DATA_STRATEGY_COMPARISON.md` with detailed input layouts, decision rules, answer merge behavior, costs, three-run dev/holdout comparisons, limitations, and selection guidance for production, R5, Hard-gate v2, and the rejected dual-all control.
