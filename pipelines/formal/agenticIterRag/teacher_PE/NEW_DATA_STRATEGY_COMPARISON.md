# New-Data Teacher Strategy Comparison

Document generated at: 2026-07-15T23:39:33+08:00

## Comparison Scope

This document compares the Teacher strategies relevant to the new 5100-step training distribution. The primary slice is the 221 dev cases where the training pipeline actually called the Teacher. All main numbers are means over three cache-free fresh runs unless explicitly marked historical.

The selection objective is:

`0.5 * I_F1 + 0.5 * gold_token_F1_coverage_on_manual_S`

A manual-S case contributes zero gold coverage when the strategy does not return S. Manual-answer F1 is reported separately because the answer supported by the retrieved evidence can differ from the dataset gold.

## 中文策略说明

### 1. 训练 Production Prompt

- 实际训练配置使用 `spad_teacher_evidence_status_answer_v2`，在 PE registry 中对应 `baseline_current_v2`。已经逐字验证两边生成的 system message 和 user message 完全一致。
- 输入包含 Original question、每轮 sub-query 和累计检索 evidence，不包含 gold answer。一次调用同时输出 reason、S/I/A status 和短答案。
- 它的主要优势是 I 判别稳定：三次 fresh dev 的 teacher-called I P/R/F1 为 `0.8970/0.8881/0.8924`，和原训练历史输出的 I F1 `0.8938` 非常接近。
- 它更倾向于输出 passage 中直接支持的答案，因此 dev manual-answer F1 最高，为 `0.6133`；但对数据集 gold 的覆盖仅 `0.3180`，等权指标只有 `0.6052`。
- 成本是一条样本一次 Teacher 调用，即 `1.0x`。适合作为无 gold 场景的生产基线，也适合作为组合策略中不可被推翻的 I gate。

### 2. 单 Prompt R5

- R5 对应 `gold_support_evidence_only_v3`。输入包含 Original question、reference gold 和完整 title/passage evidence，但隐藏 sub-query，也不在尾部重复 question。
- prompt 明确规定 gold 只是待验证假设，不是 evidence；模型必须检查问题中的实体、谓词、scope 和多跳 bridge，然后一次调用同时决定 S/I/A 并生成答案。
- gold-aware 输入把 dev Gold F1 从 production 的 `0.3180` 提高到 `0.6399`，等权指标从 `0.6052` 提高到 `0.7525`。
- 代价是 I F1 从 `0.8924` 降到 `0.8651`，其中 I precision 只有 `0.8287`；manual-answer F1 也降到 `0.4408`，说明它更倾向数据集 gold，而不一定保持人工 evidence answer 的措辞。
- 它仍然只需要一次调用，成本为 `1.0x`。当训练中可以提供 gold、但严格只允许单 prompt 时，R5 是当前最稳定的选择。

### 3. Hard-Gate v2

- Hard-Gate v2 对应 `hard_gate_r5_literal_canonical_v2`。Stage A 使用训练 production prompt 且不看 gold；只有 Stage A 给出 S/A 时，Stage B 才调用 R5。
- I 二分类边界完全归 Stage A 所有：Stage-A I 不调用 Stage B，不能被改成非 I；Stage-A S/A 即使被 Stage B 判成 I，也回退到 Stage A，不能被改成 I。因此最终 I label 与 production prompt 逐样本完全一致。
- 在非 I 路径中，如果只有一个阶段给出 S，就保留唯一 supported answer；如果两个阶段都给出 S，就选择 gold token-F1 更高的答案。只有 reference gold 的规范化字面值确实出现在 Search evidence 中、且能提高 Gold F1 时，才把答案规范化为该 gold。
- 三次 dev 的 I P/R/F1 与 production 完全相同，仍为 `0.8970/0.8881/0.8924`；Gold F1 提高到 `0.6825`，等权指标提高到 `0.7874`，同时超过 production 和 R5。
- dev 平均耗时为 `1.3558x`，实际 teacher-called 样本只有约 `38.6%` 需要 Stage B，低于允许的 `2x`。它的 manual-answer F1 为 `0.4833`，高于 R5、低于 production，仍需关注 gold 规范化带来的措辞偏移。
- 这是当前等权目标下的领先方案，但目前只在 PE harness 中实现，尚未集成进正式训练 runtime。组合 holdout 已经被使用过，正式定版前还需要新的未触碰 rollout 样本。

### 4. 被淘汰的 Dual-All v2

- Dual-All v2 对每条样本都调用 R5，并允许 Stage B 在返回 S 且 gold token-F1 不低于 `0.8` 时推翻 Stage-A I。
- dev 上它得到 I F1 `0.8812`、Gold F1 `0.7220`、等权指标 `0.8016`，表面上高于 Hard-Gate v2。
- 但在 holdout 上 I recall 降到 `0.6993`、I F1 降到 `0.8045`。根因是 gold literal 出现在 passage 中并不能证明问题要求的完整关系或 bridge 已被支持。
- 它的平均成本为 `1.7904x`，同时判别风险更高，因此已被明确淘汰。这个反例说明 gold overlap 可以用于答案规范化，但不能用于推翻 evidence-sufficiency gate。

## Main Dev Comparison

| Strategy | Gold available | Calls | I P | I R | I F1 | Gold F1 | Manual-answer F1 | Equal objective | Mean elapsed |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training production prompt, fresh | No | One call | 0.8970 | 0.8881 | 0.8924 | 0.3180 | 0.6133 | 0.6052 | 1.0000x |
| Single-prompt R5 | Yes | One call | 0.8287 | 0.9051 | 0.8651 | 0.6399 | 0.4408 | 0.7525 | 1.0000x |
| Hard-gate v2 | Stage B only | Conditional two calls | 0.8970 | 0.8881 | 0.8924 | 0.6825 | 0.4833 | 0.7874 | 1.3558x |

Hard-gate v2 has exactly the same I precision, recall, and F1 as the production prompt. This equality follows from the merge rule, not from rounding: Stage B cannot change either an I to non-I or a non-I to I.

## Strategy 1: Training Production Prompt

### Identity

- Training config version: `spad_teacher_evidence_status_answer_v2`.
- PE registry name: `baseline_current_v2`.
- The system and user messages produced by the PE builder were checked programmatically against the production `build_teacher_messages` implementation and are exactly equal.
- The source run is `260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_newdata_5100_gold_token_f1_v3_postnorm03_stage1`.

### Input and Decision

The prompt receives the Original question, every accumulated search round, each sub-query string, and up to five title/passage pairs per round. It does not receive the reference gold answer.

It makes one joint decision and emits:

```text
<reason>...</reason><status>supported_answer|insufficient_evidence|ambiguous_evidence</status><answer>...</answer>
```

The prompt asks for S only when there is one supported short answer, I when necessary facts are missing, and A when multiple incompatible answers are supported.

### Observed Behavior

- Its main strength is evidence sufficiency judgment. Fresh dev I F1 is `0.8924` and closely matches the stored training-generation dev I F1 of `0.8938`.
- Its manual-answer F1 is the highest of the three dev strategies at `0.6133`. Without gold, it tends to retain wording directly supported by the retrieved passages.
- Its main weakness under the new objective is gold alignment. Fresh dev gold coverage is only `0.3180`, so the equal objective is `0.6052` despite strong I judgment.
- Historical training outputs cover 293 called cases. Across those cases, I P/R/F1 is `0.9121/0.8830/0.8973`, gold coverage is `0.3086`, and the equal objective is `0.6030`. The fresh repeats therefore reproduce the historical behavior closely.

### Cost and Role

This is the `1.0x` reference cost: one Teacher request per Teacher-eligible trajectory. It is the best of the compared strategies when no gold is available, and it is used as the binding I gate in Hard-gate v2.

## Strategy 2: Single-Prompt R5

### Identity

- PE registry name: `gold_support_evidence_only_v3`.
- Informal name: R5, gold support, no sub-query, no question tail.
- It is the best stable single-call gold-aware prompt from the new-data ablation.

### Input and Decision

R5 receives the Original question, reference gold answers, and complete title/passage evidence. It deliberately hides sub-query strings and does not repeat the question at the end.

The system prompt states that gold is a hypothesis rather than evidence. It asks the model to verify the exact question relation and required bridges, then return either an evidence-supported gold equivalent, another supported answer, I, or A.

R5 performs sufficiency judgment and answer generation in one model call:

```text
question + reference gold + evidence -> S/I/A + answer
```

### Observed Behavior

- Dev gold coverage rises from the production prompt's `0.3180` to `0.6399`.
- I recall is high at `0.9051`, but I precision falls to `0.8287`; reference gold makes the model more willing to reject some supported non-gold cases or changes its decision boundary.
- I F1 is `0.8651`, lower than the production prompt's `0.8924` by `0.0273`.
- Manual-answer F1 falls from `0.6133` to `0.4408`, showing that gold-aware output is not interchangeable with the independently annotated evidence answer.
- The equal objective is `0.7525`, substantially above the production prompt because the gold gain outweighs the I loss under equal weighting.

### Cost and Role

R5 remains a single-call `1.0x` strategy. It is the appropriate baseline when only one call is allowed and reference gold is available. In Hard-gate v2 it becomes a conditional answer-stage verifier rather than the owner of the I boundary.

## Strategy 3: Hard-Gate v2

### Identity

- Composite registry name: `hard_gate_r5_literal_canonical_v2`.
- Stage A: the exact training production prompt, with no gold.
- Stage B: R5, called only after Stage A returns S or A.

### Decision Flow

```text
                         +-> I: return Stage-A I directly
question + evidence -> Stage A
                         +-> S/A: call R5 with gold + evidence
                                  |
                                  +-> never change the I/non-I boundary
                                  +-> select a supported answer
                                  +-> evidence-literal gold canonicalization
```

The merge rules are:

1. If Stage A returns I, Stage B is not called and I is final.
2. If Stage A returns S/A but Stage B returns I or fails parsing, fall back to Stage A. Stage B therefore cannot introduce a new I.
3. If only one stage returns S, preserve that supported answer.
4. If both stages return S, choose the answer with higher token-F1 against reference gold.
5. Replace the chosen answer with a reference gold string only when its normalized literal occurs in the supplied Search evidence and improves gold F1.
6. Manual labels and manual answers are never used by the runtime decision policy.

### Why It Preserves I Judgment

The final I label is equivalent to `Stage-A label == I`. Both directions are protected: Stage-A I cannot be overridden, and Stage-A non-I cannot be changed to I. Consequently, all three fresh dev runs have exactly the same I confusion matrix as their corresponding production-prompt Stage-A runs. Their teacher-called means are identical at I P/R/F1 `0.8970/0.8881/0.8924`.

### Answer Improvements

- Dev gold coverage improves from production `0.3180` and R5 `0.6399` to `0.6825`.
- The equal objective improves from production `0.6052` and R5 `0.7525` to `0.7874`.
- Manual-answer F1 is `0.4833`: better than R5's `0.4408`, but lower than production's `0.6133`. The strategy trades some evidence-answer wording for dataset-gold alignment.
- Parse rate is `1.0` in every repeat.

### Cost

On dev, Stage B is called for `51.6%-52.1%` of all cases and `37.6%-40.3%` of actual Teacher-called cases. The mean per-case elapsed ratio is `1.3558x`, below the allowed `2x`; mean teacher-called summed inference time is `9.28s` in the four-replica no-queue PE measurement.

Formal training uses one replica. The ratio describes additional per-sample generation work and does not claim that four-replica PE wall time is the formal single-replica training wall time.

### Current Role and Limitation

Hard-gate v2 is the current leader for the requested equal I/gold objective. It is implemented and evaluated in the PE harness but is not yet integrated into the formal training Teacher runtime.

The 128-case combination holdout has already been used for strategy diagnosis. A new untouched rollout sample is required before treating the result as an unbiased final production estimate.

## Holdout Diagnostics

These are three-run teacher-called means. R5 is the original frozen holdout result. The production and Hard-gate rows are useful diagnostics, but the holdout is no longer untouched for further selection.

| Strategy | I P | I R | I F1 | Gold F1 | Manual-answer F1 | Equal objective | Mean elapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Training production prompt | 0.9478 | 0.8235 | 0.8812 | 0.4141 | 0.7615 | 0.6476 | 1.0000x |
| Single-prompt R5 | 0.9321 | 0.8039 | 0.8632 | 0.7667 | 0.6156 | 0.8149 | 1.0000x |
| Hard-gate v2 | 0.9478 | 0.8235 | 0.8812 | 0.9000 | 0.5267 | 0.8906 | 1.2307x |

Again, Hard-gate v2 exactly preserves production I metrics while changing only the non-I answer path.

## Rejected Control: Dual-All v2

`dual_all_r5_gold_f1_08_literal_canonical_v2` called Stage B for every case and allowed it to override Stage-A I when Stage B returned S with gold token-F1 at least `0.8`.

Its dev teacher-called result looked strong: I F1 `0.8812`, gold coverage `0.7220`, objective `0.8016`, and elapsed `1.7904x`. However, holdout I recall collapsed to `0.6993` and I F1 to `0.8045`. Gold coverage reached `0.9222`, but the model frequently treated a literal gold mention as proof of the requested relation even when an evidence bridge was missing.

This control establishes the reason for the hard gate: gold overlap can improve answer wording, but it is not a reliable sufficiency override.

## Selection Guidance

| Requirement | Recommended strategy | Reason |
| --- | --- | --- |
| No reference gold available | Training production prompt | Strongest preserved evidence-only I behavior and one call. |
| Gold available, strictly one call | Single-prompt R5 | Best stable single-call equal objective. |
| Gold available, up to `2x`, equal I/gold priority | Hard-gate v2 | Preserves production I exactly and gives the best balanced objective. |

The next scientifically valid step is production-runtime integration followed by validation on a newly sampled, untouched rollout set. The 3500 evaluation set should remain a final actor-policy generalization evaluation rather than a prompt-tuning set.
