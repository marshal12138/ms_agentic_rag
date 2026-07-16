"""Second-stage prompt variants for hard-gated Teacher composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prompt_variants import (
    GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
    build_messages,
    build_user_prompt_evidence_only,
)


NON_I_OUTPUT_CONTRACT = (
    "Output only three XML blocks in this exact order: "
    "<reason>...</reason><status>...</status><answer>...</answer>. "
    "The first character must be <. Do not use markdown or repeat these instructions. "
    "The status must be exactly supported_answer or ambiguous_evidence; never output "
    "insufficient_evidence. For ambiguous_evidence, answer exactly 证据不足无法作答. "
    "For supported_answer, answer with only the shortest evidence span that fills the Original "
    "question's requested answer type."
)

GOLD_NON_I_EXTRACTOR_SYSTEM_PROMPT = (
    "You are the answer stage of a two-stage evidence-grounded factoid QA Teacher. A separate "
    "no-gold gate has already made the binding decision that this case is non-insufficient. Do not "
    "reconsider that gate and do not output insufficient_evidence. The Reference gold answer and "
    "the Stage-A draft are hypotheses, never evidence. Use only Search evidence. "
    + NON_I_OUTPUT_CONTRACT
    + " Identify the Original question's exact entity, predicate, scope, and requested answer type. "
    "If the evidence supports a Reference gold answer or an equivalent alias for that exact relation, "
    "return the shortest supported passage span matching gold-answer style. Otherwise return the "
    "shortest span for the complete different answer supported by the evidence. Use ambiguous_evidence "
    "only when multiple incompatible complete answers equally satisfy the Original question. Do not "
    "return a description, sentence, label, explanation, or unsupported memorized alias. In reason, "
    "briefly name the selected candidate and decisive passage relation."
)

GOLD_DRAFT_SELECTOR_SYSTEM_PROMPT = (
    "You are the candidate-selection stage of a two-stage evidence-grounded factoid QA Teacher. The "
    "Stage-A no-gold gate has already fixed this case as non-insufficient, so you must not output "
    "insufficient_evidence. The Stage-A answer and Reference gold answers are untrusted candidates, "
    "not evidence. "
    + NON_I_OUTPUT_CONTRACT
    + " Compare three candidate sources: the Stage-A answer, each Reference gold answer, and any other "
    "short answer explicitly supported by Search evidence. Verify the exact Original-question entity, "
    "predicate, scope, and every required bridge. Prefer a gold-equivalent candidate only when passages "
    "support that complete relation. If the Stage-A wording is descriptive but points to the right "
    "candidate, replace it with the shortest canonical passage span. If gold is unsupported but another "
    "candidate is supported, return that other candidate. Use ambiguous_evidence only for multiple "
    "incompatible complete candidates. In reason, state which candidate source won and cite the decisive "
    "passage relation."
)


@dataclass(frozen=True)
class CompositePromptVariant:
    name: str
    system_prompt: str
    description: str
    include_stage_a_draft: bool
    reuse_single_prompt_variant: str = ""
    stage_b_scope: str = "stage_a_non_i"
    i_override_min_gold_f1: float | None = None
    prefer_higher_gold_f1_between_supported_stages: bool = False
    canonicalize_evidence_literal_gold: bool = False


COMPOSITE_PROMPT_VARIANTS: dict[str, CompositePromptVariant] = {
    "hard_gate_r5_v1": CompositePromptVariant(
        name="hard_gate_r5_v1",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        description="Binding production I gate followed by the selected R5 verifier, with I fallback blocked.",
        include_stage_a_draft=False,
        reuse_single_prompt_variant="gold_support_evidence_only_v3",
    ),
    "hard_gate_r5_literal_canonical_v2": CompositePromptVariant(
        name="hard_gate_r5_literal_canonical_v2",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        description=(
            "Binding production I gate followed by R5 only for Stage-A non-I cases; preserve the "
            "sole supported answer or prefer the higher-gold-F1 answer when both stages support "
            "one, then canonicalize only to reference literals present in Search evidence."
        ),
        include_stage_a_draft=False,
        reuse_single_prompt_variant="gold_support_evidence_only_v3",
        prefer_higher_gold_f1_between_supported_stages=True,
        canonicalize_evidence_literal_gold=True,
    ),
    "hard_gate_gold_extractor_v1": CompositePromptVariant(
        name="hard_gate_gold_extractor_v1",
        system_prompt=GOLD_NON_I_EXTRACTOR_SYSTEM_PROMPT,
        description="Binding production I gate followed by a dedicated non-I gold-aware answer extractor.",
        include_stage_a_draft=True,
    ),
    "hard_gate_gold_draft_selector_v1": CompositePromptVariant(
        name="hard_gate_gold_draft_selector_v1",
        system_prompt=GOLD_DRAFT_SELECTOR_SYSTEM_PROMPT,
        description="Binding production I gate followed by evidence selection among draft, gold, and other candidates.",
        include_stage_a_draft=True,
    ),
    "dual_all_r5_gold_f1_08_override_v1": CompositePromptVariant(
        name="dual_all_r5_gold_f1_08_override_v1",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        description=(
            "Production no-gold gate plus R5 on every case; override Stage-A I only when R5 returns "
            "supported_answer with gold token-F1 at least 0.8."
        ),
        include_stage_a_draft=False,
        reuse_single_prompt_variant="gold_support_evidence_only_v3",
        stage_b_scope="all",
        i_override_min_gold_f1=0.8,
    ),
    "dual_all_r5_gold_f1_08_literal_canonical_v2": CompositePromptVariant(
        name="dual_all_r5_gold_f1_08_literal_canonical_v2",
        system_prompt=GOLD_SUPPORT_CHECK_SYSTEM_PROMPT,
        description=(
            "Production no-gold gate plus R5 on every case; preserve the 0.8 gold-F1 I override, "
            "preserve the sole supported answer or prefer the higher-gold-F1 answer when both "
            "stages support one, and canonicalize to a reference answer only when its normalized "
            "literal occurs in Search evidence."
        ),
        include_stage_a_draft=False,
        reuse_single_prompt_variant="gold_support_evidence_only_v3",
        stage_b_scope="all",
        i_override_min_gold_f1=0.8,
        prefer_higher_gold_f1_between_supported_stages=True,
        canonicalize_evidence_literal_gold=True,
    ),
}


def composite_strategy_spec(variant: CompositePromptVariant) -> dict[str, Any]:
    return {
        "name": variant.name,
        "system_prompt": variant.system_prompt,
        "include_stage_a_draft": variant.include_stage_a_draft,
        "reuse_single_prompt_variant": variant.reuse_single_prompt_variant,
        "stage_b_scope": variant.stage_b_scope,
        "i_override_min_gold_f1": variant.i_override_min_gold_f1,
        "prefer_higher_gold_f1_between_supported_stages": (
            variant.prefer_higher_gold_f1_between_supported_stages
        ),
        "canonicalize_evidence_literal_gold": variant.canonicalize_evidence_literal_gold,
    }


def build_composite_stage_b_messages(
    case: dict[str, Any], stage_a: dict[str, Any], variant_name: str
) -> list[dict[str, str]]:
    variant = COMPOSITE_PROMPT_VARIANTS[variant_name]
    if variant.reuse_single_prompt_variant:
        return build_messages(case, variant.reuse_single_prompt_variant)

    user_prompt = build_user_prompt_evidence_only(case, include_gold=True)
    marker = "\n   Search evidence:"
    prefix, separator, suffix = user_prompt.partition(marker)
    if not separator:
        raise ValueError("Could not locate Search evidence block in Stage-B prompt")
    draft = (
        "\n\n   Binding Stage-A non-I judgment:\n"
        f"      status: {stage_a.get('predicted_status') or ''}\n"
        f"      proposed answer: {stage_a.get('answer') or ''}\n"
        f"      reason: {stage_a.get('reason') or ''}"
    )
    return [
        {"role": "system", "content": variant.system_prompt},
        {"role": "user", "content": prefix + draft + separator + suffix},
    ]
