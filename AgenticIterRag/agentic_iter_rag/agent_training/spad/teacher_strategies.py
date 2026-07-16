"""Frozen production Teacher strategies for SPAD Stage 1 reward."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from agentic_iter_rag.metrics.answer_metrics import legacy_token_f1, normalize_answer


SINGLE_PROMPT_STRATEGY_ID = "spad_teacher_single_prompt_v1"
HARD_GATE_R5_LITERAL_CANONICAL_V2 = "spad_teacher_hard_gate_r5_literal_canonical_v2"
HARD_GATE_STAGE_A_PROMPT_VERSION = "spad_teacher_evidence_status_answer_v2"
HARD_GATE_STAGE_B_PROMPT_VERSION = "gold_support_evidence_only_v3"

_OUTPUT_CONTRACT_BASE = (
    "Output only three XML blocks in this exact order: "
    "<reason>...</reason><status>...</status><answer>...</answer>. "
    "The first character must be <. Do not use markdown or repeat these instructions. "
    "The status must be exactly supported_answer, insufficient_evidence, or ambiguous_evidence. "
    "For insufficient_evidence or ambiguous_evidence, answer exactly 证据不足无法作答. "
    "For supported_answer, answer with the shortest evidence span that matches gold-answer style: no explanation, "
    "no label or prefix, no sentence, and no alternative list unless the Original question explicitly requests a list."
)
_OUTPUT_CONTRACT_STRICT = (
    _OUTPUT_CONTRACT_BASE
    + " The reason must be one short sentence under 60 words and must close with </reason> before status."
)
GOLD_SUPPORT_EVIDENCE_ONLY_SYSTEM_PROMPT = (
    "You are an evidence-grounded judge for factoid QA. The user supplies an Original question, a Reference gold "
    "answer, and Search evidence. The gold is only a candidate to verify and is never evidence. Do not claim the gold "
    "is supported unless the Search evidence establishes the exact Original-question relation. "
    + _OUTPUT_CONTRACT_STRICT
    + " Identify the target entity, predicate, scope, and complete candidates. If evidence completely supports the "
    "gold and no equally matching competitor exists, use supported_answer. If it supports the gold and one or more "
    "equally matching incompatible candidates, use ambiguous_evidence. If it does not support the gold but supports "
    "one complete different answer, use supported_answer with that evidence answer. If it supports multiple complete "
    "different answers, use ambiguous_evidence. If neither the gold nor any other complete answer is supported, use "
    "insufficient_evidence. Different predicates and incomplete bridge chains are not candidates."
)


@dataclass(frozen=True)
class TeacherStrategySpec:
    strategy_id: str
    strategy_type: str
    stage_a_prompt_version: str
    stage_b_prompt_version: str = ""
    stage_b_scope: str = ""
    preserve_i_boundary: bool = False
    prefer_higher_gold_f1_between_supported: bool = False
    canonicalize_evidence_literal_gold: bool = False
    fallback_to_stage_a_on_stage_b_failure: bool = False


TEACHER_STRATEGY_REGISTRY: dict[str, TeacherStrategySpec] = {
    SINGLE_PROMPT_STRATEGY_ID: TeacherStrategySpec(
        strategy_id=SINGLE_PROMPT_STRATEGY_ID,
        strategy_type="single_prompt",
        stage_a_prompt_version=HARD_GATE_STAGE_A_PROMPT_VERSION,
    ),
    HARD_GATE_R5_LITERAL_CANONICAL_V2: TeacherStrategySpec(
        strategy_id=HARD_GATE_R5_LITERAL_CANONICAL_V2,
        strategy_type="hard_gate",
        stage_a_prompt_version=HARD_GATE_STAGE_A_PROMPT_VERSION,
        stage_b_prompt_version=HARD_GATE_STAGE_B_PROMPT_VERSION,
        stage_b_scope="stage_a_non_i",
        preserve_i_boundary=True,
        prefer_higher_gold_f1_between_supported=True,
        canonicalize_evidence_literal_gold=True,
        fallback_to_stage_a_on_stage_b_failure=True,
    ),
}


def resolve_teacher_strategy(strategy_id: str | None) -> TeacherStrategySpec:
    resolved = str(strategy_id or SINGLE_PROMPT_STRATEGY_ID).strip()
    try:
        return TEACHER_STRATEGY_REGISTRY[resolved]
    except KeyError as exc:
        available = ", ".join(sorted(TEACHER_STRATEGY_REGISTRY))
        raise ValueError(
            f"Unknown SPAD teacher strategy_id {resolved!r}; available: {available}"
        ) from exc


def validate_teacher_strategy_config(teacher_cfg: Mapping[str, Any]) -> TeacherStrategySpec:
    """Resolve a strategy and reject config that changes its frozen semantics."""

    spec = resolve_teacher_strategy(str(teacher_cfg.get("strategy_id") or ""))
    configured_prompt = str(teacher_cfg.get("prompt_version") or spec.stage_a_prompt_version)
    if spec.strategy_type == "hard_gate" and configured_prompt != spec.stage_a_prompt_version:
        raise ValueError(
            f"SPAD teacher strategy {spec.strategy_id!r} requires stage-A prompt "
            f"{spec.stage_a_prompt_version!r}, got {configured_prompt!r}"
        )
    raw_strategy = teacher_cfg.get("strategy")
    if raw_strategy is None:
        return spec
    if not isinstance(raw_strategy, Mapping):
        raise ValueError("teacher_answerer.strategy must be a mapping")
    expected = {
        "type": spec.strategy_type,
        "stage_a_prompt_version": spec.stage_a_prompt_version,
        "stage_b_prompt_version": spec.stage_b_prompt_version,
        "stage_b_scope": spec.stage_b_scope,
        "preserve_i_boundary": spec.preserve_i_boundary,
        "prefer_higher_gold_f1_between_supported": (
            spec.prefer_higher_gold_f1_between_supported
        ),
        "canonicalize_evidence_literal_gold": spec.canonicalize_evidence_literal_gold,
        "fallback_to_stage_a_on_stage_b_failure": (
            spec.fallback_to_stage_a_on_stage_b_failure
        ),
    }
    for key, expected_value in expected.items():
        actual = raw_strategy.get(key, expected_value)
        if actual != expected_value:
            raise ValueError(
                f"SPAD teacher strategy {spec.strategy_id!r} freezes {key}="
                f"{expected_value!r}, got {actual!r}"
            )
    unknown = sorted(set(raw_strategy) - set(expected))
    if unknown:
        raise ValueError(f"Unknown teacher_answerer.strategy keys: {unknown}")
    return spec


def strategy_config_dict(spec: TeacherStrategySpec) -> dict[str, Any]:
    return {
        "type": spec.strategy_type,
        "stage_a_prompt_version": spec.stage_a_prompt_version,
        "stage_b_prompt_version": spec.stage_b_prompt_version,
        "stage_b_scope": spec.stage_b_scope,
        "preserve_i_boundary": spec.preserve_i_boundary,
        "prefer_higher_gold_f1_between_supported": (
            spec.prefer_higher_gold_f1_between_supported
        ),
        "canonicalize_evidence_literal_gold": spec.canonicalize_evidence_literal_gold,
        "fallback_to_stage_a_on_stage_b_failure": (
            spec.fallback_to_stage_a_on_stage_b_failure
        ),
    }


def _indent(value: Any, spaces: int) -> str:
    prefix = " " * spaces
    return prefix + str(value or "").replace("\n", f"\n{prefix}")


def build_gold_support_evidence_only_messages(
    *,
    question: str,
    gold_answers: Sequence[Any],
    evidence_steps: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the frozen R5 gold-aware prompt without sub-query strings."""

    lines = [
        "   Original question:",
        _indent(question, 6),
        "",
        "   Reference gold answer:",
        _indent(json.dumps(list(gold_answers), ensure_ascii=False), 6),
        "",
        "   Search evidence:",
    ]
    if not evidence_steps:
        lines.extend(["", "      (no search evidence provided)"])
    for index, step in enumerate(evidence_steps, start=1):
        lines.extend(["", f"      Round {index}:", "         retrieved contents:"])
        for doc_index, doc in enumerate((step.get("docs") or [])[:5], start=1):
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_index}"
            contents = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            lines.append(f"            [{doc_index}] {title}")
            lines.append(_indent(contents, 15))
    lines.extend(
        [
            "",
            "   Now output the final result directly. "
            "Do not analyze the instruction. Do not repeat rules. Begin with <reason>.",
        ]
    )
    return [
        {"role": "system", "content": GOLD_SUPPORT_EVIDENCE_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def find_evidence_literal_gold(
    evidence_steps: list[dict[str, Any]], gold_answers: Sequence[Any]
) -> str:
    evidence_parts: list[str] = []
    for step in evidence_steps:
        for doc in step.get("docs") or []:
            contents = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            evidence_parts.extend((str(doc.get("title") or ""), str(contents)))
    normalized_evidence = f" {normalize_answer(' '.join(evidence_parts))} "
    for gold in gold_answers:
        normalized_gold = normalize_answer(gold)
        if normalized_gold and f" {normalized_gold} " in normalized_evidence:
            return str(gold)
    return ""


def _parsed_non_i(detail: Mapping[str, Any] | None) -> bool:
    return bool(
        detail
        and not detail.get("teacher_format_error")
        and detail.get("teacher_parse_status") == "parsed"
        and detail.get("teacher_evidence_status")
        in {"supported_answer", "ambiguous_evidence"}
    )


def select_hard_gate_output(
    *,
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    gold_answers: Sequence[Any],
    evidence_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the PE Hard-Gate v2 merge rules deterministically."""

    stage_a_status = str(stage_a.get("teacher_evidence_status") or "")
    stage_a_parsed = bool(
        not stage_a.get("teacher_format_error")
        and stage_a.get("teacher_parse_status") == "parsed"
    )
    stage_b_non_i = _parsed_non_i(stage_b)
    use_stage_b = bool(stage_a_parsed and stage_a_status in {
        "supported_answer",
        "ambiguous_evidence",
    } and stage_b_non_i)
    selection_reason = "stage_a_i_or_format"
    if use_stage_b:
        selection_reason = "stage_b_non_i"
    elif stage_b is not None and stage_a_status != "insufficient_evidence":
        selection_reason = "stage_b_invalid_fallback"

    if (
        use_stage_b
        and stage_a_status == "supported_answer"
        and str(stage_b.get("teacher_evidence_status") or "") != "supported_answer"
    ):
        use_stage_b = False
        selection_reason = "stage_a_only_supported"
    elif use_stage_b and stage_a_status == "supported_answer":
        stage_a_f1 = legacy_token_f1(stage_a.get("teacher_answer"), gold_answers)
        stage_b_f1 = legacy_token_f1(stage_b.get("teacher_answer"), gold_answers)
        if stage_a_f1 > stage_b_f1:
            use_stage_b = False
            selection_reason = "stage_a_supported_higher_gold_f1"

    selected = dict(stage_b if use_stage_b else stage_a)
    answer = str(selected.get("teacher_answer") or "")
    canonical_gold = ""
    if (
        selected.get("teacher_parse_status") == "parsed"
        and not selected.get("teacher_format_error")
        and selected.get("teacher_evidence_status") == "supported_answer"
    ):
        candidate = find_evidence_literal_gold(evidence_steps, gold_answers)
        if candidate and legacy_token_f1(candidate, gold_answers) > legacy_token_f1(
            answer, gold_answers
        ):
            answer = candidate
            canonical_gold = candidate
            selection_reason += "+evidence_literal_gold"
    selected["teacher_answer"] = answer
    return {
        "selected": selected,
        "stage_b_used": use_stage_b,
        "canonical_gold": canonical_gold,
        "selection_reason": selection_reason,
    }
