"""Prompt builders for SPAD-RAG actor and teacher roles."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any


TEACHER_SYSTEM_PROMPT = (
    "You are an evidence-grounded QA model. Output only two XML tag blocks: "
    "<reason>...</reason><answer>...</answer>. The first character must be <. "
    "Do not repeat instructions. Do not use markdown or numbered lists. Use only the evidence. "
    "The answer must be only the final short answer span, usually one person, date, place, number, or title. "
    "Keep names, dates, places, titles, and numbers exactly as written in the evidence when possible. "
    "If the evidence is insufficient, explain what evidence is missing and why it is insufficient in reason, "
    "and answer exactly 证据不足无法作答."
)

TEACHER_STATUS_SYSTEM_PROMPT = (
    "You are an evidence-grounded QA model. Output only three XML tag blocks: "
    "<reason>...</reason><status>...</status><answer>...</answer>. The first character must be <. "
    "Do not repeat instructions. Do not use markdown or numbered lists. Use only the evidence. "
    "The status must be exactly one of supported_answer, insufficient_evidence, ambiguous_evidence. "
    "Use supported_answer only when the evidence supports a single short answer. "
    "Use insufficient_evidence when the evidence is missing necessary facts. "
    "Use ambiguous_evidence when the evidence supports multiple incompatible answers. "
    "The reason must briefly state the supporting evidence, or state what evidence is missing and why the current evidence is insufficient. "
    "The answer must be only the final short answer span, usually one person, date, place, number, or title. "
    "Keep names, dates, places, titles, and numbers exactly as written in the evidence when possible. "
    "If the evidence is insufficient or ambiguous, answer exactly 证据不足无法作答."
)

LEGACY_TEACHER_ANSWER_PROMPT_VERSION = "spad_teacher_answer_v1"
TEACHER_ANSWER_PROMPT_VERSION = "spad_teacher_answer_v2"
HISTORICAL_TEACHER_STATUS_PROMPT_VERSION = "spad_teacher_evidence_status_answer_v1"
DEFAULT_TEACHER_STATUS_PROMPT_VERSION = "spad_teacher_evidence_status_answer_v2"


def _indent_prompt_block(value: Any, spaces: int) -> str:
    prefix = " " * spaces
    return prefix + str(value or "").replace("\n", f"\n{prefix}")


def _build_teacher_user_prompt_v1(question: str, evidence_steps: list[dict[str, Any]]) -> str:
    lines = [f"Original question:\n{question}", "", "Search evidence:"]
    if not evidence_steps:
        lines.append("(no search evidence provided)")
    for idx, step in enumerate(evidence_steps, start=1):
        lines.append(f"\nRound {idx} sub_query:\n{step.get('sub_query') or ''}")
        docs = step.get("docs") or []
        for doc_idx, doc in enumerate(docs[:5], start=1):
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_idx}"
            text = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            lines.append(f"[{doc_idx}] {title}\n{text}")
    lines.append(
        "\nNow output the final result directly. "
        "Do not analyze the instruction. Do not repeat rules. Begin with <reason>."
    )
    return "\n".join(lines)


def _build_teacher_user_prompt_v2(question: str, evidence_steps: list[dict[str, Any]]) -> str:
    lines = ["   Original question:", _indent_prompt_block(question, 6), "", "   Search evidence:"]
    if not evidence_steps:
        lines.extend(["", "      (no search evidence provided)"])
    for idx, step in enumerate(evidence_steps, start=1):
        lines.extend(
            [
                "",
                f"      Round {idx}:",
                f"         sub_query: {step.get('sub_query') or ''}",
                "         retrieved contents:",
            ]
        )
        docs = step.get("docs") or []
        for doc_idx, doc in enumerate(docs[:5], start=1):
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_idx}"
            text = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            lines.append(f"            [{doc_idx}] {title}")
            lines.append(_indent_prompt_block(text, 15))
    lines.extend(
        [
            "",
            "   Now output the final result directly. "
            "Do not analyze the instruction. Do not repeat rules. Begin with <reason>.",
        ]
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class TeacherPromptSpec:
    system_prompt: str
    include_status: bool
    user_prompt_builder: Callable[[str, list[dict[str, Any]]], str]


TEACHER_PROMPT_REGISTRY: dict[str, TeacherPromptSpec] = {
    LEGACY_TEACHER_ANSWER_PROMPT_VERSION: TeacherPromptSpec(
        system_prompt=TEACHER_SYSTEM_PROMPT,
        include_status=False,
        user_prompt_builder=_build_teacher_user_prompt_v1,
    ),
    TEACHER_ANSWER_PROMPT_VERSION: TeacherPromptSpec(
        system_prompt=TEACHER_SYSTEM_PROMPT,
        include_status=False,
        user_prompt_builder=_build_teacher_user_prompt_v2,
    ),
    HISTORICAL_TEACHER_STATUS_PROMPT_VERSION: TeacherPromptSpec(
        system_prompt=TEACHER_STATUS_SYSTEM_PROMPT,
        include_status=True,
        user_prompt_builder=_build_teacher_user_prompt_v1,
    ),
    DEFAULT_TEACHER_STATUS_PROMPT_VERSION: TeacherPromptSpec(
        system_prompt=TEACHER_STATUS_SYSTEM_PROMPT,
        include_status=True,
        user_prompt_builder=_build_teacher_user_prompt_v2,
    ),
}


def resolve_teacher_prompt(
    prompt_version: str | None,
    *,
    include_status: bool | None = None,
) -> tuple[str, TeacherPromptSpec]:
    """Resolve a configured prompt version and validate its output contract."""

    version = str(prompt_version or "").strip()
    if not version:
        version = DEFAULT_TEACHER_STATUS_PROMPT_VERSION if include_status else TEACHER_ANSWER_PROMPT_VERSION
    spec = TEACHER_PROMPT_REGISTRY.get(version)
    if spec is None:
        available = ", ".join(sorted(TEACHER_PROMPT_REGISTRY))
        raise ValueError(f"Unknown SPAD teacher prompt_version {version!r}; available: {available}")
    if include_status is not None and bool(include_status) != spec.include_status:
        raise ValueError(
            f"SPAD teacher prompt_version {version!r} has include_status={spec.include_status}, "
            f"but the caller requested include_status={bool(include_status)}"
        )
    return version, spec


def build_teacher_messages(
    question: str,
    evidence_steps: list[dict[str, Any]],
    *,
    include_status: bool | None = None,
    prompt_version: str | None = None,
) -> list[dict[str, str]]:
    """Build an evidence-only QA prompt for the teacher answerer."""

    _, spec = resolve_teacher_prompt(prompt_version, include_status=include_status)
    return [
        {"role": "system", "content": spec.system_prompt},
        {"role": "user", "content": spec.user_prompt_builder(question, evidence_steps)},
    ]


def smoke_actor_answer(question: str) -> str:
    """Return a deterministic actor answer used only by the smoke backend."""

    del question
    return "<reason>Smoke backend does not run actor generation.</reason>\n<answer>__SMOKE_ACTOR_ANSWER__</answer>"
