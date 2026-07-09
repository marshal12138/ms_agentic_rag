"""Prompt builders for SPAD-RAG actor and teacher roles."""

from __future__ import annotations

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


def build_teacher_messages(question: str, evidence_steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build an evidence-only QA prompt for the teacher answerer."""

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
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def smoke_actor_answer(question: str) -> str:
    """Return a deterministic actor answer used only by the smoke backend."""

    del question
    return "<reason>Smoke backend does not run actor generation.</reason>\n<answer>__SMOKE_ACTOR_ANSWER__</answer>"
