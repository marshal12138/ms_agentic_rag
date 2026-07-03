"""Canonical trajectory record schema for AgenticIterRag v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TraceRecord:
    trace_id: str
    sample_id: str
    question: str
    gold_answers: list[str]
    sub_query: str
    recall_topn_docs: list[dict[str, Any]]
    ranked_docs: list[dict[str, Any]]
    visible_docs: list[dict[str, Any]]
    final_answer: str | None = None
    reward: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    raw_trace_ref: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


REQUIRED_TRACE_FIELDS = (
    "trace_id",
    "sample_id",
    "question",
    "gold_answers",
    "sub_query",
    "recall_topn_docs",
    "ranked_docs",
    "visible_docs",
)


def validate_trace_record(record: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_TRACE_FIELDS if field not in record]
    if missing:
        raise ValueError(f"missing trace fields: {missing}")
    if not str(record["sub_query"]).strip():
        raise ValueError("trace sub_query must be non-empty")
    for field in ("recall_topn_docs", "ranked_docs", "visible_docs"):
        if not isinstance(record[field], list):
            raise TypeError(f"trace field {field} must be a list")


def normalize_doc(raw: dict[str, Any], rank: int | None = None) -> dict[str, Any]:
    doc_id = raw.get("doc_id") or raw.get("id") or raw.get("document_id")
    text = raw.get("text") or raw.get("contents") or raw.get("passage") or raw.get("content")
    title = raw.get("title")
    out = dict(raw)
    out["doc_id"] = str(doc_id) if doc_id is not None else f"doc-{rank or 0}"
    out["text"] = str(text) if text is not None else ""
    if title is not None:
        out["title"] = str(title)
    if rank is not None and "rank" not in out:
        out["rank"] = rank
    return out
