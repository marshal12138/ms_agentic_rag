"""Canonical LLM reranker dataset schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RerankerSample:
    query_id: str
    question: str
    sub_query: str
    candidate_docs: list[dict[str, Any]]
    label_policy: str | None
    target_ranking: list[str] = field(default_factory=list)
    positive_doc_ids: list[str] = field(default_factory=list)
    source_trace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_reranker_sample(sample: dict[str, Any]) -> None:
    for field in ("query_id", "question", "sub_query", "candidate_docs", "label_policy", "source_trace_id"):
        if field not in sample:
            raise ValueError(f"missing reranker sample field: {field}")
    docs = sample["candidate_docs"]
    if not isinstance(docs, list) or not docs:
        raise ValueError("candidate_docs must be a non-empty list")
    doc_ids = [str(doc.get("doc_id", "")) for doc in docs if isinstance(doc, dict)]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError(f"candidate_docs contain duplicated doc_id for query_id={sample['query_id']}")
