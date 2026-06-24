"""Plain schemas for async ranker signal labeling.

These dataclasses are intentionally independent from VERL runtime classes so
they can be serialized, logged and tested in isolation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class CandidateChunk:
    doc_id: str
    text: str
    title: str | None = None
    recall_rank: int | None = None
    recall_score: float | None = None
    rank_rank: int | None = None
    rank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AsyncLabelRequest:
    request_id: str
    created_global_step: int
    origin_query: str
    sub_query: str
    trajectory_id: str
    tool_call_id: str
    ranked_chunk_list: list[CandidateChunk]
    turn_idx: int = 0
    trajectory_score: float = 0.0
    score_type: str = "unknown"
    trace_metadata: dict[str, Any] = field(default_factory=dict)
    label_policy: str = "llm_judge_rank50"
    prompt_version: str = ""

    def validate_rank50(self) -> None:
        if len(self.ranked_chunk_list) != 50:
            raise ValueError(f"ranked_chunk_list must contain exactly 50 chunks, got {len(self.ranked_chunk_list)}")
        seen: set[str] = set()
        for idx, chunk in enumerate(self.ranked_chunk_list):
            if not chunk.doc_id:
                raise ValueError(f"ranked_chunk_list[{idx}] is missing doc_id")
            if chunk.doc_id in seen:
                raise ValueError(f"duplicated doc_id in ranked_chunk_list: {chunk.doc_id}")
            if not chunk.text:
                raise ValueError(f"ranked_chunk_list[{idx}] is missing text")
            seen.add(chunk.doc_id)


@dataclass(slots=True)
class JudgeChunkScore:
    doc_id: str
    judge_rank: int
    judge_score: float
    raw_score: float | None = None


@dataclass(slots=True)
class RankedPassage:
    doc_id: str
    rank: int
    text: str
    title: str | None = None
    recall_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ContrastiveSample:
    sample_id: str
    query_input: str
    origin_query: str
    sub_query: str
    positive: RankedPassage
    negatives: list[RankedPassage]
    positive_doc_index: int
    label_source: str
    trajectory_id: str
    tool_call_id: str
    recall_top50_docs: list[RankedPassage] = field(default_factory=list)
    rank_top50_docs: list[RankedPassage] = field(default_factory=list)
    rank_top5_docs: list[RankedPassage] = field(default_factory=list)
    turn_idx: int = 0
    loss_weight: float = 1.0
    sample_source: str = "async_labeling"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def documents(self) -> list[RankedPassage]:
        return [self.positive, *self.negatives]


@dataclass(slots=True)
class CandidateSignalData:
    signal_id: str
    created_global_step: int
    completed_at: float
    origin_query: str
    sub_query: str
    trajectory_id: str
    tool_call_id: str
    ranked_chunk_list: list[CandidateChunk]
    judge_scores: list[JudgeChunkScore]
    extra_scores: dict[str, Any] | None = None
    final_scores: list[JudgeChunkScore] = field(default_factory=list)
    label_source: str = "llm_judge"
    score_version: str = "rank50_linear_v1"
    prompt_version: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AsyncLabelFailure:
    request_id: str
    created_global_step: int
    failed_at: float
    error_type: str
    error_message: str
    trajectory_id: str = ""
    tool_call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
