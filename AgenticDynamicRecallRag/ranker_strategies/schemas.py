"""Schemas for ranker contrastive training.

The schemas are intentionally plain dataclasses so they can be serialized by
Ray and unit-tested without importing the full VERL stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RankedPassage:
    doc_id: str
    rank: int
    text: str
    title: str | None = None
    recall_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCallContext:
    trajectory_id: str
    tool_call_id: str
    turn_idx: int
    origin_query: str
    sub_query: str
    golden_answers: list[str]
    trajectory_score: float
    score_type: str
    ranked_passages: list[RankedPassage]
    recall_top50_docs: list[RankedPassage] = field(default_factory=list)
    rank_top50_docs: list[RankedPassage] = field(default_factory=list)
    rank_top5_docs: list[RankedPassage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RankerTrajectory:
    trajectory_id: str
    origin_query: str
    golden_answers: list[str]
    final_answer: str
    score: float
    score_type: str = "f1"
    is_valid: bool = True
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LabeledPassage(RankedPassage):
    label: int = 0
    label_source: str = ""


@dataclass(slots=True)
class LabeledRankingContext:
    trajectory_id: str
    tool_call_id: str
    turn_idx: int
    origin_query: str
    sub_query: str
    trajectory_score: float
    score_type: str
    passages: list[LabeledPassage]
    golden_answers: list[str] = field(default_factory=list)
    recall_top50_docs: list[RankedPassage] = field(default_factory=list)
    rank_top50_docs: list[RankedPassage] = field(default_factory=list)
    rank_top5_docs: list[RankedPassage] = field(default_factory=list)
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
    sample_source: str = "fresh"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def documents(self) -> list[RankedPassage]:
        return [self.positive, *self.negatives]
