"""Trajectory selection and DataProto extraction for ranker training."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..schemas import RankedPassage, RankerTrajectory, ToolCallContext


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _text_from_doc(doc: dict[str, Any]) -> str:
    return str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")


def _passage_from_raw(raw: dict[str, Any], rank: int) -> RankedPassage | None:
    if not isinstance(raw, dict):
        return None
    doc_id = str(raw.get("doc_id") or raw.get("id") or raw.get("_id") or "")
    text = _text_from_doc(raw)
    if not doc_id and text:
        doc_id = f"rank-{rank}"
    if not text:
        return None
    score = raw.get("recall_score", raw.get("score"))
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    return RankedPassage(
        doc_id=doc_id,
        rank=int(raw.get("rank", rank)),
        title=raw.get("title") or None,
        text=text,
        recall_score=score,
        metadata={k: v for k, v in raw.items() if k not in {"id", "doc_id", "contents", "text", "passage", "title", "score", "rank"}},
    )


def _ranked_passage_from_raw(raw: dict[str, Any], rank: int) -> RankedPassage | None:
    """Parse a ranker output doc.

    For ranked passages, ``rank`` must mean the ranker order, not the
    original recall rank. Recall rank is preserved in metadata.
    """
    if not isinstance(raw, dict):
        return None
    doc_id = str(raw.get("doc_id") or raw.get("id") or raw.get("_id") or "")
    if not doc_id:
        raise ValueError("missing required ranker trace field: doc_id")
    text = _text_from_doc(raw)
    if not text:
        return None
    metadata = dict(raw.get("metadata") or {})
    rank_score = raw.get("rank_score", metadata.get("rank_score"))
    if rank_score in (None, ""):
        raise ValueError("missing required ranker trace field: rank_score")
    try:
        score = float(rank_score) if rank_score is not None else None
    except (TypeError, ValueError):
        score = None
    rank_rank = raw.get("rank_rank", metadata.get("rank_rank"))
    if rank_rank in (None, ""):
        raise ValueError("missing required ranker trace field: rank_rank")
    rank_value = int(rank_rank)
    metadata["rank_score"] = score
    metadata["rank_rank"] = rank_value
    return RankedPassage(
        doc_id=doc_id,
        rank=rank_value,
        title=raw.get("title") or None,
        text=text,
        recall_score=score,
        metadata={
            **metadata,
            **{
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "id",
                    "doc_id",
                    "contents",
                    "text",
                    "passage",
                    "title",
                    "score",
                    "rank",
                    "metadata",
                }
            },
        },
    )


def build_fresh_trajectories_from_dataproto(main_batch, global_steps: int) -> list[dict[str, Any]]:
    """Extract ranker trajectories from a VERL DataProto rollout batch.

    CoAgenticRetrieverAgentLoop stores tool-call details in ``non_tensor_batch``. This
    function normalizes those object arrays into the trajectory schema expected
    by ranker strategy modules.
    """
    non_tensor = getattr(main_batch, "non_tensor_batch", {}) or {}
    if "tool_call_details" not in non_tensor:
        return []

    details_list = _as_list(non_tensor.get("tool_call_details"))
    initial_queries = _as_list(non_tensor.get("initial_query"))
    answers_list = _as_list(non_tensor.get("answers"))
    messages_list = _as_list(non_tensor.get("messages"))
    uids = _as_list(non_tensor.get("uid"))
    reward_values = _as_list(non_tensor.get("reward"))

    trajectories: list[dict[str, Any]] = []
    for idx, tool_details in enumerate(details_list):
        if not tool_details:
            continue
        if idx >= len(initial_queries) or initial_queries[idx] in (None, ""):
            raise ValueError("missing required ranker trajectory field: initial_query")
        origin_query = str(initial_queries[idx])

        answers = []
        if idx < len(answers_list):
            answers = [str(x) for x in _as_list(answers_list[idx])]

        score = _trajectory_score(tool_details, reward_values[idx] if idx < len(reward_values) else None)
        if idx >= len(uids) or uids[idx] in (None, ""):
            raise ValueError("missing required ranker trajectory field: uid")
        trajectory_id = str(uids[idx])
        messages = messages_list[idx] if idx < len(messages_list) and isinstance(messages_list[idx], list) else []

        tool_calls = []
        trajectory_score_type: str | None = None
        for call_idx, detail in enumerate(_as_list(tool_details)):
            if not isinstance(detail, dict):
                continue
            if str(_require_trajectory_field(detail, "trajectory_id")) != trajectory_id:
                raise ValueError("ranker trajectory_id mismatch between detail and uid")
            if str(_require_trajectory_field(detail, "origin_query")) != origin_query:
                raise ValueError("ranker origin_query mismatch between detail and non_tensor_batch")
            detail_score_type = str(_require_trajectory_field(detail, "score_type"))
            if trajectory_score_type is None:
                trajectory_score_type = detail_score_type
            elif trajectory_score_type != detail_score_type:
                raise ValueError("ranker score_type mismatch inside trajectory")
            documents = (
                detail.get("recall_top50_docs")
                or detail.get("top_50_documents")
                or detail.get("top_n_documents")
                or []
            )
            recall_passages = []
            for rank, raw_doc in enumerate(_as_list(documents), start=1):
                passage = _passage_from_raw(raw_doc, rank)
                if passage is not None:
                    recall_passages.append(
                        {
                            "doc_id": passage.doc_id,
                            "rank": passage.rank,
                            "title": passage.title,
                            "text": passage.text,
                            "recall_score": passage.recall_score,
                            "metadata": passage.metadata,
                        }
                    )
            rank_documents = detail.get("rank_top50_docs") or detail.get("ranked_passages") or []
            rank_docs = []
            for rank, raw_doc in enumerate(_as_list(rank_documents), start=1):
                passage = _ranked_passage_from_raw(raw_doc, rank)
                if passage is not None:
                    rank_docs.append(
                        {
                            "doc_id": passage.doc_id,
                            "rank": passage.rank,
                            "rank_rank": passage.rank,
                            "rank_score": passage.metadata["rank_score"],
                            "title": passage.title,
                            "text": passage.text,
                            "recall_score": passage.recall_score,
                            "metadata": passage.metadata,
                        }
                    )
            rank_top5_docs = []
            for rank, raw_doc in enumerate(_as_list(detail.get("rank_top5_docs")), start=1):
                passage = _ranked_passage_from_raw(raw_doc, rank)
                if passage is not None:
                    rank_top5_docs.append(
                        {
                            "doc_id": passage.doc_id,
                            "rank": passage.rank,
                            "rank_rank": passage.rank,
                            "rank_score": passage.metadata["rank_score"],
                            "title": passage.title,
                            "text": passage.text,
                            "recall_score": passage.recall_score,
                            "metadata": passage.metadata,
                        }
                    )
            tool_calls.append(
                {
                    "tool_call_id": str(_require_trajectory_field(detail, "tool_call_id")),
                    "turn_idx": int(_require_trajectory_field(detail, "turn_idx")),
                    "tool_name": "search",
                    "sub_query": str(detail.get("sub_query") or ""),
                    "recall_top50_docs": recall_passages,
                    "rank_top50_docs": rank_docs,
                    "ranked_passages": rank_docs,
                    "rank_top5_docs": rank_top5_docs,
                    "metadata": {
                        "tool_score": detail.get("tool_score"),
                        "answer_in_docs": detail.get("answer_in_docs"),
                        "search_tool_error": detail.get("search_tool_error"),
                    },
                }
            )

        if trajectory_score_type in (None, ""):
            raise ValueError("missing required ranker trajectory field: score_type")

        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "origin_query": origin_query,
                "golden_answers": answers,
                "final_answer": "",
                "score": score,
                "score_type": trajectory_score_type,
                "is_valid": bool(origin_query and tool_calls),
                "messages": messages,
                "tool_calls": tool_calls,
                "metadata": {"global_steps": global_steps},
            }
        )
    return trajectories


def _trajectory_score(tool_details: Any, reward_value: Any) -> float:
    try:
        if reward_value is not None and not (isinstance(reward_value, float) and math.isnan(reward_value)):
            return float(reward_value)
    except (TypeError, ValueError):
        pass
    scores = []
    for detail in _as_list(tool_details):
        if isinstance(detail, dict):
            try:
                if "tool_score" not in detail or detail["tool_score"] in (None, ""):
                    continue
                scores.append(float(detail["tool_score"]))
            except (TypeError, ValueError):
                continue
    if not scores:
        raise ValueError("missing required ranker trajectory score: reward/tool_score")
    return max(scores)


def _require_trajectory_field(item: dict[str, Any], key: str):
    if key not in item or item[key] in (None, ""):
        raise ValueError(f"missing required ranker trajectory field: {key}")
    return item[key]


class TrajectoryContextParser:
    """Parse raw rollout records into ranker trajectory/context objects."""

    def select(self, fresh_trajectories: list[dict[str, Any] | RankerTrajectory]) -> list[ToolCallContext]:
        raise NotImplementedError

    def _parse_trajectory(self, item: dict[str, Any] | RankerTrajectory) -> RankerTrajectory | None:
        if isinstance(item, RankerTrajectory):
            return item
        if not isinstance(item, dict):
            return None
        return RankerTrajectory(
            trajectory_id=str(_require_trajectory_field(item, "trajectory_id")),
            origin_query=str(item.get("origin_query") or ""),
            golden_answers=[str(x) for x in _as_list(item.get("golden_answers"))],
            final_answer=str(item.get("final_answer") or ""),
            score=float(_require_trajectory_field(item, "score")),
            score_type=str(_require_trajectory_field(item, "score_type")),
            is_valid=bool(_require_trajectory_field(item, "is_valid")),
            messages=_as_list(item.get("messages")),
            tool_calls=_as_list(item.get("tool_calls")),
            metadata=dict(item.get("metadata") or {}),
        )

    def _parse_context(
        self,
        trajectory: RankerTrajectory,
        tool_call: dict[str, Any],
        call_idx: int,
    ) -> ToolCallContext | None:
        if not isinstance(tool_call, dict):
            return None
        sub_query = str(tool_call.get("sub_query") or tool_call.get("query") or "")
        raw_recall_passages = _as_list(
            tool_call.get("recall_top50_docs")
            or tool_call.get("retrieved_passages")
            or tool_call.get("passages")
        )
        recall_passages = []
        for rank, raw in enumerate(raw_recall_passages, start=1):
            passage = _passage_from_raw(raw, rank)
            if passage is not None:
                recall_passages.append(passage)

        raw_rank_passages = _as_list(tool_call.get("rank_top50_docs") or tool_call.get("ranked_passages"))
        ranked_passages = []
        for rank, raw in enumerate(raw_rank_passages, start=1):
            passage = _ranked_passage_from_raw(raw, rank)
            if passage is not None:
                ranked_passages.append(passage)

        rank_top5_passages = []
        for rank, raw in enumerate(_as_list(tool_call.get("rank_top5_docs")), start=1):
            passage = _ranked_passage_from_raw(raw, rank)
            if passage is not None:
                rank_top5_passages.append(passage)

        if not trajectory.origin_query or not sub_query or not recall_passages or not ranked_passages or not rank_top5_passages:
            return None

        return ToolCallContext(
            trajectory_id=trajectory.trajectory_id,
            tool_call_id=str(_require_trajectory_field(tool_call, "tool_call_id")),
            turn_idx=int(_require_trajectory_field(tool_call, "turn_idx")),
            origin_query=trajectory.origin_query,
            sub_query=sub_query,
            golden_answers=trajectory.golden_answers,
            trajectory_score=trajectory.score,
            score_type=trajectory.score_type,
            ranked_passages=ranked_passages[:50],
            recall_top50_docs=recall_passages[:50],
            rank_top50_docs=ranked_passages[:50],
            rank_top5_docs=rank_top5_passages[:5],
            metadata=dict(tool_call.get("metadata") or {}),
        )


class TopF1TrajectorySelector(TrajectoryContextParser):
    """Select the top scoring valid trajectories and return all search contexts."""

    def __init__(self, max_selected_trajectories: int, min_final_reward: float):
        self.max_selected_trajectories = max(1, int(max_selected_trajectories))
        self.min_final_reward = float(min_final_reward)

    def select(self, fresh_trajectories: list[dict[str, Any] | RankerTrajectory]) -> list[ToolCallContext]:
        parsed = [self._parse_trajectory(item) for item in fresh_trajectories]
        valid = [
            trajectory
            for trajectory in parsed
            if trajectory is not None and trajectory.is_valid and trajectory.score >= self.min_final_reward
        ]
        valid.sort(key=lambda item: item.score, reverse=True)
        selected = valid[: self.max_selected_trajectories]

        contexts: list[ToolCallContext] = []
        for trajectory in selected:
            for call_idx, tool_call in enumerate(trajectory.tool_calls):
                context = self._parse_context(trajectory, tool_call, call_idx)
                if context is not None:
                    contexts.append(context)
        return contexts
