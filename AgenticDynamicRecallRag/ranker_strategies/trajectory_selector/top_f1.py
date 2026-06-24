"""Trajectory selection and DataProto extraction for ranker training."""

from __future__ import annotations

import math
import uuid
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
    text = _text_from_doc(raw)
    if not doc_id and text:
        doc_id = f"rank-{rank}"
    if not text:
        return None
    score = raw.get("rank_score", raw.get("recall_score", raw.get("score")))
    try:
        score = float(score) if score is not None else None
    except (TypeError, ValueError):
        score = None
    try:
        rank_value = int(raw.get("rank_rank") or rank)
    except (TypeError, ValueError):
        rank_value = rank
    return RankedPassage(
        doc_id=doc_id,
        rank=rank_value,
        title=raw.get("title") or None,
        text=text,
        recall_score=score,
        metadata={
            k: v
            for k, v in raw.items()
            if k not in {"id", "doc_id", "contents", "text", "passage", "title", "score", "rank"}
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
    raw_prompts = _as_list(non_tensor.get("raw_prompt"))
    initial_queries = _as_list(non_tensor.get("initial_query"))
    answers_list = _as_list(non_tensor.get("answers"))
    messages_list = _as_list(non_tensor.get("messages"))
    uids = _as_list(non_tensor.get("uid"))
    reward_values = _as_list(non_tensor.get("reward"))

    trajectories: list[dict[str, Any]] = []
    for idx, tool_details in enumerate(details_list):
        if not tool_details:
            continue
        origin_query = ""
        if idx < len(initial_queries) and initial_queries[idx] is not None:
            origin_query = str(initial_queries[idx])
        elif idx < len(raw_prompts):
            origin_query = _origin_query_from_raw_prompt(raw_prompts[idx])

        answers = []
        if idx < len(answers_list):
            answers = [str(x) for x in _as_list(answers_list[idx])]

        score = _trajectory_score(tool_details, reward_values[idx] if idx < len(reward_values) else None)
        trajectory_id = str(uids[idx]) if idx < len(uids) and uids[idx] is not None else uuid.uuid4().hex
        messages = messages_list[idx] if idx < len(messages_list) and isinstance(messages_list[idx], list) else []

        tool_calls = []
        for call_idx, detail in enumerate(_as_list(tool_details)):
            if not isinstance(detail, dict):
                continue
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
                            "title": passage.title,
                            "text": passage.text,
                            "recall_score": passage.recall_score,
                            "metadata": passage.metadata,
                        }
                    )
            tool_calls.append(
                {
                    "tool_call_id": f"{trajectory_id}:search:{call_idx}",
                    "turn_idx": int(detail.get("step_index", call_idx)),
                    "tool_name": "search",
                    "sub_query": str(detail.get("sub_query") or ""),
                    "recall_top50_docs": recall_passages,
                    "rank_top50_docs": rank_docs,
                    "ranked_passages": rank_docs,
                    "rank_top5_docs": rank_top5_docs or rank_docs[:5],
                    "metadata": {
                        "tool_score": detail.get("tool_score"),
                        "answer_in_docs": detail.get("answer_in_docs"),
                        "ranker_failed": detail.get("ranker_failed"),
                    },
                }
            )

        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "origin_query": origin_query,
                "golden_answers": answers,
                "final_answer": "",
                "score": score,
                "score_type": "f1",
                "is_valid": bool(origin_query and tool_calls),
                "messages": messages,
                "tool_calls": tool_calls,
                "metadata": {"global_steps": global_steps},
            }
        )
    return trajectories


def _origin_query_from_raw_prompt(raw_prompt: Any) -> str:
    if isinstance(raw_prompt, list) and raw_prompt:
        for message in raw_prompt:
            if isinstance(message, dict) and message.get("role") == "user":
                return str(message.get("content") or "")
    return ""


def _trajectory_score(tool_details: Any, fallback: Any) -> float:
    try:
        if fallback is not None and not (isinstance(fallback, float) and math.isnan(fallback)):
            return float(fallback)
    except (TypeError, ValueError):
        pass
    scores = []
    for detail in _as_list(tool_details):
        if isinstance(detail, dict):
            try:
                scores.append(float(detail.get("tool_score", 0.0)))
            except (TypeError, ValueError):
                continue
    return max(scores) if scores else 0.0


class TopF1TrajectorySelector:
    """Select the top scoring valid trajectories and return all search contexts."""

    def __init__(self, max_selected_trajectories: int = 1, min_final_reward: float = 0.0):
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

    def _parse_trajectory(self, item: dict[str, Any] | RankerTrajectory) -> RankerTrajectory | None:
        if isinstance(item, RankerTrajectory):
            return item
        if not isinstance(item, dict):
            return None
        return RankerTrajectory(
            trajectory_id=str(item.get("trajectory_id") or uuid.uuid4().hex),
            origin_query=str(item.get("origin_query") or ""),
            golden_answers=[str(x) for x in _as_list(item.get("golden_answers"))],
            final_answer=str(item.get("final_answer") or ""),
            score=float(item.get("score") or 0.0),
            score_type=str(item.get("score_type") or "f1"),
            is_valid=bool(item.get("is_valid", True)),
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

        if not trajectory.origin_query or not sub_query or not recall_passages or not ranked_passages:
            return None

        return ToolCallContext(
            trajectory_id=trajectory.trajectory_id,
            tool_call_id=str(tool_call.get("tool_call_id") or f"{trajectory.trajectory_id}:search:{call_idx}"),
            turn_idx=int(tool_call.get("turn_idx", call_idx)),
            origin_query=trajectory.origin_query,
            sub_query=sub_query,
            golden_answers=trajectory.golden_answers,
            trajectory_score=trajectory.score,
            score_type=trajectory.score_type,
            ranked_passages=ranked_passages[:50],
            recall_top50_docs=recall_passages[:50],
            rank_top50_docs=ranked_passages[:50],
            rank_top5_docs=(rank_top5_passages or ranked_passages[:5])[:5],
            metadata=dict(tool_call.get("metadata") or {}),
        )
