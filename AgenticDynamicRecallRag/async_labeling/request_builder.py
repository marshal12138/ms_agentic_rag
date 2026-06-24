"""Build async label requests from rollout tool-call records."""

from __future__ import annotations

import uuid
import random
from typing import Any

import numpy as np

from ranker_strategies.schemas import ToolCallContext

from .schemas import AsyncLabelRequest, CandidateChunk


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _text_from_doc(obj: dict[str, Any]) -> str:
    return str(obj.get("text") or obj.get("contents") or obj.get("passage") or "")


def chunk_from_mapping(obj: dict[str, Any], fallback_rank: int) -> CandidateChunk:
    return CandidateChunk(
        doc_id=str(obj.get("doc_id") or obj.get("id") or ""),
        title=obj.get("title"),
        text=_text_from_doc(obj),
        recall_rank=obj.get("recall_rank"),
        recall_score=obj.get("recall_score"),
        rank_rank=obj.get("rank_rank") or obj.get("rank") or fallback_rank,
        rank_score=obj.get("rank_score") or obj.get("score"),
        metadata=dict(obj.get("metadata") or {}),
    )


def chunk_from_ranked_passage(passage: Any, fallback_rank: int) -> CandidateChunk:
    metadata = dict(getattr(passage, "metadata", {}) or {})
    return CandidateChunk(
        doc_id=str(getattr(passage, "doc_id", "") or ""),
        title=getattr(passage, "title", None),
        text=str(getattr(passage, "text", "") or ""),
        recall_rank=metadata.get("recall_rank"),
        recall_score=getattr(passage, "recall_score", None),
        rank_rank=int(getattr(passage, "rank", fallback_rank) or fallback_rank),
        rank_score=metadata.get("rank_score", getattr(passage, "recall_score", None)),
        metadata=metadata,
    )


def build_request_from_tool_call(
    tool_call: dict[str, Any],
    *,
    global_step: int,
    prompt_version: str = "",
    origin_query: str = "",
    trajectory_id: str = "",
    trajectory_score: float = 0.0,
    score_type: str = "unknown",
) -> AsyncLabelRequest:
    ranked_docs = tool_call.get("rank_top50_docs") or tool_call.get("ranked_chunk_list")
    if ranked_docs is None:
        raise ValueError("missing_ranked_chunk_list")
    chunks = [chunk_from_mapping(item, idx) for idx, item in enumerate(_as_list(ranked_docs), start=1)]
    request = AsyncLabelRequest(
        request_id=str(tool_call.get("request_id") or uuid.uuid4().hex),
        created_global_step=int(global_step),
        origin_query=str(tool_call.get("origin_query") or tool_call.get("question") or origin_query or ""),
        sub_query=str(tool_call.get("sub_query") or tool_call.get("query") or ""),
        trajectory_id=str(tool_call.get("trajectory_id") or trajectory_id or ""),
        tool_call_id=str(tool_call.get("tool_call_id") or f"{trajectory_id}:search:{uuid.uuid4().hex[:8]}"),
        ranked_chunk_list=chunks,
        turn_idx=int(tool_call.get("turn_idx") or 0),
        trajectory_score=float(tool_call.get("trajectory_score") or trajectory_score or 0.0),
        score_type=str(tool_call.get("score_type") or score_type or "unknown"),
        trace_metadata=dict(tool_call.get("metadata") or {}),
        prompt_version=prompt_version,
    )
    request.validate_rank50()
    return request


def build_request_from_context(
    context: ToolCallContext,
    *,
    global_step: int,
    prompt_version: str = "",
) -> AsyncLabelRequest:
    ranked_docs = context.rank_top50_docs or context.ranked_passages
    chunks = [chunk_from_ranked_passage(item, idx) for idx, item in enumerate(_as_list(ranked_docs), start=1)]
    request = AsyncLabelRequest(
        request_id=uuid.uuid4().hex,
        created_global_step=int(global_step),
        origin_query=context.origin_query,
        sub_query=context.sub_query,
        trajectory_id=context.trajectory_id,
        tool_call_id=context.tool_call_id,
        ranked_chunk_list=chunks,
        turn_idx=context.turn_idx,
        trajectory_score=context.trajectory_score,
        score_type=context.score_type,
        trace_metadata=dict(context.metadata or {}),
        prompt_version=prompt_version,
    )
    request.validate_rank50()
    return request


def select_tool_calls(
    tool_calls: list[dict[str, Any]],
    max_sub_query: int,
    *,
    selection_policy: str = "random",
    global_step: int = 0,
    seed: int = 42,
) -> list[dict[str, Any]]:
    limit = max(0, int(max_sub_query))
    candidates = list(tool_calls)
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        return candidates

    policy = str(selection_policy or "random").lower()
    if policy in {"random", "uniform_random", "random_sample"}:
        # Deterministic per global step, so resumed/debug runs are reproducible while
        # still avoiding systematic preference for earlier trajectories/tool calls.
        rng = random.Random(int(seed) + int(global_step) * 1_000_003)
        return rng.sample(candidates, limit)
    if policy in {"first", "head", "high_value_first"}:
        return candidates[:limit]
    raise ValueError(f"unsupported sub_query_selection_policy: {selection_policy}")


def build_requests_from_contexts(
    contexts: list[ToolCallContext],
    *,
    global_step: int,
    max_sub_query: int,
    prompt_version: str = "",
    sub_query_selection_policy: str = "random",
    selection_seed: int = 42,
) -> tuple[list[AsyncLabelRequest], dict[str, int]]:
    selected_contexts = select_tool_calls(
        contexts,
        max_sub_query=max_sub_query,
        selection_policy=sub_query_selection_policy,
        global_step=global_step,
        seed=selection_seed,
    )
    requests: list[AsyncLabelRequest] = []
    invalid = 0
    invalid_reasons: dict[str, int] = {}
    for context in selected_contexts:
        try:
            requests.append(
                build_request_from_context(
                    context,
                    global_step=global_step,
                    prompt_version=prompt_version,
                )
            )
        except Exception as exc:
            invalid += 1
            reason = str(exc) or type(exc).__name__
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

    metrics = {
        "candidate_tool_calls": len(contexts),
        "selected_tool_calls": len(selected_contexts),
        "submitted_candidates": len(requests),
        "invalid_requests": invalid,
    }
    for reason, count in invalid_reasons.items():
        safe_reason = "".join(ch if ch.isalnum() else "_" for ch in reason.lower()).strip("_")[:80]
        if safe_reason:
            metrics[f"invalid_reason/{safe_reason}"] = count
    return requests, metrics


def build_requests_from_dataproto(
    main_batch: Any,
    *,
    global_step: int,
    max_sub_query: int,
    prompt_version: str = "",
    sub_query_selection_policy: str = "random",
    selection_seed: int = 42,
) -> tuple[list[AsyncLabelRequest], dict[str, int]]:
    non_tensor = getattr(main_batch, "non_tensor_batch", {}) or {}
    details_list = _as_list(non_tensor.get("tool_call_details"))
    raw_prompts = _as_list(non_tensor.get("raw_prompt"))
    initial_queries = _as_list(non_tensor.get("initial_query"))
    uids = _as_list(non_tensor.get("uid"))
    rewards = _as_list(non_tensor.get("reward"))

    candidates: list[dict[str, Any]] = []
    invalid = 0
    invalid_reasons: dict[str, int] = {}
    for idx, tool_details in enumerate(details_list):
        origin_query = ""
        if idx < len(initial_queries) and initial_queries[idx] is not None:
            origin_query = str(initial_queries[idx])
        elif idx < len(raw_prompts):
            origin_query = _origin_query_from_raw_prompt(raw_prompts[idx])
        trajectory_id = str(uids[idx]) if idx < len(uids) and uids[idx] is not None else uuid.uuid4().hex
        try:
            trajectory_score = float(rewards[idx]) if idx < len(rewards) and rewards[idx] is not None else 0.0
        except (TypeError, ValueError):
            trajectory_score = 0.0
        for call_idx, detail in enumerate(_as_list(tool_details)):
            if not isinstance(detail, dict):
                continue
            detail = dict(detail)
            detail.setdefault("origin_query", origin_query)
            detail.setdefault("trajectory_id", trajectory_id)
            detail.setdefault("tool_call_id", f"{trajectory_id}:search:{call_idx}")
            detail.setdefault("trajectory_score", trajectory_score)
            candidates.append(detail)

    requests: list[AsyncLabelRequest] = []
    selected_candidates = select_tool_calls(
        candidates,
        max_sub_query=max_sub_query,
        selection_policy=sub_query_selection_policy,
        global_step=global_step,
        seed=selection_seed,
    )
    for detail in selected_candidates:
        try:
            requests.append(
                build_request_from_tool_call(
                    detail,
                    global_step=global_step,
                    prompt_version=prompt_version,
                    origin_query=str(detail.get("origin_query") or ""),
                    trajectory_id=str(detail.get("trajectory_id") or ""),
                    trajectory_score=float(detail.get("trajectory_score") or 0.0),
                    score_type=str(detail.get("score_type") or "unknown"),
                )
            )
        except Exception as exc:
            invalid += 1
            reason = str(exc) or type(exc).__name__
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
    metrics = {
        "candidate_tool_calls": len(candidates),
        "selected_tool_calls": len(selected_candidates),
        "submitted_candidates": len(requests),
        "invalid_requests": invalid,
    }
    for reason, count in invalid_reasons.items():
        safe_reason = "".join(ch if ch.isalnum() else "_" for ch in reason.lower()).strip("_")[:80]
        if safe_reason:
            metrics[f"invalid_reason/{safe_reason}"] = count
    return requests, {
        **metrics,
    }


def _origin_query_from_raw_prompt(raw_prompt: Any) -> str:
    for message in _as_list(raw_prompt):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""
