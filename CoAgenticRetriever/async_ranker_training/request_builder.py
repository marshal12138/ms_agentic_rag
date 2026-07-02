"""Build async ranker training requests from rollout tool-call records."""

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


def _require_rank(value: Any, field_name: str) -> int:
    if value is None or value == "":
        raise ValueError(f"missing_required_rank:{field_name}")
    return int(value)


def _require_field(mapping: dict[str, Any], key: str):
    if key not in mapping or mapping[key] in (None, ""):
        raise ValueError(f"missing_required_field:{key}")
    return mapping[key]


def chunk_from_mapping(obj: dict[str, Any]) -> CandidateChunk:
    return CandidateChunk(
        doc_id=str(obj.get("doc_id") or obj.get("id") or ""),
        title=obj.get("title"),
        text=_text_from_doc(obj),
        recall_rank=obj.get("recall_rank"),
        recall_score=obj.get("recall_score"),
        rank_rank=_require_rank(_require_field(obj, "rank_rank"), "rank_rank"),
        rank_score=_require_field(obj, "rank_score"),
        metadata=dict(obj.get("metadata") or {}),
    )


def chunk_from_ranked_passage(passage: Any) -> CandidateChunk:
    metadata = dict(getattr(passage, "metadata", {}) or {})
    return CandidateChunk(
        doc_id=str(getattr(passage, "doc_id", "") or ""),
        title=getattr(passage, "title", None),
        text=str(getattr(passage, "text", "") or ""),
        recall_rank=metadata.get("recall_rank"),
        recall_score=getattr(passage, "recall_score", None),
        rank_rank=_require_rank(getattr(passage, "rank", None), "rank"),
        rank_score=_require_field(metadata, "rank_score"),
        metadata=metadata,
    )


def build_request_from_tool_call(
    tool_call: dict[str, Any],
    *,
    global_step: int,
    prompt_version: str,
    origin_query: str,
    trajectory_id: str,
    trajectory_score: float,
    score_type: str,
    label_policy: str,
) -> AsyncLabelRequest:
    ranked_docs = tool_call.get("rank_top50_docs") or tool_call.get("ranked_chunk_list")
    if ranked_docs is None:
        raise ValueError("missing_ranked_chunk_list")
    chunks = [chunk_from_mapping(item) for item in _as_list(ranked_docs)]
    resolved_trajectory_score = (
        _require_field(tool_call, "trajectory_score")
        if "trajectory_score" in tool_call and tool_call["trajectory_score"] not in (None, "")
        else trajectory_score
    )
    resolved_score_type = (
        _require_field(tool_call, "score_type")
        if "score_type" in tool_call and tool_call["score_type"] not in (None, "")
        else score_type
    )
    request = AsyncLabelRequest(
        request_id=str(tool_call.get("request_id") or uuid.uuid4().hex),
        created_global_step=int(global_step),
        origin_query=str(tool_call.get("origin_query") or tool_call.get("question") or origin_query),
        sub_query=str(_require_field(tool_call, "sub_query")),
        trajectory_id=str(tool_call.get("trajectory_id") or trajectory_id),
        tool_call_id=str(_require_field(tool_call, "tool_call_id")),
        ranked_chunk_list=chunks,
        turn_idx=int(_require_field(tool_call, "turn_idx")),
        trajectory_score=float(resolved_trajectory_score),
        score_type=str(resolved_score_type),
        trace_metadata=dict(tool_call.get("metadata") or {}),
        label_policy=str(_require_field(tool_call, "label_policy"))
        if "label_policy" in tool_call and tool_call["label_policy"] not in (None, "")
        else label_policy,
        prompt_version=prompt_version,
    )
    request.validate_rank50()
    return request


def build_request_from_context(
    context: ToolCallContext,
    *,
    global_step: int,
    prompt_version: str,
    label_policy: str,
) -> AsyncLabelRequest:
    ranked_docs = context.rank_top50_docs or context.ranked_passages
    chunks = [chunk_from_ranked_passage(item) for item in _as_list(ranked_docs)]
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
        label_policy=label_policy,
        prompt_version=prompt_version,
    )
    request.validate_rank50()
    return request


def select_tool_calls(
    tool_calls: list[dict[str, Any]],
    max_sub_query: int,
    *,
    selection_policy: str,
    global_step: int,
    seed: int,
) -> list[dict[str, Any]]:
    limit = max(0, int(max_sub_query))
    candidates = list(tool_calls)
    if limit <= 0 or not candidates:
        return []
    if len(candidates) <= limit:
        return candidates

    if not selection_policy:
        raise ValueError("sub_query_selection_policy must be explicitly configured")
    policy = str(selection_policy).lower()
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
    prompt_version: str,
    sub_query_selection_policy: str,
    selection_seed: int,
    label_policy: str,
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
                    label_policy=label_policy,
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
    prompt_version: str,
    sub_query_selection_policy: str,
    selection_seed: int,
    label_policy: str,
) -> tuple[list[AsyncLabelRequest], dict[str, int]]:
    non_tensor = getattr(main_batch, "non_tensor_batch", {}) or {}
    details_list = _as_list(non_tensor.get("tool_call_details"))
    initial_queries = _as_list(non_tensor.get("initial_query"))
    uids = _as_list(non_tensor.get("uid"))
    rewards = _as_list(non_tensor.get("reward"))

    candidates: list[dict[str, Any]] = []
    invalid = 0
    invalid_reasons: dict[str, int] = {}
    for idx, tool_details in enumerate(details_list):
        if idx >= len(initial_queries) or initial_queries[idx] in (None, ""):
            raise ValueError("missing_required_field:initial_query")
        origin_query = str(initial_queries[idx])
        if idx >= len(uids) or uids[idx] in (None, ""):
            raise ValueError("missing_required_field:uid")
        trajectory_id = str(uids[idx])
        if idx >= len(rewards) or rewards[idx] is None:
            raise ValueError("missing_required_field:reward")
        trajectory_score = float(rewards[idx])
        for call_idx, detail in enumerate(_as_list(tool_details)):
            if not isinstance(detail, dict):
                continue
            detail = dict(detail)
            if _require_field(detail, "trajectory_id") != trajectory_id:
                raise ValueError("ranker trajectory_id mismatch between detail and uid")
            if _require_field(detail, "origin_query") != origin_query:
                raise ValueError("ranker origin_query mismatch between detail and non_tensor_batch")
            if "trajectory_score" in detail and detail["trajectory_score"] not in (None, ""):
                if float(detail["trajectory_score"]) != trajectory_score:
                    raise ValueError("ranker trajectory_score mismatch between detail and reward")
            else:
                detail["trajectory_score"] = trajectory_score
            _require_field(detail, "tool_call_id")
            _require_field(detail, "turn_idx")
            _require_field(detail, "score_type")
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
                    origin_query=str(_require_field(detail, "origin_query")),
                    trajectory_id=str(_require_field(detail, "trajectory_id")),
                    trajectory_score=float(_require_field(detail, "trajectory_score")),
                    score_type=str(_require_field(detail, "score_type")),
                    label_policy=label_policy,
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
