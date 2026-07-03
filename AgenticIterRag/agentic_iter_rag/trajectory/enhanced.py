"""Enhanced trajectory schema helpers for AgenticIterRag v1."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ENHANCED_TRAJECTORY_SCHEMA_VERSION = "air_enhanced_trajectory_v1"
CONTEXT_FORMAT_VERSION = "air_agent_messages_v1"
TOOL_RESPONSE_FORMAT_VERSION = "air_search_tool_response_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _as_list(value: Any, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a list")
    return value


def _doc_identity(raw: dict[str, Any], rank: int | None) -> str:
    for key in ("doc_id", "id", "document_id", "chunk_id", "passage_id", "_id"):
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    return f"doc-{rank or 0}"


def normalize_enhanced_doc(raw: dict[str, Any], rank: int | None = None) -> dict[str, Any]:
    """Normalize retrieved/ranked docs for strict step-level alignment."""

    doc_id = _doc_identity(raw, rank)
    text = raw.get("text") or raw.get("contents") or raw.get("passage") or raw.get("content") or ""
    score = raw.get("score", raw.get("recall_score", 0.0))
    out = dict(raw)
    out["doc_id"] = doc_id
    out["id"] = str(out.get("id") or doc_id)
    out["text"] = str(text)
    out["contents"] = str(out.get("contents") or text)
    if rank is not None:
        out["rank"] = int(out.get("rank") or rank)
        out["recall_rank"] = int(out.get("recall_rank") or out.get("rank") or rank)
    else:
        out.setdefault("rank", None)
        out.setdefault("recall_rank", out.get("rank"))
    try:
        out["score"] = float(score or 0.0)
    except (TypeError, ValueError):
        out["score"] = 0.0
    try:
        out["recall_score"] = float(out.get("recall_score", out["score"]) or 0.0)
    except (TypeError, ValueError):
        out["recall_score"] = out["score"]
    return out


def normalize_doc_list(docs: Any, *, limit: int | None = None) -> list[dict[str, Any]]:
    values = _as_list(docs or [], "docs")
    if limit is not None:
        values = values[:limit]
    return [normalize_enhanced_doc(doc, rank=i + 1) for i, doc in enumerate(values) if isinstance(doc, dict)]


def doc_id_order(docs: list[dict[str, Any]]) -> list[str]:
    return [str(doc.get("doc_id") or doc.get("id") or "") for doc in docs]


def validate_enhanced_step(step: dict[str, Any], *, no_ranker: bool = False, top_m: int | None = None) -> None:
    step_index = step.get("step_index", "?")
    sub_query = str(step.get("sub_query") or "").strip()
    if not sub_query:
        raise ValueError(f"enhanced step {step_index} sub_query must be non-empty")

    tool_call = _as_dict(step.get("tool_call"), f"step {step_index}.tool_call")
    arguments = _as_dict(tool_call.get("arguments"), f"step {step_index}.tool_call.arguments")
    if sub_query != str(arguments.get("query") or "").strip():
        raise ValueError(f"enhanced step {step_index} sub_query does not match tool_call.arguments.query")

    messages_before = _as_list(
        step.get("messages_before_tool_response"),
        f"step {step_index}.messages_before_tool_response",
    )
    if not messages_before:
        raise ValueError(f"enhanced step {step_index} messages_before_tool_response is empty")
    last_message = _as_dict(messages_before[-1], f"step {step_index}.messages_before_tool_response[-1]")
    if last_message.get("role") != "assistant":
        raise ValueError(f"enhanced step {step_index} messages_before_tool_response must end with assistant")
    if "<tool_call>" not in str(last_message.get("content") or ""):
        raise ValueError(f"enhanced step {step_index} assistant message must contain <tool_call>")

    assistant_message = _as_dict(step.get("assistant_tool_call_message"), f"step {step_index}.assistant_tool_call_message")
    if assistant_message != last_message:
        raise ValueError(f"enhanced step {step_index} assistant_tool_call_message must equal messages_before last item")

    original_tool = _as_dict(step.get("original_tool_message"), f"step {step_index}.original_tool_message")
    if original_tool.get("role") != "tool":
        raise ValueError(f"enhanced step {step_index} original_tool_message.role must be tool")

    messages_after = _as_list(
        step.get("messages_after_original_tool_response"),
        f"step {step_index}.messages_after_original_tool_response",
    )
    if messages_after != messages_before + [original_tool]:
        raise ValueError(
            f"enhanced step {step_index} messages_after_original_tool_response must equal "
            "messages_before_tool_response + [original_tool_message]"
        )

    recall_docs = _as_list(step.get("recall_topn_docs"), f"step {step_index}.recall_topn_docs")
    if not recall_docs:
        raise ValueError(f"enhanced step {step_index} recall_topn_docs must be non-empty")
    visible_docs = _as_list(step.get("original_visible_docs"), f"step {step_index}.original_visible_docs")
    if top_m is not None and len(visible_docs) > top_m:
        raise ValueError(f"enhanced step {step_index} original_visible_docs exceeds top_m={top_m}")

    order = [str(item) for item in _as_list(step.get("doc_id_order"), f"step {step_index}.doc_id_order")]
    recall_order = doc_id_order([_as_dict(doc, f"step {step_index}.recall_topn_docs[]") for doc in recall_docs])
    if order != recall_order:
        raise ValueError(f"enhanced step {step_index} doc_id_order must match recall_topn_docs order")

    visible_ids = [
        str(item)
        for item in _as_list(step.get("original_visible_doc_ids"), f"step {step_index}.original_visible_doc_ids")
    ]
    actual_visible_ids = doc_id_order([_as_dict(doc, f"step {step_index}.original_visible_docs[]") for doc in visible_docs])
    if visible_ids != actual_visible_ids:
        raise ValueError(f"enhanced step {step_index} original_visible_doc_ids must match original_visible_docs")
    if no_ranker and visible_ids != order[: len(visible_ids)]:
        raise ValueError(f"enhanced step {step_index} no-ranker visible doc ids must follow recall order")


def validate_enhanced_record(record: dict[str, Any], *, no_ranker: bool = False, top_m: int | None = None) -> None:
    if record.get("schema_version") != ENHANCED_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("enhanced trajectory schema_version is invalid")
    for field in ("trajectory_id", "sample_id", "question", "context_format_version", "tool_response_format_version"):
        if not str(record.get(field) or "").strip():
            raise ValueError(f"enhanced trajectory missing required field: {field}")
    if record.get("context_format_version") != CONTEXT_FORMAT_VERSION:
        raise ValueError("enhanced trajectory context_format_version is invalid")
    if record.get("tool_response_format_version") != TOOL_RESPONSE_FORMAT_VERSION:
        raise ValueError("enhanced trajectory tool_response_format_version is invalid")
    gold_answers = _as_list(record.get("gold_answers"), "gold_answers")
    if not gold_answers:
        raise ValueError("enhanced trajectory gold_answers must be non-empty")
    if record.get("baseline_reward") is None:
        raise ValueError("enhanced trajectory baseline_reward must exist")
    baseline_metrics = _as_dict(record.get("baseline_metrics"), "baseline_metrics")
    steps = _as_list(record.get("steps"), "steps")
    for expected_idx, step in enumerate(steps):
        step_obj = _as_dict(step, f"steps[{expected_idx}]")
        if int(step_obj.get("step_index", -1)) != expected_idx:
            raise ValueError(f"enhanced step index mismatch at position {expected_idx}")
        validate_enhanced_step(step_obj, no_ranker=no_ranker, top_m=top_m)
    metric_tool_calls = baseline_metrics.get("tool_calls")
    status = str(baseline_metrics.get("status") or record.get("status") or "")
    if isinstance(metric_tool_calls, int) and len(steps) != metric_tool_calls and status not in {"failed", "error"}:
        raise ValueError(
            f"enhanced trajectory step count does not match baseline_metrics.tool_calls: "
            f"{len(steps)} != {metric_tool_calls}"
        )


def summarize_enhanced_records(
    records: list[dict[str, Any]],
    *,
    top_n: int,
    top_m: int,
) -> dict[str, Any]:
    step_counts = [len(record.get("steps") or []) for record in records]
    record_count = len(records)
    search_step_count = sum(step_counts)
    return {
        "dataset_type": "enhanced_trajectory",
        "schema_version": ENHANCED_TRAJECTORY_SCHEMA_VERSION,
        "record_count": record_count,
        "search_step_count": search_step_count,
        "records_without_search": sum(1 for count in step_counts if count == 0),
        "max_steps_per_record": max(step_counts) if step_counts else 0,
        "avg_steps_per_record": search_step_count / record_count if record_count else 0.0,
        "top_n": top_n,
        "top_m": top_m,
        "context_format_version": CONTEXT_FORMAT_VERSION,
        "tool_response_format_version": TOOL_RESPONSE_FORMAT_VERSION,
        "created_at": utc_now(),
    }
