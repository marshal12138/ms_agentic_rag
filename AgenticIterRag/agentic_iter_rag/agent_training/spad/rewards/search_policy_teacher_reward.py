"""VERL custom reward for SPAD-RAG Stage 1 search-policy RL."""

from __future__ import annotations

import json
import hashlib
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from typing import Any

from agentic_iter_rag.agent_training.spad.prompts import (
    DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
    build_teacher_messages,
    resolve_teacher_prompt,
)
from agentic_iter_rag.agent_training.spad.reward import compute_search_policy_reward
from agentic_iter_rag.metrics.answer_metrics import answer_group_metrics, groups_from_ground_truth, legacy_exact_match


REWARD_VERSION = "spad_em_teacher_backoff"


_DEFAULT_EXTRA: dict[str, Any] = {
    "score": 0.0,
    "reward_type": "spad_teacher_f1",
    "teacher_f1": 0.0,
    "teacher_called": False,
    "teacher_called_count": 0,
    "teacher_skip_reason": "",
    "format_status": "",
    "stop_status": "",
    "search_count": 0,
    "free_search_count": 1,
    "paid_search_count": 0,
    "second_plus_search_count": 0,
    "duplicate_query_count": 0,
    "effective_search_cost": 0.0,
    "teacher_elapsed_s": 0.0,
    "teacher_answer": "",
    "teacher_evidence_status": "",
    "teacher_raw_content": "",
    "teacher_parse_status": "",
    "teacher_format_error": False,
    "teacher_format_error_count": 0,
    "teacher_enable_thinking": "",
    "teacher_prompt_version": "",
    "teacher_messages": [],
    "teacher_request_hash": "",
    "teacher_model": "",
    "teacher_endpoint": "",
    "teacher_temperature": 0.0,
    "teacher_top_p": 1.0,
    "teacher_max_tokens": 0,
    "supported_answer_count": 0,
    "insufficient_evidence_count": 0,
    "ambiguous_evidence_count": 0,
    "bad_stop_applied": False,
    "bad_stop_count": 0,
    "bad_stop_reason": "",
    "search_r1_answer_em": 0.0,
    "search_r1_extracted_answer": "",
    "legacy_em": 0.0,
    "legacy_f1": 0.0,
    "structured_em": 0.0,
    "answer_group_f1": 0.0,
    "answer_group_recall": 0.0,
    "matched_group_count": 0,
    "required_group_count": 0,
    "structured_reward_eligible": True,
    "answer_semantics": "",
    "actor_answer": "",
    "actor_answer_parse_status": "",
    "em_reward": 0.0,
    "teacher_status_reward": 0.0,
    "group_uid": "",
    "group_size": 0,
    "group_all_em_zero": False,
    "partial_reward_applied": False,
}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else [converted]
    return [value]


def _to_plain(value: Any) -> Any:
    """Convert OmegaConf-like containers into JSON-serializable Python values."""

    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


def _gold_answers(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, dict):
        value = ground_truth.get("target")
        if value is None:
            value = ground_truth.get("answers")
        if value is None:
            value = ground_truth.get("answer")
    else:
        value = ground_truth
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _question(extra_info: dict[str, Any]) -> str:
    return str(extra_info.get("question") or extra_info.get("initial_query") or "").strip()


def _detail_docs(detail: dict[str, Any], *, visible_top_m: int) -> list[dict[str, Any]]:
    for key in ("top_5_documents", "rank_top5_docs", "rank_top50_docs", "recall_top50_docs"):
        value = detail.get(key)
        if isinstance(value, list):
            return value[:visible_top_m]
        if isinstance(value, str) and value.strip():
            return [{"text": value.strip()}]
    return []


def _search_evidence(extra_info: dict[str, Any], *, visible_top_m: int) -> list[dict[str, Any]]:
    details = _as_list(extra_info.get("tool_call_details"))
    evidence: list[dict[str, Any]] = []
    for turn, raw_detail in enumerate(details, start=1):
        if not isinstance(raw_detail, dict):
            continue
        sub_query = str(raw_detail.get("sub_query") or "").strip()
        docs = _detail_docs(raw_detail, visible_top_m=visible_top_m)
        evidence.append({"turn": turn, "sub_query": sub_query, "docs": docs})
    return evidence


def _count_duplicate_queries(evidence: list[dict[str, Any]]) -> int:
    seen: set[str] = set()
    duplicate = 0
    for item in evidence:
        query = re.sub(r"\s+", " ", str(item.get("sub_query") or "").strip().lower())
        if not query:
            continue
        if query in seen:
            duplicate += 1
        seen.add(query)
    return duplicate


def _extract_last_complete_answer(text: str) -> tuple[str, str]:
    answer_start = text.rfind("<answer>")
    if answer_start < 0:
        return "", "missing_answer_close"
    body_start = answer_start + len("<answer>")
    answer_end = text.find("</answer>", body_start)
    if answer_end < 0:
        return "", "missing_answer_close"
    answer = text[body_start:answer_end].strip()
    if not answer:
        return "", "empty_answer"
    return answer, "parsed"


def _answer_em(prediction: str, gold_answers: list[str]) -> float:
    return legacy_exact_match(prediction, gold_answers)


def _compute_search_r1_original_reward_details(
    *,
    solution_str: str,
    gold_answers: list[str],
    search_count: int,
    duplicate_query_count: int,
    reward_cfg: dict[str, Any],
) -> dict[str, Any]:
    search_r1_cfg = reward_cfg.get("search_r1_original")
    if not isinstance(search_r1_cfg, Mapping):
        search_r1_cfg = {}
    answer, parse_status = _extract_last_complete_answer(solution_str)
    answer_em = _answer_em(answer, gold_answers) if parse_status == "parsed" else 0.0
    score = float(search_r1_cfg.get("score", 1.0))
    format_score = float(search_r1_cfg.get("format_score", 0.0))
    final_reward = score if answer_em >= 1.0 else format_score
    return _sanitize(
        {
            "final_reward": final_reward,
            "reward_type": "search_r1_original",
            "teacher_f1": answer_em,
            "teacher_called": False,
            "teacher_skip_reason": "search_r1_original_no_teacher",
            "format_status": "valid" if parse_status == "parsed" else "invalid",
            "stop_status": "answer_closed" if parse_status == "parsed" else parse_status,
            "search_count": search_count,
            "duplicate_query_count": duplicate_query_count,
            "teacher_evidence_status": "",
            "teacher_format_error": False,
            "teacher_parse_status": parse_status,
            "search_r1_answer_em": answer_em,
            "search_r1_extracted_answer": answer,
        }
    )


def _compute_search_r1_structured_reward_details(
    *,
    solution_str: str,
    ground_truth: Any,
    search_count: int,
    duplicate_query_count: int,
    reward_cfg: dict[str, Any],
) -> dict[str, Any]:
    structured_cfg = reward_cfg.get("search_r1_structured")
    if not isinstance(structured_cfg, Mapping):
        structured_cfg = {}
    answers, groups, eligible, semantics = groups_from_ground_truth(ground_truth)
    answer, parse_status = _extract_last_complete_answer(solution_str)
    metrics = answer_group_metrics(
        answer,
        answers,
        groups,
        structured_eligible=eligible,
    )
    score = float(structured_cfg.get("score", 1.0))
    format_score = float(structured_cfg.get("format_score", 0.0))
    final_reward = score if metrics.structured_em >= 1.0 else format_score
    return _sanitize(
        {
            "final_reward": final_reward,
            "reward_type": "search_r1_structured",
            "teacher_f1": metrics.structured_em,
            "teacher_called": False,
            "teacher_skip_reason": "search_r1_structured_no_teacher",
            "format_status": "valid" if parse_status == "parsed" else "invalid",
            "stop_status": "answer_closed" if parse_status == "parsed" else parse_status,
            "search_count": search_count,
            "duplicate_query_count": duplicate_query_count,
            "teacher_parse_status": parse_status,
            "search_r1_answer_em": metrics.structured_em,
            "search_r1_extracted_answer": answer,
            "legacy_em": metrics.legacy_em,
            "legacy_f1": metrics.legacy_f1,
            "structured_em": metrics.structured_em,
            "answer_group_f1": metrics.answer_group_f1,
            "answer_group_recall": metrics.answer_group_recall,
            "matched_group_count": metrics.matched_group_count,
            "required_group_count": metrics.required_group_count,
            "structured_reward_eligible": metrics.structured_eligible,
            "answer_semantics": semantics,
        }
    )


def _extract_teacher_answer_legacy(text: str) -> tuple[str, str]:
    # Require a real reason+answer block so prompt examples such as
    # "<answer>...</answer>" in a model's rule recap are not treated as answers.
    matches = list(re.finditer(r"<reason>.*?</reason>\s*<answer>(.*?)</answer>", text, re.S))
    parse_status = "parsed"
    if matches:
        answer = matches[-1].group(1).strip()
    else:
        open_answer_matches = list(re.finditer(r"<reason>.*?</reason>\s*<answer>\s*(.*)$", text, re.S))
        if not open_answer_matches:
            return "", "missing_reason_answer_tags"
        answer = open_answer_matches[-1].group(1).strip()
        parse_status = "missing_answer_close_tag"
    if answer in {"...", "…"}:
        return "", "placeholder_answer"
    if not answer:
        return "", "empty_answer"
    return answer, parse_status


def _extract_teacher_result(text: str) -> tuple[str, str, str, bool]:
    """Parse Stage 1 teacher output with evidence status.

    Stage 1 treats missing/invalid status as a teacher format error instead of
    inferring status from the answer text. That keeps bad-stop reward decisions
    tied to the explicit teacher contract.
    """

    allowed_statuses = {"supported_answer", "insufficient_evidence", "ambiguous_evidence"}
    matches = list(
        re.finditer(
            r"<reason>(.*?)</reason>\s*<status>(.*?)</status>\s*<answer>(.*?)</answer>",
            text,
            re.S,
        )
    )
    if not matches:
        if "<reason>" not in text or "</reason>" not in text:
            return "", "", "missing_reason_tag", True
        if "<status>" not in text or "</status>" not in text:
            return "", "", "missing_status_tag", True
        if "<answer>" not in text or "</answer>" not in text:
            return "", "", "missing_answer_tag", True
        return "", "", "missing_reason_status_answer_tags", True

    _, raw_status, raw_answer = matches[-1].groups()
    status = raw_status.strip()
    answer = raw_answer.strip()
    if status not in allowed_statuses:
        return answer, status, "invalid_status", True
    if answer in {"...", "…"}:
        return "", status, "placeholder_answer", True
    if not answer:
        return "", status, "empty_answer", True
    return answer, status, "parsed", False


def _extract_teacher_answer(text: str) -> tuple[str, str]:
    """Compatibility parser used by Stage 2 answer refresh.

    New Stage 1 reward uses ``_extract_teacher_result`` directly so missing
    ``<status>`` remains a format error there. Stage 2 can still consume the
    previous two-tag teacher answer prompt.
    """

    answer, _status, parse_status, format_error = _extract_teacher_result(text)
    if not format_error:
        return answer, parse_status
    return _extract_teacher_answer_legacy(text)


def _post_teacher(endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_teacher(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    request_cfg: dict[str, Any],
    prompt_version: str,
    messages: list[dict[str, str]] | None = None,
) -> tuple[str, str, bool, float, str, str, str]:
    endpoint = str(request_cfg.get("endpoint") or os.environ.get("SPAD_TEACHER_ENDPOINT") or "").strip()
    model = str(request_cfg.get("model") or os.environ.get("SPAD_TEACHER_MODEL") or "").strip()
    if not endpoint:
        raise ValueError("SPAD teacher endpoint is empty")
    if not model:
        raise ValueError("SPAD teacher model is empty")
    started = time.perf_counter()
    payload = {
        "model": model,
        "messages": messages
        if messages is not None
        else build_teacher_messages(
            question=question,
            evidence_steps=evidence,
            include_status=True,
            prompt_version=prompt_version,
        ),
        "temperature": float(request_cfg.get("temperature", 0.0)),
        "top_p": float(request_cfg.get("top_p", 1.0)),
        "max_tokens": int(request_cfg.get("max_tokens", 512)),
    }
    chat_template_kwargs = _to_plain(request_cfg.get("chat_template_kwargs"))
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    if "include_reasoning" in request_cfg:
        payload["include_reasoning"] = bool(request_cfg.get("include_reasoning"))
    result = _post_teacher(endpoint, payload, float(request_cfg.get("timeout_seconds", 180)))
    content = (
        result.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    raw_content = str(content)
    answer, evidence_status, parse_status, format_error = _extract_teacher_result(raw_content)
    enable_thinking = ""
    if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
        enable_thinking = str(bool(chat_template_kwargs.get("enable_thinking"))).lower()
    return answer, evidence_status, format_error, time.perf_counter() - started, raw_content, parse_status, enable_thinking


def _sanitize(result: dict[str, Any]) -> dict[str, Any]:
    merged = {**_DEFAULT_EXTRA, **result}
    search_count = int(merged.get("search_count") or 0)
    free_search_count = max(0, int(merged.get("free_search_count", 1) or 0))
    default_paid_search_count = max(0, search_count - free_search_count)
    paid_search_count = int(
        result.get("paid_search_count", result.get("second_plus_search_count", default_paid_search_count)) or 0
    )
    second_plus_search_count = int(result.get("second_plus_search_count", paid_search_count) or 0)
    return {
        "score": float(merged.get("final_reward", merged.get("score", 0.0))),
        "reward_type": str(merged.get("reward_type") or "spad_teacher_f1"),
        "teacher_f1": float(merged.get("teacher_f1", 0.0)),
        "teacher_called": bool(merged.get("teacher_called", False)),
        "teacher_called_count": int(bool(merged.get("teacher_called", False))),
        "teacher_skip_reason": str(merged.get("teacher_skip_reason") or ""),
        "format_status": str(merged.get("format_status") or ""),
        "stop_status": str(merged.get("stop_status") or ""),
        "search_count": search_count,
        "free_search_count": free_search_count,
        "paid_search_count": paid_search_count,
        "second_plus_search_count": second_plus_search_count,
        "duplicate_query_count": int(merged.get("duplicate_query_count") or 0),
        "effective_search_cost": float(merged.get("effective_search_cost") or 0.0),
        "teacher_elapsed_s": float(merged.get("teacher_elapsed_s") or 0.0),
        "teacher_answer": str(merged.get("teacher_answer") or ""),
        "teacher_evidence_status": str(merged.get("teacher_evidence_status") or ""),
        "teacher_raw_content": str(merged.get("teacher_raw_content") or ""),
        "teacher_parse_status": str(merged.get("teacher_parse_status") or ""),
        "teacher_format_error": bool(merged.get("teacher_format_error", False)),
        "teacher_format_error_count": int(bool(merged.get("teacher_format_error", False))),
        "teacher_enable_thinking": str(merged.get("teacher_enable_thinking") or ""),
        "teacher_prompt_version": str(merged.get("teacher_prompt_version") or ""),
        "teacher_messages": _to_plain(merged.get("teacher_messages") or []),
        "teacher_request_hash": str(merged.get("teacher_request_hash") or ""),
        "teacher_model": str(merged.get("teacher_model") or ""),
        "teacher_endpoint": str(merged.get("teacher_endpoint") or ""),
        "teacher_temperature": float(merged.get("teacher_temperature") or 0.0),
        "teacher_top_p": float(merged.get("teacher_top_p") or 1.0),
        "teacher_max_tokens": int(merged.get("teacher_max_tokens") or 0),
        "supported_answer_count": int(str(merged.get("teacher_evidence_status") or "") == "supported_answer"),
        "insufficient_evidence_count": int(str(merged.get("teacher_evidence_status") or "") == "insufficient_evidence"),
        "ambiguous_evidence_count": int(str(merged.get("teacher_evidence_status") or "") == "ambiguous_evidence"),
        "bad_stop_applied": bool(merged.get("bad_stop_applied", False)),
        "bad_stop_count": int(bool(merged.get("bad_stop_applied", False))),
        "bad_stop_reason": str(merged.get("bad_stop_reason") or ""),
        "search_r1_answer_em": float(merged.get("search_r1_answer_em") or 0.0),
        "search_r1_extracted_answer": str(merged.get("search_r1_extracted_answer") or ""),
        "legacy_em": float(merged.get("legacy_em") or 0.0),
        "legacy_f1": float(merged.get("legacy_f1") or 0.0),
        "structured_em": float(merged.get("structured_em") or 0.0),
        "answer_group_f1": float(merged.get("answer_group_f1") or 0.0),
        "answer_group_recall": float(merged.get("answer_group_recall") or 0.0),
        "matched_group_count": int(merged.get("matched_group_count") or 0),
        "required_group_count": int(merged.get("required_group_count") or 0),
        "structured_reward_eligible": bool(merged.get("structured_reward_eligible", True)),
        "answer_semantics": str(merged.get("answer_semantics") or ""),
        "actor_answer": str(merged.get("actor_answer") or ""),
        "actor_answer_parse_status": str(merged.get("actor_answer_parse_status") or ""),
        "em_reward": float(merged.get("em_reward") or 0.0),
        "teacher_status_reward": float(merged.get("teacher_status_reward") or 0.0),
        "group_uid": str(merged.get("group_uid") or ""),
        "group_size": int(merged.get("group_size") or 0),
        "group_all_em_zero": bool(merged.get("group_all_em_zero", False)),
        "partial_reward_applied": bool(merged.get("partial_reward_applied", False)),
    }


def _spad_group_base_details(
    *,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    visible_top_m: int,
) -> dict[str, Any]:
    gold_answers = _gold_answers(ground_truth)
    answer, parse_status = _extract_last_complete_answer(solution_str)
    evidence = _search_evidence(extra_info, visible_top_m=visible_top_m)
    return {
        "actor_answer": answer,
        "actor_answer_parse_status": parse_status,
        "em_reward": _answer_em(answer, gold_answers) if parse_status == "parsed" else 0.0,
        "evidence": evidence,
        "search_count": len(evidence),
        "duplicate_query_count": _count_duplicate_queries(evidence),
        "group_uid": str(extra_info.get("uid") or ""),
        "question": _question(extra_info),
    }


def _teacher_audit_details(
    *,
    base: dict[str, Any],
    request_cfg: dict[str, Any],
    teacher_prompt_version: str,
) -> dict[str, Any]:
    messages = (
        build_teacher_messages(
            question=base["question"],
            evidence_steps=base["evidence"],
            include_status=True,
            prompt_version=teacher_prompt_version,
        )
        if base["evidence"]
        else []
    )
    request_identity = {
        "model": str(request_cfg.get("model") or os.environ.get("SPAD_TEACHER_MODEL") or ""),
        "endpoint": str(request_cfg.get("endpoint") or os.environ.get("SPAD_TEACHER_ENDPOINT") or ""),
        "messages": messages,
        "temperature": float(request_cfg.get("temperature", 0.0)),
        "top_p": float(request_cfg.get("top_p", 1.0)),
        "max_tokens": int(request_cfg.get("max_tokens", 512)),
        "chat_template_kwargs": _to_plain(request_cfg.get("chat_template_kwargs") or {}),
        "prompt_version": teacher_prompt_version,
    }
    request_hash = hashlib.sha256(
        json.dumps(request_identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "teacher_messages": messages,
        "teacher_request_hash": request_hash,
        "teacher_model": request_identity["model"],
        "teacher_endpoint": request_identity["endpoint"],
        "teacher_temperature": request_identity["temperature"],
        "teacher_top_p": request_identity["top_p"],
        "teacher_max_tokens": request_identity["max_tokens"],
    }


def _spad_teacher_status_details(
    *,
    base: dict[str, Any],
    request_cfg: dict[str, Any],
    teacher_prompt_version: str,
) -> dict[str, Any]:
    audit = _teacher_audit_details(
        base=base,
        request_cfg=request_cfg,
        teacher_prompt_version=teacher_prompt_version,
    )
    if not base["evidence"]:
        return {
            **audit,
            "teacher_called": False,
            "teacher_skip_reason": "no_search_evidence",
            "teacher_parse_status": "not_called",
            "teacher_status_reward": 0.0,
        }
    try:
        (
            teacher_answer,
            teacher_evidence_status,
            teacher_format_error,
            teacher_elapsed_s,
            teacher_raw_content,
            teacher_parse_status,
            teacher_enable_thinking,
        ) = _call_teacher(
            question=base["question"],
            evidence=base["evidence"],
            request_cfg=request_cfg,
            prompt_version=teacher_prompt_version,
            messages=audit["teacher_messages"],
        )
    except Exception as exc:
        return {
            **audit,
            "teacher_called": True,
            "teacher_skip_reason": "",
            "teacher_format_error": True,
            "teacher_parse_status": f"teacher_error:{type(exc).__name__}",
            "teacher_status_reward": 0.0,
        }
    status_reward = float(
        not teacher_format_error
        and teacher_evidence_status in {"supported_answer", "ambiguous_evidence"}
    )
    return {
        **audit,
        "teacher_called": True,
        "teacher_skip_reason": "",
        "teacher_answer": teacher_answer,
        "teacher_evidence_status": teacher_evidence_status,
        "teacher_format_error": teacher_format_error,
        "teacher_elapsed_s": teacher_elapsed_s,
        "teacher_raw_content": teacher_raw_content,
        "teacher_parse_status": teacher_parse_status,
        "teacher_enable_thinking": teacher_enable_thinking,
        "teacher_status_reward": status_reward,
    }


def compute_spad_search_policy_reward_details(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute Stage 1 reward for one trajectory.

    The teacher is called only after the trajectory has a valid search history
    and a legal stop at the opening <answer> tag.
    """

    del data_source
    reward_cfg = dict(kwargs.get("reward_cfg") or {})
    request_cfg = dict(kwargs.get("teacher_request") or {})
    teacher_prompt_version, _ = resolve_teacher_prompt(
        str(kwargs.get("teacher_prompt_version") or DEFAULT_TEACHER_STATUS_PROMPT_VERSION),
        include_status=True,
    )
    visible_top_m = int(kwargs.get("visible_top_m") or 5)
    gold_answers = _gold_answers(ground_truth)
    evidence = _search_evidence(extra_info, visible_top_m=visible_top_m)
    search_count = len(evidence)
    duplicate_query_count = _count_duplicate_queries(evidence)
    reward_type = str(reward_cfg.get("type") or "spad_teacher_f1")
    if reward_type == "search_r1_original":
        return _compute_search_r1_original_reward_details(
            solution_str=solution_str,
            gold_answers=gold_answers,
            search_count=search_count,
            duplicate_query_count=duplicate_query_count,
            reward_cfg=reward_cfg,
        )
    if reward_type == "search_r1_structured":
        return _compute_search_r1_structured_reward_details(
            solution_str=solution_str,
            ground_truth=ground_truth,
            search_count=search_count,
            duplicate_query_count=duplicate_query_count,
            reward_cfg=reward_cfg,
        )
    if reward_type == "spad_em_teacher_backoff":
        raise ValueError("spad_em_teacher_backoff requires BatchRewardManager")
    question = _question(extra_info)
    legal_stop = "<answer>" in solution_str and "</answer>" not in solution_str

    if search_count <= 0:
        penalty = float(reward_cfg.get("no_finish_penalty", -0.5))
        return _sanitize(
            {
                "final_reward": penalty,
                "teacher_called": False,
                "teacher_skip_reason": "no_search_evidence",
                "format_status": "unknown",
                "stop_status": "no_search_evidence",
                "search_count": 0,
                "duplicate_query_count": 0,
            }
        )

    # First pass parses the actor stop action and short-circuits invalid formats.
    pre = compute_search_policy_reward(
        actor_output=solution_str,
        teacher_answer="",
        gold_answers=gold_answers,
        search_count=search_count,
        duplicate_query_count=duplicate_query_count,
        reward_cfg=reward_cfg,
        legal_stop=legal_stop,
        stop_at_answer_opening=True,
    )
    if not pre.get("teacher_called"):
        pre["search_count"] = search_count
        pre["duplicate_query_count"] = duplicate_query_count
        return _sanitize(pre)

    try:
        (
            teacher_answer,
            teacher_evidence_status,
            teacher_format_error,
            teacher_elapsed_s,
            teacher_raw_content,
            teacher_parse_status,
            teacher_enable_thinking,
        ) = _call_teacher(
            question=question,
            evidence=evidence,
            request_cfg=request_cfg,
            prompt_version=teacher_prompt_version,
        )
    except Exception as exc:
        teacher_answer = ""
        teacher_evidence_status = ""
        teacher_format_error = True
        teacher_elapsed_s = 0.0
        teacher_raw_content = ""
        teacher_parse_status = f"teacher_error:{type(exc).__name__}"
        teacher_enable_thinking = ""
    result = compute_search_policy_reward(
        actor_output=solution_str,
        teacher_answer=teacher_answer,
        gold_answers=gold_answers,
        search_count=search_count,
        duplicate_query_count=duplicate_query_count,
        reward_cfg=reward_cfg,
        legal_stop=True,
        stop_at_answer_opening=True,
        teacher_evidence_status=teacher_evidence_status,
        teacher_format_error=teacher_format_error,
    )
    result["search_count"] = search_count
    result["duplicate_query_count"] = duplicate_query_count
    result["teacher_elapsed_s"] = teacher_elapsed_s
    result["teacher_answer"] = teacher_answer
    result["teacher_evidence_status"] = teacher_evidence_status
    result["teacher_raw_content"] = teacher_raw_content
    result["teacher_parse_status"] = teacher_parse_status
    result["teacher_format_error"] = teacher_format_error
    result["teacher_enable_thinking"] = teacher_enable_thinking
    result["teacher_prompt_version"] = teacher_prompt_version
    return _sanitize(result)


def compute_spad_search_policy_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> float:
    return float(
        compute_spad_search_policy_reward_details(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )["score"]
    )


def compute_spad_search_policy_reward_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict[str, Any]],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    item_count = len(solution_strs)
    if not (len(data_sources) == len(ground_truths) == len(extra_infos) == item_count):
        raise ValueError("SPAD Stage 1 batch reward inputs must have the same length")
    if item_count == 0:
        return []
    reward_cfg = dict(kwargs.get("reward_cfg") or {})
    if str(reward_cfg.get("type") or "") == "spad_em_teacher_backoff":
        visible_top_m = int(kwargs.get("visible_top_m") or 5)
        expected_group_size = int(kwargs.get("n_samples_per_prompt") or 8)
        request_cfg = dict(kwargs.get("teacher_request") or {})
        teacher_prompt_version, _ = resolve_teacher_prompt(
            str(kwargs.get("teacher_prompt_version") or DEFAULT_TEACHER_STATUS_PROMPT_VERSION),
            include_status=True,
        )
        backoff_cfg = reward_cfg.get("spad_em_teacher_backoff")
        if not isinstance(backoff_cfg, Mapping):
            backoff_cfg = {}
        partial_weight = float(backoff_cfg.get("partial_reward", 0.1))
        bases = [
            _spad_group_base_details(
                solution_str=str(solution_strs[index]),
                ground_truth=ground_truths[index],
                extra_info=extra_infos[index],
                visible_top_m=visible_top_m,
            )
            for index in range(item_count)
        ]
        grouped_indices: dict[str, list[int]] = {}
        for index, base in enumerate(bases):
            uid = str(base["group_uid"])
            if not uid:
                raise ValueError(f"SPAD batch reward item {index} is missing uid")
            grouped_indices.setdefault(uid, []).append(index)
        invalid_groups = {
            uid: len(indices)
            for uid, indices in grouped_indices.items()
            if len(indices) != expected_group_size
        }
        if invalid_groups:
            raise ValueError(
                f"SPAD batch reward requires exactly {expected_group_size} rollouts per uid; "
                f"got {invalid_groups}"
            )

        all_zero_by_uid = {
            uid: not any(bases[index]["em_reward"] >= 1.0 for index in indices)
            for uid, indices in grouped_indices.items()
        }
        teacher_indices = [
            index
            for index, base in enumerate(bases)
            if all_zero_by_uid[str(base["group_uid"])] and base["evidence"]
        ]
        teacher_details: dict[int, dict[str, Any]] = {}
        max_workers = int(
            kwargs.get("batch_workers")
            or os.environ.get("SPAD_TEACHER_BATCH_WORKERS")
            or min(16, max(1, len(teacher_indices)))
        )
        max_workers = max(1, min(max_workers, max(1, len(teacher_indices))))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _spad_teacher_status_details,
                    base=bases[index],
                    request_cfg=request_cfg,
                    teacher_prompt_version=teacher_prompt_version,
                ): index
                for index in teacher_indices
            }
            for future in as_completed(futures):
                teacher_details[futures[future]] = dict(future.result())

        results: list[dict[str, Any]] = []
        for index, base in enumerate(bases):
            uid = str(base["group_uid"])
            all_zero = all_zero_by_uid[uid]
            teacher_audit = _teacher_audit_details(
                base=base,
                request_cfg=request_cfg,
                teacher_prompt_version=teacher_prompt_version,
            )
            detail = teacher_details.get(
                index,
                {
                    **teacher_audit,
                    "teacher_called": False,
                    "teacher_skip_reason": (
                        "group_has_positive_em" if not all_zero else "no_search_evidence"
                    ),
                    "teacher_parse_status": "not_called",
                    "teacher_status_reward": 0.0,
                },
            )
            final_reward = (
                partial_weight * float(detail["teacher_status_reward"])
                if all_zero
                else float(base["em_reward"])
            )
            answers = _gold_answers(ground_truths[index])
            results.append(
                _sanitize(
                    {
                        **detail,
                        "final_reward": final_reward,
                        "reward_type": REWARD_VERSION,
                        "teacher_f1": final_reward,
                        "format_status": (
                            "valid" if base["actor_answer_parse_status"] == "parsed" else "invalid"
                        ),
                        "stop_status": (
                            "answer_closed"
                            if base["actor_answer_parse_status"] == "parsed"
                            else base["actor_answer_parse_status"]
                        ),
                        "search_count": base["search_count"],
                        "duplicate_query_count": base["duplicate_query_count"],
                        "teacher_prompt_version": teacher_prompt_version,
                        "actor_answer": base["actor_answer"],
                        "actor_answer_parse_status": base["actor_answer_parse_status"],
                        "em_reward": base["em_reward"],
                        "search_r1_answer_em": base["em_reward"],
                        "search_r1_extracted_answer": base["actor_answer"],
                        "legacy_em": base["em_reward"],
                        "legacy_f1": answer_group_metrics(
                            base["actor_answer"], answers
                        ).legacy_f1,
                        "group_uid": uid,
                        "group_size": len(grouped_indices[uid]),
                        "group_all_em_zero": all_zero,
                        "partial_reward_applied": all_zero,
                    }
                )
            )
        return results
    max_workers = int(kwargs.get("batch_workers") or os.environ.get("SPAD_TEACHER_BATCH_WORKERS") or min(16, item_count))
    max_workers = max(1, min(max_workers, item_count))
    results = [_sanitize({}) for _ in range(item_count)]

    def compute_one(index: int) -> dict[str, Any]:
        try:
            return compute_spad_search_policy_reward_details(
                data_source=str(data_sources[index]),
                solution_str=str(solution_strs[index]),
                ground_truth=ground_truths[index],
                extra_info=extra_infos[index],
                **kwargs,
            )
        except Exception as exc:
            reward_cfg = dict(kwargs.get("reward_cfg") or {})
            penalty = float(reward_cfg.get("invalid_format_penalty", -0.5))
            return _sanitize(
                {
                    "final_reward": penalty,
                    "teacher_called": False,
                    "teacher_skip_reason": f"reward_error:{type(exc).__name__}",
                    "format_status": "error",
                    "stop_status": "reward_error",
                    "teacher_parse_status": f"reward_error:{type(exc).__name__}",
                }
            )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_one, idx): idx for idx in range(item_count)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = _sanitize(dict(future.result()))
    return results


def compute_spad_em_teacher_backoff_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict[str, Any]],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Stable named entry point for the UID-group EM/teacher-backoff reward."""

    reward_cfg = dict(kwargs.get("reward_cfg") or {})
    configured_type = str(reward_cfg.get("type") or REWARD_VERSION)
    if configured_type != REWARD_VERSION:
        raise ValueError(
            f"{REWARD_VERSION} entry point received reward type {configured_type!r}"
        )
    reward_cfg["type"] = REWARD_VERSION
    return compute_spad_search_policy_reward_batch(
        data_sources,
        solution_strs,
        ground_truths,
        extra_infos,
        **{**kwargs, "reward_cfg": reward_cfg},
    )
