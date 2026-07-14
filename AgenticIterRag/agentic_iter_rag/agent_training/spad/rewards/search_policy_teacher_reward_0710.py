"""Frozen SPAD Stage 1 reward used by the 2026-07-10 formal run.

This module intentionally remains separate from the UID-group EM/backoff
reward. Its public entry point, prompt, teacher parser, and per-rollout
scheduling contract reproduce the 0710 teacher-answer-F1 experiment.
"""

from __future__ import annotations

import json
import os
import re
import string
import time
import urllib.request
from collections import Counter
from collections.abc import Mapping
from typing import Any

from agentic_iter_rag.agent_training.spad.parsers import parse_reason_answer_opening_stop


REWARD_VERSION = "spad_teacher_f1_0710"
TEACHER_SYSTEM_PROMPT_0710 = (
    "You are an evidence-grounded QA model. Output only two XML tag blocks: "
    "<reason>...</reason><answer>...</answer>. The first character must be <. "
    "Do not repeat instructions. Do not use markdown or numbered lists. Use only the evidence. "
    "The answer must be only the final short answer span, usually one person, date, place, number, or title. "
    "Keep names, dates, places, titles, and numbers exactly as written in the evidence when possible. "
    "If the evidence is insufficient, explain what evidence is missing and why it is insufficient in reason, "
    "and answer exactly 证据不足无法作答."
)

_DEFAULT_EXTRA: dict[str, Any] = {
    "score": 0.0,
    "reward_type": REWARD_VERSION,
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
    "teacher_raw_content": "",
    "teacher_parse_status": "",
    "teacher_format_error": False,
    "teacher_format_error_count": 0,
    "teacher_evidence_status": "",
    "teacher_enable_thinking": "",
    "bad_stop_applied": False,
    "bad_stop_count": 0,
    "bad_stop_reason": "",
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
    if isinstance(value, Mapping):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if hasattr(value, "items"):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


def _gold_answers(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, dict):
        value = ground_truth.get("target") or ground_truth.get("answers") or ground_truth.get("answer")
    else:
        value = ground_truth
    return [str(item) for item in _as_list(value) if str(item).strip()]


def _normalize_answer(text: str) -> str:
    value = text.lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def _answer_f1(prediction: str, gold_answers: list[str]) -> float:
    prediction_tokens = _normalize_answer(prediction).split()
    best = 0.0
    for answer in gold_answers:
        answer_tokens = _normalize_answer(answer).split()
        if not prediction_tokens or not answer_tokens:
            continue
        overlap = sum((Counter(prediction_tokens) & Counter(answer_tokens)).values())
        if overlap <= 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


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
    evidence: list[dict[str, Any]] = []
    for turn, raw_detail in enumerate(_as_list(extra_info.get("tool_call_details")), start=1):
        if not isinstance(raw_detail, dict):
            continue
        evidence.append(
            {
                "turn": turn,
                "sub_query": str(raw_detail.get("sub_query") or "").strip(),
                "docs": _detail_docs(raw_detail, visible_top_m=visible_top_m),
            }
        )
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


def _build_teacher_messages_0710(question: str, evidence_steps: list[dict[str, Any]]) -> list[dict[str, str]]:
    lines = [f"Original question:\n{question}", "", "Search evidence:"]
    if not evidence_steps:
        lines.append("(no search evidence provided)")
    for index, step in enumerate(evidence_steps, start=1):
        lines.append(f"\nRound {index} sub_query:\n{step.get('sub_query') or ''}")
        for doc_index, doc in enumerate((step.get("docs") or [])[:5], start=1):
            title = doc.get("title") or doc.get("doc_id") or f"doc-{doc_index}"
            text = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
            lines.append(f"[{doc_index}] {title}\n{text}")
    lines.append(
        "\nNow output the final result directly. "
        "Do not analyze the instruction. Do not repeat rules. Begin with <reason>."
    )
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT_0710},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _extract_teacher_answer_0710(text: str) -> tuple[str, str]:
    matches = list(re.finditer(r"<reason>.*?</reason>\s*<answer>(.*?)</answer>", text, re.S))
    parse_status = "parsed"
    if matches:
        answer = matches[-1].group(1).strip()
    else:
        open_matches = list(re.finditer(r"<reason>.*?</reason>\s*<answer>\s*(.*)$", text, re.S))
        if not open_matches:
            return "", "missing_reason_tag"
        answer = open_matches[-1].group(1).strip()
        parse_status = "missing_answer_close_tag"
    if answer in {"...", "…"}:
        return "", "placeholder_answer"
    if not answer:
        return "", "empty_answer"
    return answer, parse_status


def _post_teacher(endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_teacher_0710(
    *,
    question: str,
    evidence: list[dict[str, Any]],
    request_cfg: dict[str, Any],
) -> tuple[str, float, str, str, str]:
    endpoint = str(request_cfg.get("endpoint") or os.environ.get("SPAD_TEACHER_ENDPOINT") or "").strip()
    model = str(request_cfg.get("model") or os.environ.get("SPAD_TEACHER_MODEL") or "").strip()
    if not endpoint:
        raise ValueError("SPAD teacher endpoint is empty")
    if not model:
        raise ValueError("SPAD teacher model is empty")
    started = time.perf_counter()
    payload = {
        "model": model,
        "messages": _build_teacher_messages_0710(question, evidence),
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
    raw_content = str(result.get("choices", [{}])[0].get("message", {}).get("content", ""))
    answer, parse_status = _extract_teacher_answer_0710(raw_content)
    enable_thinking = ""
    if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
        enable_thinking = str(bool(chat_template_kwargs.get("enable_thinking"))).lower()
    return answer, time.perf_counter() - started, raw_content, parse_status, enable_thinking


def _sanitize(result: dict[str, Any]) -> dict[str, Any]:
    merged = {**_DEFAULT_EXTRA, **result}
    search_count = int(merged.get("search_count") or 0)
    free_search_count = max(0, int(merged.get("free_search_count", 1) or 0))
    paid_search_count = int(
        merged.get("paid_search_count", merged.get("second_plus_search_count", max(0, search_count - free_search_count)))
        or 0
    )
    return {
        "score": float(merged.get("final_reward", merged.get("score", 0.0))),
        "reward_type": REWARD_VERSION,
        "teacher_f1": float(merged.get("teacher_f1", 0.0)),
        "teacher_called": bool(merged.get("teacher_called", False)),
        "teacher_called_count": int(bool(merged.get("teacher_called", False))),
        "teacher_skip_reason": str(merged.get("teacher_skip_reason") or ""),
        "format_status": str(merged.get("format_status") or ""),
        "stop_status": str(merged.get("stop_status") or ""),
        "search_count": search_count,
        "free_search_count": free_search_count,
        "paid_search_count": paid_search_count,
        "second_plus_search_count": int(merged.get("second_plus_search_count", paid_search_count) or 0),
        "duplicate_query_count": int(merged.get("duplicate_query_count") or 0),
        "effective_search_cost": float(merged.get("effective_search_cost") or 0.0),
        "teacher_elapsed_s": float(merged.get("teacher_elapsed_s") or 0.0),
        "teacher_answer": str(merged.get("teacher_answer") or ""),
        "teacher_raw_content": str(merged.get("teacher_raw_content") or ""),
        "teacher_parse_status": str(merged.get("teacher_parse_status") or ""),
        "teacher_format_error": bool(merged.get("teacher_format_error", False)),
        "teacher_format_error_count": int(bool(merged.get("teacher_format_error", False))),
        "teacher_evidence_status": str(merged.get("teacher_evidence_status") or ""),
        "teacher_enable_thinking": str(merged.get("teacher_enable_thinking") or ""),
        "bad_stop_applied": bool(merged.get("bad_stop_applied", False)),
        "bad_stop_count": int(bool(merged.get("bad_stop_applied", False))),
        "bad_stop_reason": str(merged.get("bad_stop_reason") or ""),
    }


def compute_spad_teacher_f1_0710_details(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute the frozen 0710 per-rollout teacher-answer-F1 reward."""

    del data_source
    reward_cfg = dict(kwargs.get("reward_cfg") or {})
    request_cfg = dict(kwargs.get("teacher_request") or {})
    visible_top_m = int(kwargs.get("visible_top_m") or 5)
    evidence = _search_evidence(extra_info, visible_top_m=visible_top_m)
    search_count = len(evidence)
    duplicate_query_count = _count_duplicate_queries(evidence)
    legal_stop = "<answer>" in solution_str and "</answer>" not in solution_str

    if search_count <= 0:
        return _sanitize(
            {
                "final_reward": float(reward_cfg.get("no_finish_penalty", -0.5)),
                "teacher_skip_reason": "no_search_evidence",
                "format_status": "unknown",
                "stop_status": "no_search_evidence",
            }
        )

    parsed = parse_reason_answer_opening_stop(solution_str)
    if not parsed.valid:
        return _sanitize(
            {
                "final_reward": float(reward_cfg.get("invalid_format_penalty", -0.5)),
                "teacher_skip_reason": str(parsed.error_code or "invalid_format"),
                "format_status": "invalid",
                "stop_status": "unknown",
                "search_count": search_count,
                "duplicate_query_count": duplicate_query_count,
            }
        )
    if not legal_stop:
        return _sanitize(
            {
                "final_reward": float(reward_cfg.get("no_finish_penalty", -0.5)),
                "teacher_skip_reason": "no_finish",
                "format_status": "valid",
                "stop_status": "no_finish",
                "search_count": search_count,
                "duplicate_query_count": duplicate_query_count,
            }
        )

    try:
        teacher_answer, elapsed_s, raw_content, parse_status, enable_thinking = _call_teacher_0710(
            question=_question(extra_info),
            evidence=evidence,
            request_cfg=request_cfg,
        )
    except Exception as exc:
        teacher_answer = ""
        elapsed_s = 0.0
        raw_content = ""
        parse_status = f"teacher_error:{type(exc).__name__}"
        enable_thinking = ""

    teacher_format_error = parse_status != "parsed"
    free_search_count = max(0, int(reward_cfg.get("free_search_count", 1) or 0))
    paid_search_count = max(0, search_count - free_search_count)
    effective_search_cost = float(reward_cfg.get("search_cost", 0.02)) * paid_search_count
    missing_reason_penalty = float(reward_cfg.get("missing_reason_penalty", -0.02)) * int(
        getattr(parsed, "missing_reason_count", 0) or 0
    )
    insufficient_answer = str(
        (reward_cfg.get("bad_stop") or {}).get("insufficient_answer") or "证据不足无法作答"
    )
    evidence_status = (
        "insufficient_evidence" if teacher_answer.strip() == insufficient_answer else "supported_answer"
    )
    bad_stop_cfg = dict(reward_cfg.get("bad_stop") or {})
    bad_stop_applied = False
    bad_stop_reason = ""

    if teacher_format_error:
        final_reward = float(bad_stop_cfg.get("teacher_format_error_penalty", -0.1))
        teacher_f1 = 0.0
    elif bool(bad_stop_cfg.get("enabled", True)) and evidence_status == "insufficient_evidence":
        if search_count < int(reward_cfg.get("max_search_turns", 5) or 5):
            final_reward = float(bad_stop_cfg.get("penalty", -0.35))
            bad_stop_applied = True
            bad_stop_reason = "early_stop_insufficient_evidence"
        else:
            final_reward = float(bad_stop_cfg.get("max_budget_failed_penalty", -0.15))
            bad_stop_reason = "max_budget_insufficient_evidence"
        teacher_f1 = 0.0
    else:
        teacher_f1 = _answer_f1(teacher_answer, _gold_answers(ground_truth))
        duplicate_penalty = float(reward_cfg.get("duplicate_query_penalty", -0.1)) * duplicate_query_count
        final_reward = (
            float(reward_cfg.get("teacher_f1_weight", 1.0)) * teacher_f1
            - effective_search_cost
            + duplicate_penalty
            + missing_reason_penalty
        )

    return _sanitize(
        {
            "final_reward": final_reward,
            "teacher_f1": teacher_f1,
            "teacher_called": True,
            "format_status": "valid",
            "stop_status": "legal_finish",
            "search_count": search_count,
            "free_search_count": free_search_count,
            "paid_search_count": paid_search_count,
            "second_plus_search_count": paid_search_count,
            "effective_search_cost": effective_search_cost,
            "duplicate_query_count": duplicate_query_count,
            "teacher_elapsed_s": elapsed_s,
            "teacher_answer": teacher_answer,
            "teacher_raw_content": raw_content,
            "teacher_parse_status": parse_status,
            "teacher_enable_thinking": enable_thinking,
            "teacher_format_error": teacher_format_error,
            "teacher_evidence_status": evidence_status,
            "bad_stop_applied": bad_stop_applied,
            "bad_stop_reason": bad_stop_reason,
        }
    )


def compute_spad_teacher_f1_0710(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> float:
    return float(
        compute_spad_teacher_f1_0710_details(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )["score"]
    )
