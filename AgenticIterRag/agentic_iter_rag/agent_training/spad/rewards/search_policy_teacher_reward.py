"""VERL custom reward for SPAD-RAG Stage 1 search-policy RL."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Mapping
from typing import Any

from agentic_iter_rag.agent_training.spad.prompts import build_teacher_messages
from agentic_iter_rag.agent_training.spad.reward import compute_search_policy_reward


_DEFAULT_EXTRA: dict[str, Any] = {
    "score": 0.0,
    "teacher_f1": 0.0,
    "teacher_called": False,
    "teacher_skip_reason": "",
    "format_status": "",
    "stop_status": "",
    "search_count": 0,
    "duplicate_query_count": 0,
    "teacher_elapsed_s": 0.0,
    "teacher_answer": "",
    "teacher_raw_content": "",
    "teacher_parse_status": "",
    "teacher_enable_thinking": "",
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
        value = ground_truth.get("target") or ground_truth.get("answers") or ground_truth.get("answer")
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


def _extract_teacher_answer(text: str) -> tuple[str, str]:
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
        "messages": build_teacher_messages(question=question, evidence_steps=evidence),
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
    answer, parse_status = _extract_teacher_answer(raw_content)
    enable_thinking = ""
    if isinstance(chat_template_kwargs, dict) and "enable_thinking" in chat_template_kwargs:
        enable_thinking = str(bool(chat_template_kwargs.get("enable_thinking"))).lower()
    return answer, time.perf_counter() - started, raw_content, parse_status, enable_thinking


def _sanitize(result: dict[str, Any]) -> dict[str, Any]:
    merged = {**_DEFAULT_EXTRA, **result}
    return {
        "score": float(merged.get("final_reward", merged.get("score", 0.0))),
        "teacher_f1": float(merged.get("teacher_f1", 0.0)),
        "teacher_called": bool(merged.get("teacher_called", False)),
        "teacher_skip_reason": str(merged.get("teacher_skip_reason") or ""),
        "format_status": str(merged.get("format_status") or ""),
        "stop_status": str(merged.get("stop_status") or ""),
        "search_count": int(merged.get("search_count") or 0),
        "duplicate_query_count": int(merged.get("duplicate_query_count") or 0),
        "teacher_elapsed_s": float(merged.get("teacher_elapsed_s") or 0.0),
        "teacher_answer": str(merged.get("teacher_answer") or ""),
        "teacher_raw_content": str(merged.get("teacher_raw_content") or ""),
        "teacher_parse_status": str(merged.get("teacher_parse_status") or ""),
        "teacher_enable_thinking": str(merged.get("teacher_enable_thinking") or ""),
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
    visible_top_m = int(kwargs.get("visible_top_m") or 5)
    gold_answers = _gold_answers(ground_truth)
    evidence = _search_evidence(extra_info, visible_top_m=visible_top_m)
    search_count = len(evidence)
    duplicate_query_count = _count_duplicate_queries(evidence)
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

    (
        teacher_answer,
        teacher_elapsed_s,
        teacher_raw_content,
        teacher_parse_status,
        teacher_enable_thinking,
    ) = _call_teacher(
        question=question,
        evidence=evidence,
        request_cfg=request_cfg,
    )
    result = compute_search_policy_reward(
        actor_output=solution_str,
        teacher_answer=teacher_answer,
        gold_answers=gold_answers,
        search_count=search_count,
        duplicate_query_count=duplicate_query_count,
        reward_cfg=reward_cfg,
        legal_stop=True,
        stop_at_answer_opening=True,
    )
    result["search_count"] = search_count
    result["duplicate_query_count"] = duplicate_query_count
    result["teacher_elapsed_s"] = teacher_elapsed_s
    result["teacher_answer"] = teacher_answer
    result["teacher_raw_content"] = teacher_raw_content
    result["teacher_parse_status"] = teacher_parse_status
    result["teacher_enable_thinking"] = teacher_enable_thinking
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
    max_workers = int(kwargs.get("batch_workers") or os.environ.get("SPAD_TEACHER_BATCH_WORKERS") or min(16, item_count))
    max_workers = max(1, min(max_workers, item_count))
    results = [_sanitize({}) for _ in range(item_count)]

    def compute_one(index: int) -> dict[str, Any]:
        return compute_spad_search_policy_reward_details(
            data_source=str(data_sources[index]),
            solution_str=str(solution_strs[index]),
            ground_truth=ground_truths[index],
            extra_info=extra_infos[index],
            **kwargs,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_one, idx): idx for idx in range(item_count)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = _sanitize(dict(future.result()))
    return results
