"""Reward utilities for SPAD-RAG search-policy training."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping
from typing import Any

from agentic_iter_rag.agent_training.spad.parsers import parse_reason_answer, parse_reason_answer_opening_stop


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ans_tokens = normalize_answer(answer).split()
    if not pred_tokens or not ans_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ans_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ans_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(prediction: str, gold_answers: list[str]) -> float:
    return max((f1(prediction, item) for item in gold_answers), default=0.0)


def _ground_truth_answers(ground_truth: Any) -> list[str]:
    if isinstance(ground_truth, Mapping):
        value = ground_truth.get("target")
        if value is None:
            value = ground_truth.get("answers")
        if value is None:
            value = ground_truth.get("answer")
    else:
        value = ground_truth
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    return [str(item) for item in value if item is not None and str(item).strip()]


def compute_gold_answer_f1_reward_details(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Stage 3 answer-only GRPO reward using the last complete answer block."""

    del data_source, extra_info, kwargs
    parsed = parse_reason_answer(solution_str)
    answers = _ground_truth_answers(ground_truth)
    score = compute_f1(parsed.answer, answers) if parsed.valid and answers else 0.0
    return {
        "score": float(score),
        "gold_answer_f1": float(score),
        "actor_answer": str(parsed.answer or ""),
        "actor_answer_parse_status": "parsed" if parsed.valid else str(parsed.error_code or "invalid"),
        "reward_type": "gold_answer_f1",
    }


def compute_search_policy_reward(
    *,
    actor_output: str,
    gold_answers: list[str],
    search_count: int,
    duplicate_query_count: int,
    reward_cfg: dict[str, Any],
    teacher_answer: str | None = None,
    teacher_evidence_status: str | None = None,
    teacher_format_error: bool = False,
    legal_stop: bool = True,
    stop_at_answer_opening: bool = False,
) -> dict[str, Any]:
    """Compute Stage 1 reward with teacher short-circuit for invalid trajectories."""

    parsed = (
        parse_reason_answer_opening_stop(actor_output)
        if stop_at_answer_opening
        else parse_reason_answer(actor_output)
    )
    if not parsed.valid:
        penalty = float(reward_cfg.get("invalid_format_penalty", -0.5))
        free_search_count = max(0, int(reward_cfg.get("free_search_count", 1) or 0))
        paid_search_count = max(0, int(search_count) - free_search_count)
        return {
            "final_reward": penalty,
            "teacher_called": False,
            "teacher_skip_reason": parsed.error_code,
            "teacher_f1": 0.0,
            "teacher_evidence_status": "",
            "teacher_format_error": False,
            "bad_stop_applied": False,
            "bad_stop_reason": "",
            "free_search_count": free_search_count,
            "paid_search_count": paid_search_count,
            "second_plus_search_count": paid_search_count,
            "effective_search_cost": 0.0,
            "format_status": "invalid",
            "action_status": "skipped_format_error",
            "stop_status": "unknown",
            "reward_breakdown": {"invalid_format_penalty": penalty},
            "parse": parsed.to_dict(),
        }
    if not legal_stop:
        penalty = float(reward_cfg.get("no_finish_penalty", -0.5))
        free_search_count = max(0, int(reward_cfg.get("free_search_count", 1) or 0))
        paid_search_count = max(0, int(search_count) - free_search_count)
        return {
            "final_reward": penalty,
            "teacher_called": False,
            "teacher_skip_reason": "no_finish",
            "teacher_f1": 0.0,
            "teacher_evidence_status": "",
            "teacher_format_error": False,
            "bad_stop_applied": False,
            "bad_stop_reason": "",
            "free_search_count": free_search_count,
            "paid_search_count": paid_search_count,
            "second_plus_search_count": paid_search_count,
            "effective_search_cost": 0.0,
            "format_status": "valid",
            "action_status": "valid",
            "stop_status": "no_finish",
            "reward_breakdown": {"no_finish_penalty": penalty},
            "parse": parsed.to_dict(),
        }

    raw_bad_stop_cfg = reward_cfg.get("bad_stop")
    bad_stop_cfg = dict(raw_bad_stop_cfg) if isinstance(raw_bad_stop_cfg, Mapping) else {}
    bad_stop_enabled = bool(bad_stop_cfg.get("enabled", True))
    max_search_turns = int(reward_cfg.get("max_search_turns", 3) or 3)
    evidence_status = (teacher_evidence_status or "supported_answer").strip()
    search_cost = float(reward_cfg.get("search_cost", 0.02))
    free_search_count = max(0, int(reward_cfg.get("free_search_count", 1) or 0))
    paid_search_count = max(0, int(search_count) - free_search_count)
    effective_search_cost = search_cost * paid_search_count
    missing_reason_penalty = float(reward_cfg.get("missing_reason_penalty", -0.02)) * int(
        getattr(parsed, "missing_reason_count", 0) or 0
    )

    if teacher_format_error:
        penalty = float(bad_stop_cfg.get("teacher_format_error_penalty", -0.1))
        return {
            "final_reward": penalty,
            "teacher_called": True,
            "teacher_skip_reason": None,
            "teacher_f1": 0.0,
            "teacher_evidence_status": evidence_status,
            "teacher_format_error": True,
            "bad_stop_applied": False,
            "bad_stop_reason": "",
            "free_search_count": free_search_count,
            "paid_search_count": paid_search_count,
            "effective_search_cost": effective_search_cost,
            "format_status": "valid",
            "action_status": "valid",
            "stop_status": "legal_finish",
            "reward_breakdown": {
                "teacher_format_error_penalty": penalty,
                "effective_search_cost": effective_search_cost,
                "paid_search_count": paid_search_count,
                "second_plus_search_count": paid_search_count,
            },
            "parse": parsed.to_dict(),
        }

    if bad_stop_enabled and evidence_status in {"insufficient_evidence", "ambiguous_evidence"}:
        if int(search_count) < max_search_turns:
            penalty = float(bad_stop_cfg.get("penalty", -0.20))
            bad_stop_applied = True
            bad_stop_reason = f"early_stop_{evidence_status}"
        else:
            penalty = float(bad_stop_cfg.get("max_budget_failed_penalty", -0.15))
            bad_stop_applied = False
            bad_stop_reason = f"max_budget_{evidence_status}"
        return {
            "final_reward": penalty,
            "teacher_called": True,
            "teacher_skip_reason": None,
            "teacher_f1": 0.0,
            "teacher_evidence_status": evidence_status,
            "teacher_format_error": False,
            "bad_stop_applied": bad_stop_applied,
            "bad_stop_reason": bad_stop_reason,
            "free_search_count": free_search_count,
            "paid_search_count": paid_search_count,
            "effective_search_cost": effective_search_cost,
            "format_status": "valid",
            "action_status": "valid",
            "stop_status": "legal_finish",
            "reward_breakdown": {
                "bad_stop_penalty": penalty if bad_stop_applied else 0.0,
                "max_budget_failed_penalty": penalty if not bad_stop_applied else 0.0,
                "effective_search_cost": effective_search_cost,
                "paid_search_count": paid_search_count,
                "second_plus_search_count": paid_search_count,
                "missing_reason_penalty": missing_reason_penalty,
            },
            "parse": parsed.to_dict(),
        }

    answer_for_score = teacher_answer if teacher_answer is not None else getattr(parsed, "answer", "") or ""
    teacher_f1 = compute_f1(answer_for_score, gold_answers)
    duplicate_penalty = float(reward_cfg.get("duplicate_query_penalty", -0.1)) * int(duplicate_query_count)
    base = float(reward_cfg.get("teacher_f1_weight", 1.0)) * teacher_f1
    final_reward = base - effective_search_cost + duplicate_penalty + missing_reason_penalty
    return {
        "final_reward": final_reward,
        "teacher_called": True,
        "teacher_skip_reason": None,
        "teacher_f1": teacher_f1,
        "teacher_evidence_status": evidence_status,
        "teacher_format_error": False,
        "bad_stop_applied": False,
        "bad_stop_reason": "",
        "free_search_count": free_search_count,
        "paid_search_count": paid_search_count,
        "effective_search_cost": effective_search_cost,
        "format_status": "valid",
        "action_status": "valid",
        "stop_status": "legal_finish",
        "reward_breakdown": {
            "teacher_f1_reward": base,
            "search_cost": -effective_search_cost,
            "effective_search_cost": effective_search_cost,
            "paid_search_count": paid_search_count,
            "second_plus_search_count": paid_search_count,
            "duplicate_query_penalty": duplicate_penalty,
            "missing_reason_penalty": missing_reason_penalty,
        },
        "parse": parsed.to_dict(),
    }
