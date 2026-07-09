"""Reward utilities for SPAD-RAG search-policy training."""

from __future__ import annotations

import re
import string
from collections import Counter
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


def compute_search_policy_reward(
    *,
    actor_output: str,
    gold_answers: list[str],
    search_count: int,
    duplicate_query_count: int,
    reward_cfg: dict[str, Any],
    teacher_answer: str | None = None,
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
        return {
            "final_reward": penalty,
            "teacher_called": False,
            "teacher_skip_reason": parsed.error_code,
            "teacher_f1": 0.0,
            "format_status": "invalid",
            "action_status": "skipped_format_error",
            "stop_status": "unknown",
            "reward_breakdown": {"invalid_format_penalty": penalty},
            "parse": parsed.to_dict(),
        }
    if not legal_stop:
        penalty = float(reward_cfg.get("no_finish_penalty", -0.5))
        return {
            "final_reward": penalty,
            "teacher_called": False,
            "teacher_skip_reason": "no_finish",
            "teacher_f1": 0.0,
            "format_status": "valid",
            "action_status": "valid",
            "stop_status": "no_finish",
            "reward_breakdown": {"no_finish_penalty": penalty},
            "parse": parsed.to_dict(),
        }

    answer_for_score = teacher_answer if teacher_answer is not None else getattr(parsed, "answer", "") or ""
    teacher_f1 = compute_f1(answer_for_score, gold_answers)
    search_penalty = float(reward_cfg.get("search_cost", 0.02)) * int(search_count)
    duplicate_penalty = float(reward_cfg.get("duplicate_query_penalty", -0.1)) * int(duplicate_query_count)
    missing_reason_penalty = float(reward_cfg.get("missing_reason_penalty", -0.02)) * int(
        getattr(parsed, "missing_reason_count", 0) or 0
    )
    base = float(reward_cfg.get("teacher_f1_weight", 1.0)) * teacher_f1
    final_reward = base - search_penalty + duplicate_penalty + missing_reason_penalty
    return {
        "final_reward": final_reward,
        "teacher_called": True,
        "teacher_skip_reason": None,
        "teacher_f1": teacher_f1,
        "format_status": "valid",
        "action_status": "valid",
        "stop_status": "legal_finish",
        "reward_breakdown": {
            "teacher_f1_reward": base,
            "search_cost": -search_penalty,
            "duplicate_query_penalty": duplicate_penalty,
            "missing_reason_penalty": missing_reason_penalty,
        },
        "parse": parsed.to_dict(),
    }
