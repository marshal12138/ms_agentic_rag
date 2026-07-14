"""SPAD Stage 1 reward with an audited Teacher-answer gold token-F1 bonus.

This module deliberately composes the stable ``spad_em_teacher_backoff``
implementation instead of changing it.  The only added behavior is a fixed
bonus proportional to the Teacher answer's maximum token F1 against the gold
aliases.  The extra bonus requires both a complete Actor answer and a
Teacher-supported evidence status; the stable base backoff remains unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import (
    compute_spad_em_teacher_backoff_batch,
)
from agentic_iter_rag.metrics.answer_metrics import groups_from_ground_truth, legacy_token_f1


REWARD_VERSION = "spad_em_teacher_backoff_gold_token_f1_bonus"
BASE_REWARD_VERSION = "spad_em_teacher_backoff"
BONUS_ELIGIBILITY_VERSION = "actor_answer_closed_teacher_supported_v2"
BONUS_TEACHER_STATUSES = frozenset({"supported_answer", "ambiguous_evidence"})


def apply_teacher_gold_token_f1_bonus(
    result: Mapping[str, Any],
    ground_truth: Any,
    *,
    bonus_weight: float,
) -> dict[str, Any]:
    """Return one copied reward detail with the independently audited bonus."""

    updated = dict(result)
    base_reward = float(updated.get("score", updated.get("final_reward", 0.0)))
    teacher_answer = str(updated.get("teacher_answer") or "").strip()
    actor_answer_parse_status = str(updated.get("actor_answer_parse_status") or "")
    teacher_evidence_status = str(updated.get("teacher_evidence_status") or "")
    answers, _, _, _ = groups_from_ground_truth(ground_truth)
    eligible = bool(
        updated.get("group_all_em_zero")
        and updated.get("teacher_called")
        and not updated.get("teacher_format_error")
        and str(updated.get("teacher_parse_status") or "") == "parsed"
        and teacher_answer
        and actor_answer_parse_status == "parsed"
        and teacher_evidence_status in BONUS_TEACHER_STATUSES
    )
    teacher_gold_token_f1 = legacy_token_f1(teacher_answer, answers) if eligible else 0.0
    bonus = float(bonus_weight) * teacher_gold_token_f1
    final_reward = base_reward + bonus
    updated.update(
        {
            "score": final_reward,
            "final_reward": final_reward,
            "teacher_f1": final_reward,
            "reward_type": REWARD_VERSION,
            "base_reward": base_reward,
            "teacher_gold_token_f1": teacher_gold_token_f1,
            "teacher_gold_token_f1_bonus": bonus,
            "teacher_gold_token_f1_bonus_applied": bool(bonus > 0.0),
            "teacher_gold_token_f1_bonus_applied_count": int(bonus > 0.0),
            "teacher_gold_token_f1_bonus_weight": float(bonus_weight),
            "teacher_gold_token_f1_bonus_eligible": eligible,
            "teacher_gold_token_f1_bonus_eligibility_version": BONUS_ELIGIBILITY_VERSION,
        }
    )
    return updated


def compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict[str, Any]],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run stable group reward, then add the Teacher-answer gold token-F1 bonus."""

    reward_cfg = dict(kwargs.get("reward_cfg") or {})
    configured_type = str(reward_cfg.get("type") or REWARD_VERSION)
    if configured_type != REWARD_VERSION:
        raise ValueError(
            f"{REWARD_VERSION} entry point received reward type {configured_type!r}"
        )
    variant_cfg = reward_cfg.get(REWARD_VERSION)
    if not isinstance(variant_cfg, Mapping):
        variant_cfg = {}
    partial_reward = float(variant_cfg.get("partial_reward", 0.1))
    bonus_weight = float(variant_cfg.get("gold_token_f1_bonus", 0.1))
    if bonus_weight < 0.0:
        raise ValueError("gold_token_f1_bonus must be non-negative")

    base_reward_cfg = dict(reward_cfg)
    base_reward_cfg["type"] = BASE_REWARD_VERSION
    base_cfg = base_reward_cfg.get(BASE_REWARD_VERSION)
    base_cfg = dict(base_cfg) if isinstance(base_cfg, Mapping) else {}
    base_cfg["partial_reward"] = partial_reward
    base_reward_cfg[BASE_REWARD_VERSION] = base_cfg
    base_results = compute_spad_em_teacher_backoff_batch(
        data_sources,
        solution_strs,
        ground_truths,
        extra_infos,
        **{**kwargs, "reward_cfg": base_reward_cfg},
    )
    if len(base_results) != len(ground_truths):
        raise ValueError("base SPAD reward returned a different number of results")
    return [
        apply_teacher_gold_token_f1_bonus(
            result,
            ground_truths[index],
            bonus_weight=bonus_weight,
        )
        for index, result in enumerate(base_results)
    ]
