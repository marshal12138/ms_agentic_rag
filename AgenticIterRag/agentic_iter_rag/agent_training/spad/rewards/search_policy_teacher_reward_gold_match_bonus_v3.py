"""SPAD Stage1 reward V3 with post-normalization Teacher-group scaling.

The reward values and V2 bonus eligibility remain unchanged. This independent
variant adds an explicit per-group advantage scale audit field. The GRPO
trainer consumes that field only after standard group mean/std normalization.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import (
    compute_spad_em_teacher_backoff_batch,
)
from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_gold_match_bonus import (
    BONUS_ELIGIBILITY_VERSION,
    apply_teacher_gold_token_f1_bonus,
)


REWARD_VERSION = "spad_em_teacher_backoff_gold_token_f1_bonus_v3"
BASE_REWARD_VERSION = "spad_em_teacher_backoff"
ADVANTAGE_SCALE_KEY = "advantage_postnorm_scale"
ADVANTAGE_SCALE_VERSION = "teacher_fallback_v1"


def apply_teacher_gold_token_f1_bonus_v3(
    result: Mapping[str, Any],
    ground_truth: Any,
    *,
    bonus_weight: float,
    teacher_group_postnorm_scale: float,
) -> dict[str, Any]:
    """Apply the V2 bonus and attach the V3 whole-group advantage scale."""

    updated = apply_teacher_gold_token_f1_bonus(
        result,
        ground_truth,
        bonus_weight=bonus_weight,
    )
    teacher_fallback = bool(updated.get("group_all_em_zero"))
    updated.update(
        {
            "reward_type": REWARD_VERSION,
            "advantage_source": "teacher_fallback" if teacher_fallback else "actor_em",
            ADVANTAGE_SCALE_KEY: (
                float(teacher_group_postnorm_scale) if teacher_fallback else 1.0
            ),
            "advantage_postnorm_scale_version": ADVANTAGE_SCALE_VERSION,
            "teacher_gold_token_f1_bonus_eligibility_version": BONUS_ELIGIBILITY_VERSION,
        }
    )
    return updated


def compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict[str, Any]],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Compute stable reward plus V2 bonus and emit V3 group-scale metadata."""

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
    teacher_group_postnorm_scale = float(
        variant_cfg.get("teacher_group_postnorm_scale", 0.1)
    )
    if bonus_weight < 0.0:
        raise ValueError("gold_token_f1_bonus must be non-negative")
    if not 0.0 < teacher_group_postnorm_scale <= 1.0:
        raise ValueError("teacher_group_postnorm_scale must be in (0, 1]")

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
        apply_teacher_gold_token_f1_bonus_v3(
            result,
            ground_truths[index],
            bonus_weight=bonus_weight,
            teacher_group_postnorm_scale=teacher_group_postnorm_scale,
        )
        for index, result in enumerate(base_results)
    ]
