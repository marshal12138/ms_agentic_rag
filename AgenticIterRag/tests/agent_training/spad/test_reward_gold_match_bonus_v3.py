"""Tests for Gold Token-F1 V3 reward metadata and configuration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_gold_match_bonus_v3 import (
    ADVANTAGE_SCALE_KEY,
    ADVANTAGE_SCALE_VERSION,
    REWARD_VERSION,
    apply_teacher_gold_token_f1_bonus_v3,
    compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_batch,
)


def _detail(**overrides: object) -> dict[str, object]:
    detail: dict[str, object] = {
        "score": 0.1,
        "final_reward": 0.1,
        "teacher_f1": 0.1,
        "reward_type": "spad_em_teacher_backoff",
        "group_all_em_zero": True,
        "teacher_called": True,
        "teacher_format_error": False,
        "teacher_parse_status": "parsed",
        "teacher_answer": "Paris",
        "teacher_evidence_status": "supported_answer",
        "actor_answer": "Lyon",
        "actor_answer_parse_status": "parsed",
    }
    detail.update(overrides)
    return detail


class GoldMatchBonusV3PostprocessorTest(unittest.TestCase):
    def test_teacher_fallback_group_gets_weak_postnorm_scale(self) -> None:
        result = apply_teacher_gold_token_f1_bonus_v3(
            _detail(),
            {"target": ["Paris"]},
            bonus_weight=0.1,
            teacher_group_postnorm_scale=0.1,
        )

        self.assertEqual(result["reward_type"], REWARD_VERSION)
        self.assertEqual(result["advantage_source"], "teacher_fallback")
        self.assertAlmostEqual(result[ADVANTAGE_SCALE_KEY], 0.1)
        self.assertEqual(
            result["advantage_postnorm_scale_version"], ADVANTAGE_SCALE_VERSION
        )
        self.assertAlmostEqual(result["score"], 0.2)

    def test_actor_em_group_keeps_unit_postnorm_scale(self) -> None:
        result = apply_teacher_gold_token_f1_bonus_v3(
            _detail(group_all_em_zero=False, teacher_called=False),
            {"target": ["Paris"]},
            bonus_weight=0.1,
            teacher_group_postnorm_scale=0.1,
        )

        self.assertEqual(result["advantage_source"], "actor_em")
        self.assertAlmostEqual(result[ADVANTAGE_SCALE_KEY], 1.0)


class GoldMatchBonusV3BatchEntryTest(unittest.TestCase):
    @patch(
        "agentic_iter_rag.agent_training.spad.rewards."
        "search_policy_teacher_reward_gold_match_bonus_v3."
        "compute_spad_em_teacher_backoff_batch"
    )
    def test_batch_entry_delegates_to_stable_and_emits_scale(self, base) -> None:
        base.return_value = [_detail()]
        results = compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_batch(
            ["source"],
            ["<answer>Lyon</answer>"],
            [{"target": ["Paris"]}],
            [{"uid": "q1"}],
            reward_cfg={
                "type": REWARD_VERSION,
                REWARD_VERSION: {
                    "partial_reward": 0.1,
                    "gold_token_f1_bonus": 0.1,
                    "teacher_group_postnorm_scale": 0.1,
                },
            },
        )

        delegated_cfg = base.call_args.kwargs["reward_cfg"]
        self.assertEqual(delegated_cfg["type"], "spad_em_teacher_backoff")
        self.assertAlmostEqual(results[0][ADVANTAGE_SCALE_KEY], 0.1)

    def test_invalid_teacher_group_scale_is_rejected(self) -> None:
        for scale in (0.0, -0.1, 1.1):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError, "must be in"
            ):
                compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_batch(
                    [],
                    [],
                    [],
                    [],
                    reward_cfg={
                        "type": REWARD_VERSION,
                        REWARD_VERSION: {"teacher_group_postnorm_scale": scale},
                    },
                )


if __name__ == "__main__":
    unittest.main()
