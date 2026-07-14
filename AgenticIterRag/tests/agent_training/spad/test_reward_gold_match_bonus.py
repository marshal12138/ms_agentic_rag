"""Tests for the independent SPAD Teacher-answer gold token-F1 reward variant."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_gold_match_bonus import (
    BONUS_ELIGIBILITY_VERSION,
    REWARD_VERSION,
    apply_teacher_gold_token_f1_bonus,
    compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch,
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


class GoldMatchBonusPostprocessorTest(unittest.TestCase):
    def test_matching_teacher_answer_adds_bonus_and_preserves_base_audit(self) -> None:
        result = apply_teacher_gold_token_f1_bonus(
            _detail(), {"target": ["Paris", "City of Paris"]}, bonus_weight=0.1
        )

        self.assertAlmostEqual(result["base_reward"], 0.1)
        self.assertAlmostEqual(result["teacher_gold_token_f1"], 1.0)
        self.assertAlmostEqual(result["teacher_gold_token_f1_bonus"], 0.1)
        self.assertAlmostEqual(result["score"], 0.2)
        self.assertAlmostEqual(result["final_reward"], 0.2)
        self.assertEqual(result["reward_type"], REWARD_VERSION)
        self.assertTrue(result["teacher_gold_token_f1_bonus_applied"])
        self.assertTrue(result["teacher_gold_token_f1_bonus_eligible"])
        self.assertEqual(
            result["teacher_gold_token_f1_bonus_eligibility_version"],
            BONUS_ELIGIBILITY_VERSION,
        )

    def test_normalized_alias_matching_is_used(self) -> None:
        result = apply_teacher_gold_token_f1_bonus(
            _detail(teacher_answer="The City of Paris!"),
            {"target": ["city of paris"]},
            bonus_weight=0.1,
        )
        self.assertAlmostEqual(result["score"], 0.2)

    def test_nonmatching_teacher_answer_keeps_stable_backoff_reward(self) -> None:
        result = apply_teacher_gold_token_f1_bonus(
            _detail(teacher_answer="Lyon"), {"target": ["Paris"]}, bonus_weight=0.1
        )
        self.assertAlmostEqual(result["base_reward"], 0.1)
        self.assertAlmostEqual(result["teacher_gold_token_f1_bonus"], 0.0)
        self.assertAlmostEqual(result["score"], 0.1)

    def test_matching_answer_can_receive_bonus_when_evidence_status_base_is_zero(self) -> None:
        result = apply_teacher_gold_token_f1_bonus(
            _detail(score=0.0, final_reward=0.0, teacher_f1=0.0),
            {"target": ["Paris"]},
            bonus_weight=0.1,
        )
        self.assertAlmostEqual(result["base_reward"], 0.0)
        self.assertAlmostEqual(result["score"], 0.1)

    def test_ineligible_teacher_or_group_never_receives_bonus(self) -> None:
        for detail in (
            _detail(teacher_parse_status="format_error", teacher_format_error=True),
            _detail(teacher_called=False, teacher_parse_status="not_called"),
            _detail(group_all_em_zero=False),
            _detail(teacher_evidence_status="insufficient_evidence"),
        ):
            with self.subTest(detail=detail):
                result = apply_teacher_gold_token_f1_bonus(
                    detail, {"target": ["Paris"]}, bonus_weight=0.1
                )
                self.assertAlmostEqual(result["teacher_gold_token_f1_bonus"], 0.0)
                self.assertFalse(result["teacher_gold_token_f1_bonus_eligible"])

    def test_actor_without_complete_answer_keeps_base_backoff_only(self) -> None:
        result = apply_teacher_gold_token_f1_bonus(
            _detail(
                actor_answer="",
                actor_answer_parse_status="missing_answer_close",
            ),
            {"target": ["Paris"]},
            bonus_weight=0.1,
        )

        self.assertAlmostEqual(result["base_reward"], 0.1)
        self.assertAlmostEqual(result["score"], 0.1)
        self.assertAlmostEqual(result["teacher_gold_token_f1_bonus"], 0.0)
        self.assertFalse(result["teacher_gold_token_f1_bonus_applied"])
        self.assertFalse(result["teacher_gold_token_f1_bonus_eligible"])

    def test_ambiguous_evidence_remains_eligible_for_bonus(self) -> None:
        result = apply_teacher_gold_token_f1_bonus(
            _detail(teacher_evidence_status="ambiguous_evidence"),
            {"target": ["Paris"]},
            bonus_weight=0.1,
        )

        self.assertAlmostEqual(result["score"], 0.2)
        self.assertTrue(result["teacher_gold_token_f1_bonus_eligible"])

    def test_partial_word_overlap_receives_proportional_bonus(self) -> None:
        result = apply_teacher_gold_token_f1_bonus(
            _detail(teacher_answer="Paris France"),
            {"target": ["Paris"]},
            bonus_weight=0.1,
        )
        self.assertAlmostEqual(result["teacher_gold_token_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(result["teacher_gold_token_f1_bonus"], 1.0 / 15.0)
        self.assertAlmostEqual(result["score"], 1.0 / 6.0)


class GoldMatchBonusBatchEntryTest(unittest.TestCase):
    @patch(
        "agentic_iter_rag.agent_training.spad.rewards."
        "search_policy_teacher_reward_gold_match_bonus.compute_spad_em_teacher_backoff_batch"
    )
    def test_entry_delegates_to_stable_reward_without_mutating_variant_config(self, base) -> None:
        base.return_value = [_detail()]
        reward_cfg = {
            "type": REWARD_VERSION,
            REWARD_VERSION: {"partial_reward": 0.2, "gold_token_f1_bonus": 0.1},
        }

        results = compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch(
            ["source"],
            ["<answer>Lyon</answer>"],
            [{"target": ["Paris"]}],
            [{"uid": "q1"}],
            reward_cfg=reward_cfg,
        )

        delegated_cfg = base.call_args.kwargs["reward_cfg"]
        self.assertEqual(delegated_cfg["type"], "spad_em_teacher_backoff")
        self.assertAlmostEqual(
            delegated_cfg["spad_em_teacher_backoff"]["partial_reward"], 0.2
        )
        self.assertEqual(reward_cfg["type"], REWARD_VERSION)
        self.assertAlmostEqual(results[0]["score"], 0.2)

    def test_entry_rejects_another_reward_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "received reward type"):
            compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch(
                [], [], [], [], reward_cfg={"type": "spad_em_teacher_backoff"}
            )

    def test_negative_bonus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch(
                [],
                [],
                [],
                [],
                reward_cfg={
                    "type": REWARD_VERSION,
                    REWARD_VERSION: {"gold_token_f1_bonus": -0.1},
                },
            )


if __name__ == "__main__":
    unittest.main()
