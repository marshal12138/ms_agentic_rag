"""Structured AND-of-OR answer contract tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import (
    REWARD_VERSION,
    compute_spad_em_teacher_backoff_batch,
    compute_spad_search_policy_reward_batch,
    compute_spad_search_policy_reward_details,
)
from agentic_iter_rag.metrics.answer_metrics import answer_group_metrics
from agentic_iter_rag.agent_training.spad.reward import compute_gold_answer_f1_reward_details


class AnswerGroupMetricsTest(unittest.TestCase):
    def test_alias_group_preserves_legacy_or_semantics(self) -> None:
        result = answer_group_metrics(
            "Muhammad",
            ["the Islamic prophet Muhammad", "Muhammad"],
            [["the Islamic prophet Muhammad", "Muhammad"]],
        )
        self.assertEqual(result.legacy_em, 1.0)
        self.assertEqual(result.structured_em, 1.0)
        self.assertEqual(result.answer_group_f1, 1.0)

    def test_required_set_rejects_one_member(self) -> None:
        groups = [["legislative"], ["executive"], ["judicial"]]
        result = answer_group_metrics("legislative", [item[0] for item in groups], groups)
        self.assertEqual(result.legacy_em, 1.0)
        self.assertEqual(result.structured_em, 0.0)
        self.assertAlmostEqual(result.answer_group_recall, 1 / 3)

    def test_required_set_accepts_complete_answer_in_any_order(self) -> None:
        groups = [["legislative"], ["executive"], ["judicial"]]
        first = answer_group_metrics("legislative, executive, judicial", [item[0] for item in groups], groups)
        reordered = answer_group_metrics("judicial; legislative; executive", [item[0] for item in groups], groups)
        self.assertEqual(first.structured_em, 1.0)
        self.assertEqual(reordered.structured_em, 1.0)
        self.assertEqual(first.answer_group_f1, 1.0)
        self.assertEqual(reordered.answer_group_f1, 1.0)

    def test_multi_slot_requires_person_and_studied_subject(self) -> None:
        groups = [["Gregor Mendel", "Mendel"], ["pea plants", "the common edible pea"]]
        partial = answer_group_metrics("Mendel", [alias for group in groups for alias in group], groups)
        complete = answer_group_metrics("Mendel studied pea plants", [alias for group in groups for alias in group], groups)
        self.assertEqual(partial.structured_em, 0.0)
        self.assertEqual(complete.structured_em, 1.0)

    def test_contaminated_intermediate_answer_is_not_a_valid_alias(self) -> None:
        result = answer_group_metrics(
            "University of Zurich",
            ["nearly 25,000", "University of Zurich"],
            [["nearly 25,000"]],
        )
        self.assertEqual(result.legacy_em, 1.0)
        self.assertEqual(result.structured_em, 0.0)

    def test_ambiguous_row_can_be_excluded_from_structured_metric(self) -> None:
        result = answer_group_metrics(
            "November 2016",
            ["November 2016", "July 2017"],
            [["November 2016", "July 2017"]],
            structured_eligible=False,
        )
        self.assertEqual(result.legacy_em, 1.0)
        self.assertFalse(result.structured_eligible)
        self.assertEqual(result.structured_em, 0.0)


class StructuredSearchR1RewardTest(unittest.TestCase):
    def reward(self, answer: str, groups: list[list[str]], *, eligible: bool = True) -> dict:
        return compute_spad_search_policy_reward_details(
            data_source="unit-test",
            solution_str=f"<answer>{answer}</answer>",
            ground_truth={
                "target": [alias for group in groups for alias in group],
                "required_answer_groups": groups,
                "answer_semantics": "required_set",
                "structured_reward_eligible": eligible,
            },
            extra_info={"question": "test", "tool_call_details": []},
            reward_cfg={
                "type": "search_r1_structured",
                "search_r1_structured": {"score": 1.0, "format_score": 0.0},
            },
        )

    def test_structured_reward_uses_all_required_groups(self) -> None:
        groups = [["Ghana"], ["Algeria"], ["Mali"]]
        partial = self.reward("Ghana", groups)
        complete = self.reward("Mali, Ghana, Algeria", groups)
        self.assertEqual(partial["legacy_em"], 1.0)
        self.assertEqual(partial["score"], 0.0)
        self.assertEqual(complete["score"], 1.0)
        self.assertEqual(complete["reward_type"], "search_r1_structured")
        self.assertFalse(complete["teacher_called"])

    def test_ineligible_row_produces_no_positive_reward(self) -> None:
        result = self.reward("November 2016", [["November 2016", "July 2017"]], eligible=False)
        self.assertEqual(result["score"], 0.0)
        self.assertFalse(result["structured_reward_eligible"])


class SpadGroupRewardTest(unittest.TestCase):
    def test_named_entry_rejects_a_different_reward_type(self) -> None:
        with self.assertRaisesRegex(ValueError, REWARD_VERSION):
            compute_spad_em_teacher_backoff_batch(
                [],
                [],
                [],
                [],
                reward_cfg={"type": "search_r1_structured"},
            )

    def test_teacher_is_only_called_for_all_zero_groups(self) -> None:
        data_sources = ["unit-test"] * 16
        solutions = ["<answer>Paris</answer>", *(["<answer>wrong</answer>"] * 15)]
        ground_truths = [{"target": ["Paris"]} for _ in range(16)]
        extra_infos = []
        for index in range(16):
            extra_infos.append(
                {
                    "uid": "has-em" if index < 8 else "all-zero",
                    "question": f"q{index}",
                    "tool_call_details": [
                        {"sub_query": "France capital", "top_5_documents": [{"text": "Paris"}]}
                    ],
                }
            )

        def teacher(**kwargs):
            status = "supported_answer" if kwargs["question"] == "q8" else "insufficient_evidence"
            return "Paris", status, False, 0.01, "raw", "parsed", "false"

        with patch(
            "agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward._call_teacher",
            side_effect=teacher,
        ) as call_teacher:
            results = compute_spad_em_teacher_backoff_batch(
                data_sources,
                solutions,
                ground_truths,
                extra_infos,
                reward_cfg={
                    "type": "spad_em_teacher_backoff",
                    "spad_em_teacher_backoff": {"partial_reward": 0.1},
                },
                n_samples_per_prompt=8,
            )
        self.assertEqual(call_teacher.call_count, 8)
        self.assertEqual([item["score"] for item in results[:8]], [1.0, *([0.0] * 7)])
        self.assertEqual(results[8]["score"], 0.1)
        self.assertEqual([item["score"] for item in results[9:]], [0.0] * 7)
        self.assertTrue(all(not item["teacher_called"] for item in results[:8]))
        self.assertTrue(all(item["teacher_called"] for item in results[8:]))
        self.assertTrue(all(item["group_all_em_zero"] for item in results[8:]))
        self.assertTrue(all(len(item["teacher_messages"]) == 2 for item in results))
        self.assertTrue(all(len(item["teacher_request_hash"]) == 64 for item in results))
        self.assertTrue(
            all(item["teacher_skip_reason"] == "group_has_positive_em" for item in results[:8])
        )

    def test_group_size_mismatch_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 8 rollouts"):
            compute_spad_search_policy_reward_batch(
                ["unit-test"],
                ["<answer>Paris</answer>"],
                [{"target": ["Paris"]}],
                [{"uid": "short-group", "question": "q", "tool_call_details": []}],
                reward_cfg={"type": "spad_em_teacher_backoff"},
                n_samples_per_prompt=8,
            )


class Stage3GoldAnswerF1Test(unittest.TestCase):
    def test_complete_answer_receives_token_f1(self) -> None:
        result = compute_gold_answer_f1_reward_details(
            "unit-test",
            "<reason>Known.</reason><answer>Paris France</answer>",
            {"target": ["Paris"]},
            {},
        )
        self.assertAlmostEqual(result["score"], 2 / 3)

    def test_missing_close_tag_receives_zero(self) -> None:
        result = compute_gold_answer_f1_reward_details(
            "unit-test", "<reason>Known.</reason><answer>Paris", {"target": ["Paris"]}, {}
        )
        self.assertEqual(result["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
