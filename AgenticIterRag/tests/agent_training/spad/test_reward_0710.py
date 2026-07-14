"""Frozen 0710 teacher-F1 reward contract tests."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_0710 import (
    REWARD_VERSION,
    compute_spad_teacher_f1_0710_details,
)


REWARD_CFG = {
    "teacher_f1_weight": 1.0,
    "search_cost": 0.02,
    "free_search_count": 1,
    "max_search_turns": 5,
    "invalid_format_penalty": -0.5,
    "missing_reason_penalty": -0.02,
    "duplicate_query_penalty": -0.1,
    "no_finish_penalty": -0.5,
    "bad_stop": {
        "enabled": True,
        "insufficient_answer": "证据不足无法作答",
        "penalty": -0.35,
        "max_budget_failed_penalty": -0.15,
        "teacher_format_error_penalty": -0.1,
    },
}


def score_with_teacher(teacher_result):
    with patch(
        "agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_0710._call_teacher_0710",
        return_value=teacher_result,
    ):
        return compute_spad_teacher_f1_0710_details(
            data_source="unit-test",
            solution_str="<reason>search complete</reason><answer>",
            ground_truth={"target": ["Paris"]},
            extra_info={
                "question": "Capital of France?",
                "tool_call_details": [
                    {"sub_query": "France capital", "top_5_documents": [{"text": "Paris"}]}
                ],
            },
            reward_cfg=REWARD_CFG,
            teacher_request={},
        )


class Reward0710Test(unittest.TestCase):
    def test_supported_answer_uses_teacher_f1(self) -> None:
        result = score_with_teacher(("Paris", 1.0, "<reason>x</reason><answer>Paris</answer>", "parsed", "false"))
        self.assertEqual(result["reward_type"], REWARD_VERSION)
        self.assertEqual(result["teacher_f1"], 1.0)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["effective_search_cost"], 0.0)

    def test_insufficient_answer_uses_0710_bad_stop_penalty(self) -> None:
        result = score_with_teacher(
            ("证据不足无法作答", 1.0, "<reason>x</reason><answer>证据不足无法作答</answer>", "parsed", "false")
        )
        self.assertEqual(result["score"], -0.35)
        self.assertTrue(result["bad_stop_applied"])
        self.assertEqual(result["bad_stop_reason"], "early_stop_insufficient_evidence")

    def test_teacher_xml_error_uses_0710_format_penalty(self) -> None:
        result = score_with_teacher(("", 1.0, "not xml", "missing_reason_tag", "false"))
        self.assertEqual(result["score"], -0.1)
        self.assertTrue(result["teacher_format_error"])


if __name__ == "__main__":
    unittest.main()
