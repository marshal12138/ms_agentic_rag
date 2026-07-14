"""Focused tests for SPAD-RAG Stage 1 reward shaping."""

from __future__ import annotations

import unittest

from agentic_iter_rag.agent_training.spad.reward import compute_search_policy_reward
from agentic_iter_rag.agent_training.spad.refresh_rollout import _strip_teacher_status_block
from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import (
    _extract_teacher_result,
    compute_spad_search_policy_reward_details,
)


ACTOR_STOP = "<reason>Evidence is enough to answer.</reason>\n<answer>"
TOOL_TURN = '<reason>Search once.</reason>\n<tool_call>{"name":"search","arguments":{"query":"France capital"}}</tool_call>'

REWARD_CFG = {
    "teacher_f1_weight": 1.0,
    "search_cost": 0.02,
    "free_search_count": 1,
    "duplicate_query_penalty": -0.1,
    "missing_reason_penalty": -0.02,
    "invalid_format_penalty": -0.5,
    "no_finish_penalty": -0.5,
    "max_search_turns": 3,
    "bad_stop": {
        "enabled": True,
        "penalty": -0.20,
        "max_budget_failed_penalty": -0.15,
        "teacher_format_error_penalty": -0.1,
    },
}

SEARCH_R1_REWARD_CFG = {
    "type": "search_r1_original",
    "search_r1_original": {
        "score": 1.0,
        "format_score": 0.0,
    },
}


def reward_for(
    *,
    search_count: int,
    teacher_answer: str = "Paris",
    teacher_evidence_status: str = "supported_answer",
    teacher_format_error: bool = False,
    reward_cfg: dict | None = None,
) -> dict:
    return compute_search_policy_reward(
        actor_output=ACTOR_STOP,
        teacher_answer=teacher_answer,
        teacher_evidence_status=teacher_evidence_status,
        teacher_format_error=teacher_format_error,
        gold_answers=["Paris"],
        search_count=search_count,
        duplicate_query_count=0,
        reward_cfg=reward_cfg or REWARD_CFG,
        legal_stop=True,
        stop_at_answer_opening=True,
    )


class SearchPolicyRewardTest(unittest.TestCase):
    def test_first_search_is_free(self) -> None:
        result = reward_for(search_count=1)
        self.assertAlmostEqual(result["final_reward"], 1.0)
        self.assertAlmostEqual(result["effective_search_cost"], 0.0)

    def test_second_search_pays_one_cost(self) -> None:
        result = reward_for(search_count=2)
        self.assertAlmostEqual(result["final_reward"], 0.98)
        self.assertAlmostEqual(result["effective_search_cost"], 0.02)

    def test_third_search_pays_two_costs(self) -> None:
        result = reward_for(search_count=3)
        self.assertAlmostEqual(result["final_reward"], 0.96)
        self.assertAlmostEqual(result["effective_search_cost"], 0.04)

    def test_free_search_count_zero_charges_first_search(self) -> None:
        reward_cfg = {**REWARD_CFG, "free_search_count": 0}
        result = reward_for(search_count=1, reward_cfg=reward_cfg)
        self.assertAlmostEqual(result["final_reward"], 0.98)
        self.assertAlmostEqual(result["effective_search_cost"], 0.02)

    def test_free_search_count_two_charges_from_third_search(self) -> None:
        reward_cfg = {**REWARD_CFG, "free_search_count": 2}
        result = reward_for(search_count=3, reward_cfg=reward_cfg)
        self.assertAlmostEqual(result["final_reward"], 0.98)
        self.assertAlmostEqual(result["effective_search_cost"], 0.02)

    def test_insufficient_evidence_before_budget_is_bad_stop(self) -> None:
        result = reward_for(
            search_count=2,
            teacher_answer="证据不足无法作答",
            teacher_evidence_status="insufficient_evidence",
        )
        self.assertAlmostEqual(result["final_reward"], -0.20)
        self.assertTrue(result["bad_stop_applied"])
        self.assertEqual(result["bad_stop_reason"], "early_stop_insufficient_evidence")

    def test_default_bad_stop_penalty_is_minus_point_two(self) -> None:
        reward_cfg = {key: value for key, value in REWARD_CFG.items() if key != "bad_stop"}
        result = reward_for(
            search_count=2,
            teacher_answer="证据不足无法作答",
            teacher_evidence_status="insufficient_evidence",
            reward_cfg=reward_cfg,
        )
        self.assertAlmostEqual(result["final_reward"], -0.20)
        self.assertTrue(result["bad_stop_applied"])

    def test_insufficient_evidence_at_budget_uses_budget_penalty(self) -> None:
        result = reward_for(
            search_count=3,
            teacher_answer="证据不足无法作答",
            teacher_evidence_status="insufficient_evidence",
        )
        self.assertAlmostEqual(result["final_reward"], -0.15)
        self.assertFalse(result["bad_stop_applied"])
        self.assertEqual(result["bad_stop_reason"], "max_budget_insufficient_evidence")

    def test_teacher_format_error_uses_format_penalty(self) -> None:
        result = reward_for(search_count=2, teacher_answer="", teacher_format_error=True)
        self.assertAlmostEqual(result["final_reward"], -0.1)
        self.assertTrue(result["teacher_format_error"])
        self.assertFalse(result["bad_stop_applied"])
        self.assertEqual(result["teacher_f1"], 0.0)

    def test_invalid_format_records_paid_search_count_without_extra_cost(self) -> None:
        result = compute_search_policy_reward(
            actor_output=f"{TOOL_TURN}\nuser\n<tool_response>doc</tool_response>\nassistant\n<think>broken\n<answer>",
            teacher_answer="",
            gold_answers=["Paris"],
            search_count=2,
            duplicate_query_count=0,
            reward_cfg=REWARD_CFG,
            legal_stop=True,
            stop_at_answer_opening=True,
        )
        self.assertAlmostEqual(result["final_reward"], -0.5)
        self.assertFalse(result["teacher_called"])
        self.assertEqual(result["paid_search_count"], 1)
        self.assertEqual(result["second_plus_search_count"], 1)
        self.assertAlmostEqual(result["effective_search_cost"], 0.0)

    def test_no_finish_records_paid_search_count_without_extra_cost(self) -> None:
        result = compute_search_policy_reward(
            actor_output=f"{TOOL_TURN}\nuser\n<tool_response>doc</tool_response>\nassistant\n<reason>Need more.</reason>",
            teacher_answer="",
            gold_answers=["Paris"],
            search_count=2,
            duplicate_query_count=0,
            reward_cfg=REWARD_CFG,
            legal_stop=True,
            stop_at_answer_opening=True,
        )
        self.assertAlmostEqual(result["final_reward"], -0.5)
        self.assertFalse(result["teacher_called"])
        self.assertEqual(result["teacher_skip_reason"], "no_finish")
        self.assertEqual(result["paid_search_count"], 1)
        self.assertEqual(result["second_plus_search_count"], 1)
        self.assertAlmostEqual(result["effective_search_cost"], 0.0)


class SearchR1OriginalRewardTest(unittest.TestCase):
    def reward_for(
        self,
        solution_str: str,
        *,
        gold_answers: list[str] | None = None,
        tool_call_details: list[dict] | None = None,
    ) -> dict:
        return compute_spad_search_policy_reward_details(
            data_source="unit-test",
            solution_str=solution_str,
            ground_truth={"target": gold_answers or ["Paris"]},
            extra_info={
                "question": "What is the capital of France?",
                "tool_call_details": tool_call_details or [],
            },
            reward_cfg=SEARCH_R1_REWARD_CFG,
        )

    def test_exact_match_scores_one_without_search_evidence(self) -> None:
        result = self.reward_for("<answer>Paris</answer>")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["reward_type"], "search_r1_original")
        self.assertFalse(result["teacher_called"])
        self.assertEqual(result["teacher_skip_reason"], "search_r1_original_no_teacher")
        self.assertEqual(result["search_count"], 0)
        self.assertEqual(result["search_r1_answer_em"], 1.0)
        self.assertEqual(result["search_r1_extracted_answer"], "Paris")

    def test_mismatch_scores_zero(self) -> None:
        result = self.reward_for("<answer>Lyon</answer>")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["search_r1_answer_em"], 0.0)
        self.assertFalse(result["teacher_called"])

    def test_missing_answer_close_scores_zero(self) -> None:
        result = self.reward_for("<answer>Paris")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["format_status"], "invalid")
        self.assertEqual(result["stop_status"], "missing_answer_close")
        self.assertEqual(result["search_r1_answer_em"], 0.0)

    def test_multiple_answers_use_last_answer(self) -> None:
        result = self.reward_for("<answer>Lyon</answer>\n<answer>Paris</answer>")
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["search_r1_extracted_answer"], "Paris")

    def test_correct_answer_with_search_evidence_scores_one(self) -> None:
        result = self.reward_for(
            "<answer>Paris</answer>",
            tool_call_details=[{"sub_query": "France capital", "top_5_documents": [{"text": "Paris is the capital."}]}],
        )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["search_count"], 1)


class TeacherResultParserTest(unittest.TestCase):
    def test_parse_supported_answer(self) -> None:
        answer, status, parse_status, format_error = _extract_teacher_result(
            "<reason>Doc 1 supports it.</reason>\n"
            "<status>supported_answer</status>\n"
            "<answer>Paris</answer>"
        )
        self.assertEqual(answer, "Paris")
        self.assertEqual(status, "supported_answer")
        self.assertEqual(parse_status, "parsed")
        self.assertFalse(format_error)

    def test_missing_status_is_format_error(self) -> None:
        answer, status, parse_status, format_error = _extract_teacher_result(
            "<reason>Doc 1 supports it.</reason>\n<answer>Paris</answer>"
        )
        self.assertEqual(answer, "")
        self.assertEqual(status, "")
        self.assertEqual(parse_status, "missing_status_tag")
        self.assertTrue(format_error)

    def test_invalid_status_is_format_error(self) -> None:
        answer, status, parse_status, format_error = _extract_teacher_result(
            "<reason>Doc 1 supports it.</reason>\n"
            "<status>supported</status>\n"
            "<answer>Paris</answer>"
        )
        self.assertEqual(answer, "Paris")
        self.assertEqual(status, "supported")
        self.assertEqual(parse_status, "invalid_status")
        self.assertTrue(format_error)


class Stage2TeacherOutputTest(unittest.TestCase):
    def test_strip_status_from_chosen_answer(self) -> None:
        chosen = _strip_teacher_status_block(
            "<reason>Doc 1 supports it.</reason>\n"
            "<status>supported_answer</status>\n"
            "<answer>Paris</answer>"
        )
        self.assertEqual(chosen, "<reason>Doc 1 supports it.</reason>\n<answer>Paris</answer>")
        self.assertNotIn("<status>", chosen)


if __name__ == "__main__":
    unittest.main()
