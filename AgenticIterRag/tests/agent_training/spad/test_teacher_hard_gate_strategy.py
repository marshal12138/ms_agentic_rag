"""Tests for the production Hard-Gate v2 Teacher strategy."""

from __future__ import annotations

import unittest
from unittest import mock

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2 import (
    compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2_batch,
)
from agentic_iter_rag.agent_training.spad.teacher_strategies import (
    GOLD_SUPPORT_EVIDENCE_ONLY_SYSTEM_PROMPT,
    HARD_GATE_R5_LITERAL_CANONICAL_V2,
    build_gold_support_evidence_only_messages,
    select_hard_gate_output,
)


def _teacher_detail(status: str, answer: str, *, parsed: bool = True) -> dict:
    return {
        "teacher_answer": answer,
        "teacher_evidence_status": status,
        "teacher_parse_status": "parsed" if parsed else "missing_status_tag",
        "teacher_format_error": not parsed,
    }


class TeacherHardGateStrategyTest(unittest.TestCase):
    def test_r5_prompt_hides_subquery_and_contains_gold_and_evidence(self) -> None:
        messages = build_gold_support_evidence_only_messages(
            question="What is the capital of France?",
            gold_answers=["Paris"],
            evidence_steps=[
                {
                    "sub_query": "DISTRACTING QUERY",
                    "docs": [{"title": "France", "contents": "Paris is the capital."}],
                }
            ],
        )

        self.assertEqual(messages[0]["content"], GOLD_SUPPORT_EVIDENCE_ONLY_SYSTEM_PROMPT)
        self.assertIn('["Paris"]', messages[1]["content"])
        self.assertIn("Paris is the capital.", messages[1]["content"])
        self.assertNotIn("DISTRACTING QUERY", messages[1]["content"])

    def test_stage_a_i_is_binding(self) -> None:
        stage_a = _teacher_detail("insufficient_evidence", "证据不足无法作答")
        stage_b = _teacher_detail("supported_answer", "Paris")

        selected = select_hard_gate_output(
            stage_a=stage_a,
            stage_b=stage_b,
            gold_answers=["Paris"],
            evidence_steps=[],
        )

        self.assertFalse(selected["stage_b_used"])
        self.assertEqual(
            selected["selected"]["teacher_evidence_status"], "insufficient_evidence"
        )

    def test_stage_b_failure_falls_back_to_stage_a(self) -> None:
        selected = select_hard_gate_output(
            stage_a=_teacher_detail("supported_answer", "Paris"),
            stage_b=_teacher_detail("", "", parsed=False),
            gold_answers=["Paris"],
            evidence_steps=[],
        )

        self.assertFalse(selected["stage_b_used"])
        self.assertEqual(selected["selected"]["teacher_answer"], "Paris")
        self.assertEqual(selected["selection_reason"], "stage_b_invalid_fallback")

    def test_two_supported_answers_choose_higher_gold_f1(self) -> None:
        selected = select_hard_gate_output(
            stage_a=_teacher_detail("supported_answer", "Paris, France"),
            stage_b=_teacher_detail("supported_answer", "Paris"),
            gold_answers=["Paris"],
            evidence_steps=[],
        )

        self.assertTrue(selected["stage_b_used"])
        self.assertEqual(selected["selected"]["teacher_answer"], "Paris")

    def test_literal_guard_canonicalizes_only_from_evidence(self) -> None:
        without_literal = select_hard_gate_output(
            stage_a=_teacher_detail("supported_answer", "City of Light"),
            stage_b=None,
            gold_answers=["Paris"],
            evidence_steps=[{"docs": [{"contents": "The capital is the City of Light."}]}],
        )
        with_literal = select_hard_gate_output(
            stage_a=_teacher_detail("supported_answer", "City of Light"),
            stage_b=None,
            gold_answers=["Paris"],
            evidence_steps=[{"docs": [{"contents": "Paris is called the City of Light."}]}],
        )

        self.assertEqual(without_literal["canonical_gold"], "")
        self.assertEqual(without_literal["selected"]["teacher_answer"], "City of Light")
        self.assertEqual(with_literal["canonical_gold"], "Paris")
        self.assertEqual(with_literal["selected"]["teacher_answer"], "Paris")

    def test_v3_batch_preserves_bonus_and_postnorm_scale(self) -> None:
        def fake_call_teacher(**kwargs):
            if kwargs["prompt_version"] == "gold_support_evidence_only_v3":
                return "Paris", "supported_answer", False, 2.0, "stage-b", "parsed", "false"
            return "City of Light", "supported_answer", False, 1.0, "stage-a", "parsed", "false"

        ground_truths = [{"target": ["Paris"]} for _ in range(8)]
        extra_infos = [
            {
                "uid": "question-1",
                "question": "What is the capital of France?",
                "tool_call_details": [
                    {
                        "sub_query": "France capital",
                        "top_5_documents": [
                            {"title": "France", "contents": "Paris is the capital of France."}
                        ],
                    }
                ],
            }
            for _ in range(8)
        ]
        with mock.patch(
            "agentic_iter_rag.agent_training.spad.rewards."
            "search_policy_teacher_reward._call_teacher",
            side_effect=fake_call_teacher,
        ), mock.patch(
            "agentic_iter_rag.agent_training.spad.rewards."
            "search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2._call_teacher",
            side_effect=fake_call_teacher,
        ):
            results = compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2_batch(
                ["co_search"] * 8,
                ["<reason>done</reason><answer>Lyon</answer>"] * 8,
                ground_truths,
                extra_infos,
                reward_cfg={
                    "type": "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2",
                    "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2": {
                        "partial_reward": 0.1,
                        "gold_token_f1_bonus": 0.1,
                        "teacher_group_postnorm_scale": 0.1,
                    },
                },
                teacher_request={"model": "teacher", "endpoint": "http://teacher"},
                teacher_prompt_version="spad_teacher_evidence_status_answer_v2",
                teacher_strategy_id=HARD_GATE_R5_LITERAL_CANONICAL_V2,
                n_samples_per_prompt=8,
            )

        self.assertEqual(len(results), 8)
        for result in results:
            self.assertEqual(result["teacher_strategy_id"], HARD_GATE_R5_LITERAL_CANONICAL_V2)
            self.assertEqual(result["teacher_total_call_count"], 2)
            self.assertTrue(result["teacher_stage_b_called"])
            self.assertTrue(result["teacher_stage_b_used"])
            self.assertTrue(result["teacher_i_boundary_preserved"])
            self.assertEqual(result["teacher_answer"], "Paris")
            self.assertAlmostEqual(result["score"], 0.2)
            self.assertAlmostEqual(result["teacher_gold_token_f1_bonus"], 0.1)
            self.assertAlmostEqual(result["advantage_postnorm_scale"], 0.1)

    def test_v3_batch_does_not_call_stage_b_after_stage_a_i(self) -> None:
        def fake_call_teacher(**kwargs):
            self.assertNotEqual(kwargs["prompt_version"], "gold_support_evidence_only_v3")
            return (
                "证据不足无法作答",
                "insufficient_evidence",
                False,
                1.0,
                "stage-a",
                "parsed",
                "false",
            )

        with mock.patch(
            "agentic_iter_rag.agent_training.spad.rewards."
            "search_policy_teacher_reward._call_teacher",
            side_effect=fake_call_teacher,
        ) as mocked_stage_a, mock.patch(
            "agentic_iter_rag.agent_training.spad.rewards."
            "search_policy_teacher_reward_gold_match_bonus_v3_hard_gate_v2._call_teacher",
            side_effect=fake_call_teacher,
        ) as mocked_stage_b:
            results = compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2_batch(
                ["co_search"] * 8,
                ["<reason>done</reason><answer>Lyon</answer>"] * 8,
                [{"target": ["Paris"]} for _ in range(8)],
                [
                    {
                        "uid": "question-1",
                        "question": "What is the capital of France?",
                        "tool_call_details": [
                            {
                                "sub_query": "France capital",
                                "top_5_documents": [{"contents": "No capital is named."}],
                            }
                        ],
                    }
                    for _ in range(8)
                ],
                reward_cfg={
                    "type": "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2",
                    "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2": {},
                },
                teacher_request={"model": "teacher", "endpoint": "http://teacher"},
                teacher_prompt_version="spad_teacher_evidence_status_answer_v2",
                teacher_strategy_id=HARD_GATE_R5_LITERAL_CANONICAL_V2,
                n_samples_per_prompt=8,
            )

        self.assertEqual(mocked_stage_a.call_count, 8)
        self.assertEqual(mocked_stage_b.call_count, 0)
        self.assertTrue(all(not row["teacher_stage_b_called"] for row in results))
        self.assertTrue(all(row["teacher_total_call_count"] == 1 for row in results))
        self.assertTrue(all(row["teacher_i_boundary_preserved"] for row in results))


if __name__ == "__main__":
    unittest.main()
