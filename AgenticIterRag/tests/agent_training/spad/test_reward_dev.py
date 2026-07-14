"""Contract tests for speculative SPAD teacher prefetch."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_dev import (
    PREFETCH_DETAIL_KEY,
    PREFETCH_ONLY_KEY,
    _clear_prefetch_cache_for_tests,
    compute_spad_em_teacher_backoff_dev,
)


class SpadDevRewardTest(unittest.TestCase):
    def setUp(self) -> None:
        _clear_prefetch_cache_for_tests()

    @staticmethod
    def kwargs() -> dict:
        return {
            "reward_cfg": {
                "type": "spad_em_teacher_backoff_dev",
                "spad_em_teacher_backoff_dev": {"partial_reward": 0.1},
            },
            "teacher_request": {
                "endpoint": "http://teacher/v1/chat/completions",
                "model": "teacher",
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 512,
            },
            "teacher_prompt_version": "spad_teacher_evidence_status_answer_v2",
            "visible_top_m": 5,
            "n_samples_per_prompt": 8,
        }

    @staticmethod
    def extra(uid: str) -> dict:
        return {
            "uid": uid,
            "question": "What is the capital of France?",
            "tool_call_details": [
                {
                    "sub_query": "France capital",
                    "top_5_documents": [{"text": "Paris is the capital of France."}],
                }
            ],
        }

    def test_prefetch_coalesces_identical_requests_and_group_uses_results(self) -> None:
        def teacher(**kwargs):
            del kwargs
            return "Paris", "supported_answer", False, 0.01, "raw", "parsed", "false"

        extras = [self.extra("all-zero") for _ in range(8)]
        with patch(
            "agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward._call_teacher",
            side_effect=teacher,
        ) as call_teacher:
            for index in range(8):
                prefetch = compute_spad_em_teacher_backoff_dev(
                    data_source="unit-test",
                    solution_str="<answer>wrong</answer>",
                    ground_truth={"target": ["Paris"]},
                    extra_info=extras[index],
                    **self.kwargs(),
                )
                self.assertTrue(prefetch[PREFETCH_ONLY_KEY])
                extras[index][PREFETCH_DETAIL_KEY] = prefetch[PREFETCH_DETAIL_KEY]

            results = compute_spad_em_teacher_backoff_dev(
                data_sources=["unit-test"] * 8,
                solution_strs=["<answer>wrong</answer>"] * 8,
                ground_truths=[{"target": ["Paris"]}] * 8,
                extra_infos=extras,
                **self.kwargs(),
            )

        self.assertEqual(call_teacher.call_count, 1)
        self.assertEqual([item["score"] for item in results], [0.1] * 8)
        self.assertTrue(all(item["teacher_called"] for item in results))
        self.assertTrue(all(item["reward_type"] == "spad_em_teacher_backoff_dev" for item in results))

    def test_positive_em_group_discards_prefetched_teacher_results(self) -> None:
        extras = [self.extra("has-em") for _ in range(8)]
        prefetched = {
            "teacher_called": True,
            "teacher_status_reward": 1.0,
            "teacher_evidence_status": "supported_answer",
        }
        for extra in extras:
            extra[PREFETCH_DETAIL_KEY] = prefetched
        results = compute_spad_em_teacher_backoff_dev(
            data_sources=["unit-test"] * 8,
            solution_strs=["<answer>Paris</answer>", *(["<answer>wrong</answer>"] * 7)],
            ground_truths=[{"target": ["Paris"]}] * 8,
            extra_infos=extras,
            **self.kwargs(),
        )
        self.assertEqual([item["score"] for item in results], [1.0, *([0.0] * 7)])
        self.assertTrue(all(not item["teacher_called"] for item in results))
        self.assertTrue(all(item["teacher_skip_reason"] == "group_has_positive_em" for item in results))


if __name__ == "__main__":
    unittest.main()
