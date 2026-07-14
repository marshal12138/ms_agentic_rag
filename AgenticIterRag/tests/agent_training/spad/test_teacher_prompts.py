"""Regression tests for SPAD teacher prompt formatting."""

from __future__ import annotations

import unittest

from agentic_iter_rag.agent_training.spad.prompts import (
    DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
    HISTORICAL_TEACHER_STATUS_PROMPT_VERSION,
    TEACHER_ANSWER_PROMPT_VERSION,
    TEACHER_STATUS_SYSTEM_PROMPT,
    build_teacher_messages,
    resolve_teacher_prompt,
)


class TeacherPromptFormattingTest(unittest.TestCase):
    def test_user_prompt_uses_hierarchical_evidence_layout(self) -> None:
        messages = build_teacher_messages(
            question="What is the capital of France?",
            evidence_steps=[
                {
                    "sub_query": "France capital",
                    "docs": [
                        {
                            "title": "Paris",
                            "contents": "Paris is the capital.\nIt is in France.",
                        },
                        {
                            "doc_id": "doc-2-id",
                            "text": "A second passage.",
                        },
                    ],
                }
            ],
            include_status=True,
        )

        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(
            messages[1]["content"],
            "   Original question:\n"
            "      What is the capital of France?\n"
            "\n"
            "   Search evidence:\n"
            "\n"
            "      Round 1:\n"
            "         sub_query: France capital\n"
            "         retrieved contents:\n"
            "            [1] Paris\n"
            "               Paris is the capital.\n"
            "               It is in France.\n"
            "            [2] doc-2-id\n"
            "               A second passage.\n"
            "\n"
            "   Now output the final result directly. Do not analyze the instruction. "
            "Do not repeat rules. Begin with <reason>.",
        )

    def test_user_prompt_keeps_only_top_five_documents(self) -> None:
        messages = build_teacher_messages(
            question="Question",
            evidence_steps=[
                {
                    "sub_query": "query",
                    "docs": [
                        {"title": f"title-{index}", "contents": f"contents-{index}"}
                        for index in range(1, 7)
                    ],
                }
            ],
        )

        content = messages[1]["content"]
        self.assertIn("            [5] title-5\n               contents-5", content)
        self.assertNotIn("title-6", content)
        self.assertNotIn("contents-6", content)

    def test_no_evidence_uses_indented_placeholder(self) -> None:
        messages = build_teacher_messages(question="Question", evidence_steps=[])
        self.assertIn("   Search evidence:\n\n      (no search evidence provided)", messages[1]["content"])

    def test_explicit_status_prompt_version_matches_legacy_status_selector(self) -> None:
        legacy = build_teacher_messages(question="Question", evidence_steps=[], include_status=True)
        versioned = build_teacher_messages(
            question="Question",
            evidence_steps=[],
            prompt_version=DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
        )

        self.assertEqual(versioned, legacy)
        self.assertEqual(versioned[0]["content"], TEACHER_STATUS_SYSTEM_PROMPT)

    def test_historical_v1_keeps_pre_hierarchical_user_layout(self) -> None:
        messages = build_teacher_messages(
            question="Question",
            evidence_steps=[
                {
                    "sub_query": "query",
                    "docs": [{"title": "title", "contents": "contents"}],
                }
            ],
            include_status=True,
            prompt_version=HISTORICAL_TEACHER_STATUS_PROMPT_VERSION,
        )

        self.assertEqual(
            messages[1]["content"],
            "Original question:\n"
            "Question\n"
            "\n"
            "Search evidence:\n"
            "\n"
            "Round 1 sub_query:\n"
            "query\n"
            "[1] title\n"
            "contents\n"
            "\n"
            "Now output the final result directly. Do not analyze the instruction. "
            "Do not repeat rules. Begin with <reason>.",
        )

    def test_unknown_prompt_version_fails_instead_of_falling_back(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown SPAD teacher prompt_version"):
            build_teacher_messages(
                question="Question",
                evidence_steps=[],
                prompt_version="spad_teacher_missing_v99",
            )

    def test_prompt_version_rejects_conflicting_output_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "has include_status=False"):
            build_teacher_messages(
                question="Question",
                evidence_steps=[],
                include_status=True,
                prompt_version=TEACHER_ANSWER_PROMPT_VERSION,
            )

    def test_resolver_returns_canonical_configured_version(self) -> None:
        version, spec = resolve_teacher_prompt(
            f"  {DEFAULT_TEACHER_STATUS_PROMPT_VERSION}  ",
            include_status=True,
        )
        self.assertEqual(version, DEFAULT_TEACHER_STATUS_PROMPT_VERSION)
        self.assertTrue(spec.include_status)


if __name__ == "__main__":
    unittest.main()
