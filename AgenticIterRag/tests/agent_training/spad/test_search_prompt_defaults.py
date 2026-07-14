"""Regression tests for aligned SPAD/Search-R1 search prompts."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REQUIRED_GUIDANCE = (
    "On the first assistant turn for every question, you MUST call search.",
    "If the evidence is sufficient, answer immediately.",
    "identify the missing fact or bridge entity",
    "Never repeat or paraphrase a previous query",
    "newly discovered entities or relations",
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SearchPromptDefaultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[4]
        cls.data_prompt = _load_module(
            "prepare_cosearch_data_for_test",
            repo_root / "scripts" / "cosearch_local" / "prepare_cosearch_data.py",
        ).SEARCH_R1_PROMPT
        runtime_prompts = _load_module(
            "runtime_search_prompts_for_test",
            repo_root / "AgenticIterRag" / "verl" / "verl" / "tools" / "utils" / "prompts.py",
        )
        cls.runtime_prompt = runtime_prompts.SEARCH_R1_PROMPT
        cls.runtime_cot_prompt = runtime_prompts.SEARCH_R1_CoT_PROMPT

    def test_training_eval_and_runtime_prompts_share_search_guidance(self) -> None:
        for prompt in (self.data_prompt, self.runtime_prompt, self.runtime_cot_prompt):
            for guidance in REQUIRED_GUIDANCE:
                self.assertIn(guidance, prompt)

    def test_materialized_data_prompt_keeps_question_placeholder(self) -> None:
        rendered = self.data_prompt.format(question="What is the capital of France?")
        self.assertTrue(rendered.endswith("Question: What is the capital of France?\n"))


if __name__ == "__main__":
    unittest.main()
