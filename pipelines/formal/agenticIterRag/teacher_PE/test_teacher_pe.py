"""CPU-only regression tests for the teacher prompt-engineering harness."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from build_benchmark import build_cases, validate_cases
from prompt_variants import BASELINE_SYSTEM_PROMPT, PROMPT_VARIANTS, build_messages
from run_ablation import parse_teacher_response, score_predictions


HERE = Path(__file__).resolve().parent


class BenchmarkTest(unittest.TestCase):
    def test_frozen_benchmark_counts_and_split(self) -> None:
        manifest = validate_cases(build_cases())
        self.assertEqual(manifest["case_count"], 237)
        self.assertEqual(manifest["label_counts"], {"S": 104, "I": 105, "A": 28})
        self.assertEqual(manifest["split_counts"], {"dev": 178, "holdout": 59})


class PromptVariantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with (HERE / "benchmark_237.jsonl").open("r", encoding="utf-8") as handle:
            cls.case = json.loads(next(handle))

    def test_baseline_uses_current_system_and_no_gold_field(self) -> None:
        messages = build_messages(self.case, "baseline_current_v2")
        self.assertEqual(messages[0]["content"], BASELINE_SYSTEM_PROMPT)
        self.assertNotIn("Reference gold answer", messages[1]["content"])
        self.assertIn("   Original question:\n      ", messages[1]["content"])
        self.assertIn("   Search evidence:\n\n      Round 1:", messages[1]["content"])

    def test_instruction_only_variants_have_identical_user_input(self) -> None:
        baseline_user = build_messages(self.case, "baseline_current_v2")[1]["content"]
        for name, variant in PROMPT_VARIANTS.items():
            if variant.family == "instruction_only":
                self.assertEqual(build_messages(self.case, name)[1]["content"], baseline_user)

    def test_gold_variants_add_gold_but_keep_evidence_layout(self) -> None:
        messages = build_messages(self.case, "gold_support_check")
        self.assertIn("   Reference gold answer:", messages[1]["content"])
        self.assertIn("   Search evidence:\n\n      Round 1:", messages[1]["content"])


class ParserAndMetricsTest(unittest.TestCase):
    def test_parser_accepts_production_xml(self) -> None:
        parsed = parse_teacher_response(
            "<reason>Evidence [1].</reason><status>supported_answer</status><answer>Paris</answer>"
        )
        self.assertTrue(parsed["parsed"])
        self.assertEqual(parsed["predicted_label"], "S")

    def test_parser_accepts_generation_stopped_after_status(self) -> None:
        parsed = parse_teacher_response(
            "<reason>No complete candidate.</reason><status>insufficient_evidence</status>"
        )
        self.assertTrue(parsed["parsed"])
        self.assertEqual(parsed["parse_status"], "parsed_through_status")
        self.assertEqual(parsed["predicted_label"], "I")

    def test_parser_tolerates_backend_stripping_status_stop_string(self) -> None:
        parsed = parse_teacher_response(
            "<reason>Two candidates.</reason><status>ambiguous_evidence"
        )
        self.assertTrue(parsed["parsed"])
        self.assertEqual(parsed["parse_status"], "parsed_status_stop_without_close")
        self.assertEqual(parsed["predicted_label"], "A")

    def test_i_metrics_and_sa_tolerance(self) -> None:
        predictions = [
            {"manual_label": "I", "predicted_label": "I", "parsed": True},
            {"manual_label": "S", "predicted_label": "A", "parsed": True},
            {"manual_label": "A", "predicted_label": "I", "parsed": True},
        ]
        metrics = score_predictions(predictions)
        self.assertEqual(metrics["tolerated_sa_confusion"], 1)
        self.assertEqual(metrics["involved_i_errors"], 1)
        self.assertEqual(metrics["i_binary"]["precision"], 0.5)
        self.assertEqual(metrics["i_binary"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
