"""CPU-only regression tests for the teacher prompt-engineering harness."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from build_benchmark import build_cases, validate_cases
from composite_prompt_variants import COMPOSITE_PROMPT_VARIANTS, build_composite_stage_b_messages
from prompt_variants import BASELINE_SYSTEM_PROMPT, PROMPT_VARIANTS, build_messages
from run_ablation import answer_exact_match, answer_token_f1, parse_teacher_response, score_predictions
from run_composite_ablation import find_evidence_literal_gold, select_composite_output


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

    def test_gold_question_tail_cross_uses_successful_layout(self) -> None:
        messages = build_messages(self.case, "gold_support_question_tail_v3")
        self.assertIn("   Reference gold answer:", messages[1]["content"])
        self.assertNotIn("sub_query:", messages[1]["content"])
        self.assertIn("   Original question to judge:\n      ", messages[1]["content"])

    def test_gold_layout_factorial_variants(self) -> None:
        evidence_only = build_messages(self.case, "gold_support_evidence_only_v3")[1]["content"]
        subquery_tail = build_messages(
            self.case, "gold_support_subquery_question_tail_v3"
        )[1]["content"]
        self.assertIn("   Reference gold answer:", evidence_only)
        self.assertNotIn("sub_query:", evidence_only)
        self.assertNotIn("Original question to judge:", evidence_only)
        self.assertIn("   Reference gold answer:", subquery_tail)
        self.assertIn("sub_query:", subquery_tail)
        self.assertIn("Original question to judge:", subquery_tail)

    def test_gold_decoupled_variant_reuses_best_gold_layout(self) -> None:
        leader = build_messages(self.case, "gold_support_evidence_only_v3")
        decoupled = build_messages(self.case, "gold_decoupled_status_answer_v3")
        self.assertEqual(leader[1]["content"], decoupled[1]["content"])
        self.assertIn("Stage 1", decoupled[0]["content"])
        self.assertIn("Stage 2", decoupled[0]["content"])

    def test_gold_instruction_crosses_reuse_best_layout(self) -> None:
        leader_user = build_messages(
            self.case, "gold_support_evidence_only_v3"
        )[1]["content"]
        for variant in (
            "gold_i_guard_evidence_only_v3",
            "gold_binary_support_evidence_only_v3",
            "gold_compact_balanced_v3",
        ):
            self.assertEqual(build_messages(self.case, variant)[1]["content"], leader_user)

    def test_composite_stage_b_contracts_keep_i_gate_outside_prompt(self) -> None:
        stage_a = {
            "predicted_status": "supported_answer",
            "answer": "descriptive draft",
            "reason": "Evidence supports one answer.",
        }
        for name, variant in COMPOSITE_PROMPT_VARIANTS.items():
            messages = build_composite_stage_b_messages(self.case, stage_a, name)
            self.assertIn("Reference gold answer", messages[1]["content"])
            self.assertNotIn("sub_query:", messages[1]["content"])
            if variant.include_stage_a_draft:
                self.assertIn("Binding Stage-A non-I judgment", messages[1]["content"])
                self.assertIn("never output insufficient_evidence", messages[0]["content"])

    def test_composite_r5_reuses_selected_single_prompt_exactly(self) -> None:
        stage_a = {"predicted_status": "supported_answer", "answer": "draft", "reason": "r"}
        self.assertEqual(
            build_composite_stage_b_messages(self.case, stage_a, "hard_gate_r5_v1"),
            build_messages(self.case, "gold_support_evidence_only_v3"),
        )

    def test_dual_all_override_has_bounded_gold_rule(self) -> None:
        variant = COMPOSITE_PROMPT_VARIANTS["dual_all_r5_gold_f1_08_override_v1"]
        self.assertEqual(variant.stage_b_scope, "all")
        self.assertEqual(variant.i_override_min_gold_f1, 0.8)
        self.assertEqual(variant.reuse_single_prompt_variant, "gold_support_evidence_only_v3")

    def test_literal_canonical_strategy_is_bounded_and_evidence_grounded(self) -> None:
        variant = COMPOSITE_PROMPT_VARIANTS[
            "dual_all_r5_gold_f1_08_literal_canonical_v2"
        ]
        self.assertTrue(variant.prefer_higher_gold_f1_between_supported_stages)
        self.assertTrue(variant.canonicalize_evidence_literal_gold)
        case = {
            "gold_answers": ["Fairfield County"],
            "evidence_steps": [
                {
                    "docs": [
                        {
                            "title": "Stamford, Connecticut",
                            "contents": "Stamford is a city in Fairfield County, Connecticut.",
                        }
                    ]
                }
            ],
        }
        self.assertEqual(find_evidence_literal_gold(case), "Fairfield County")
        stage_a = {
            "parsed": True,
            "predicted_label": "S",
            "answer": "Fairfield County in Connecticut",
        }
        stage_b = {
            "parsed": True,
            "predicted_label": "S",
            "answer": "Connecticut",
        }
        selected = select_composite_output(case, stage_a, stage_b, variant)
        self.assertFalse(selected["stage_b_used"])
        self.assertEqual(selected["answer"], "Fairfield County")
        self.assertEqual(selected["canonical_gold"], "Fairfield County")

        stage_b_ambiguous = {
            "parsed": True,
            "predicted_label": "A",
            "answer": "证据不足无法作答",
        }
        selected = select_composite_output(case, stage_a, stage_b_ambiguous, variant)
        self.assertFalse(selected["stage_b_used"])
        self.assertEqual(selected["selected"]["predicted_label"], "S")
        self.assertEqual(selected["selection_reason"], "stage_a_only_supported+evidence_literal_gold")

        hard_gate = COMPOSITE_PROMPT_VARIANTS["hard_gate_r5_literal_canonical_v2"]
        self.assertEqual(hard_gate.stage_b_scope, "stage_a_non_i")
        self.assertIsNone(hard_gate.i_override_min_gold_f1)
        self.assertTrue(hard_gate.canonicalize_evidence_literal_gold)

    def test_literal_canonical_strategy_does_not_inject_absent_gold(self) -> None:
        variant = COMPOSITE_PROMPT_VARIANTS[
            "dual_all_r5_gold_f1_08_literal_canonical_v2"
        ]
        case = {
            "gold_answers": ["Fairfield County"],
            "evidence_steps": [{"docs": [{"title": "Stamford", "contents": "A city."}]}],
        }
        stage_a = {"parsed": True, "predicted_label": "S", "answer": "Stamford"}
        stage_b = {"parsed": True, "predicted_label": "I", "answer": "无法作答"}
        selected = select_composite_output(case, stage_a, stage_b, variant)
        self.assertEqual(selected["answer"], "Stamford")
        self.assertEqual(selected["canonical_gold"], "")


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

    def test_answer_metrics_and_equal_weight_objective(self) -> None:
        predictions = [
            {
                "manual_label": "I",
                "predicted_label": "I",
                "parsed": True,
                "answer": "证据不足无法作答",
                "gold_answers": ["missing gold"],
                "teacher_called": True,
                "step_layer": "L1",
            },
            {
                "manual_label": "S",
                "predicted_label": "S",
                "parsed": True,
                "answer": "The Eiffel Tower",
                "gold_answers": ["Eiffel Tower"],
                "manual_answer": "Eiffel Tower",
                "teacher_called": True,
                "step_layer": "L1",
            },
            {
                "manual_label": "S",
                "predicted_label": "I",
                "parsed": True,
                "answer": "证据不足无法作答",
                "gold_answers": ["Paris"],
                "manual_answer": "Paris",
                "teacher_called": False,
                "step_layer": "L2",
            },
        ]
        metrics = score_predictions(predictions)
        self.assertEqual(answer_exact_match("The Eiffel Tower", ["Eiffel Tower"]), 1.0)
        self.assertEqual(answer_token_f1("New York City", ["York City"]), 0.8)
        self.assertEqual(metrics["answer_gold"]["token_f1_coverage"], 0.5)
        self.assertAlmostEqual(
            metrics["equal_weight_objective"],
            0.5 * metrics["i_binary"]["f1"] + 0.25,
        )

    def test_answer_alignment_variant_keeps_successful_layout(self) -> None:
        with (HERE / "benchmark_237.jsonl").open("r", encoding="utf-8") as handle:
            case = json.loads(next(handle))
        baseline = build_messages(case, "baseline_question_tail_evidence_only_v2")
        aligned = build_messages(case, "question_tail_answer_alignment_v3")
        self.assertEqual(baseline[1]["content"], aligned[1]["content"])
        self.assertNotEqual(baseline[0]["content"], aligned[0]["content"])


if __name__ == "__main__":
    unittest.main()
