"""Focused tests for the new-data evaluation aggregator."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/cosearch_local/aggregate_newdata_model_eval.py"
SPEC = importlib.util.spec_from_file_location("aggregate_newdata_model_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
aggregate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(aggregate)


class NewDataEvalAggregateTest(unittest.TestCase):
    def test_metric_row_preserves_answer_group_metrics(self) -> None:
        row = aggregate.metric_row(
            {
                "index": 0,
                "data_source": "nq",
                "status": "answered",
                "final_answer": "Paris",
                "sub_queries": ["France capital"],
                "enhanced_trajectory": {"question": "What is the capital of France?"},
                "metrics": {
                    "em": 1.0,
                    "f1": 1.0,
                    "structured_em": 0.75,
                    "answer_group_f1": 0.8,
                    "answer_group_recall": 0.5,
                    "tool_calls": 1,
                },
            },
            "nq",
            "What is the capital of France?",
        )

        self.assertEqual(row["structured_em"], 0.75)
        self.assertEqual(row["answer_group_f1"], 0.8)
        self.assertEqual(row["answer_group_recall"], 0.5)
        summary = aggregate.scalar_summary([row])
        self.assertEqual(summary["structured_em"], 0.75)
        self.assertEqual(summary["answer_group_f1"], 0.8)
        self.assertEqual(summary["answer_group_recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
