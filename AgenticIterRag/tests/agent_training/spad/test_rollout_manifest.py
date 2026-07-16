"""Rollout manifest completion and audit tests."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agentic_iter_rag.agent_training.spad.search_policy_rl import _validate_rollout_manifest


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RolloutManifestTest(unittest.TestCase):
    def build_manifest(
        self,
        root: Path,
        *,
        row_count: int = 8,
        invalid_output_count: int = 0,
        teacher_message_count: int | None = None,
    ) -> Path:
        shard = root / "1.jsonl"
        rows = [
            {
                "uid": "group",
                "index": index,
                "input": "question",
                "output": "" if index < invalid_output_count else "answer",
                "gts": {"target": ["answer"]},
                "raw_prompt": "prompt",
                "assistant_turn_records": [{"turn_index": 0}],
            }
            for index in range(row_count)
        ]
        shard.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        teacher_message_count = row_count if teacher_message_count is None else teacher_message_count
        manifest = {
            "completed": True,
            "expected_steps": 1,
            "actual_step_count": 1,
            "expected_prompt_count": 1,
            "actual_prompt_count": 1,
            "expected_group_count": 1,
            "actual_group_count": 1,
            "expected_rollout_count": row_count,
            "actual_rollout_count": row_count,
            "teacher_called_count": 4,
            "shards": [
                {
                    "path": str(shard),
                    "record_count": row_count,
                    "sha256": sha256_file(shard),
                    "field_nonempty_counts": {
                        "input": row_count,
                        "output": row_count - invalid_output_count,
                        "gts": row_count,
                        "raw_prompt": row_count,
                        "assistant_turn_records": row_count,
                        "search_count": row_count,
                        "tool_call_details": row_count,
                        "teacher_messages": teacher_message_count,
                    },
                }
            ],
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_complete_manifest_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.build_manifest(root)
            result = _validate_rollout_manifest(root, require_teacher_audit=True)
            self.assertEqual(result["path"], str(path))
            self.assertEqual(len(result["sha256"]), 64)

    def test_missing_teacher_messages_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_manifest(root, teacher_message_count=7)
            with self.assertRaisesRegex(ValueError, "teacher_messages"):
                _validate_rollout_manifest(root, require_teacher_audit=True)

    def test_invalid_trajectories_at_or_below_half_percent_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_manifest(root, row_count=1000, invalid_output_count=5)
            result = _validate_rollout_manifest(root, require_teacher_audit=False)
            self.assertEqual(result["summary"]["invalid_trajectory_count"], 5)
            self.assertEqual(result["summary"]["invalid_trajectory_rate_limit"], 0.005)

    def test_invalid_trajectories_above_half_percent_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.build_manifest(root, row_count=1000, invalid_output_count=6)
            with self.assertRaisesRegex(ValueError, "semi-strict limit"):
                _validate_rollout_manifest(root, require_teacher_audit=False)


if __name__ == "__main__":
    unittest.main()
