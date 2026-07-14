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
    def build_manifest(self, root: Path, *, teacher_message_count: int = 8) -> Path:
        shard = root / "1.jsonl"
        rows = [{"uid": "group", "index": index} for index in range(8)]
        shard.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest = {
            "completed": True,
            "expected_steps": 1,
            "actual_step_count": 1,
            "expected_prompt_count": 1,
            "actual_prompt_count": 1,
            "expected_group_count": 1,
            "actual_group_count": 1,
            "expected_rollout_count": 8,
            "actual_rollout_count": 8,
            "teacher_called_count": 4,
            "shards": [
                {
                    "path": str(shard),
                    "record_count": 8,
                    "sha256": sha256_file(shard),
                    "field_nonempty_counts": {
                        "input": 8,
                        "output": 8,
                        "gts": 8,
                        "raw_prompt": 8,
                        "assistant_turn_records": 8,
                        "search_count": 8,
                        "tool_call_details": 8,
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


if __name__ == "__main__":
    unittest.main()
