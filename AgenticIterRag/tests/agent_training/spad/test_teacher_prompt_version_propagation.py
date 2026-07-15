"""Tests that configured SPAD teacher prompt versions reach runtime calls."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agentic_iter_rag.agent_training.spad.prompts import DEFAULT_TEACHER_STATUS_PROMPT_VERSION
from agentic_iter_rag.agent_training.spad.refresh_rollout import (
    _run_rollout_backend,
    _teacher_label_result_from_raw,
)
from agentic_iter_rag.agent_training.spad.search_policy_rl import _build_verl_plan


class TeacherPromptVersionPropagationTest(unittest.TestCase):
    def build_plan(
        self,
        root: Path,
        prompt_version: str,
        reward_type: str = "spad_em_teacher_backoff",
        trainer_overrides: dict | None = None,
    ) -> dict:
        config = {
            "data": {
                "train_files": ["/data/train.parquet"],
                "val_files": ["/data/val.parquet"],
                "max_prompt_length": 1024,
                "max_response_length": 512,
            },
            "model": {"path": "/models/qwen"},
            "main_run": {"project": {"experiment_name": "prompt-version-test"}},
            "infer_runtime": {"retriever": {"recall_final_top_n": 50}},
        }
        spad_cfg = {
            "teacher_answerer": {
                "prompt_version": prompt_version,
                "request": {"temperature": 0.0},
            },
            "reward": {"type": reward_type},
            "sub_stages": {
                "search_policy_rl": {
                    "trainer": dict(trainer_overrides or {}),
                    "rollout": {},
                }
            },
        }
        return _build_verl_plan(
            config=config,
            spad_cfg=spad_cfg,
            stage_dir=root / "stage",
            log_dir=root / "logs",
            checkpoint_dir=root / "checkpoints",
            resource_plan={"trainer": {"gpu_ids": [0], "tensor_parallel_size": 1}},
            teacher_output={"endpoint": "http://127.0.0.1:8067/v1/chat/completions", "model": "GLM-4.7-Flash"},
            recall_output={"retrieval_url": "http://127.0.0.1:8130/retrieve"},
        )

    def test_stage1_hydra_overrides_include_configured_prompt_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build_plan(Path(tmp), DEFAULT_TEACHER_STATUS_PROMPT_VERSION)

        self.assertIn(
            "+custom_reward_function.reward_kwargs.teacher_prompt_version="
            + DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
            plan["hydra_overrides"],
        )

    def test_stage1_disables_per_uid_advantage_std_normalization_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build_plan(Path(tmp), DEFAULT_TEACHER_STATUS_PROMPT_VERSION)

        self.assertIn(
            "algorithm.norm_adv_by_std_in_grpo=False",
            plan["hydra_overrides"],
        )

    def test_stage1_rejects_unknown_prompt_version_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown SPAD teacher prompt_version"):
                self.build_plan(Path(tmp), "spad_teacher_missing_v99")

    def test_0710_reward_name_selects_frozen_module_and_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build_plan(
                Path(tmp),
                DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
                reward_type="spad_teacher_f1_0710",
            )

        overrides = plan["hydra_overrides"]
        self.assertIn("reward_model.reward_manager=naive", overrides)
        self.assertIn("+reward_model.use_reward_loop=True", overrides)
        self.assertIn("+reward_model.stream_group_reward=False", overrides)
        self.assertIn("custom_reward_function.name=compute_spad_teacher_f1_0710_details", overrides)
        self.assertIn("+actor_rollout_ref.rollout.stop=['</tool_call>','<answer>']", overrides)
        self.assertTrue(plan["reward_path"].endswith("search_policy_teacher_reward_0710.py"))

    def test_current_reward_name_selects_named_batch_entry_and_runtime_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build_plan(Path(tmp), DEFAULT_TEACHER_STATUS_PROMPT_VERSION)

        overrides = plan["hydra_overrides"]
        self.assertIn("reward_model.reward_manager=batch", overrides)
        self.assertIn("+reward_model.use_reward_loop=False", overrides)
        self.assertIn("+reward_model.stream_group_reward=True", overrides)
        self.assertIn(
            "custom_reward_function.name=compute_spad_em_teacher_backoff_batch",
            overrides,
        )
        self.assertIn("+actor_rollout_ref.rollout.stop=['</tool_call>','</answer>']", overrides)
        self.assertTrue(plan["reward_path"].endswith("search_policy_teacher_reward.py"))

    def test_dev_reward_uses_prefetch_loop_and_group_batch_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build_plan(
                Path(tmp),
                DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
                reward_type="spad_em_teacher_backoff_dev",
            )

        overrides = plan["hydra_overrides"]
        self.assertIn("reward_model.reward_manager=batch", overrides)
        self.assertIn("+reward_model.reward_loop_manager=naive", overrides)
        self.assertIn("+reward_model.use_reward_loop=True", overrides)
        self.assertIn("+reward_model.stream_group_reward=True", overrides)
        self.assertIn(
            "custom_reward_function.name=compute_spad_em_teacher_backoff_dev",
            overrides,
        )
        self.assertIn("+actor_rollout_ref.rollout.stop=['</tool_call>','</answer>']", overrides)
        self.assertTrue(plan["reward_path"].endswith("search_policy_teacher_reward_dev.py"))

    def test_gold_token_f1_bonus_reward_selects_independent_batch_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build_plan(
                Path(tmp),
                DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
                reward_type="spad_em_teacher_backoff_gold_token_f1_bonus",
            )

        overrides = plan["hydra_overrides"]
        self.assertIn("reward_model.reward_manager=batch", overrides)
        self.assertIn("+reward_model.use_reward_loop=False", overrides)
        self.assertIn("+reward_model.stream_group_reward=True", overrides)
        self.assertIn(
            "custom_reward_function.name="
            "compute_spad_em_teacher_backoff_gold_token_f1_bonus_batch",
            overrides,
        )
        self.assertTrue(
            plan["reward_path"].endswith(
                "search_policy_teacher_reward_gold_match_bonus.py"
            )
        )

    def test_gold_token_f1_v3_selects_postnorm_scaled_batch_module(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plan = self.build_plan(
                Path(tmp),
                DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
                reward_type="spad_em_teacher_backoff_gold_token_f1_bonus_v3",
                trainer_overrides={"norm_adv_by_std_in_grpo": True},
            )

        overrides = plan["hydra_overrides"]
        self.assertIn("algorithm.norm_adv_by_std_in_grpo=True", overrides)
        self.assertIn(
            "+algorithm.group_postnorm_advantage_scale_key=advantage_postnorm_scale",
            overrides,
        )
        self.assertIn(
            "+algorithm.group_postnorm_advantage_scale_version=teacher_fallback_v1",
            overrides,
        )
        self.assertIn(
            "custom_reward_function.name="
            "compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_batch",
            overrides,
        )
        self.assertTrue(
            plan["reward_path"].endswith(
                "search_policy_teacher_reward_gold_match_bonus_v3.py"
            )
        )

    def test_gold_token_f1_v3_rejects_norm_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(
            ValueError, "requires norm_adv_by_std_in_grpo=true"
        ):
            self.build_plan(
                Path(tmp),
                DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
                reward_type="spad_em_teacher_backoff_gold_token_f1_bonus_v3",
            )

    def test_stage2_records_configured_prompt_version_in_sample_metadata(self) -> None:
        refresh_record, pair_record, _ = _teacher_label_result_from_raw(
            trajectory={
                "index": 1,
                "question": "What is the capital of France?",
                "gold_answers": ["Paris"],
                "evidence_steps": [{"sub_query": "France capital", "docs": []}],
                "messages_before_final_answer": [{"role": "user", "content": "Question"}],
                "actor_answer": "Lyon",
                "rejected": "<reason>Wrong.</reason><answer>Lyon</answer>",
            },
            teacher_raw=(
                "<reason>Evidence [1] supports Paris.</reason>"
                "<status>supported_answer</status><answer>Paris</answer>"
            ),
            teacher_elapsed=1.0,
            spad_cfg={
                "teacher_answerer": {
                    "prompt_version": DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
                    "insufficient_answer": "证据不足无法作答",
                }
            },
            filter_cfg={},
        )

        self.assertEqual(
            refresh_record["metadata"]["teacher_prompt_version"],
            DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
        )
        self.assertIsNotNone(pair_record)
        self.assertEqual(pair_record["teacher_prompt_version"], DEFAULT_TEACHER_STATUS_PROMPT_VERSION)

    def test_stage2_rejects_unknown_prompt_version_before_services(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "Unknown SPAD teacher prompt_version"):
                _run_rollout_backend(
                    config={},
                    spad_cfg={
                        "teacher_answerer": {
                            "prompt_version": "spad_teacher_missing_v99",
                            "request": {},
                        }
                    },
                    sub_cfg={"rollout": {}},
                    stage_dir=root / "stage",
                    log_dir=root / "logs",
                    checkpoint_dir=root / "checkpoints",
                    dataset_jsonl=root / "dataset.jsonl",
                    resource_plan={},
                    actor_checkpoint="/models/qwen",
                    dry_run=True,
                )


if __name__ == "__main__":
    unittest.main()
