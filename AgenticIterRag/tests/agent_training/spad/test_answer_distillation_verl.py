"""Tests for SPAD-RAG Stage 3 VERL planning."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agentic_iter_rag.agent_training.spad.answer_distillation import (
    _convert_pairs_to_grpo_parquet,
    run_answer_distillation,
)
from agentic_iter_rag.agent_training.spad.refresh_rollout import run_answer_refresh_data


def _spad_cfg_for_dpo(backend: str = "verl") -> dict:
    return {
        "sub_stages": {
            "answer_distillation": {
                "phase_order": ["dpo"],
                "inputs": {},
                "phases": {
                    "dpo": {
                        "enabled": True,
                        "backend": backend,
                        "train_batch_size": 64,
                        "micro_batch_size_per_gpu": 4,
                        "learning_rate": 1.0e-6,
                        "total_epochs": 1,
                        "total_training_steps": 1,
                        "max_length": 4096,
                        "beta": 0.1,
                        "pairwise_loss_weight": 1.0,
                        "chosen_sft_loss_weight": 0.2,
                        "apply_chat_template_kwargs": {"enable_thinking": False},
                    }
                },
            }
        }
    }


def _spad_cfg_for_grpo() -> dict:
    return {
        "sub_stages": {
            "answer_distillation": {
                "phase_order": ["grpo"],
                "inputs": {},
                "phases": {
                    "grpo": {
                        "enabled": True,
                        "backend": "verl",
                        "reward_type": "gold_answer_f1",
                        "train_batch_size": 64,
                        "ppo_mini_batch_size": 64,
                        "n_samples_per_prompt": 8,
                        "total_training_steps": 1,
                    }
                },
            }
        }
    }


def _write_dataset_manifest(root: Path) -> Path:
    dataset_jsonl = root / "answer_distill_pairs.jsonl"
    dataset_jsonl.write_text(
        json.dumps(
            {
                "index": 7,
                "question": "Question?",
                "gold_answers": ["answer"],
                "messages_before_final_answer": [{"role": "user", "content": "Question?"}],
                "chosen": "answer",
                "rejected": "wrong",
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = root / "answer_distill_dataset_manifest.json"
    manifest.write_text(json.dumps({"outputs": {"dataset_jsonl": str(dataset_jsonl)}}), encoding="utf-8")
    return manifest


class AnswerDistillationVerlPlanTest(unittest.TestCase):
    def test_stage2_pair_converts_to_grpo_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = _write_dataset_manifest(root)
            dataset_jsonl = json.loads(manifest.read_text())["outputs"]["dataset_jsonl"]
            conversion = _convert_pairs_to_grpo_parquet(dataset_jsonl, root / "grpo")
            import pandas as pd

            frame = pd.read_parquet(conversion["train_parquet"])
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0].prompt.tolist()[0]["content"], "Question?")
            self.assertEqual(frame.iloc[0].reward_model["ground_truth"]["target"].tolist(), ["answer"])
            self.assertEqual(frame.iloc[0].extra_info["stage2_index"], 7)

    def test_grpo_dry_run_uses_main_ppo_without_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest = _write_dataset_manifest(root)
            outputs = run_answer_distillation(
                spad_cfg=_spad_cfg_for_grpo(),
                stage_dir=root / "stage",
                log_dir=root / "logs",
                checkpoint_dir=root / "checkpoints",
                resource_plan={
                    "phases": {
                        "grpo": {
                            "trainer": {
                                "gpu_ids": list(range(8)),
                                "n_gpus_per_node": 8,
                                "tensor_parallel_size": 1,
                            }
                        }
                    }
                },
                dry_run=True,
                init_actor_checkpoint="/models/qwen3-1.7b",
                dataset_manifest=str(dataset_manifest),
            )
            plan = json.loads(
                Path(outputs["phase_outputs"]["grpo"]["verl_command_plan"]).read_text()
            )
            overrides = plan["planned_hydra_overrides"]
            self.assertEqual(plan["entry"], "python -m verl.trainer.main_ppo")
            self.assertIn("actor_rollout_ref.rollout.multi_turn.enable=False", overrides)
            self.assertIn(
                "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4",
                overrides,
            )
            self.assertIn("trainer.save_freq=1000000", overrides)
            self.assertTrue(
                any("compute_gold_answer_f1_reward_details" in item for item in overrides)
            )
            self.assertEqual(plan["training_params"]["n_samples_per_prompt"], 8)

    def test_verl_backend_writes_dry_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest = _write_dataset_manifest(root)
            outputs = run_answer_distillation(
                spad_cfg=_spad_cfg_for_dpo("verl"),
                stage_dir=root / "stage",
                log_dir=root / "logs",
                checkpoint_dir=root / "checkpoints",
                resource_plan={"phases": {"dpo": {"trainer": {"gpu_ids": [0], "n_gpus_per_node": 1}}}},
                dry_run=True,
                init_actor_checkpoint="/models/qwen3-1.7b",
                dataset_manifest=str(dataset_manifest),
            )

            phase_outputs = outputs["phase_outputs"]["dpo"]
            self.assertEqual(phase_outputs["status"], "planned")
            self.assertEqual(phase_outputs["backend"], "verl")
            self.assertIn("verl_command_plan", phase_outputs)
            plan_path = Path(phase_outputs["verl_command_plan"])
            self.assertTrue(plan_path.exists())
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual(plan["implementation_status"], "executable")
            self.assertEqual(plan["phase"], "dpo")
            self.assertEqual(plan["dataset_jsonl"], phase_outputs["dataset_jsonl"])
            self.assertEqual(plan["training_params"]["train_batch_size"], 64)
            self.assertIn("spad_offline_dpo", plan["candidate_entries"][0])
            self.assertTrue(plan["command"][0].endswith("python") or plan["command"][0].endswith("python3"))
            self.assertEqual(plan["command"][1:3], ["-m", "torch.distributed.run"])
            self.assertEqual(plan["training_params"]["tensor_parallel_size"], 1)

    def test_dry_run_plan_uses_single_gpu_when_resource_is_single_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_manifest = _write_dataset_manifest(root)
            outputs = run_answer_distillation(
                spad_cfg=_spad_cfg_for_dpo("verl"),
                stage_dir=root / "stage",
                log_dir=root / "logs",
                checkpoint_dir=root / "checkpoints",
                resource_plan={"phases": {"dpo": {"trainer": {"gpu_ids": [0], "n_gpus_per_node": 1, "tensor_parallel_size": 1}}}},
                dry_run=True,
                init_actor_checkpoint="/models/qwen3-1.7b",
                dataset_manifest=str(dataset_manifest),
            )
            plan_path = Path(outputs["phase_outputs"]["dpo"]["verl_command_plan"])
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertIn("--nproc_per_node=1", plan["command"])


class AnswerRefreshDryRunManifestTest(unittest.TestCase):
    def test_dry_run_outputs_dataset_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = run_answer_refresh_data(
                config={},
                spad_cfg={"sub_stages": {"answer_refresh_data": {"backend": "smoke"}}},
                stage_dir=root / "stage",
                log_dir=root / "logs",
                checkpoint_dir=root / "checkpoints",
                resource_plan={},
                dry_run=True,
                actor_checkpoint="/checkpoints/actor",
            )

            self.assertIn("dataset_manifest", outputs)
            manifest_path = Path(outputs["dataset_manifest"])
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["outputs"]["dataset_jsonl"], outputs["dataset_jsonl"])


if __name__ == "__main__":
    unittest.main()
