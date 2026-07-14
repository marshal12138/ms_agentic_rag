"""Stage 3 answer distillation for SPAD-RAG."""

from __future__ import annotations

import json
import hashlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.manifest import write_sub_stage_manifest
from agentic_iter_rag.agent_training.spad.checkpoint_finalizer import finalize_actor_checkpoint
from agentic_iter_rag.agent_training.spad.search_policy_rl import (
    _find_latest_checkpoint,
    _list_override,
    _run_shell_script,
    _scalar_override,
    _write_verl_script,
)
from agentic_iter_rag.agent_training.spad.service_manager import project_root, repo_root, tail_text
from agentic_iter_rag.utils.io import iter_jsonl, write_json


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_pairs_to_grpo_parquet(
    dataset_jsonl: str,
    phase_dir: Path,
    *,
    tokenizer_path: str | None = None,
    max_prompt_length: int = 12000,
) -> dict[str, Any]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    skipped = {"empty_prompt": 0, "empty_gold": 0, "invalid_row": 0, "prompt_too_long": 0}
    tokenizer = None
    if tokenizer_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    for raw in iter_jsonl(dataset_jsonl):
        if not isinstance(raw, dict):
            skipped["invalid_row"] += 1
            continue
        prompt = raw.get("messages_before_final_answer")
        gold_answers = raw.get("gold_answers")
        if hasattr(gold_answers, "tolist"):
            gold_answers = gold_answers.tolist()
        if not isinstance(prompt, list) or not prompt:
            skipped["empty_prompt"] += 1
            continue
        if not isinstance(gold_answers, list) or not any(str(item).strip() for item in gold_answers):
            skipped["empty_gold"] += 1
            continue
        if tokenizer is not None:
            prompt_ids = tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=False,
            )
            if len(prompt_ids) > max_prompt_length:
                skipped["prompt_too_long"] += 1
                continue
        rows.append(
            {
                "data_source": str(raw.get("data_source") or "spad_answer_distillation"),
                "prompt": prompt,
                "ability": "qa",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "target": [str(item) for item in gold_answers if str(item).strip()]
                    },
                },
                "extra_info": {
                    "question": str(raw.get("question") or ""),
                    "stage2_index": int(raw.get("index") or 0),
                },
            }
        )
    if not rows:
        raise ValueError(f"Stage3 GRPO conversion produced no rows: skipped={skipped}")
    dataset_dir = phase_dir / "verl_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    train_path = dataset_dir / "grpo_train.parquet"
    val_path = dataset_dir / "grpo_val.parquet"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    pd.DataFrame(rows[: min(8, len(rows))]).to_parquet(val_path, index=False)
    manifest = {
        "source_jsonl": str(dataset_jsonl),
        "source_sha256": _sha256_file(Path(dataset_jsonl)),
        "input_count": len(rows) + sum(skipped.values()),
        "kept_count": len(rows),
        "skipped": skipped,
        "train_parquet": str(train_path),
        "train_sha256": _sha256_file(train_path),
        "val_parquet": str(val_path),
        "val_sha256": _sha256_file(val_path),
    }
    manifest_path = dataset_dir / "manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def _build_stage3_grpo_plan(
    *,
    phase_cfg: dict[str, Any],
    init_actor_checkpoint: str,
    train_parquet: str,
    val_parquet: str,
    phase_dir: Path,
    phase_log_dir: Path,
    phase_checkpoint_dir: Path,
    resource_plan: dict[str, Any],
) -> dict[str, Any]:
    trainer_resource = _phase_trainer_resource(resource_plan, "grpo")
    gpu_ids = [int(item) for item in trainer_resource.get("gpu_ids", [])]
    n_gpus = int(trainer_resource.get("n_gpus_per_node", len(gpu_ids) or 1))
    if not gpu_ids:
        gpu_ids = list(range(n_gpus))
    tensor_parallel_size = int(trainer_resource.get("tensor_parallel_size", 1))
    if tensor_parallel_size != 1:
        raise ValueError("SPAD Stage3 Qwen3-1.7B GRPO must use tensor_parallel_size=1")
    train_batch_size = int(phase_cfg.get("train_batch_size", 64))
    ppo_mini_batch_size = int(phase_cfg.get("ppo_mini_batch_size", 64))
    rollout_n = int(phase_cfg.get("n_samples_per_prompt", 8))
    if (train_batch_size, ppo_mini_batch_size, rollout_n) != (64, 64, 8):
        raise ValueError("Stage3 formal GRPO requires train_batch=64, mini_batch=64, rollout_n=8")
    max_prompt_length = int(phase_cfg.get("max_prompt_length", 12000))
    max_response_length = int(phase_cfg.get("max_response_length", 1024))
    max_model_len = int(phase_cfg.get("rollout_max_model_len", max_prompt_length + max_response_length))
    output_dir = phase_checkpoint_dir / "actor_model_verl"
    rollout_dir = phase_dir / "rollout_data"
    validation_dir = phase_dir / "validation_data"
    reward_path = project_root() / "agentic_iter_rag" / "agent_training" / "spad" / "reward.py"
    total_steps = _normal_training_steps(phase_cfg.get("total_training_steps"))
    hydra_overrides = [
        _scalar_override("algorithm.adv_estimator", "grpo"),
        _scalar_override("algorithm.use_kl_in_reward", False),
        _scalar_override("critic.enable", False),
        _scalar_override("reward_model.enable", False),
        _scalar_override("+reward_model.use_reward_loop", True),
        _scalar_override("reward_model.reward_manager", "naive"),
        _list_override("data.train_files", [train_parquet]),
        _list_override("data.val_files", [val_parquet]),
        _scalar_override("data.train_batch_size", train_batch_size),
        _scalar_override("data.val_batch_size", min(8, train_batch_size)),
        _scalar_override("data.shuffle", True),
        _scalar_override("data.seed", int(phase_cfg.get("data_seed", 42))),
        _scalar_override("data.max_prompt_length", max_prompt_length),
        _scalar_override("data.max_response_length", max_response_length),
        _scalar_override("data.truncation", "error"),
        _scalar_override("data.return_raw_chat", True),
        _scalar_override("data.trust_remote_code", True),
        _scalar_override("data.dataloader_num_workers", 0),
        _scalar_override("+data.apply_chat_template_kwargs.enable_thinking", False),
        _scalar_override("actor_rollout_ref.model.path", init_actor_checkpoint),
        _scalar_override("actor_rollout_ref.model.trust_remote_code", True),
        _scalar_override("actor_rollout_ref.model.use_remove_padding", True),
        _scalar_override("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        _scalar_override("actor_rollout_ref.actor.optim.lr", float(phase_cfg.get("learning_rate", 1.0e-6))),
        _scalar_override("actor_rollout_ref.actor.use_torch_compile", False),
        _scalar_override("actor_rollout_ref.actor.ppo_mini_batch_size", ppo_mini_batch_size),
        _scalar_override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", int(phase_cfg.get("actor_micro_batch_size_per_gpu", 2))),
        _scalar_override("actor_rollout_ref.actor.use_kl_loss", True),
        _scalar_override("actor_rollout_ref.actor.kl_loss_coef", float(phase_cfg.get("kl_loss_coef", 0.001))),
        _scalar_override("actor_rollout_ref.actor.kl_loss_type", "low_var_kl"),
        _scalar_override("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", int(phase_cfg.get("log_prob_micro_batch_size_per_gpu", 4))),
        _scalar_override("actor_rollout_ref.rollout.name", "vllm"),
        _scalar_override("actor_rollout_ref.rollout.mode", "async"),
        _scalar_override("actor_rollout_ref.rollout.tensor_model_parallel_size", 1),
        _scalar_override("actor_rollout_ref.rollout.n", rollout_n),
        _scalar_override("actor_rollout_ref.rollout.temperature", float(phase_cfg.get("rollout_temperature", 1.0))),
        _scalar_override("actor_rollout_ref.rollout.top_p", float(phase_cfg.get("rollout_top_p", 1.0))),
        _scalar_override("actor_rollout_ref.rollout.gpu_memory_utilization", float(phase_cfg.get("rollout_gpu_memory_utilization", 0.6))),
        _scalar_override("actor_rollout_ref.rollout.max_model_len", max_model_len),
        _scalar_override("actor_rollout_ref.rollout.prompt_length", max_prompt_length),
        _scalar_override("actor_rollout_ref.rollout.response_length", max_response_length),
        _scalar_override("actor_rollout_ref.rollout.max_num_batched_tokens", int(phase_cfg.get("max_num_batched_tokens", max_model_len))),
        _scalar_override("actor_rollout_ref.rollout.max_num_seqs", int(phase_cfg.get("max_num_seqs", 32))),
        _scalar_override(
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu",
            int(phase_cfg.get("log_prob_micro_batch_size_per_gpu", 4)),
        ),
        _scalar_override("actor_rollout_ref.rollout.enable_chunked_prefill", True),
        _scalar_override("actor_rollout_ref.rollout.enable_prefix_caching", True),
        _scalar_override("actor_rollout_ref.rollout.multi_turn.enable", False),
        _list_override("+actor_rollout_ref.rollout.stop", [str(item) for item in phase_cfg.get("stop_sequences", ["</answer>"])]),
        _scalar_override("+actor_rollout_ref.rollout.include_stop_str_in_output", True),
        _scalar_override("custom_reward_function.path", str(reward_path)),
        _scalar_override("custom_reward_function.name", "compute_gold_answer_f1_reward_details"),
        _scalar_override("trainer.n_gpus_per_node", n_gpus),
        _scalar_override("trainer.nnodes", 1),
        _scalar_override("trainer.device", "npu"),
        _list_override("trainer.logger", ["console"]),
        _scalar_override("trainer.total_epochs", int(phase_cfg.get("total_epochs", 1))),
        _scalar_override("trainer.total_training_steps", total_steps),
        _scalar_override("trainer.project_name", "spad_rag_stage3"),
        _scalar_override("trainer.experiment_name", str(phase_cfg.get("experiment_name", "spad_stage3_grpo"))),
        _scalar_override("trainer.default_local_dir", str(output_dir)),
        _scalar_override("trainer.val_before_train", False),
        # VERL guards even its is_last_step save behind save_freq > 0. A large
        # positive interval saves only the final step for these dataset sizes.
        _scalar_override("trainer.save_freq", int(phase_cfg.get("save_freq", 1_000_000))),
        _scalar_override("trainer.test_freq", -1),
        _scalar_override("trainer.max_actor_ckpt_to_keep", 1),
        _scalar_override("trainer.rollout_data_dir", str(rollout_dir)),
        _scalar_override("trainer.validation_data_dir", str(validation_dir)),
        _scalar_override("+trainer.num_examine", 0),
    ]
    return {
        "backend": "verl",
        "phase": "grpo",
        "entry": "python -m verl.trainer.main_ppo",
        "verl_root": str(project_root() / "verl"),
        "model_path": init_actor_checkpoint,
        "train_parquet": train_parquet,
        "val_parquet": val_parquet,
        "output_dir": str(output_dir),
        "rollout_data_dir": str(rollout_dir),
        "validation_data_dir": str(validation_dir),
        "actor_gpu_ids": gpu_ids[:n_gpus],
        "tensor_parallel_size": tensor_parallel_size,
        "hydra_overrides": hydra_overrides,
        "runtime_dir": str(phase_log_dir),
    }


def _dataset_jsonl_from_manifest(dataset_manifest: str | None) -> str | None:
    if not dataset_manifest:
        return None
    dataset_path = Path(dataset_manifest)
    if dataset_path.suffix == ".jsonl":
        return str(dataset_path)
    if not dataset_path.exists():
        return None
    import json

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return payload.get("outputs", {}).get("dataset_jsonl") or payload.get("dataset_jsonl")


def _count_dataset_samples(dataset_manifest: str | None) -> int:
    dataset_jsonl = _dataset_jsonl_from_manifest(dataset_manifest)
    if not dataset_jsonl:
        return 0
    return sum(1 for _ in iter_jsonl(dataset_jsonl))


def _phase_trainer_resource(resource_plan: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_plan = (resource_plan.get("phases") or {}).get(phase, {})
    if not isinstance(phase_plan, dict):
        return {}
    trainer = phase_plan.get("trainer") or {}
    return trainer if isinstance(trainer, dict) else {}


def _normal_training_steps(value: Any) -> Any:
    if value is None:
        return None
    steps = int(value)
    return None if steps < 0 else steps


def _hydra_literal(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _quote(value: Any) -> str:
    return shlex.quote(str(value))


def _build_verl_dpo_command(
    *,
    phase_cfg: dict[str, Any],
    init_actor_checkpoint: str,
    dataset_jsonl: str,
    output_dir: Path,
    metrics_path: Path,
    n_gpus: int,
) -> list[str]:
    entry = project_root() / "verl" / "recipe" / "spad_offline_dpo" / "train_spad_offline_dpo.py"
    total_steps = _normal_training_steps(phase_cfg.get("total_training_steps"))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={max(1, int(n_gpus))}",
        str(entry),
        "--model-path",
        str(init_actor_checkpoint),
        "--dataset-jsonl",
        str(dataset_jsonl),
        "--output-dir",
        str(output_dir),
        "--metrics-path",
        str(metrics_path),
        "--train-batch-size",
        str(int(phase_cfg.get("train_batch_size", 64))),
        "--micro-batch-size-per-gpu",
        str(int(phase_cfg.get("micro_batch_size_per_gpu", 4))),
        "--learning-rate",
        str(float(phase_cfg.get("learning_rate", 1.0e-6))),
        "--total-epochs",
        str(int(phase_cfg.get("total_epochs", 1))),
        "--total-training-steps",
        str(-1 if total_steps is None else int(total_steps)),
        "--max-samples",
        str(int(phase_cfg.get("max_samples", -1))),
        "--max-length",
        str(int(phase_cfg.get("max_length", 4096))),
        "--beta",
        str(float(phase_cfg.get("beta", 0.1))),
        "--pairwise-loss-weight",
        str(float(phase_cfg.get("pairwise_loss_weight", 1.0))),
        "--chosen-sft-loss-weight",
        str(float(phase_cfg.get("chosen_sft_loss_weight", 0.2))),
        "--clip-grad-norm",
        str(float(phase_cfg.get("clip_grad_norm", 1.0))),
    ]
    apply_kwargs = dict(phase_cfg.get("apply_chat_template_kwargs") or {})
    if bool(apply_kwargs.get("enable_thinking", False)):
        cmd.append("--enable-thinking")
    return cmd


def _run_verl_dpo(
    *,
    phase_cfg: dict[str, Any],
    init_actor_checkpoint: str,
    dataset_jsonl: str,
    phase_checkpoint_path: Path,
    phase_log_dir: Path,
    resource_plan: dict[str, Any],
) -> dict[str, Any]:
    trainer_resource = _phase_trainer_resource(resource_plan, "dpo")
    gpu_ids = [int(item) for item in trainer_resource.get("gpu_ids", [])]
    n_gpus = int(trainer_resource.get("n_gpus_per_node", len(gpu_ids) or 1))
    if not gpu_ids:
        gpu_ids = list(range(n_gpus))
    if int(trainer_resource.get("tensor_parallel_size", 1)) != 1:
        raise ValueError("SPAD Stage 3 Qwen3-1.7B DPO must use tensor_parallel_size=1; parameter parallel is disabled")
    phase_log_dir.mkdir(parents=True, exist_ok=True)
    phase_checkpoint_path.mkdir(parents=True, exist_ok=True)
    metrics_path = phase_log_dir / "spad_verl_dpo_metrics.json"
    cmd = _build_verl_dpo_command(
        phase_cfg=phase_cfg,
        init_actor_checkpoint=init_actor_checkpoint,
        dataset_jsonl=dataset_jsonl,
        output_dir=phase_checkpoint_path,
        metrics_path=metrics_path,
        n_gpus=n_gpus,
    )
    env = dict(os.environ)
    visible = ",".join(str(item) for item in gpu_ids[:n_gpus])
    env["ASCEND_RT_VISIBLE_DEVICES"] = visible
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["PYTHONPATH"] = f"{project_root() / 'verl'}:{project_root()}:{env.get('PYTHONPATH', '')}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
    env["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
    command_path = phase_log_dir / "verl_dpo_command.json"
    write_json(
        command_path,
        {
            "backend": "verl",
            "entry": "recipe/spad_offline_dpo/train_spad_offline_dpo.py",
            "command": cmd,
            "gpu_ids": gpu_ids[:n_gpus],
            "n_gpus_per_node": n_gpus,
            "tensor_parallel_size": 1,
            "dataset_jsonl": dataset_jsonl,
            "model_path": init_actor_checkpoint,
            "output_dir": str(phase_checkpoint_path),
        },
    )
    log_path = phase_log_dir / "verl_dpo_train.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, cwd=str(repo_root()), env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"SPAD Stage 3 VERL DPO failed with code {result.returncode}; log={log_path}\n{tail_text(log_path)}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return {
        "checkpoint": str(phase_checkpoint_path),
        "metrics_path": str(metrics_path),
        "command_path": str(command_path),
        "log_path": str(log_path),
        "elapsed_s": metrics.get("elapsed_s"),
        "sample_count": metrics.get("sample_count"),
        "world_size": metrics.get("world_size", n_gpus),
        "tensor_parallel_size": 1,
        "backend_detail": metrics.get("backend", "verl_spad_offline_dpo"),
    }


def _build_verl_dry_run_plan(
    *,
    phase: str,
    phase_cfg: dict[str, Any],
    phase_dir: Path,
    phase_checkpoint_path: Path,
    init_actor_checkpoint: str,
    dataset_manifest: str,
    dataset_jsonl: str | None,
    resource_plan: dict[str, Any],
) -> dict[str, Any]:
    trainer_resource = _phase_trainer_resource(resource_plan, phase)
    gpu_ids = [int(item) for item in trainer_resource.get("gpu_ids", [])]
    dataset_dir = phase_dir / "verl_dataset"
    train_parquet = dataset_dir / f"{phase}_train.parquet"
    val_parquet = dataset_dir / f"{phase}_val.parquet"
    total_training_steps = _normal_training_steps(phase_cfg.get("total_training_steps"))
    common_training_params = {
        "train_batch_size": int(phase_cfg.get("train_batch_size", 64)),
        "micro_batch_size_per_gpu": int(phase_cfg.get("micro_batch_size_per_gpu", 4)),
        "learning_rate": float(phase_cfg.get("learning_rate", 1.0e-6)),
        "total_epochs": int(phase_cfg.get("total_epochs", 1)),
        "total_training_steps": total_training_steps,
        "max_length": int(phase_cfg.get("max_length", 4096)),
        "apply_chat_template_kwargs": dict(phase_cfg.get("apply_chat_template_kwargs") or {}),
    }
    if phase == "grpo":
        grpo_plan = _build_stage3_grpo_plan(
            phase_cfg=phase_cfg,
            init_actor_checkpoint=init_actor_checkpoint,
            train_parquet=str(train_parquet),
            val_parquet=str(val_parquet),
            phase_dir=phase_dir,
            phase_log_dir=phase_dir,
            phase_checkpoint_dir=phase_checkpoint_path,
            resource_plan=resource_plan,
        )
        phase_training_params = {
            "train_batch_size": int(phase_cfg.get("train_batch_size", 64)),
            "ppo_mini_batch_size": int(phase_cfg.get("ppo_mini_batch_size", 64)),
            "n_samples_per_prompt": int(phase_cfg.get("n_samples_per_prompt", 8)),
            "learning_rate": float(phase_cfg.get("learning_rate", 1.0e-6)),
            "total_epochs": int(phase_cfg.get("total_epochs", 1)),
            "total_training_steps": total_training_steps,
            "max_prompt_length": int(phase_cfg.get("max_prompt_length", 12000)),
            "max_response_length": int(phase_cfg.get("max_response_length", 1024)),
            "reward_type": str(phase_cfg.get("reward_type", "gold_answer_f1")),
            "tensor_parallel_size": 1,
            "world_size": len(grpo_plan["actor_gpu_ids"]),
        }
        entry = "python -m verl.trainer.main_ppo"
        planned_hydra_overrides = grpo_plan["hydra_overrides"]
        dry_run_reason = ""
        candidate_entries = ["verl.trainer.main_ppo"]
        target_schema = "verl_rl_prompt_reward_parquet"
        command = None
    elif phase == "dpo":
        n_gpus = int(trainer_resource.get("n_gpus_per_node", len(gpu_ids) or 1))
        phase_training_params = {
            **common_training_params,
            "beta": float(phase_cfg.get("beta", 0.1)),
            "pairwise_loss_weight": float(phase_cfg.get("pairwise_loss_weight", 1.0)),
            "chosen_sft_loss_weight": float(phase_cfg.get("chosen_sft_loss_weight", 0.0)),
            "clip_grad_norm": phase_cfg.get("clip_grad_norm"),
            "max_samples": int(phase_cfg.get("max_samples", -1)),
            "tensor_parallel_size": 1,
            "world_size": n_gpus,
        }
        entry = "recipe/spad_offline_dpo/train_spad_offline_dpo.py"
        metrics_path = phase_dir / "spad_verl_dpo_metrics.json"
        command = _build_verl_dpo_command(
            phase_cfg=phase_cfg,
            init_actor_checkpoint=init_actor_checkpoint,
            dataset_jsonl=dataset_jsonl or "",
            output_dir=phase_checkpoint_path,
            metrics_path=metrics_path,
            n_gpus=n_gpus,
        )
        planned_hydra_overrides = []
        dry_run_reason = ""
        candidate_entries = ["verl/recipe/spad_offline_dpo/train_spad_offline_dpo.py"]
        target_schema = "spad_answer_distill_pair_v1"
    elif phase == "sft":
        phase_training_params = {
            **common_training_params,
            "loss_weight": float(phase_cfg.get("loss_weight", 1.0)),
        }
        entry = "python -m verl.trainer.fsdp_sft_trainer"
        planned_hydra_overrides = [
            f"model.partial_pretrain={init_actor_checkpoint}",
            f"data.train_files=[{train_parquet}]",
            f"data.val_files=[{val_parquet}]",
            f"data.max_length={phase_training_params['max_length']}",
            f"data.micro_batch_size_per_gpu={phase_training_params['micro_batch_size_per_gpu']}",
            f"trainer.default_local_dir={phase_checkpoint_path}",
            f"trainer.n_gpus_per_node={trainer_resource.get('n_gpus_per_node', len(gpu_ids) or 1)}",
            f"trainer.total_epochs={phase_training_params['total_epochs']}",
            f"trainer.total_training_steps={_hydra_literal(total_training_steps)}",
        ]
        dry_run_reason = (
            "SFT can target VERL's FSDP SFT trainer after converting Stage2 pairs to chosen-answer "
            "prompt/response parquet. This path is currently plan-only in AIR SPAD."
        )
        candidate_entries = ["verl.trainer.fsdp_sft_trainer"]
        target_schema = "verl_sft_prompt_response_parquet"
    else:
        phase_training_params = common_training_params
        entry = None
        planned_hydra_overrides = []
        dry_run_reason = f"Stage 3 phase={phase!r} has no VERL wiring yet."
        candidate_entries = []
        target_schema = "unknown"

    return {
        "status": "planned",
        "implementation_status": "executable",
        "phase": phase,
        "backend": "verl",
        "entry": entry,
        "candidate_entries": candidate_entries,
        "dry_run_reason": dry_run_reason,
        "model_path": init_actor_checkpoint,
        "dataset_manifest": dataset_manifest,
        "dataset_jsonl": dataset_jsonl,
        "dataset_jsonl_exists": bool(dataset_jsonl and Path(dataset_jsonl).exists()),
        "required_data_conversion": {
            "source_schema": "spad_answer_distill_pair_v1",
            "source_fields": ["messages_before_final_answer", "chosen", "rejected"],
            "target_schema": target_schema,
            "planned_train_file": str(train_parquet),
            "planned_val_file": str(val_parquet),
            "notes": [
                "Teacher status tags must not be included in the chosen answer text.",
                "DPO must preserve the same messages_before_final_answer context used by Stage2 refresh.",
            ],
        },
        "training_params": phase_training_params,
        "resource_plan": resource_plan,
        "trainer_resource": trainer_resource,
        "gpu_ids": gpu_ids,
        "output_dir": str(phase_checkpoint_path),
        "planned_hydra_overrides": planned_hydra_overrides,
        "command": command if phase in {"dpo", "grpo"} else None,
    }


def run_answer_distillation(
    *,
    spad_cfg: dict[str, Any],
    stage_dir: Path,
    log_dir: Path,
    checkpoint_dir: Path,
    resource_plan: dict[str, Any],
    dry_run: bool,
    init_actor_checkpoint: str | None,
    dataset_manifest: str | None,
) -> dict[str, Any]:
    """Run Stage 3. The smoke backend validates dataset readability and writes phase manifests."""

    sub_cfg = spad_cfg["sub_stages"]["answer_distillation"]
    init_actor_checkpoint = str(init_actor_checkpoint or sub_cfg.get("inputs", {}).get("init_actor_checkpoint") or "")
    dataset_manifest = str(dataset_manifest or sub_cfg.get("inputs", {}).get("dataset_manifest") or "")
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    phase_outputs: dict[str, Any] = {}
    final_checkpoint = init_actor_checkpoint
    phase_order = [str(item) for item in sub_cfg.get("phase_order", ["sft", "dpo"])]
    for phase in phase_order:
        phase_cfg = sub_cfg.get("phases", {}).get(phase, {})
        if not isinstance(phase_cfg, dict) or not bool(phase_cfg.get("enabled", True)):
            continue
        backend = str(phase_cfg.get("backend") or "smoke")
        phase_dir = stage_dir / phase
        phase_log_dir = log_dir / phase
        phase_checkpoint_dir = checkpoint_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        phase_log_dir.mkdir(parents=True, exist_ok=True)
        phase_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        phase_checkpoint_path = phase_checkpoint_dir / f"{phase}_checkpoint_{backend}"
        local_outputs: dict[str, Any] | None = None
        verl_plan_path: str | None = None
        dataset_jsonl = _dataset_jsonl_from_manifest(dataset_manifest)
        if dry_run:
            sample_count = None
            warning = ""
            if backend == "verl":
                plan = _build_verl_dry_run_plan(
                    phase=phase,
                    phase_cfg=phase_cfg,
                    phase_dir=phase_dir,
                    phase_checkpoint_path=phase_checkpoint_path,
                    init_actor_checkpoint=init_actor_checkpoint,
                    dataset_manifest=dataset_manifest,
                    dataset_jsonl=dataset_jsonl,
                    resource_plan=resource_plan,
                )
                verl_plan_path = str(phase_log_dir / "verl_command_plan.json")
                write_json(verl_plan_path, plan)
                warning = "verl backend dry-run wrote executable SPAD offline DPO plan"
        else:
            sample_count = _count_dataset_samples(dataset_manifest)
            if backend == "smoke":
                phase_checkpoint_path.mkdir(parents=True, exist_ok=True)
                (phase_checkpoint_path / "README.txt").write_text(
                    f"SPAD-RAG Stage 3 {phase} smoke checkpoint placeholder. Real training requires backend=local_dpo or backend=verl after VERL wiring is implemented.\n",
                    encoding="utf-8",
                )
                warning = "smoke backend does not update actor parameters"
            elif phase == "dpo" and backend == "local_dpo":
                if not dataset_jsonl:
                    raise ValueError("Stage 3 local_dpo requires a Stage 2 dataset_jsonl")
                if not init_actor_checkpoint:
                    raise ValueError("Stage 3 local_dpo requires init_actor_checkpoint")
                from agentic_iter_rag.agent_training.spad.local_dpo import run_local_dpo

                local_outputs = run_local_dpo(
                    model_path=str(init_actor_checkpoint),
                    dataset_jsonl=dataset_jsonl,
                    output_dir=phase_checkpoint_path,
                    log_dir=phase_log_dir,
                    phase_cfg=phase_cfg,
                    resource_plan=resource_plan,
                )
                phase_checkpoint_path = Path(local_outputs["checkpoint"])
                sample_count = int(local_outputs.get("sample_count") or sample_count)
                warning = "local_dpo is a first-pass AIR-internal ablation trainer"
            elif phase == "dpo" and backend == "verl":
                if not dataset_jsonl:
                    raise ValueError("Stage 3 VERL DPO requires a Stage 2 dataset_jsonl")
                if not init_actor_checkpoint:
                    raise ValueError("Stage 3 VERL DPO requires init_actor_checkpoint")
                local_outputs = _run_verl_dpo(
                    phase_cfg=phase_cfg,
                    init_actor_checkpoint=init_actor_checkpoint,
                    dataset_jsonl=dataset_jsonl,
                    phase_checkpoint_path=phase_checkpoint_path,
                    phase_log_dir=phase_log_dir,
                    resource_plan=resource_plan,
                )
                phase_checkpoint_path = Path(local_outputs["checkpoint"])
                sample_count = int(local_outputs.get("sample_count") or sample_count)
                warning = "verl_spad_offline_dpo uses data parallel DDP with tensor_parallel_size=1 for Qwen3-1.7B"
            elif phase == "grpo" and backend == "verl":
                if not dataset_jsonl:
                    raise ValueError("Stage3 GRPO requires a Stage2 dataset_jsonl")
                if not init_actor_checkpoint:
                    raise ValueError("Stage3 GRPO requires init_actor_checkpoint")
                conversion = _convert_pairs_to_grpo_parquet(
                    dataset_jsonl,
                    phase_dir,
                    tokenizer_path=init_actor_checkpoint,
                    max_prompt_length=int(phase_cfg.get("max_prompt_length", 12000)),
                )
                sample_count = int(conversion["kept_count"])
                if sample_count < int(phase_cfg.get("train_batch_size", 64)):
                    raise RuntimeError(
                        f"Stage3 GRPO has {sample_count} kept pairs, fewer than formal batch 64"
                    )
                plan = _build_stage3_grpo_plan(
                    phase_cfg=phase_cfg,
                    init_actor_checkpoint=init_actor_checkpoint,
                    train_parquet=str(conversion["train_parquet"]),
                    val_parquet=str(conversion["val_parquet"]),
                    phase_dir=phase_dir,
                    phase_log_dir=phase_log_dir,
                    phase_checkpoint_dir=phase_checkpoint_path,
                    resource_plan=resource_plan,
                )
                command_path = phase_log_dir / "verl_command_plan.json"
                write_json(command_path, plan)
                script_path = _write_verl_script(
                    script_path=phase_log_dir / "run_spad_stage3_grpo.sh",
                    plan=plan,
                    actor_gpu_ids=plan["actor_gpu_ids"],
                )
                log_path = phase_log_dir / "verl_grpo_train.log"
                run_info = _run_shell_script(
                    script_path,
                    log_path,
                    timeout_s=phase_cfg.get("timeout_seconds"),
                )
                if int(run_info["return_code"]) != 0:
                    raise RuntimeError(
                        f"SPAD Stage3 GRPO exited with code {run_info['return_code']}; "
                        f"log={log_path}\n{tail_text(log_path)}"
                    )
                raw_checkpoint = _find_latest_checkpoint(Path(plan["output_dir"]))
                finalizer = finalize_actor_checkpoint(
                    raw_checkpoint,
                    hf_root=phase_checkpoint_path / "actor_model_hf",
                    log_dir=phase_log_dir,
                )
                phase_checkpoint_path = Path(finalizer["hf_actor_checkpoint"])
                local_outputs = {
                    "checkpoint": str(phase_checkpoint_path),
                    "raw_actor_checkpoint": str(raw_checkpoint),
                    "hf_actor_checkpoint": str(phase_checkpoint_path),
                    "checkpoint_finalizer": finalizer,
                    "dataset_conversion": conversion,
                    "command_path": str(command_path),
                    "log_path": str(log_path),
                    "run_info": run_info,
                    "world_size": len(plan["actor_gpu_ids"]),
                    "tensor_parallel_size": 1,
                    "backend_detail": "verl_grpo_gold_answer_f1",
                }
                warning = "Stage3 GRPO trains final-answer generation with Gold token F1"
            else:
                raise NotImplementedError(f"Stage 3 {phase} backend={backend} is not supported")
        outputs = {
            "status": "planned" if dry_run else "completed",
            "phase": phase,
            "backend": backend,
            "init_actor_checkpoint": init_actor_checkpoint,
            "dataset_manifest": dataset_manifest,
            "sample_count": sample_count,
            "checkpoint": str(phase_checkpoint_path),
            "runtime_dir": str(phase_log_dir),
            "resource_plan": resource_plan,
            "warning": warning,
        }
        if dataset_jsonl:
            outputs["dataset_jsonl"] = dataset_jsonl
        if verl_plan_path:
            outputs["verl_command_plan"] = verl_plan_path
        if local_outputs is not None:
            metrics_path = local_outputs.get("metrics_path")
            if metrics_path:
                outputs["metrics_path"] = metrics_path
            for key in (
                "command_path",
                "log_path",
                "world_size",
                "tensor_parallel_size",
                "backend_detail",
                "raw_actor_checkpoint",
                "hf_actor_checkpoint",
                "checkpoint_finalizer",
                "dataset_conversion",
                "run_info",
            ):
                if key in local_outputs:
                    outputs[key] = local_outputs[key]
        outputs["manifest"] = str(phase_dir / "manifest.json")
        write_sub_stage_manifest(outputs["manifest"], sub_stage=f"answer_distillation.{phase}", outputs=outputs)
        phase_outputs[phase] = outputs
        final_checkpoint = str(phase_checkpoint_path)
    outputs = {
        "status": "planned" if dry_run else "completed",
        "init_actor_checkpoint": init_actor_checkpoint,
        "dataset_manifest": dataset_manifest,
        "phase_outputs": phase_outputs,
        "checkpoint": final_checkpoint,
        "runtime_dir": str(log_dir),
        "resource_plan": resource_plan,
    }
    outputs["manifest"] = str(stage_dir / "manifest.json")
    write_sub_stage_manifest(outputs["manifest"], sub_stage="answer_distillation", outputs=outputs)
    return outputs
