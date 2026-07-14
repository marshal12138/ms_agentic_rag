"""SPAD-RAG internal orchestrator for AIR train_agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.answer_distillation import run_answer_distillation
from agentic_iter_rag.agent_training.spad.config import (
    selected_sub_stages,
    spad_checkpoint_root,
    spad_runtime_log_root,
    spad_runtime_root,
)
from agentic_iter_rag.agent_training.spad.manifest import write_spad_manifest
from agentic_iter_rag.agent_training.spad.refresh_rollout import run_answer_refresh_data
from agentic_iter_rag.agent_training.spad.resource import resolve_sub_stage_resource
from agentic_iter_rag.agent_training.spad.search_policy_rl import run_search_policy_rl


def run_spad_rag(
    config: dict[str, Any],
    train_agent_stage_cfg: dict[str, Any],
    spad_cfg: dict[str, Any],
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run SPAD-RAG sub-stages under AIR train_agent."""

    del train_agent_stage_cfg
    root = spad_runtime_root(config)
    log_root = spad_runtime_log_root(config)
    checkpoint_root = spad_checkpoint_root(config)
    root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    selected = selected_sub_stages(spad_cfg)
    sub_stage_outputs: dict[str, Any] = {}
    search_policy_checkpoint: str | None = None
    answer_actor_checkpoint: str | None = None
    refresh_dataset_manifest: str | None = None
    final_checkpoint: str | None = None

    for sub_stage in selected:
        resource_plan = resolve_sub_stage_resource(config, sub_stage)
        stage_dir = root / sub_stage
        stage_log_dir = log_root / sub_stage
        stage_checkpoint_dir = checkpoint_root / sub_stage
        if sub_stage == "search_policy_rl":
            outputs = run_search_policy_rl(
                config=config,
                spad_cfg=spad_cfg,
                stage_dir=stage_dir,
                log_dir=stage_log_dir,
                checkpoint_dir=stage_checkpoint_dir,
                resource_plan=resource_plan,
                dry_run=dry_run,
            )
            search_policy_checkpoint = outputs.get("actor_checkpoint")
            answer_actor_checkpoint = outputs.get("hf_actor_checkpoint")
            if not dry_run and not answer_actor_checkpoint:
                raise RuntimeError("SPAD Stage1 completed without hf_actor_checkpoint")
            final_checkpoint = answer_actor_checkpoint or search_policy_checkpoint
        elif sub_stage == "answer_refresh_data":
            outputs = run_answer_refresh_data(
                config=config,
                spad_cfg=spad_cfg,
                stage_dir=stage_dir,
                log_dir=stage_log_dir,
                checkpoint_dir=stage_checkpoint_dir,
                resource_plan=resource_plan,
                dry_run=dry_run,
                actor_checkpoint=search_policy_checkpoint,
            )
            refresh_dataset_manifest = outputs.get("dataset_manifest")
            answer_actor_checkpoint = outputs.get("hf_actor_checkpoint") or answer_actor_checkpoint
        elif sub_stage == "answer_distillation":
            outputs = run_answer_distillation(
                spad_cfg=spad_cfg,
                stage_dir=stage_dir,
                log_dir=stage_log_dir,
                checkpoint_dir=stage_checkpoint_dir,
                resource_plan=resource_plan,
                dry_run=dry_run,
                init_actor_checkpoint=answer_actor_checkpoint,
                dataset_manifest=refresh_dataset_manifest,
            )
            if outputs.get("checkpoint"):
                final_checkpoint = outputs["checkpoint"]
        else:
            raise ValueError(f"unsupported SPAD sub-stage: {sub_stage}")
        sub_stage_outputs[sub_stage] = outputs

    spad_manifest_path = root / "spad_manifest.json"
    manifest = write_spad_manifest(
        spad_manifest_path,
        outputs={
            "status": "planned" if dry_run else "completed",
            "selected_sub_stages": selected,
            "sub_stage_outputs": sub_stage_outputs,
            "final_agent_checkpoint": final_checkpoint,
        },
    )
    return {
        "status": "planned" if dry_run else "completed",
        "impl": "spad_rag",
        "selected_sub_stages": selected,
        "sub_stage_outputs": sub_stage_outputs,
        "agent_checkpoint": final_checkpoint,
        "agent_training_manifest": str(spad_manifest_path),
        "spad_manifest": manifest,
    }
