"""Stage 3 answer distillation for SPAD-RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.local_dpo import run_local_dpo
from agentic_iter_rag.agent_training.spad.manifest import write_sub_stage_manifest
from agentic_iter_rag.utils.io import iter_jsonl


def _dataset_jsonl_from_manifest(dataset_manifest: str | None) -> str | None:
    if not dataset_manifest:
        return None
    dataset_path = Path(dataset_manifest)
    if dataset_path.suffix == ".jsonl":
        return str(dataset_path)
    import json

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return payload.get("outputs", {}).get("dataset_jsonl") or payload.get("dataset_jsonl")


def _count_dataset_samples(dataset_manifest: str | None) -> int:
    dataset_jsonl = _dataset_jsonl_from_manifest(dataset_manifest)
    if not dataset_jsonl:
        return 0
    return sum(1 for _ in iter_jsonl(dataset_jsonl))


def run_answer_distillation(
    *,
    spad_cfg: dict[str, Any],
    stage_dir: Path,
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
    phase_outputs: dict[str, Any] = {}
    final_checkpoint = init_actor_checkpoint
    phase_order = [str(item) for item in sub_cfg.get("phase_order", ["sft", "dpo"])]
    for phase in phase_order:
        phase_cfg = sub_cfg.get("phases", {}).get(phase, {})
        if not isinstance(phase_cfg, dict) or not bool(phase_cfg.get("enabled", True)):
            continue
        backend = str(phase_cfg.get("backend") or "smoke")
        phase_dir = stage_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = phase_dir / f"{phase}_checkpoint_{backend}"
        if dry_run:
            sample_count = None
            warning = ""
        else:
            sample_count = _count_dataset_samples(dataset_manifest)
            if backend == "smoke":
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / "README.txt").write_text(
                    f"SPAD-RAG Stage 3 {phase} smoke checkpoint placeholder. Real training requires backend=verl.\n",
                    encoding="utf-8",
                )
                warning = "smoke backend does not update actor parameters"
            elif phase == "dpo" and backend == "local_dpo":
                dataset_jsonl = _dataset_jsonl_from_manifest(dataset_manifest)
                if not dataset_jsonl:
                    raise ValueError("Stage 3 local_dpo requires a Stage 2 dataset_jsonl")
                if not init_actor_checkpoint:
                    raise ValueError("Stage 3 local_dpo requires init_actor_checkpoint")
                local_outputs = run_local_dpo(
                    model_path=str(init_actor_checkpoint),
                    dataset_jsonl=dataset_jsonl,
                    output_dir=checkpoint_dir,
                    phase_cfg=phase_cfg,
                    resource_plan=resource_plan,
                )
                checkpoint_dir = Path(local_outputs["checkpoint"])
                sample_count = int(local_outputs.get("sample_count") or sample_count)
                warning = "local_dpo is a first-pass AIR-internal ablation trainer"
            else:
                raise NotImplementedError(f"Stage 3 {phase} backend={backend} is not supported")
        outputs = {
            "status": "planned" if dry_run else "completed",
            "phase": phase,
            "backend": backend,
            "init_actor_checkpoint": init_actor_checkpoint,
            "dataset_manifest": dataset_manifest,
            "sample_count": sample_count,
            "checkpoint": str(checkpoint_dir),
            "resource_plan": resource_plan,
            "warning": warning,
        }
        outputs["manifest"] = str(phase_dir / "manifest.json")
        write_sub_stage_manifest(outputs["manifest"], sub_stage=f"answer_distillation.{phase}", outputs=outputs)
        phase_outputs[phase] = outputs
        final_checkpoint = str(checkpoint_dir)
    outputs = {
        "status": "planned" if dry_run else "completed",
        "init_actor_checkpoint": init_actor_checkpoint,
        "dataset_manifest": dataset_manifest,
        "phase_outputs": phase_outputs,
        "checkpoint": final_checkpoint,
        "resource_plan": resource_plan,
    }
    outputs["manifest"] = str(stage_dir / "manifest.json")
    write_sub_stage_manifest(outputs["manifest"], sub_stage="answer_distillation", outputs=outputs)
    return outputs
