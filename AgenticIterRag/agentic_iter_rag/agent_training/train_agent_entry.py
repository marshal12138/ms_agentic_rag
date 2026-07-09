"""Thin entry helpers for AIR train_agent stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.registry import get_train_agent_runner
from agentic_iter_rag.pipeline.manifest import write_stage_manifest
from agentic_iter_rag.utils.io import read_yaml, write_json


def resolve_impl_config(config: dict[str, Any], impl_config_ref: str) -> dict[str, Any]:
    """Resolve train_agent impl_config_ref from the compiled config."""

    impl_cfg = config.get(impl_config_ref)
    if not isinstance(impl_cfg, dict):
        raise ValueError(f"train_agent.impl_config_ref={impl_config_ref!r} does not point to a config mapping")
    return impl_cfg


def run_from_config(config_path: str | Path, manifest_path: str | Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Run the selected train_agent implementation from a compiled AIR config."""

    config_path = Path(config_path)
    manifest_path = Path(manifest_path)
    config = read_yaml(config_path)
    stage_cfg = config["pipeline"]["stage_configs"]["train_agent"]
    impl = str(stage_cfg.get("impl") or "placeholder")
    if impl == "placeholder":
        outputs = {
            "status": "placeholder",
            "entry": "AgenticIterRag/main_train_agent.py",
            "note": "train_agent.impl is placeholder; no agent training was executed.",
        }
        write_stage_manifest(manifest_path, stage="train_agent", config=stage_cfg, outputs=outputs)
        return outputs

    impl_cfg = resolve_impl_config(config, str(stage_cfg.get("impl_config_ref") or "agent_training"))
    runner = get_train_agent_runner(impl)
    outputs = runner(config, stage_cfg, impl_cfg, dry_run=dry_run)
    outputs.setdefault("status", "planned" if dry_run else "completed")
    outputs.setdefault("impl", impl)
    outputs.setdefault("entry", "AgenticIterRag/main_train_agent.py")
    outputs.setdefault("manifest", str(manifest_path))
    write_stage_manifest(manifest_path, stage="train_agent", config=stage_cfg, outputs=outputs)
    return outputs


def write_outputs_to_config(config: dict[str, Any], outputs: dict[str, Any]) -> None:
    """Update compiled config with train_agent outputs for downstream stages."""

    stage_outputs = config["pipeline"]["stage_configs"]["train_agent"].setdefault("outputs", {})
    stage_outputs.update(
        {
            "agent_checkpoint": outputs.get("agent_checkpoint"),
            "agent_training_manifest": outputs.get("agent_training_manifest"),
            "manifest": outputs.get("manifest"),
        }
    )
    if outputs.get("agent_checkpoint"):
        config.setdefault("infer_runtime", {}).setdefault("models", {})["trained_agent_model"] = outputs["agent_checkpoint"]
    final_json = config.get("runtime_compiled", {}).get("FINAL_CONFIG_JSON")
    if final_json:
        write_json(final_json, config)
