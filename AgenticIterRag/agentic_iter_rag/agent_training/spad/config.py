"""Config helpers for SPAD-RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agentic_iter_rag.utils.io import deep_get


def resolve_ref(config: dict[str, Any], value: Any) -> Any:
    """Resolve a dotted config reference when value is a config path string."""

    if not isinstance(value, str):
        return value
    if "." not in value:
        return value
    resolved = deep_get(config, value, None)
    return value if resolved is None else resolved


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def selected_sub_stages(spad_cfg: dict[str, Any]) -> list[str]:
    """Apply SPAD resume/stop/skip controls to sub_stage_order."""

    stages = [str(item) for item in as_list(spad_cfg.get("sub_stage_order"))]
    resume_from = spad_cfg.get("resume_from_sub_stage")
    stop_after = spad_cfg.get("stop_after_sub_stage")
    skip = {str(item) for item in as_list(spad_cfg.get("skip_sub_stages"))}
    if resume_from:
        if resume_from not in stages:
            raise ValueError(f"agent_training.resume_from_sub_stage is not in sub_stage_order: {resume_from}")
        stages = stages[stages.index(str(resume_from)) :]
    if stop_after:
        if stop_after not in stages:
            raise ValueError(f"agent_training.stop_after_sub_stage is not in selected sub stages: {stop_after}")
        stages = stages[: stages.index(str(stop_after)) + 1]
    out: list[str] = []
    sub_stage_cfgs = spad_cfg.get("sub_stages", {})
    for stage in stages:
        if stage in skip:
            continue
        cfg = sub_stage_cfgs.get(stage, {})
        if isinstance(cfg, dict) and not bool(cfg.get("enabled", True)):
            continue
        out.append(stage)
    return out


def spad_runtime_root(config: dict[str, Any]) -> Path:
    """Return the run-local output root for SPAD-RAG train_agent artifacts."""

    artifact_root = config.get("runtime_compiled", {}).get("ARTIFACT_ROOT")
    if not artifact_root:
        raise ValueError("runtime_compiled.ARTIFACT_ROOT is required for SPAD-RAG")
    return Path(str(artifact_root)) / "stages" / "train_agent" / "spad_rag"


def input_train_files(config: dict[str, Any], spad_cfg: dict[str, Any]) -> list[str]:
    refs = spad_cfg.get("refs", {})
    value = resolve_ref(config, refs.get("train_files", "data.train_files"))
    return [str(item) for item in as_list(value)]


def input_val_files(config: dict[str, Any], spad_cfg: dict[str, Any]) -> list[str]:
    refs = spad_cfg.get("refs", {})
    value = resolve_ref(config, refs.get("val_files", "data.val_files"))
    return [str(item) for item in as_list(value)]


def init_actor_model(config: dict[str, Any], spad_cfg: dict[str, Any]) -> str:
    refs = spad_cfg.get("refs", {})
    return str(resolve_ref(config, refs.get("init_actor_model", "model.path")))
