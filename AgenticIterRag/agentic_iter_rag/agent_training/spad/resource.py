"""SPAD-RAG resource resolution."""

from __future__ import annotations

from typing import Any


def resolve_sub_stage_resource(config: dict[str, Any], sub_stage: str) -> dict[str, Any]:
    """Return resource plan for one SPAD sub-stage."""

    train_agent_resource = config["resource"]["stage_resources"]["train_agent"]
    spad_resource = train_agent_resource.get("impls", {}).get("spad_rag", {})
    sub_stages = spad_resource.get("sub_stages", {})
    plan = sub_stages.get(sub_stage)
    if not isinstance(plan, dict):
        raise ValueError(f"resource.stage_resources.train_agent.impls.spad_rag.sub_stages.{sub_stage} must be set")
    return plan
