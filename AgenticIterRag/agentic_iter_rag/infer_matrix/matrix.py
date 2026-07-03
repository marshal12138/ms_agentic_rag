"""Build a baseline matrix manifest for AgenticIterRag v1."""

from __future__ import annotations

from typing import Any


DEFAULT_BASELINES = (
    "origin_agent",
    "trained_agent",
    "trained_agent_original_llm_reranker",
    "trained_agent_trained_llm_reranker",
)


def build_infer_matrix(config: dict[str, Any]) -> list[dict[str, Any]]:
    baselines = config.get("baselines") or list(DEFAULT_BASELINES)
    return [{"name": str(name), "enabled": True} for name in baselines]

