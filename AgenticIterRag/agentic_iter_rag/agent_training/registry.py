"""Registry for AIR train_agent implementations."""

from __future__ import annotations

from typing import Any, Callable

from agentic_iter_rag.agent_training.spad.orchestrator import run_spad_rag


TrainAgentRunner = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]]


def get_train_agent_runner(impl: str) -> TrainAgentRunner:
    """Return the runner for a train_agent implementation name."""

    if impl == "spad_rag":
        return run_spad_rag
    raise ValueError(f"unsupported train_agent impl={impl!r}")
