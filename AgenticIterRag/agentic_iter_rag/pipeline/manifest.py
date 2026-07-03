"""Manifest helpers for AgenticIterRag v1 stages."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_iter_rag.utils.io import write_json


def write_stage_manifest(path: str | Path, *, stage: str, config: dict[str, Any], outputs: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "type": "agentic_iter_rag_stage_manifest",
            "stage": stage,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": config,
            "outputs": outputs,
        },
    )

