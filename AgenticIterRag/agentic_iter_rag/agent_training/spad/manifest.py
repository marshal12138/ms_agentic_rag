"""Manifest helpers for SPAD-RAG."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_iter_rag.utils.io import write_json, write_jsonl


CORE_ROLLOUT_AUDIT_FIELDS = (
    "input",
    "output",
    "gts",
    "raw_prompt",
    "assistant_turn_records",
)
SEMI_STRICT_INVALID_ROLLOUT_RATE = 0.005


def is_invalid_rollout_record(record: dict[str, Any]) -> bool:
    return any(not bool(record.get(field)) for field in CORE_ROLLOUT_AUDIT_FIELDS)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_sub_stage_manifest(path: str | Path, *, sub_stage: str, outputs: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "type": "spad_rag_sub_stage_manifest",
        "sub_stage": sub_stage,
        "created_at": utc_now(),
        "outputs": outputs,
    }
    write_json(path, payload)
    return payload


def write_spad_manifest(path: str | Path, *, outputs: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "type": "spad_rag_train_agent_manifest",
        "impl": "spad_rag",
        "created_at": utc_now(),
        **outputs,
    }
    write_json(path, payload)
    return payload


def write_records(path: str | Path, records: list[dict[str, Any]]) -> int:
    return write_jsonl(path, records)
