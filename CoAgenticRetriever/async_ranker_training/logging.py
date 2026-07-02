"""Observation logging helpers for async ranker training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils.jsonl import append_jsonl


class AsyncRankerTrainingLogger:
    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, record: dict[str, Any]) -> None:
        append_jsonl(self.log_dir / name, record)
