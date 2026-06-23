"""Metric counters for async labeling."""

from __future__ import annotations

from collections import Counter
from typing import Any


class AsyncLabelingMetrics:
    def __init__(self) -> None:
        self.counters: Counter[str] = Counter()

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] += value

    def to_dict(self) -> dict[str, Any]:
        return dict(self.counters)
