"""Base types for async labeling stages."""

from __future__ import annotations

from typing import Protocol

from ..schemas import AsyncLabelRequest
from .llm_judge_rank50 import StageResult


class AsyncLabelStage(Protocol):
    def score(self, request: AsyncLabelRequest) -> StageResult:
        ...
