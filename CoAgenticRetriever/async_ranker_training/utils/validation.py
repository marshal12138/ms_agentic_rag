"""Validation helpers for async ranker training."""

from __future__ import annotations

from ..schemas import AsyncLabelRequest


def validate_rank50_request(request: AsyncLabelRequest) -> None:
    request.validate_rank50()
