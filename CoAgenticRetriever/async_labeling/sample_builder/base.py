"""Base protocol for async-labeling sample builders."""

from __future__ import annotations

from typing import Protocol

from ..schemas import CandidateSignalData, ContrastiveSample


class CandidateSignalSampleBuilder(Protocol):
    def build(self, signals: list[CandidateSignalData]) -> list[ContrastiveSample]:
        ...
