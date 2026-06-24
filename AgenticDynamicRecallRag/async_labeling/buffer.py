"""Destructive completed signal queue for async labeling."""

from __future__ import annotations

from collections import deque
from threading import Condition
from time import monotonic

from .schemas import CandidateSignalData


class CompletedSignalBuffer:
    def __init__(self, max_size: int = 4096, drop_policy: str = "drop_oldest"):
        self.max_size = max(1, int(max_size))
        self.drop_policy = drop_policy
        self._items: deque[CandidateSignalData] = deque()
        self._condition = Condition()
        self.dropped_count = 0

    def __len__(self) -> int:
        with self._condition:
            return len(self._items)

    def push(self, signal: CandidateSignalData) -> None:
        with self._condition:
            if len(self._items) >= self.max_size:
                if self.drop_policy == "drop_newest":
                    self.dropped_count += 1
                    return
                self._items.popleft()
                self.dropped_count += 1
            self._items.append(signal)
            self._condition.notify_all()

    def pop_latest(self, n: int = 1, wait: bool = True, timeout: float | None = None) -> list[CandidateSignalData]:
        n = max(1, int(n))
        deadline = None if timeout is None else monotonic() + float(timeout)
        with self._condition:
            while len(self._items) < n:
                if not wait:
                    return []
                if deadline is not None:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        return []
                    self._condition.wait(remaining)
                    continue
                self._condition.wait()

            items: list[CandidateSignalData] = []
            for _ in range(n):
                items.append(self._items.pop())
            items.reverse()
            return items
