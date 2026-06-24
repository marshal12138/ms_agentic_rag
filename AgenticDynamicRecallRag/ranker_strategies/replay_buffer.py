"""Replay buffer for ranker contrastive samples."""

from __future__ import annotations

import random
from collections import deque

from .schemas import ContrastiveSample


class RankerContrastiveReplayBuffer:
    def __init__(self, max_size: int = 200000, seed: int | None = None):
        self.max_size = max(1, int(max_size))
        self.rng = random.Random(seed)
        self._buffer: deque[ContrastiveSample] = deque(maxlen=self.max_size)
        self._fresh: list[ContrastiveSample] = []
        self._seen: set[tuple[str, str, str]] = set()

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, samples: list[ContrastiveSample], source_step: int) -> int:
        added = 0
        fresh: list[ContrastiveSample] = []
        for sample in samples:
            key = (str(source_step), sample.trajectory_id, sample.sample_id)
            if key in self._seen:
                continue
            self._seen.add(key)
            sample.metadata = {**sample.metadata, "source_step": source_step}
            sample.sample_source = "fresh"
            self._buffer.append(sample)
            fresh.append(sample)
            added += 1
        self._fresh = fresh
        return added

    def sample(self, batch_size: int, fresh_ratio: float = 0.5) -> list[ContrastiveSample]:
        batch_size = max(1, int(batch_size))
        if not self._buffer:
            return []

        fresh_count = min(len(self._fresh), int(round(batch_size * float(fresh_ratio))))
        historical_count = batch_size - fresh_count
        samples: list[ContrastiveSample] = []
        if fresh_count:
            samples.extend(self._sample_with_replacement(self._fresh, fresh_count, source="fresh"))
        if historical_count:
            samples.extend(self._sample_with_replacement(list(self._buffer), historical_count, source="replay"))
        self.rng.shuffle(samples)
        return samples

    def _sample_with_replacement(self, pool: list[ContrastiveSample], count: int, source: str) -> list[ContrastiveSample]:
        if not pool or count <= 0:
            return []
        out = []
        for _ in range(count):
            item = self.rng.choice(pool)
            item.sample_source = source
            out.append(item)
        return out
