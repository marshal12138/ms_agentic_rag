"""Logging helpers for contrastive data construction."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .collator import serialize_sample_for_log
from .schemas import ContrastiveSample


logger = logging.getLogger(__name__)


class ContrastiveConstructionLogger:
    def __init__(self, every_n_steps: int, force_first: bool, jsonl_path: str | None):
        self.every_n_steps = max(1, int(every_n_steps))
        self.force_first = bool(force_first)
        self.jsonl_path = jsonl_path or os.getenv("COAGENTIC_RETRIEVER_CONSTRUCTION_JSONL", "")
        if self.jsonl_path:
            os.makedirs(os.path.dirname(self.jsonl_path), exist_ok=True)

    def should_log(self, global_steps: int, ranker_step_idx: int) -> bool:
        if self.force_first and global_steps <= 1 and ranker_step_idx == 0:
            return True
        return ranker_step_idx == 0 and global_steps % self.every_n_steps == 0

    def log(
        self,
        *,
        samples: list[ContrastiveSample],
        global_steps: int,
        ranker_step_idx: int,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        if not samples or not self.should_log(global_steps, ranker_step_idx):
            return
        payload = {
            "global_step": global_steps,
            "ranker_step_idx": ranker_step_idx,
            "metrics": metrics or {},
            "example": serialize_sample_for_log(samples[0]),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(f"[ranker-contrastive-construction]\n{text}", flush=True)
        logger.info("ranker contrastive construction example: %s", text)
        if self.jsonl_path:
            with open(self.jsonl_path, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
