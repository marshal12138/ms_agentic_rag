"""Orchestration: generate queries, retrieve, fuse across alphas, score positions.

A sample is collected only when it is *valid*:
  - the model emitted a tool call (otherwise the record is skipped), and
  - at least one alpha keeps an answer-containing doc inside the top-k window
    (if every alpha pushes the answer out, the record is invalid -- readme rule).

The pipeline stops once `num_samples` valid samples are collected.
"""

from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Dict, Iterator, List, Optional

from .config import Config
from .corpus_lookup import CorpusLookup, contains_answer
from .fusion import answer_positions, rrf_fuse
from .llm_client import LLMClient
from .retriever_client import RetrieverClient


def _alpha_key(alpha: float) -> str:
    return f"alpha_{alpha:g}"


def iter_dataset(path: str, max_scan: Optional[int]) -> Iterator[dict]:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if max_scan is not None and count >= max_scan:
                return
            count += 1
            yield json.loads(line)


class Pipeline:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config)
        self.retriever = RetrieverClient(config)
        self.corpus = CorpusLookup(config)

    def _process(self, record: dict) -> Optional[dict]:
        """Return a result row for a valid sample, else None (skipped/invalid)."""
        question = record.get("question", "")
        golden = record.get("golden_answers") or []
        if not question or not golden:
            return None

        query = self.llm.generate_query(question)
        if query is None:
            return None  # model did not call the tool -> skip

        result = self.retriever.retrieve([query])
        if not result:
            return None
        per_query = result[0]
        bm25_ids = list(per_query[0]) if len(per_query) > 0 else []
        dense_ids = list(per_query[1]) if len(per_query) > 1 else []
        if not bm25_ids and not dense_ids:
            return None

        all_ids = list(dict.fromkeys(list(bm25_ids) + list(dense_ids)))
        texts = self.corpus.texts_for(all_ids)
        answer_ids = {
            idx for idx in all_ids if contains_answer(texts.get(idx, ""), golden)
        }

        positions_by_alpha: Dict[str, List[int]] = {}
        any_hit = False
        for alpha in self.config.alphas:
            ranked = rrf_fuse(bm25_ids, dense_ids, alpha, self.config.rrf_k)
            positions = answer_positions(ranked, answer_ids, self.config.topk)
            positions_by_alpha[_alpha_key(alpha)] = positions
            if positions != [-1]:
                any_hit = True

        if not any_hit:
            return None  # answer pushed out under every alpha -> invalid sample

        row = dict(record)
        row["generated_query"] = query
        row.update(positions_by_alpha)
        return row

    def run(self, on_progress=None) -> List[dict]:
        """Stream records through the worker pool, stopping at num_samples valid rows.

        A bounded window of futures is kept in flight so we never materialize the
        whole dataset; submission stops as soon as enough valid samples land.
        """
        cfg = self.config
        collected: List[dict] = []
        scanned = 0
        records = iter_dataset(cfg.dataset_path, cfg.max_scan)
        window = max(1, cfg.workers * 2)

        with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
            pending = set()
            exhausted = False
            while True:
                while not exhausted and len(pending) < window:
                    try:
                        rec = next(records)
                    except StopIteration:
                        exhausted = True
                        break
                    pending.add(pool.submit(self._process, rec))

                if not pending:
                    break

                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    scanned += 1
                    try:
                        row = fut.result()
                    except Exception:
                        row = None
                    if row is not None:
                        collected.append(row)
                    if on_progress is not None:
                        on_progress(len(collected), scanned)

                if len(collected) >= cfg.num_samples:
                    for fut in pending:
                        fut.cancel()
                    break

        return collected[: cfg.num_samples]
