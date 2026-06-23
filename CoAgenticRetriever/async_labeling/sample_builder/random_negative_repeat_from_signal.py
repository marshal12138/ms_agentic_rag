"""Default async-labeling sample builder.

The strategy mirrors ranker_strategies/sample_builder/random_negative_repeat.py:
build 1 positive + N negative contrastive groups and repeat positives/negative
sampling until num_groups_per_step is reached.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field

from ..schemas import CandidateChunk, CandidateSignalData, ContrastiveSample, RankedPassage


def _chunk_to_ranked_passage(chunk: CandidateChunk) -> RankedPassage:
    if chunk.rank_rank is None:
        raise ValueError("missing required async sample chunk rank_rank")
    return RankedPassage(
        doc_id=chunk.doc_id,
        rank=int(chunk.rank_rank),
        title=chunk.title,
        text=chunk.text,
        recall_score=chunk.recall_score,
        metadata=chunk.metadata,
    )


@dataclass(slots=True)
class RandomNegativeRepeatFromSignalBuilder:
    num_groups_per_step: int
    neg_per_pos: int
    allow_repeat_negative_sampling: bool
    seed: int | None
    rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.num_groups_per_step = max(1, int(self.num_groups_per_step))
        self.neg_per_pos = max(1, int(self.neg_per_pos))
        self.rng = random.Random(self.seed)

    def build(self, signals: list[CandidateSignalData]) -> list[ContrastiveSample]:
        candidates: list[tuple[CandidateSignalData, CandidateChunk, list[CandidateChunk]]] = []
        for signal in signals:
            sorted_scores = sorted(signal.final_scores or signal.judge_scores, key=lambda item: item.judge_rank)
            if not sorted_scores:
                continue
            chunk_by_id = {chunk.doc_id: chunk for chunk in signal.ranked_chunk_list}
            positive_ids = [sorted_scores[0].doc_id]
            negative_ids = [score.doc_id for score in sorted_scores[1:] if score.doc_id in chunk_by_id]
            negatives = [chunk_by_id[doc_id] for doc_id in negative_ids]
            for doc_id in positive_ids:
                positive = chunk_by_id.get(doc_id)
                if positive is not None and negatives:
                    candidates.append((signal, positive, negatives))

        samples: list[ContrastiveSample] = []
        if not candidates:
            return samples

        idx = 0
        while len(samples) < self.num_groups_per_step:
            signal, positive, negatives = candidates[idx % len(candidates)]
            idx += 1
            sampled_negatives = self._sample_negatives(negatives)
            if len(sampled_negatives) < self.neg_per_pos:
                continue
            sample_id = f"{signal.signal_id}:{positive.doc_id}:{uuid.uuid4().hex[:8]}"
            query_input = f"{signal.origin_query} [SEP] {signal.sub_query}"
            ranked_docs = [_chunk_to_ranked_passage(chunk) for chunk in signal.ranked_chunk_list]
            samples.append(
                ContrastiveSample(
                    sample_id=sample_id,
                    query_input=query_input,
                    origin_query=signal.origin_query,
                    sub_query=signal.sub_query,
                    positive=_chunk_to_ranked_passage(positive),
                    negatives=[_chunk_to_ranked_passage(item) for item in sampled_negatives],
                    positive_doc_index=0,
                    label_source=signal.label_source,
                    trajectory_id=signal.trajectory_id,
                    tool_call_id=signal.tool_call_id,
                    turn_idx=int(signal.metadata["turn_idx"]),
                    loss_weight=max(float(signal.metadata["trajectory_score"]), 1.0),
                    sample_source="async_labeling",
                    recall_top50_docs=ranked_docs[:50],
                    rank_top50_docs=ranked_docs[:50],
                    rank_top5_docs=ranked_docs[:5],
                    metadata={
                        **dict(signal.metadata or {}),
                        "score_version": signal.score_version,
                        "prompt_version": signal.prompt_version,
                    },
                )
            )
            if not self.allow_repeat_negative_sampling and idx >= len(candidates):
                break
        return samples

    def _sample_negatives(self, negatives: list[CandidateChunk]) -> list[CandidateChunk]:
        if len(negatives) >= self.neg_per_pos:
            return self.rng.sample(negatives, self.neg_per_pos)
        if not self.allow_repeat_negative_sampling or not negatives:
            return []
        return [self.rng.choice(negatives) for _ in range(self.neg_per_pos)]
