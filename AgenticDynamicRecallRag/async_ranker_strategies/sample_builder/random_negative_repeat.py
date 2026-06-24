"""Contrastive sample builders for async ranker labeling."""

from __future__ import annotations

import random
import uuid

from ranker_strategies.schemas import ContrastiveSample, LabeledPassage, LabeledRankingContext, RankedPassage


def _as_retrieved(passage: LabeledPassage) -> RankedPassage:
    return RankedPassage(
        doc_id=passage.doc_id,
        rank=passage.rank,
        title=passage.title,
        text=passage.text,
        recall_score=passage.recall_score,
        metadata=passage.metadata,
    )


class RandomNegativeRepeatSampleBuilder:
    """Build 1 positive + N negatives groups, repeating positives if needed."""

    def __init__(
        self,
        num_groups_per_step: int = 32,
        neg_per_pos: int = 15,
        allow_repeat_negative_sampling: bool = True,
        seed: int | None = None,
    ):
        self.num_groups_per_step = max(1, int(num_groups_per_step))
        self.neg_per_pos = max(1, int(neg_per_pos))
        self.allow_repeat_negative_sampling = bool(allow_repeat_negative_sampling)
        self.rng = random.Random(seed)

    def build(self, labeled_contexts: list[LabeledRankingContext]) -> list[ContrastiveSample]:
        candidates: list[tuple[LabeledRankingContext, LabeledPassage, list[LabeledPassage]]] = []
        for context in labeled_contexts:
            positives = [passage for passage in context.passages if passage.label == 1]
            negatives = [passage for passage in context.passages if passage.label == 0]
            if not positives or not negatives:
                continue
            for positive in positives:
                candidates.append((context, positive, negatives))

        if not candidates:
            return []

        samples: list[ContrastiveSample] = []
        idx = 0
        while len(samples) < self.num_groups_per_step:
            context, positive, negatives = candidates[idx % len(candidates)]
            idx += 1
            sampled_negatives = self._sample_negatives(negatives)
            if len(sampled_negatives) < self.neg_per_pos:
                continue
            label_source = positive.label_source
            sample_id = (
                f"{context.trajectory_id}:{context.tool_call_id}:"
                f"{positive.doc_id}:{uuid.uuid4().hex[:8]}"
            )
            query_input = f"{context.origin_query} [SEP] {context.sub_query}"
            samples.append(
                ContrastiveSample(
                    sample_id=sample_id,
                    query_input=query_input,
                    origin_query=context.origin_query,
                    sub_query=context.sub_query,
                    positive=_as_retrieved(positive),
                    negatives=[_as_retrieved(item) for item in sampled_negatives],
                    positive_doc_index=0,
                    label_source=label_source,
                    trajectory_id=context.trajectory_id,
                    tool_call_id=context.tool_call_id,
                    recall_top50_docs=context.recall_top50_docs[:50],
                    rank_top50_docs=context.rank_top50_docs[:50],
                    rank_top5_docs=context.rank_top5_docs[:5],
                    turn_idx=context.turn_idx,
                    loss_weight=max(float(context.trajectory_score), 1.0),
                    sample_source="fresh",
                    metadata={
                        **dict(context.metadata or {}),
                        "score_type": context.score_type,
                    },
                )
            )

            if not self.allow_repeat_negative_sampling and idx >= len(candidates):
                break

        return samples

    def _sample_negatives(self, negatives: list[LabeledPassage]) -> list[LabeledPassage]:
        if len(negatives) >= self.neg_per_pos:
            return self.rng.sample(negatives, self.neg_per_pos)
        if not self.allow_repeat_negative_sampling:
            return []
        return [self.rng.choice(negatives) for _ in range(self.neg_per_pos)]
