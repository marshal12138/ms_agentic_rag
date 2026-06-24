"""Convert async LLM-judge top-k signals into standard ranker labeled contexts."""

from __future__ import annotations

from async_labeling.schemas import CandidateChunk, CandidateSignalData
from ranker_strategies.schemas import LabeledPassage, LabeledRankingContext, RankedPassage


def _chunk_to_ranked_passage(chunk: CandidateChunk) -> RankedPassage:
    return RankedPassage(
        doc_id=chunk.doc_id,
        rank=int(chunk.rank_rank or chunk.recall_rank or 0),
        title=chunk.title,
        text=chunk.text,
        recall_score=chunk.recall_score,
        metadata={
            **dict(chunk.metadata or {}),
            "recall_rank": chunk.recall_rank,
            "rank_rank": chunk.rank_rank,
            "rank_score": chunk.rank_score,
        },
    )


def _chunk_to_labeled_passage(chunk: CandidateChunk, *, label: int, label_source: str) -> LabeledPassage:
    ranked = _chunk_to_ranked_passage(chunk)
    return LabeledPassage(
        doc_id=ranked.doc_id,
        rank=ranked.rank,
        title=ranked.title,
        text=ranked.text,
        recall_score=ranked.recall_score,
        metadata=ranked.metadata,
        label=label,
        label_source=label_source,
    )


class LLMJudgeTopKSignalBuilder:
    """Use the highest-ranked K judge results as positives and the rest as negatives."""

    def __init__(self, positive_top_k: int = 5, label_source: str | None = None):
        self.positive_top_k = max(1, int(positive_top_k))
        self.label_source = label_source or f"llm_judge_top{self.positive_top_k}"

    def build(self, signals: list[CandidateSignalData]) -> list[LabeledRankingContext]:
        contexts: list[LabeledRankingContext] = []
        for signal in signals:
            scores = sorted(signal.final_scores or signal.judge_scores, key=lambda item: item.judge_rank)
            if not scores:
                continue
            chunk_by_id = {chunk.doc_id: chunk for chunk in signal.ranked_chunk_list}
            judged_ids = [score.doc_id for score in scores if score.doc_id in chunk_by_id]
            positive_ids = {
                score.doc_id
                for score in scores
                if score.judge_rank <= self.positive_top_k and score.doc_id in chunk_by_id
            }
            if not positive_ids:
                continue

            passages: list[LabeledPassage] = []
            for doc_id in judged_ids:
                passages.append(
                    _chunk_to_labeled_passage(
                        chunk_by_id[doc_id],
                        label=1 if doc_id in positive_ids else 0,
                        label_source=self.label_source,
                    )
                )
            if not any(passage.label == 0 for passage in passages):
                continue

            ranked_docs = [_chunk_to_ranked_passage(chunk) for chunk in signal.ranked_chunk_list]
            contexts.append(
                LabeledRankingContext(
                    trajectory_id=signal.trajectory_id,
                    tool_call_id=signal.tool_call_id,
                    turn_idx=int(signal.metadata.get("turn_idx", 0) or 0),
                    origin_query=signal.origin_query,
                    sub_query=signal.sub_query,
                    trajectory_score=float(signal.metadata.get("trajectory_score", 1.0) or 1.0),
                    score_type=str(signal.metadata.get("score_type", "llm_judge_rank50")),
                    passages=passages,
                    golden_answers=[],
                    recall_top50_docs=ranked_docs[:50],
                    rank_top50_docs=ranked_docs[:50],
                    rank_top5_docs=ranked_docs[:5],
                    metadata={
                        **dict(signal.metadata or {}),
                        "score_version": signal.score_version,
                        "prompt_version": signal.prompt_version,
                        "async_signal_id": signal.signal_id,
                        "positive_top_k": self.positive_top_k,
                    },
                )
            )
        return contexts
