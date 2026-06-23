"""Supervision signal builders for ranker contrastive training."""

from __future__ import annotations

from ..schemas import LabeledPassage, LabeledRankingContext, ToolCallContext


class TopKPseudoRankSignalBuilder:
    """Label ranker top-k passages as positives."""

    def __init__(self, positive_top_k: int, allow_all_negative: bool):
        self.positive_top_k = max(1, int(positive_top_k))
        self.allow_all_negative = bool(allow_all_negative)

    def build(self, contexts: list[ToolCallContext]) -> list[LabeledRankingContext]:
        labeled_contexts: list[LabeledRankingContext] = []
        label_source = f"ranker_top{self.positive_top_k}_pseudo_rank"
        for context in contexts:
            passages = []
            for passage in context.ranked_passages:
                label = 1 if passage.rank <= self.positive_top_k else 0
                passages.append(
                    LabeledPassage(
                        doc_id=passage.doc_id,
                        rank=passage.rank,
                        title=passage.title,
                        text=passage.text,
                        recall_score=passage.recall_score,
                        metadata=passage.metadata,
                        label=label,
                        label_source=label_source,
                    )
                )
            if not self.allow_all_negative and not any(p.label == 1 for p in passages):
                continue
            labeled_contexts.append(
                LabeledRankingContext(
                    trajectory_id=context.trajectory_id,
                    tool_call_id=context.tool_call_id,
                    turn_idx=context.turn_idx,
                    origin_query=context.origin_query,
                    sub_query=context.sub_query,
                    trajectory_score=context.trajectory_score,
                    score_type=context.score_type,
                    passages=passages,
                    golden_answers=context.golden_answers,
                    recall_top50_docs=context.recall_top50_docs,
                    rank_top50_docs=context.rank_top50_docs or context.ranked_passages[:50],
                    rank_top5_docs=context.rank_top5_docs,
                    metadata=context.metadata,
                )
            )
        return labeled_contexts
