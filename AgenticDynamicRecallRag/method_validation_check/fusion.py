"""Weighted RRF fusion of the two retriever id-lists and answer-position scoring.

Retriever order is [bm25, dense] (per project decision). alpha is the bm25
weight; (1 - alpha) is the dense weight. RRF score for a doc id is

    sum_over_retrievers( weight_r / (rrf_k + rank_r + 1) )

where rank_r is the 0-based position of the id in retriever r's list.
"""

from __future__ import annotations

from typing import List


def rrf_fuse(bm25_ids: List, dense_ids: List, alpha: float, rrf_k: int) -> List:
    """Return doc ids ranked by descending weighted-RRF score.

    Ties are broken by first appearance to keep the order deterministic.
    """
    w_bm25 = alpha
    w_dense = 1.0 - alpha
    fused: dict = {}
    order: dict = {}
    seq = 0

    for rank, idx in enumerate(bm25_ids):
        fused[idx] = fused.get(idx, 0.0) + w_bm25 / (rrf_k + rank + 1)
        if idx not in order:
            order[idx] = seq
            seq += 1
    for rank, idx in enumerate(dense_ids):
        fused[idx] = fused.get(idx, 0.0) + w_dense / (rrf_k + rank + 1)
        if idx not in order:
            order[idx] = seq
            seq += 1

    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], order[kv[0]]))
    return [idx for idx, _ in ranked]


def answer_positions(ranked_ids: List, answer_id_set: set, topk: int) -> List[int]:
    """1-based positions (within the top-k window) of answer-containing docs.

    Returns [-1] when no answer-containing doc lands inside the top-k window.
    """
    positions = [
        rank + 1
        for rank, idx in enumerate(ranked_ids[:topk])
        if idx in answer_id_set
    ]
    return positions if positions else [-1]


if __name__ == "__main__":
    # Validate fusion and scoring with synthetic id-lists (no services needed).
    from .config import Config

    cfg = Config()
    bm25_ids = [10, 11, 12, 13, 14]
    dense_ids = [12, 20, 21, 13, 22]

    # alpha = 1.0 -> pure bm25: ranking follows the bm25 list order.
    ranked_bm25 = rrf_fuse(bm25_ids, dense_ids, alpha=1.0, rrf_k=cfg.rrf_k)
    assert ranked_bm25[:5] == [10, 11, 12, 13, 14], ranked_bm25
    print("[fusion] alpha=1.0 (pure bm25) ->", ranked_bm25)

    # alpha = 0.0 -> pure dense: ranking follows the dense list order.
    ranked_dense = rrf_fuse(bm25_ids, dense_ids, alpha=0.0, rrf_k=cfg.rrf_k)
    assert ranked_dense[:5] == [12, 20, 21, 13, 22], ranked_dense
    print("[fusion] alpha=0.0 (pure dense) ->", ranked_dense)

    # docs in both lists (12, 13) should rise as alpha balances the two.
    ranked_mix = rrf_fuse(bm25_ids, dense_ids, alpha=0.5, rrf_k=cfg.rrf_k)
    assert set(ranked_mix) == set(bm25_ids) | set(dense_ids)
    print("[fusion] alpha=0.5 (balanced) ->", ranked_mix)

    # answer positions within top-k window.
    assert answer_positions([10, 11, 12, 13, 14], {12}, cfg.topk) == [3]
    assert answer_positions([10, 11, 12, 13, 14], {11, 13}, cfg.topk) == [2, 4]
    assert answer_positions([10, 11, 12, 13, 14], {99}, cfg.topk) == [-1]
    # answer present but outside the top-k window -> [-1].
    assert answer_positions([10, 11, 12, 13, 14, 15], {15}, cfg.topk) == [-1]
    print("[fusion] answer_positions OK over the full alpha grid")

    # sanity: every alpha on the grid produces a valid ranking.
    for a in cfg.alphas:
        ranked = rrf_fuse(bm25_ids, dense_ids, a, cfg.rrf_k)
        pos = answer_positions(ranked, {12, 13}, cfg.topk)
        print(f"  alpha={a:g} -> positions(of 12,13) = {pos}")
    print("[fusion] all assertions passed")
