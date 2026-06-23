#!/usr/bin/env python3
"""Small deterministic checks for CoSearch paper mechanisms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from cosearch_core import average_hit_at_ks, composite_ranker_reward, semantic_group_indices  # noqa: E402


def main() -> None:
    docs = [
        {"contents": "Paris is the capital and largest city of France."},
        {"contents": "Berlin is the capital of Germany."},
        {"contents": "France won the match in Paris."},
        {"contents": "Tokyo is in Japan."},
        {"contents": "The French Republic is in Europe."},
    ]
    answers = ["Paris"]
    hit = average_hit_at_ks(answers, docs, (1, 3, 5))
    ranker_items = [
        {"question_id": 1, "sub_query": "capital of france", "answer_in_docs": True},
        {"question_id": 1, "sub_query": "what is france capital", "answer_in_docs": True},
        {"question_id": 1, "sub_query": "france capital city", "answer_in_docs": True},
        {"question_id": 1, "sub_query": "who directed jaws", "answer_in_docs": False},
    ]
    groups = semantic_group_indices(ranker_items, threshold=0.5, min_size=3)
    reward_low = composite_ranker_reward(0.3333, 1.0, True, True, cond_threshold=0.5)
    reward_high = composite_ranker_reward(1.0, 1.0, True, True, cond_threshold=0.5)
    result = {
        "average_hit_at_1_3_5": hit,
        "semantic_groups": groups,
        "reward_when_tool_below_threshold": reward_low,
        "reward_when_tool_above_threshold": reward_high,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
