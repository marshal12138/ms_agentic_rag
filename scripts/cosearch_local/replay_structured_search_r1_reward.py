#!/usr/bin/env python3
"""Replay frozen Search-R1 rollouts with the structured answer reward."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import (
    compute_spad_search_policy_reward_details,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/search_r1_structured.train.parquet"
DEFAULT_ROLLOUT_DIR = ROOT / (
    "log/agenticIterRag/260710-113003-543853-pipeline-"
    "agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/outputs/"
    "stages/train_agent/spad_rag/search_policy_rl/rollout_data"
)
DEFAULT_OUTPUT = ROOT / "data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/legacy_rollout_replay"
QUESTION_RE = re.compile(r"Question: (.*?)\n(?:\nassistant|$)", re.S)
REWARD_CFG = {
    "type": "search_r1_structured",
    "search_r1_structured": {"score": 1.0, "format_score": 0.0},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data)
    by_question = {
        str(row.extra_info["question"]): {
            "ground_truth": row.reward_model["ground_truth"],
            "data_source": str(row.data_source),
            "source_id": str(row.extra_info["source_id"]),
        }
        for row in data.itertuples(index=False)
    }

    records: list[dict[str, Any]] = []
    question_scores: dict[str, list[float]] = defaultdict(list)
    for shard in sorted(args.rollout_dir.glob("*.jsonl"), key=lambda path: int(path.stem)):
        with shard.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                old = json.loads(line)
                match = QUESTION_RE.search(str(old.get("input") or ""))
                if not match:
                    raise ValueError(f"missing question in {shard}:{line_index + 1}")
                question = match.group(1).strip()
                item = by_question.get(question)
                if item is None:
                    raise KeyError(f"question is absent from structured data: {question}")
                result = compute_spad_search_policy_reward_details(
                    data_source=item["data_source"],
                    solution_str=str(old.get("output") or ""),
                    ground_truth=item["ground_truth"],
                    extra_info={"question": question, "tool_call_details": []},
                    reward_cfg=REWARD_CFG,
                )
                structured_score = float(result["score"])
                question_scores[question].append(structured_score)
                records.append(
                    {
                        "step": int(old.get("step") or int(shard.stem)),
                        "line_index": line_index,
                        "uid": str(old.get("uid") or ""),
                        "data_source": item["data_source"],
                        "source_id": item["source_id"],
                        "question": question,
                        "legacy_score": float(old.get("score") or 0.0),
                        **result,
                    }
                )

    replay_path = args.output_dir / "replay.jsonl"
    with replay_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    nonconstant_groups = sum(len(set(scores)) > 1 for scores in question_scores.values())
    all_zero_groups = sum(max(scores, default=0.0) == 0.0 for scores in question_scores.values())
    all_one_groups = sum(min(scores, default=0.0) == 1.0 for scores in question_scores.values())
    label_counts = Counter(str(record.get("answer_semantics") or "") for record in records)
    summary = {
        "version": "search-r1-structured-answer-v1",
        "data_path": str(args.data),
        "data_sha256": sha256_file(args.data),
        "rollout_dir": str(args.rollout_dir),
        "rollout_count": len(records),
        "group_count": len(question_scores),
        "legacy_positive_count": sum(record["legacy_score"] >= 1.0 for record in records),
        "structured_positive_count": sum(float(record["score"]) >= 1.0 for record in records),
        "legacy_positive_structured_zero_count": sum(
            record["legacy_score"] >= 1.0 and float(record["score"]) == 0.0 for record in records
        ),
        "structured_nonconstant_group_count": nonconstant_groups,
        "structured_all_zero_group_count": all_zero_groups,
        "structured_all_one_group_count": all_one_groups,
        "rollout_answer_semantics_counts": dict(sorted(label_counts.items())),
        "replay_path": str(replay_path),
        "replay_sha256": sha256_file(replay_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
