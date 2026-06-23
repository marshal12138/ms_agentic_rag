#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_first_rollout_trajectory(rollout_dir: Path) -> dict:
    samples = []
    for path in sorted(rollout_dir.glob("main/*.jsonl")):
        samples.extend(read_jsonl(path))
    if not samples:
        return {}
    uid = samples[0].get("uid")
    if uid:
        samples = [sample for sample in samples if sample.get("uid") == uid]
    return {
        "uid": uid,
        "steps": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-llm-io", type=Path, required=True)
    parser.add_argument("--train-rollout-dir", type=Path, required=True)
    parser.add_argument("--train-search-timing", type=Path, required=True)
    parser.add_argument("--eval-llm-io", type=Path, required=True)
    parser.add_argument("--eval-traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    train_llm = read_jsonl(args.train_llm_io)
    train_search = read_jsonl(args.train_search_timing)
    eval_llm = read_jsonl(args.eval_llm_io)
    eval_traces = read_jsonl(args.eval_traces)

    payload = {
        "train": {
            "llm_io_path": str(args.train_llm_io),
            "search_timing_path": str(args.train_search_timing),
            "first_agent_records": train_llm[:4],
            "first_search_records": train_search[:4],
            "rollout_trajectory": load_first_rollout_trajectory(args.train_rollout_dir),
        },
        "eval": {
            "llm_io_path": str(args.eval_llm_io),
            "traces_path": str(args.eval_traces),
            "first_agent_records": eval_llm[:4],
            "first_trace": eval_traces[0] if eval_traces else {},
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
