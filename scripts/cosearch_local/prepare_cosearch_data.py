#!/usr/bin/env python3
"""Prepare CoSearch-style RL and evaluation parquet files from local data.

This follows the paper's training mixture:
Natural Questions 20,480; HotpotQA 14,220; MuSiQue 9,000; 2WikiMultiHopQA 7,500.

The output schema is compatible with CoSearch/VERL RLHFDataset and includes the
extra_info["question"] field required by CoSearchAgentLoop.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SEARCH_R1_PROMPT = """You are a tool-augmented research agent for wiki-based factoid question answering.

Your task is to answer questions drawn from Wikipedia-style datasets.
The final answer is evaluated using exact match (EM) or token-level F1, so it must be short and precise.

You have ONE tool available:
- search(query: string) -> returns a list of Wikipedia passages

For EVERY assistant turn, output EXACTLY TWO TAG BLOCKS in this order:
1) <reason> ... </reason>
2) EITHER <tool_call> ... </tool_call> OR <answer> ... </answer>

On the first assistant turn for every question, you MUST call search.
Do NOT output <answer> until after a tool result has been provided by the environment.
Do NOT output <tool_call> and <answer> in the same assistant turn.

When calling the tool, the <tool_call> block MUST contain ONLY this JSON shape:
<tool_call>
{{
  "name": "search",
  "arguments": {{
    "query": "<string>"
  }}
}}
</tool_call>

Inside <answer>, output ONLY the final short answer string.

Question: {question}
"""


TRAIN_COUNTS = {
    "nq": 20480,
    "hotpotqa": 14220,
    "musique": 9000,
    "2wikimultihopqa": 7500,
}

EVAL_FILES = {
    "nq": ["test.jsonl"],
    "triviaqa": ["test.jsonl"],
    "popqa": ["test.jsonl"],
    "hotpotqa": ["dev.jsonl"],
    "2wikimultihopqa": ["dev.jsonl"],
    "musique": ["dev.jsonl"],
    "bamboogle": ["test.jsonl"],
}


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_question_mark(question: str) -> str:
    question = " ".join(question.strip().split())
    if question and question[-1] not in "?.!":
        question += "?"
    return question


def normalize_answers(item: dict) -> list[str]:
    answers = item.get("golden_answers")
    if answers is None:
        answers = item.get("answers")
    if answers is None:
        answer = item.get("answer")
        answers = [answer] if answer is not None else []
    if isinstance(answers, str):
        answers = [answers]
    return [str(a) for a in answers if str(a).strip()]


def clean_json_value(value: Any) -> Any:
    """Keep nested raw metadata parquet-friendly and JSON-serializable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(v) for v in value]
    return str(value)


def to_cosearch_row(item: dict, data_source: str, split: str, idx: int) -> dict:
    question = ensure_question_mark(item["question"])
    answers = normalize_answers(item)
    metadata = clean_json_value(item.get("metadata"))
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": SEARCH_R1_PROMPT.format(question=question)}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": {"target": answers}},
        "extra_info": {
            "split": split,
            "index": idx,
            "question": question,
            "source_id": item.get("id", str(idx)),
            "metadata": metadata,
        },
    }


def load_split_rows(root: Path, data_source: str, filenames: list[str]) -> list[dict]:
    rows: list[dict] = []
    for filename in filenames:
        path = root / data_source / filename
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(read_jsonl(path))
    return rows


def sample_rows(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(rows) < n:
        raise ValueError(f"Need {n} rows, only found {len(rows)}")
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    return [rows[i] for i in idxs[:n]]


def write_parquet(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    print(f"wrote {len(rows):,} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/data01/ms_wksp/agent_up_to_date/Agentic_R_Learn/data/raw/hhjinjiajie__FlashRAG_Dataset"),
    )
    parser.add_argument("--out-root", type=Path, default=Path("data/co_search/local_flashrag"))
    parser.add_argument("--seed", type=int, default=26041755)
    parser.add_argument("--smoke-train-per-source", type=int, default=16)
    parser.add_argument("--smoke-eval-per-source", type=int, default=8)
    parser.add_argument("--max-eval-per-source", type=int, default=-1)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    train_rows: list[dict] = []
    smoke_train_rows: list[dict] = []
    manifest = {
        "raw_root": str(args.raw_root),
        "seed": args.seed,
        "paper_train_counts": TRAIN_COUNTS,
        "train_sources": {},
        "eval_sources": {},
    }

    for data_source, count in TRAIN_COUNTS.items():
        rows = load_split_rows(args.raw_root, data_source, ["train.jsonl"])
        sampled = sample_rows(rows, count, rng)
        train_rows.extend(to_cosearch_row(row, data_source, "train", i) for i, row in enumerate(sampled))
        smoke = sample_rows(rows, min(args.smoke_train_per_source, len(rows)), rng)
        smoke_train_rows.extend(to_cosearch_row(row, data_source, "train_smoke", i) for i, row in enumerate(smoke))
        manifest["train_sources"][data_source] = {"available": len(rows), "sampled": count}

    write_parquet(train_rows, args.out_root / "co_search_rl_51k.train.parquet")
    write_parquet(smoke_train_rows, args.out_root / "co_search_rl_smoke.train.parquet")

    eval_all_rows: list[dict] = []
    smoke_eval_rows: list[dict] = []
    eval_dir = args.out_root / "eval_by_dataset"
    for data_source, files in EVAL_FILES.items():
        rows = load_split_rows(args.raw_root, data_source, files)
        if args.max_eval_per_source > 0:
            rows = rows[: args.max_eval_per_source]
        converted = [to_cosearch_row(row, data_source, "test", i) for i, row in enumerate(rows)]
        eval_all_rows.extend(converted)
        smoke_eval_rows.extend(converted[: args.smoke_eval_per_source])
        write_parquet(converted, eval_dir / f"{data_source}.parquet")
        manifest["eval_sources"][data_source] = {"files": files, "rows": len(converted)}

    write_parquet(eval_all_rows, args.out_root / "co_search_7bench.eval.parquet")
    write_parquet(smoke_eval_rows, args.out_root / "co_search_7bench_smoke.eval.parquet")

    manifest_path = args.out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
