#!/usr/bin/env python3
"""Build a lightweight BM25 index over a local Wikipedia JSONL corpus."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("/data01/ms_wksp/agent_up_to_date/Agentic_R_Learn/data/raw/lwhlwh__retrieval_corpus/wiki18_100w.jsonl"),
    )
    parser.add_argument("--out", type=Path, default=Path("data/retrieval/bm25_wiki18_20k.pkl"))
    parser.add_argument("--max-docs", type=int, default=20000)
    args = parser.parse_args()

    docs: list[dict] = []
    tokenized: list[list[str]] = []
    with args.corpus.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.max_docs:
                break
            item = json.loads(line)
            contents = item.get("contents") or item.get("text") or ""
            title = contents.split("\n", 1)[0].strip('" ') if contents else item.get("title", "")
            text = contents.split("\n", 1)[1] if "\n" in contents else contents
            doc = {"id": item.get("id", str(idx)), "title": title, "text": text, "contents": contents}
            docs.append(doc)
            tokenized.append(tokenize(contents))

    index = BM25Okapi(tokenized)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump({"docs": docs, "index": index}, f)
    print(f"indexed {len(docs):,} docs -> {args.out}")


if __name__ == "__main__":
    main()
