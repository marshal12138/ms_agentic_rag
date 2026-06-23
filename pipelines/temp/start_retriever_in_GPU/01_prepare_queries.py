#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def question_from_row(row: dict) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("question"):
        return str(extra["question"]).strip()

    prompt = row.get("prompt")
    if prompt is not None:
        try:
            first = prompt[0]
            content = first.get("content", "") if isinstance(first, dict) else str(first)
            match = re.search(r"Question:\s*(.+?)\s*$", content, re.S)
            if match:
                return match.group(1).strip()
        except Exception:
            pass
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    rows = df.to_dict("records")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    with args.output.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(rows):
            query = question_from_row(row)
            if not query or query in seen:
                continue
            seen.add(query)
            f.write(json.dumps({"id": idx, "query": query}, ensure_ascii=False) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break

    print(json.dumps({"input_rows": len(rows), "queries_written": written, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
