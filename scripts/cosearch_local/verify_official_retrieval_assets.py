#!/usr/bin/env python3
"""Verify the Search-R1 retrieval assets and optional HTTP endpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib import request


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def read_head(path: Path, n_bytes: int = 512) -> bytes:
    with path.open("rb") as f:
        return f.read(n_bytes)


def read_first_line(path: Path) -> bytes:
    with path.open("rb") as f:
        return f.readline()


def check_corpus(corpus_path: Path) -> None:
    if not corpus_path.exists():
        fail(f"corpus not found: {corpus_path}")
    if corpus_path.is_symlink():
        fail(f"corpus must be the official Search-R1 JSONL file, not a symlink: {corpus_path}")

    head = read_head(corpus_path)
    if head.startswith(b"data00/") or b"ustar" in head[:512]:
        fail(
            f"corpus looks like a tar archive, not JSONL: {corpus_path}. "
            "Extract the inner wiki_dump.jsonl to wiki-18.jsonl."
        )

    first_line = read_first_line(corpus_path)
    try:
        row = json.loads(first_line.decode("utf-8"))
    except Exception as exc:
        fail(f"corpus first line is not valid JSON: {exc}")

    if not isinstance(row, dict) or "id" not in row or "contents" not in row:
        fail("corpus rows must contain Search-R1 fields: id and contents")


def check_index(index_path: Path, expected_ntotal: int | None, full: bool) -> int:
    if not index_path.exists():
        fail(f"index not found: {index_path}")
    if index_path.is_symlink():
        fail(f"index must be the official Search-R1 FAISS index, not a symlink: {index_path}")

    import faiss

    index = faiss.read_index(str(index_path))
    ntotal = int(index.ntotal)
    dim = int(index.d)
    if dim != 768:
        fail(f"expected e5-base-v2 dim=768, got dim={dim}")
    if expected_ntotal is not None and ntotal != expected_ntotal:
        fail(f"expected index ntotal={expected_ntotal}, got {ntotal}")
    if full and ntotal <= 1_000_000:
        fail(f"index ntotal={ntotal} is too small for official wiki-18; likely a smoke index")
    return ntotal


def count_lines(path: Path) -> int:
    total = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            total += chunk.count(b"\n")
    return total


def check_url(url: str, topk: int) -> None:
    payload = json.dumps(
        {"queries": ["capital of France"], "topk": topk, "return_scores": True}
    ).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    result = data.get("result")
    if not isinstance(result, list) or not result or not isinstance(result[0], list):
        fail("retrieval endpoint response is missing result[0]")
    if len(result[0]) != topk:
        fail(f"retrieval endpoint returned {len(result[0])} docs, expected {topk}")
    item = result[0][0]
    doc = item.get("document", item) if isinstance(item, dict) else {}
    if not isinstance(doc, dict) or "contents" not in doc:
        fail("retrieval endpoint document is missing contents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/retrieval/wiki-18/e5_Flat.index"))
    parser.add_argument("--corpus", type=Path, default=Path("data/retrieval/wiki-18/wiki-18.jsonl"))
    parser.add_argument("--expected-ntotal", type=int, default=int(os.getenv("SEARCH_R1_WIKI18_NTOTAL", "21015324")))
    parser.add_argument("--full", action="store_true", help="Also count corpus lines and compare with index ntotal.")
    parser.add_argument("--url", help="Optional /retrieve endpoint to validate.")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    check_corpus(args.corpus)
    ntotal = check_index(args.index, args.expected_ntotal, full=args.full)

    line_count = None
    if args.full:
        line_count = count_lines(args.corpus)
        if line_count != ntotal:
            fail(f"corpus line count {line_count} != FAISS index ntotal {ntotal}")

    if args.url:
        check_url(args.url, args.topk)

    print("Search-R1 retrieval verification passed.")
    print(f"  index:  {args.index} ntotal={ntotal}")
    print(f"  corpus: {args.corpus}" + (f" lines={line_count}" if line_count is not None else ""))
    if args.url:
        print(f"  url:    {args.url} topk={args.topk}")


if __name__ == "__main__":
    main()
