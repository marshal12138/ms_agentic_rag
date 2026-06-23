#!/usr/bin/env python3
"""Validate the Search-R1 /retrieve HTTP response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib import request


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def post_retrieve(url: str, query: str, topk: int) -> dict:
    payload = json.dumps(
        {"queries": [query], "topk": topk, "return_scores": True}
    ).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8010/retrieve")
    parser.add_argument("--query", default="who got the first nobel prize in physics?")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--expect-contains", default="Röntgen")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    data = post_retrieve(args.url, args.query, args.topk)
    result = data.get("result")
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], list):
        fail("response must contain result as a one-item list of retrieval results")
    docs = result[0]
    if len(docs) != args.topk:
        fail(f"expected {args.topk} docs, got {len(docs)}")

    contents_blob = []
    for i, item in enumerate(docs, 1):
        if not isinstance(item, dict):
            fail(f"result item {i} is not a dict")
        doc = item.get("document")
        if not isinstance(doc, dict):
            fail(f"result item {i} is missing document")
        contents = doc.get("contents")
        if not isinstance(contents, str) or not contents.strip():
            fail(f"document {i} is missing non-empty contents")
        if "score" not in item:
            fail(f"result item {i} is missing score")
        contents_blob.append(contents)

    if args.expect_contains and args.expect_contains.lower() not in "\n".join(contents_blob).lower():
        fail(f"expected substring not found in top-{args.topk}: {args.expect_contains}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.quiet:
        print("Dense retriever HTTP verification passed.")
        print(f"  url:   {args.url}")
        print(f"  query: {args.query}")
        print(f"  topk:  {args.topk}")
        for i, item in enumerate(docs[:3], 1):
            doc = item["document"]
            first_line = doc["contents"].splitlines()[0] if doc["contents"].splitlines() else ""
            print(f"  doc{i}: score={float(item['score']):.6f} title={first_line[:120]}")


if __name__ == "__main__":
    main()
