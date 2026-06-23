#!/usr/bin/env python3
"""Validate Search-R1 inference retrieval call shape without loading the LLM."""

from __future__ import annotations

import argparse
import requests


def passages_to_string(retrieval_result: list[dict]) -> str:
    formatted = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        formatted += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return formatted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8010/retrieve")
    parser.add_argument("--query", default="who got the first nobel prize in physics?")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--expect-contains", default="Röntgen")
    args = parser.parse_args()

    payload = {"queries": [args.query], "topk": args.topk, "return_scores": True}
    response = requests.post(args.url, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    results = data["result"]
    if not isinstance(results, list) or len(results) != 1:
        raise SystemExit("ERROR: expected result to be a one-item list")
    if len(results[0]) != args.topk:
        raise SystemExit(f"ERROR: expected {args.topk} docs, got {len(results[0])}")

    formatted = passages_to_string(results[0])
    if args.expect_contains and args.expect_contains.lower() not in formatted.lower():
        raise SystemExit(f"ERROR: expected substring not found: {args.expect_contains}")

    print("Search-R1 infer retrieval verification passed.")
    print(f"  url:     {args.url}")
    print(f"  query:   {args.query}")
    print(f"  topk:    {args.topk}")
    for i, item in enumerate(results[0][:3], 1):
        title = item["document"]["contents"].splitlines()[0]
        print(f"  doc{i}:   score={float(item['score']):.6f} title={title[:120]}")


if __name__ == "__main__":
    main()
