#!/usr/bin/env python3
"""Validate that the CoAgenticRetriever recall endpoint is reachable and sane."""

from __future__ import annotations

import argparse
import json
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="CoAgenticRetriever")
    parser.add_argument("--url", default="http://127.0.0.1:8010/retrieve")
    parser.add_argument("--query", default="who got the first nobel prize in physics?")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--expect-contains", default="Röntgen")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.top_m < 1:
        raise SystemExit(f"ERROR: --top-m must be a positive integer; got {args.top_m}")
    if args.top_n < 1:
        raise SystemExit(f"ERROR: --top-n must be a positive integer; got {args.top_n}")
    if args.top_m > args.top_n:
        raise SystemExit(f"ERROR: --top-m {args.top_m} exceeds --top-n {args.top_n}")
    if args.top_m > 5:
        raise SystemExit(
            "ERROR: --top-m exceeds current reward preflight limit of 5 visible documents; "
            "use agent-visible TOP_M here, not ranker.top_k/RANK_TOP_K."
        )

    payload = json.dumps(
        {
            "queries": [args.query],
            "topk": args.top_n,
            "return_scores": True,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    raw_candidates = (data.get("result") or [[]])[0]
    documents = []
    for idx, item in enumerate(raw_candidates, start=1):
        doc = item.get("document", item) if isinstance(item, dict) else {}
        score = item.get("score", doc.get("score", 0.0)) if isinstance(item, dict) else 0.0
        documents.append(
            {
                "rank": idx,
                "id": str(doc.get("id", "")),
                "title": str(doc.get("title", "")),
                "contents": str(doc.get("contents") or doc.get("text") or doc.get("passage") or ""),
                "score": float(score or 0.0),
            }
        )
    text = "\n".join(f"{doc['title']}\n{doc['contents']}" for doc in documents[: args.top_m])
    if args.expect_contains and args.expect_contains.lower() not in text.lower():
        raise SystemExit(f"ERROR: expected substring not found in recall top-{args.top_m}: {args.expect_contains}")
    if len(documents) != args.top_n:
        raise SystemExit(f"ERROR: expected {args.top_n} recall docs, got {len(documents)}")

    print("CoAgentic retrieval verification passed.")
    print(f"  url:      {args.url}")
    print(f"  query:    {args.query}")
    print(f"  top_n:    {args.top_n}")
    print(f"  top_m:    {args.top_m}")
    print("  metrics:  " + json.dumps({"num_recall_docs": len(documents)}, ensure_ascii=False, sort_keys=True))
    print("  preview:  " + (text.splitlines()[0] if text.splitlines() else "")[:160])


if __name__ == "__main__":
    main()
