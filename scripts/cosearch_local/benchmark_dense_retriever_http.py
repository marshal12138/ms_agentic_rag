#!/usr/bin/env python3
"""Benchmark the Search-R1 /retrieve HTTP endpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import time
from pathlib import Path
from urllib import request


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
    with request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def validate_response(data: dict, topk: int) -> list[dict]:
    result = data.get("result")
    if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], list):
        raise SystemExit("ERROR: response must contain result as a one-item list")
    docs = result[0]
    if len(docs) != topk:
        raise SystemExit(f"ERROR: expected {topk} docs, got {len(docs)}")
    for i, item in enumerate(docs, 1):
        if not isinstance(item, dict) or "score" not in item:
            raise SystemExit(f"ERROR: result item {i} is missing score")
        doc = item.get("document")
        if not isinstance(doc, dict) or not isinstance(doc.get("contents"), str):
            raise SystemExit(f"ERROR: result item {i} is missing document contents")
    return docs


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8010/retrieve")
    parser.add_argument("--query", default="who got the first nobel prize in physics?")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    first_docs: list[dict] | None = None
    for _ in range(args.warmup):
        first_docs = validate_response(post_retrieve(args.url, args.query, args.topk), args.topk)

    latencies = []

    def timed_request() -> tuple[float, list[dict]]:
        start = time.perf_counter()
        data = post_retrieve(args.url, args.query, args.topk)
        elapsed = time.perf_counter() - start
        return elapsed, validate_response(data, args.topk)

    wall_start = time.perf_counter()
    if args.concurrency <= 1:
        for _ in range(args.requests):
            elapsed, first_docs = timed_request()
            latencies.append(elapsed)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(timed_request) for _ in range(args.requests)]
            for future in as_completed(futures):
                elapsed, first_docs = future.result()
                latencies.append(elapsed)
    wall_seconds = time.perf_counter() - wall_start

    summary = {
        "url": args.url,
        "query": args.query,
        "topk": args.topk,
        "warmup": args.warmup,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "latency_seconds": latencies,
        "wall_seconds": wall_seconds,
        "avg_seconds": statistics.fmean(latencies) if latencies else 0.0,
        "median_seconds": statistics.median(latencies) if latencies else 0.0,
        "min_seconds": min(latencies) if latencies else 0.0,
        "max_seconds": max(latencies) if latencies else 0.0,
        "p90_seconds": percentile(latencies, 0.90),
        "qps": (len(latencies) / wall_seconds) if latencies and wall_seconds > 0 else 0.0,
        "top_titles": [
            item["document"]["contents"].splitlines()[0]
            for item in (first_docs or [])[:3]
        ],
        "top_scores": [
            float(item["score"])
            for item in (first_docs or [])[:3]
        ],
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Dense retriever HTTP benchmark passed.")
    print(f"  url:      {args.url}")
    print(f"  query:    {args.query}")
    print(f"  topk:     {args.topk}")
    print(f"  requests: {args.requests} (+{args.warmup} warmup)")
    print(f"  concur.:  {args.concurrency}")
    print(f"  wall:     {summary['wall_seconds']:.3f}s")
    print(
        "  latency:  "
        f"avg={summary['avg_seconds']:.3f}s "
        f"median={summary['median_seconds']:.3f}s "
        f"min={summary['min_seconds']:.3f}s "
        f"max={summary['max_seconds']:.3f}s "
        f"p90={summary['p90_seconds']:.3f}s"
    )
    print(f"  qps:      {summary['qps']:.4f}")
    for i, (title, score) in enumerate(zip(summary["top_titles"], summary["top_scores"]), 1):
        print(f"  doc{i}:    score={score:.6f} title={title[:120]}")


if __name__ == "__main__":
    main()
