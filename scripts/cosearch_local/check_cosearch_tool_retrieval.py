#!/usr/bin/env python3
"""Validate that CoSearchTool talks to the configured dense retrieval endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("CoSearch"))
    parser.add_argument("--url", default="http://127.0.0.1:8010/retrieve")
    parser.add_argument("--query", default="who got the first nobel prize in physics?")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--expect-contains", default="Röntgen")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "verl"))

    from verl.tools.co_search_tool import CoSearchTool

    tool = CoSearchTool(
        {
            "retrieval_service_url": args.url,
            "timeout": 120,
            "max_retries": 1,
            "default_top_n": args.top_n,
            "default_top_m": args.top_m,
            "use_reranker": False,
        }
    )

    async def run() -> tuple[object, float, dict]:
        instance_id, _ = await tool.create(
            create_kwargs={
                "top_n": args.top_n,
                "top_m": args.top_m,
                "answers": ["Wilhelm Conrad Röntgen"],
            }
        )
        return await tool.execute(instance_id, {"query": args.query})

    response, reward, metrics = asyncio.run(run())
    text = getattr(response, "text", str(response))
    if args.expect_contains and args.expect_contains.lower() not in text.lower():
        raise SystemExit(f"ERROR: expected substring not found in tool response: {args.expect_contains}")
    if metrics.get("num_retrieved_docs") != args.top_n:
        raise SystemExit(f"ERROR: expected {args.top_n} retrieved docs, got {metrics.get('num_retrieved_docs')}")

    print("CoSearchTool retrieval verification passed.")
    print(f"  url:      {args.url}")
    print(f"  query:    {args.query}")
    print(f"  top_n:    {args.top_n}")
    print(f"  top_m:    {args.top_m}")
    print(f"  reward:   {reward}")
    print("  metrics:  " + json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    print("  preview:  " + text.splitlines()[0][:160])


if __name__ == "__main__":
    main()
