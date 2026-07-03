#!/usr/bin/env python3
"""AgenticIterRag v1 recall retriever 语义预检。

该脚本只验证 AIR infer 需要的最小事实：
1. recall 服务可以按 `/retrieve` 协议返回 `result`。
2. 单条 query 至少返回 `top_m` 条候选，且不超过 `top_n` 约束。
3. 如果配置了 `expect_contains`，可见文档内容中必须包含该字符串。
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


def post_retrieve(url: str, query: str, top_n: int, timeout: float) -> dict[str, Any]:
    payload = json.dumps({"queries": [query], "topk": top_n, "return_scores": True}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_docs(data: dict[str, Any]) -> list[dict[str, Any]]:
    result = data.get("result")
    if not isinstance(result, list) or not result:
        raise ValueError("recall response missing non-empty result")
    first = result[0]
    if not isinstance(first, list):
        raise ValueError("recall response result[0] must be a list")
    docs: list[dict[str, Any]] = []
    for item in first:
        if isinstance(item, dict) and isinstance(item.get("document"), dict):
            docs.append(item["document"])
        elif isinstance(item, dict):
            docs.append(item)
    return docs


def doc_text(doc: dict[str, Any]) -> str:
    return str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check AgenticIterRag recall retriever.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-n", type=int, required=True)
    parser.add_argument("--top-m", type=int, required=True)
    parser.add_argument("--expect-contains", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    try:
        if args.top_n < 1:
            raise ValueError(f"top_n must be positive, got {args.top_n}")
        if args.top_m < 1 or args.top_m > args.top_n:
            raise ValueError(f"top_m must be in [1, top_n], got top_m={args.top_m}, top_n={args.top_n}")
        docs = extract_docs(post_retrieve(args.url, args.query, args.top_n, args.timeout))
        if len(docs) < args.top_m:
            raise ValueError(f"recall returned {len(docs)} docs, fewer than top_m={args.top_m}")
        if args.expect_contains:
            visible_text = "\n".join(doc_text(doc) for doc in docs[: args.top_m])
            if args.expect_contains not in visible_text:
                raise ValueError(f"expect_contains={args.expect_contains!r} not found in visible top_m docs")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"AIR recall preflight passed: returned={len(docs)} top_n={args.top_n} top_m={args.top_m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
