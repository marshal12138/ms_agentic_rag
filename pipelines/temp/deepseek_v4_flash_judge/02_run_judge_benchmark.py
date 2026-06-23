#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests


def load_requests(path: Path, model: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            payload = obj["payload"]
            payload["model"] = model
            rows.append(obj)
    return rows


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int((pct / 100) * (len(values) - 1))
    return values[idx]


def call_one(session: requests.Session, url: str, row: dict[str, Any], timeout: float) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        resp = session.post(url, json=row["payload"], timeout=timeout)
        elapsed = time.perf_counter() - t0
        text = resp.text
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "")
        usage = data.get("usage", {})
        return {
            "example_id": row["example_id"],
            "mode": row["mode"],
            "ok": True,
            "elapsed_s": elapsed,
            "content": content,
            "usage": usage,
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "example_id": row["example_id"],
            "mode": row["mode"],
            "ok": False,
            "elapsed_s": elapsed,
            "content": "",
            "usage": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8067/v1/chat/completions")
    parser.add_argument("--model", default="DeepSeek-V4-Flash")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    rows = load_requests(args.requests, args.model)[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(call_one, session, args.url, row, args.timeout) for row in rows]
        for fut in as_completed(futures):
            result = fut.result()
            results.append(result)
            with args.output.open("a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    total_elapsed = time.perf_counter() - started

    ok = [r for r in results if r["ok"]]
    latencies = [r["elapsed_s"] for r in ok]
    usage = [r.get("usage") or {} for r in ok]
    completion_tokens = sum(int(u.get("completion_tokens") or 0) for u in usage)
    prompt_tokens = sum(int(u.get("prompt_tokens") or 0) for u in usage)
    total_tokens = sum(int(u.get("total_tokens") or 0) for u in usage)
    summary = {
        "mode": rows[0]["mode"] if rows else "unknown",
        "request_count": len(rows),
        "ok_count": len(ok),
        "error_count": len(results) - len(ok),
        "total_elapsed_s": total_elapsed,
        "qpm": len(ok) / total_elapsed * 60 if total_elapsed else 0.0,
        "qps": len(ok) / total_elapsed if total_elapsed else 0.0,
        "latency_s": {
            "mean": statistics.mean(latencies) if latencies else None,
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p95": percentile(latencies, 95),
            "max": max(latencies) if latencies else None,
        },
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_s": total_tokens / total_elapsed if total_elapsed else 0.0,
        },
        "error_samples": [r["error"] for r in results if not r["ok"]][:5],
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
