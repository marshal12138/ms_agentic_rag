#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests


thread_local = threading.local()


def load_queries(path: Path) -> list[str]:
    queries: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            query = str(obj.get("query", "")).strip()
            if query:
                queries.append(query)
    if not queries:
        raise SystemExit(f"no queries loaded from {path}")
    return queries


def session() -> requests.Session:
    sess = getattr(thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        thread_local.session = sess
    return sess


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = int(math.ceil(pct / 100.0 * len(values))) - 1
    idx = max(0, min(idx, len(values) - 1))
    return values[idx]


def run_worker(
    worker_id: int,
    *,
    url: str,
    queries: list[str],
    topk: int,
    batch_size: int,
    timeout: float,
    end_time: float,
    start_index: int,
) -> dict[str, Any]:
    latencies: list[float] = []
    ok_requests = 0
    ok_queries = 0
    errors = 0
    error_samples: list[str] = []
    idx = start_index
    n = len(queries)

    while time.perf_counter() < end_time:
        batch = [queries[(idx + j) % n] for j in range(batch_size)]
        idx += batch_size
        payload = {"queries": batch, "topk": topk, "return_scores": True}
        t0 = time.perf_counter()
        try:
            resp = session().post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("result")
            if not isinstance(result, list) or len(result) != len(batch):
                raise RuntimeError(f"bad result shape: {type(result).__name__}, len={len(result) if isinstance(result, list) else 'NA'}")
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            ok_requests += 1
            ok_queries += len(batch)
        except Exception as exc:
            errors += 1
            if len(error_samples) < 5:
                error_samples.append(f"{type(exc).__name__}: {exc}")

    return {
        "worker_id": worker_id,
        "latencies_s": latencies,
        "ok_requests": ok_requests,
        "ok_queries": ok_queries,
        "errors": errors,
        "error_samples": error_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    queries = load_queries(args.queries)

    # Warmup is intentionally simple and discarded.
    warmup_end = time.perf_counter() + args.warmup
    while time.perf_counter() < warmup_end:
        payload = {"queries": queries[: args.batch_size], "topk": args.topk, "return_scores": True}
        try:
            session().post(args.url, json=payload, timeout=args.timeout).raise_for_status()
        except Exception:
            pass

    start = time.perf_counter()
    end = start + args.duration
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_worker,
                i,
                url=args.url,
                queries=queries,
                topk=args.topk,
                batch_size=args.batch_size,
                timeout=args.timeout,
                end_time=end,
                start_index=i * args.batch_size * 997,
            )
            for i in range(args.concurrency)
        ]
        worker_results = [f.result() for f in futures]

    elapsed = time.perf_counter() - start
    all_latencies = [x for wr in worker_results for x in wr["latencies_s"]]
    ok_requests = sum(wr["ok_requests"] for wr in worker_results)
    ok_queries = sum(wr["ok_queries"] for wr in worker_results)
    errors = sum(wr["errors"] for wr in worker_results)
    error_samples = [e for wr in worker_results for e in wr["error_samples"]][:10]

    result = {
        "url": args.url,
        "topk": args.topk,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "duration_s": args.duration,
        "measured_elapsed_s": elapsed,
        "ok_requests": ok_requests,
        "ok_queries": ok_queries,
        "errors": errors,
        "request_qps": ok_requests / elapsed if elapsed else 0.0,
        "query_qps": ok_queries / elapsed if elapsed else 0.0,
        "error_rate": errors / max(1, errors + ok_requests),
        "latency_s": {
            "count": len(all_latencies),
            "mean": statistics.mean(all_latencies) if all_latencies else None,
            "p50": percentile(all_latencies, 50),
            "p90": percentile(all_latencies, 90),
            "p95": percentile(all_latencies, 95),
            "p99": percentile(all_latencies, 99),
            "max": max(all_latencies) if all_latencies else None,
        },
        "error_samples": error_samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
