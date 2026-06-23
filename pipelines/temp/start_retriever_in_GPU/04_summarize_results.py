#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.results_dir.glob("bench_b*_c*.json")):
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    rows.sort(key=lambda r: (r.get("batch_size", 0), r.get("concurrency", 0)))

    best = max(rows, key=lambda r: r.get("query_qps", 0.0), default=None)
    stable_rows = [r for r in rows if r.get("error_rate", 1.0) == 0 and (r.get("latency_s", {}).get("p95") or 9999) < 10]
    stable = max(stable_rows, key=lambda r: r.get("query_qps", 0.0), default=None)

    lines = [
        "# GPU 5 Retriever QPS Benchmark",
        "",
        "## Summary",
        "",
    ]
    if best:
        lines.append(
            f"- Max observed query QPS: `{fmt(best.get('query_qps'))}` "
            f"(batch={best.get('batch_size')}, concurrency={best.get('concurrency')}, "
            f"p95={fmt(best.get('latency_s', {}).get('p95'))}s, error_rate={fmt(best.get('error_rate'))})"
        )
    if stable:
        lines.append(
            f"- Recommended stable query QPS candidate: `{fmt(stable.get('query_qps'))}` "
            f"(batch={stable.get('batch_size')}, concurrency={stable.get('concurrency')}, "
            f"p95={fmt(stable.get('latency_s', {}).get('p95'))}s)"
        )
    lines.extend(["", "## Results", "", "| batch | concurrency | query_qps | request_qps | p50_s | p95_s | p99_s | errors | error_rate |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in rows:
        lat = r.get("latency_s", {})
        lines.append(
            f"| {r.get('batch_size')} | {r.get('concurrency')} | {fmt(r.get('query_qps'))} | "
            f"{fmt(r.get('request_qps'))} | {fmt(lat.get('p50'))} | {fmt(lat.get('p95'))} | "
            f"{fmt(lat.get('p99'))} | {r.get('errors')} | {fmt(r.get('error_rate'))} |"
        )

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
