#!/usr/bin/env python3
"""Load-balancing proxy for AIR dense retriever backends.

相比旧 round-robin proxy，这个代理会记录每个 backend 的：

* 当前 in-flight 请求数
* 最近成功请求延迟的 EWMA
* 失败后的短暂 cooldown

调度策略默认是 least_inflight：优先把请求发给当前最空闲、延迟较低、没有处于
cooldown 的 backend。CPU/NPU/GPU retriever 都走同一套代理协议。
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests
import uvicorn
from fastapi import Body, FastAPI, HTTPException


@dataclass
class BackendState:
    url: str
    in_flight: int = 0
    success_count: int = 0
    failure_count: int = 0
    latency_ewma: float | None = None
    cooldown_until: float = 0.0
    last_error: str | None = None
    total_elapsed_s: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        avg_latency_s = self.total_elapsed_s / self.success_count if self.success_count > 0 else None
        return {
            "url": self.url,
            "in_flight": self.in_flight,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "latency_ewma": self.latency_ewma,
            "avg_latency_s": avg_latency_s,
            "cooldown_remaining_s": max(0.0, self.cooldown_until - now),
            "last_error": self.last_error,
        }


def build_app(
    *,
    backend_urls: list[str],
    timeout: float,
    strategy: str,
    failure_cooldown_seconds: float,
    latency_ewma_alpha: float,
    max_retries_per_request: int,
) -> FastAPI:
    if strategy != "least_inflight":
        raise ValueError(f"unsupported retrieval proxy strategy: {strategy}")
    if not backend_urls:
        raise ValueError("at least one backend url is required")
    states = [BackendState(url=url) for url in backend_urls]
    lock = threading.Lock()
    app = FastAPI()

    def selectable_states(now: float, tried: set[str]) -> list[BackendState]:
        # 不把处于 cooldown 或本次请求已经失败过的 backend 再放进候选。
        candidates = [state for state in states if state.url not in tried and state.cooldown_until <= now]
        return candidates or [state for state in states if state.url not in tried]

    def pick_backend(tried: set[str]) -> BackendState | None:
        with lock:
            candidates = selectable_states(time.time(), tried)
            if not candidates:
                return None
            # 排序优先级：in-flight 最少，其次历史延迟更低，最后用 URL 稳定打散。
            state = min(
                candidates,
                key=lambda item: (
                    item.in_flight,
                    item.latency_ewma if item.latency_ewma is not None else 0.0,
                    item.url,
                ),
            )
            state.in_flight += 1
            return state

    def mark_success(state: BackendState, elapsed_s: float) -> None:
        with lock:
            state.in_flight = max(0, state.in_flight - 1)
            state.success_count += 1
            state.total_elapsed_s += elapsed_s
            state.last_error = None
            if state.latency_ewma is None:
                state.latency_ewma = elapsed_s
            else:
                alpha = latency_ewma_alpha
                state.latency_ewma = alpha * elapsed_s + (1.0 - alpha) * state.latency_ewma

    def mark_failure(state: BackendState, exc: Exception) -> None:
        with lock:
            state.in_flight = max(0, state.in_flight - 1)
            state.failure_count += 1
            state.last_error = str(exc)
            state.cooldown_until = time.time() + failure_cooldown_seconds

    @app.get("/health")
    def health() -> dict[str, Any]:
        with lock:
            return {
                "status": "ok",
                "strategy": strategy,
                "backend_count": len(states),
                "backends": [state.snapshot() for state in states],
            }

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return health()

    @app.post("/retrieve")
    def retrieve(payload: dict[str, Any] = Body(...)) -> Any:
        errors: list[str] = []
        tried: set[str] = set()
        max_attempts = min(max(int(max_retries_per_request), 1), len(states))
        for _ in range(max_attempts):
            state = pick_backend(tried)
            if state is None:
                break
            tried.add(state.url)
            start = time.perf_counter()
            try:
                response = requests.post(state.url, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                elapsed_s = time.perf_counter() - start
                mark_success(state, elapsed_s)
                print(
                    json.dumps(
                        {
                            "event": "retrieval_proxy_request",
                            "backend": state.url,
                            "elapsed_s": round(elapsed_s, 6),
                            "strategy": strategy,
                            "attempts": len(tried),
                            "query_count": len(payload.get("queries") or []),
                            "topk": payload.get("topk"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if isinstance(data, dict):
                    data.setdefault("_proxy_backend", state.url)
                    data.setdefault("_proxy_elapsed_s", elapsed_s)
                    data.setdefault("_proxy_strategy", strategy)
                return data
            except Exception as exc:
                mark_failure(state, exc)
                errors.append(f"{state.url}: {exc}")
        raise HTTPException(status_code=502, detail={"errors": errors, "tried": sorted(tried)})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--strategy", default="least_inflight")
    parser.add_argument("--failure-cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--latency-ewma-alpha", type=float, default=0.2)
    parser.add_argument("--max-retries-per-request", type=int, default=8)
    args = parser.parse_args()

    app = build_app(
        backend_urls=args.backend,
        timeout=args.timeout,
        strategy=args.strategy,
        failure_cooldown_seconds=args.failure_cooldown_seconds,
        latency_ewma_alpha=args.latency_ewma_alpha,
        max_retries_per_request=args.max_retries_per_request,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
