#!/usr/bin/env python3
"""AIR stage2 frozen agent 多实例代理。

stage2 reward 会产生大量 continuation 请求。这里不做外部显式 batch，而是把单条
OpenAI-compatible 请求并发转发给多个 frozen agent vLLM 实例，让 vLLM 实例内部做
continuous batching。代理只负责 least-inflight 负载均衡、失败 cooldown 和基础监控。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request


@dataclass
class BackendState:
    """单个 frozen agent backend 的运行状态。"""

    base_url: str
    in_flight: int = 0
    success_count: int = 0
    failure_count: int = 0
    latency_ewma: float | None = None
    total_elapsed_s: float = 0.0
    cooldown_until: float = 0.0
    last_error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        avg_latency_s = self.total_elapsed_s / self.success_count if self.success_count else None
        return {
            "base_url": self.base_url,
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
    backend_base_urls: list[str],
    timeout: float,
    strategy: str,
    failure_cooldown_seconds: float,
    latency_ewma_alpha: float,
    max_retries_per_request: int,
) -> FastAPI:
    """构造 async FastAPI 代理。

    当前只支持 least_inflight。这样做是因为 continuation 长尾很明显，round-robin
    无法感知哪个 frozen agent 实例还在处理长请求。
    """

    if strategy != "least_inflight":
        raise ValueError(f"unsupported frozen agent proxy strategy: {strategy}")
    if not backend_base_urls:
        raise ValueError("at least one frozen agent backend is required")
    states = [BackendState(base_url=url.rstrip("/")) for url in backend_base_urls]
    app = FastAPI()
    client = httpx.AsyncClient(timeout=timeout)

    async def pick_backend(tried: set[str]) -> BackendState | None:
        now = time.time()
        # FastAPI 单进程事件循环下，这段同步状态修改不会跨线程并发；后续如启多 worker 再加锁。
        candidates = [
            state
            for state in states
            if state.base_url not in tried and state.cooldown_until <= now
        ]
        if not candidates:
            candidates = [state for state in states if state.base_url not in tried]
        if not candidates:
            return None
        state = min(
            candidates,
            key=lambda item: (
                item.in_flight,
                item.latency_ewma if item.latency_ewma is not None else 0.0,
                item.base_url,
            ),
        )
        state.in_flight += 1
        return state

    def mark_success(state: BackendState, elapsed_s: float) -> None:
        state.in_flight = max(0, state.in_flight - 1)
        state.success_count += 1
        state.total_elapsed_s += elapsed_s
        state.last_error = None
        if state.latency_ewma is None:
            state.latency_ewma = elapsed_s
        else:
            state.latency_ewma = latency_ewma_alpha * elapsed_s + (1.0 - latency_ewma_alpha) * state.latency_ewma

    def mark_failure(state: BackendState, exc: Exception) -> None:
        state.in_flight = max(0, state.in_flight - 1)
        state.failure_count += 1
        state.last_error = str(exc)
        state.cooldown_until = time.time() + failure_cooldown_seconds

    async def forward_json(path: str, payload: dict[str, Any]) -> Any:
        errors: list[str] = []
        tried: set[str] = set()
        max_attempts = min(max(int(max_retries_per_request), 1), len(states))
        for _ in range(max_attempts):
            state = await pick_backend(tried)
            if state is None:
                break
            tried.add(state.base_url)
            url = f"{state.base_url}{path}"
            start = time.perf_counter()
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                elapsed_s = time.perf_counter() - start
                mark_success(state, elapsed_s)
                data = response.json()
                print(
                    json.dumps(
                        {
                            "event": "frozen_agent_proxy_request",
                            "backend": state.base_url,
                            "path": path,
                            "elapsed_s": round(elapsed_s, 6),
                            "strategy": strategy,
                            "attempts": len(tried),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if isinstance(data, dict):
                    data.setdefault("_proxy_backend", state.base_url)
                    data.setdefault("_proxy_elapsed_s", elapsed_s)
                    data.setdefault("_proxy_strategy", strategy)
                return data
            except Exception as exc:
                mark_failure(state, exc)
                errors.append(f"{state.base_url}: {exc}")
        raise HTTPException(status_code=502, detail={"errors": errors, "tried": sorted(tried)})

    @app.on_event("shutdown")
    async def shutdown() -> None:
        await client.aclose()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "strategy": strategy,
            "backend_count": len(states),
            "backends": [state.snapshot() for state in states],
        }

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        return await health()

    @app.get("/v1/models")
    async def models() -> Any:
        # /v1/models 是启动预检接口，直接转发到当前最空闲 backend。
        state = await pick_backend(set())
        if state is None:
            raise HTTPException(status_code=502, detail="no frozen agent backend available")
        start = time.perf_counter()
        try:
            response = await client.get(f"{state.base_url}/v1/models")
            response.raise_for_status()
            elapsed_s = time.perf_counter() - start
            mark_success(state, elapsed_s)
            return response.json()
        except Exception as exc:
            mark_failure(state, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/completions")
    async def completions(payload: dict[str, Any] = Body(...)) -> Any:
        return await forward_json("/v1/completions", payload)

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any] = Body(...)) -> Any:
        return await forward_json("/v1/chat/completions", payload)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def unsupported(path: str, request: Request) -> Any:
        raise HTTPException(status_code=404, detail=f"unsupported frozen agent proxy path: /{path}")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True, help="backend base URL, e.g. http://127.0.0.1:8141")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--strategy", default="least_inflight")
    parser.add_argument("--failure-cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--latency-ewma-alpha", type=float, default=0.2)
    parser.add_argument("--max-retries-per-request", type=int, default=3)
    args = parser.parse_args()

    app = build_app(
        backend_base_urls=args.backend,
        timeout=args.timeout,
        strategy=args.strategy,
        failure_cooldown_seconds=args.failure_cooldown_seconds,
        latency_ewma_alpha=args.latency_ewma_alpha,
        max_retries_per_request=args.max_retries_per_request,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
