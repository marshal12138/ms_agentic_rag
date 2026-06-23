#!/usr/bin/env python3
"""Round-robin proxy for multiple Search-R1 dense retriever instances."""

from __future__ import annotations

import argparse
import itertools
import threading
import time
from typing import Any

import requests
from fastapi import Body, FastAPI, HTTPException
import uvicorn


def build_app(backends: list[str], timeout: float) -> FastAPI:
    app = FastAPI()
    cycle = itertools.cycle(backends)
    cycle_lock = threading.Lock()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "backends": backends}

    @app.post("/retrieve")
    def retrieve(payload: dict[str, Any] = Body(...)) -> Any:
        errors: list[str] = []
        for _ in range(len(backends)):
            with cycle_lock:
                backend = next(cycle)
            start = time.perf_counter()
            try:
                response = requests.post(backend, json=payload, timeout=timeout)
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict):
                    data.setdefault("_proxy_backend", backend)
                    data.setdefault("_proxy_elapsed_s", time.perf_counter() - start)
                return data
            except Exception as exc:
                errors.append(f"{backend}: {exc}")
        raise HTTPException(status_code=502, detail={"errors": errors})

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    app = build_app(args.backend, args.timeout)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
