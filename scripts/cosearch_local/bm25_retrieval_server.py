#!/usr/bin/env python3
"""FastAPI retrieval server compatible with CoSearchTool/Search-R1."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


def create_app(index_path: Path, default_topk: int) -> FastAPI:
    with index_path.open("rb") as f:
        payload = pickle.load(f)
    docs = payload["docs"]
    index = payload["index"]

    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        topk = request.topk or default_topk
        response = []
        for query in request.queries:
            scores = index.get_scores(tokenize(query))
            if len(scores) == 0:
                response.append([])
                continue
            top_idxs = np.argsort(scores)[::-1][:topk]
            if request.return_scores:
                response.append([{"document": docs[int(i)], "score": float(scores[int(i)])} for i in top_idxs])
            else:
                response.append([docs[int(i)] for i in top_idxs])
        return {"result": response}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=Path("data/retrieval/bm25_wiki18_20k.pkl"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    app = create_app(args.index, args.topk)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
