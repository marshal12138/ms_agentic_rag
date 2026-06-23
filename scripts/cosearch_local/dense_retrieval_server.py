#!/usr/bin/env python3
"""Dense FAISS retrieval server compatible with CoSearchTool/Search-R1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer


def load_docs(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pool_mean(last_hidden_state, attention_mask):
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


def create_app(index_dir: Path, model_path: Path, default_topk: int, device: str) -> FastAPI:
    docs = load_docs(index_dir / "docs.jsonl")
    index = faiss.read_index(str(index_dir / "e5_Flat.index"))
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    if device == "cuda":
        model = model.half()

    @torch.no_grad()
    def encode(queries: list[str]) -> np.ndarray:
        batch = [f"query: {q}" for q in queries]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, return_dict=True)
        emb = pool_mean(out.last_hidden_state, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.float().cpu().numpy().astype(np.float32)

    app = FastAPI()

    @app.post("/retrieve")
    def retrieve(request: QueryRequest):
        topk = request.topk or default_topk
        query_emb = encode(request.queries)
        scores, idxs = index.search(query_emb, topk)
        response = []
        for row_scores, row_idxs in zip(scores, idxs):
            items = []
            for score, idx in zip(row_scores, row_idxs):
                doc = dict(docs[int(idx)])
                if request.return_scores:
                    items.append({"document": doc, "score": float(score)})
                else:
                    doc["score"] = float(score)
                    items.append(doc)
            response.append(items)
        return {"result": response}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path("data/retrieval/e5_wiki18_smoke"))
    parser.add_argument("--model", type=Path, default=Path("/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    app = create_app(args.index_dir, args.model, args.topk, args.device)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
