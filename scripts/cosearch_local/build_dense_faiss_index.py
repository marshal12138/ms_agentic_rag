#!/usr/bin/env python3
"""Build a smoke-only dense FAISS index for the local CoSearch retriever.

The paper path uses Search-R1's released wiki-18 E5 index. This helper is only
for resource-saving smoke runs when BUILD_LOCAL_INDEX=1 is set explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def read_corpus(path: Path, max_docs: int) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_docs > 0 and idx >= max_docs:
                break
            item = json.loads(line)
            contents = item.get("contents") or item.get("text") or ""
            title = contents.split("\n", 1)[0].strip('" ') if "\n" in contents else item.get("title", "")
            text = contents.split("\n", 1)[1] if "\n" in contents else contents
            docs.append({"id": str(item.get("id", idx)), "title": title, "text": text, "contents": contents})
    return docs


def pool_mean(last_hidden_state, attention_mask):
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def encode_passages(model, tokenizer, docs: list[dict], batch_size: int, max_length: int, device: str) -> np.ndarray:
    vectors = []
    for start in tqdm(range(0, len(docs), batch_size), desc="encode passages"):
        batch = [f"passage: {d['contents']}" for d in docs[start : start + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs, return_dict=True)
        emb = pool_mean(out.last_hidden_state, inputs["attention_mask"])
        emb = torch.nn.functional.normalize(emb, dim=-1)
        vectors.append(emb.float().cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("/data01/ms_wksp/agent_up_to_date/Agentic_R_Learn/data/raw/lwhlwh__retrieval_corpus/wiki18_100w.jsonl"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("data/retrieval/e5_wiki18_smoke"))
    parser.add_argument("--max-docs", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=180)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    docs = read_corpus(args.corpus, args.max_docs)
    docs_path = args.out_dir / "docs.jsonl"
    with docs_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
    if args.device == "cuda":
        model = model.half()
    emb = encode_passages(model, tokenizer, docs, args.batch_size, args.max_length, args.device)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(args.out_dir / "e5_Flat.index"))
    print(f"wrote {len(docs):,} docs, dim={emb.shape[1]} -> {args.out_dir}")


if __name__ == "__main__":
    main()
