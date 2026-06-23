#!/usr/bin/env python3
"""Core CoAgenticRetriever recall-top50 to rank-top5 inference smoke."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def add_project_paths(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "verl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint-dir", default="")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--candidate-docs", type=int, default=50)
    parser.add_argument("--recall-top-k", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recall-device", default="cuda:1")
    parser.add_argument("--recall-encode-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-service-url", default="")
    return parser.parse_args()


def load_corpus(path: Path, limit: int) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    docs = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(
                {
                    "id": str(row.get("id") or row.get("_id") or len(docs)),
                    "title": row.get("title") or "",
                    "contents": row.get("contents") or row.get("text") or row.get("passage") or "",
                }
            )
            if len(docs) >= limit:
                break
    return docs


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class FrozenE5RecallRetriever:
    def __init__(self, model_path: str, device: str, batch_size: int = 64):
        if not device.startswith("cuda"):
            raise RuntimeError(f"Recall retriever requires CUDA; got device={device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("Recall retriever requires CUDA, but torch.cuda.is_available() is False.")
        self.model_path = model_path
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.encoder = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(self.device)
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False
        print(f"[recall-retriever] initialized frozen=true path={model_path} device={self.device}", flush=True)

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int) -> torch.Tensor:
        chunks = []
        for start in range(0, len(texts), self.batch_size):
            tokens = self.tokenizer(
                texts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            outputs = self.encoder(**tokens)
            emb = mean_pool(outputs.last_hidden_state, tokens["attention_mask"])
            chunks.append(F.normalize(emb, dim=-1).detach())
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def retrieve_topk(
        self,
        query: str,
        docs: list[dict],
        doc_emb: torch.Tensor,
        top_k: int,
        max_query_length: int,
    ) -> list[dict]:
        query_emb = self.encode([query], max_length=max_query_length)
        scores = torch.matmul(query_emb, doc_emb.T).squeeze(0)
        top_scores, top_indices = torch.topk(scores, k=min(top_k, len(docs)))
        out = []
        for recall_rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            doc = dict(docs[idx])
            doc["recall_rank"] = recall_rank
            doc["rank"] = recall_rank
            doc["recall_score"] = float(score)
            doc["retriever_score"] = float(score)
            out.append(doc)
        return out


def row_question(row: dict) -> str:
    extra_info = row.get("extra_info") or {}
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"])
    prompt = row.get("prompt")
    try:
        first = list(prompt)[0]
        return str(first.get("content") or "")
    except Exception:
        return ""


def normalize_recall_doc(doc: dict, rank: int) -> dict:
    score = doc.get("recall_score", doc.get("retriever_score", doc.get("score", 1.0 / rank)))
    return {
        "doc_id": str(doc.get("doc_id") or doc.get("id") or doc.get("_id") or rank),
        "rank": rank,
        "recall_rank": rank,
        "title": doc.get("title") or "",
        "text": doc.get("contents") or doc.get("text") or doc.get("passage") or "",
        "retriever_score": float(score),
        "recall_score": float(score),
    }


def retrieve_topk_from_service(query: str, url: str, top_k: int, timeout: float = 120.0) -> list[dict]:
    payload = json.dumps({"queries": [query], "topk": top_k, "return_scores": True}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    raw = (data.get("result") or [[]])[0]
    docs = []
    for recall_rank, item in enumerate(raw, start=1):
        doc = dict(item.get("document") or {})
        score = float(item.get("score", doc.get("score", 0.0)) or 0.0)
        doc["recall_rank"] = recall_rank
        doc["rank"] = recall_rank
        doc["recall_score"] = score
        doc["retriever_score"] = score
        docs.append(normalize_recall_doc(doc, recall_rank))
    if len(docs) < top_k:
        raise RuntimeError(f"Recall service returned {len(docs)} docs, expected {top_k}: {url}")
    return docs


def step_number(path: Path) -> int:
    name = path.name
    if not name.startswith("global_step_"):
        return -1
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return -1


def resolve_ranker_checkpoint_dir(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    direct_rank_encoder = checkpoint_dir / "rank_encoder"
    if direct_rank_encoder.is_dir():
        return checkpoint_dir

    direct_ranker_dir = checkpoint_dir / "ranker"
    if (direct_ranker_dir / "rank_encoder").is_dir():
        return direct_ranker_dir

    direct_retriever_dir = checkpoint_dir / "retriever"
    if (direct_retriever_dir / "rank_encoder").is_dir():
        return direct_retriever_dir

    step_dirs = sorted(
        [
            path
            for path in checkpoint_dir.iterdir()
            if path.is_dir()
            and step_number(path) >= 0
            and ((path / "ranker" / "rank_encoder").is_dir() or (path / "retriever" / "rank_encoder").is_dir())
        ],
        key=step_number,
    ) if checkpoint_dir.is_dir() else []
    if step_dirs:
        step_dir = step_dirs[-1]
        if (step_dir / "ranker" / "rank_encoder").is_dir():
            return step_dir / "ranker"
        return step_dir / "retriever"

    return checkpoint_dir


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    add_project_paths(project_root)

    from omegaconf import OmegaConf
    from verl.workers.ranker.e5_ranker_worker import LocalE5RankerWorker

    checkpoint_dir = resolve_ranker_checkpoint_dir(Path(args.checkpoint_dir))
    rank_encoder_path = checkpoint_dir / "rank_encoder"
    if not rank_encoder_path.is_dir():
        raise FileNotFoundError(
            f"trained rank_encoder not found under checkpoint_dir={args.checkpoint_dir!r}; "
            f"resolved_dir={checkpoint_dir}"
        )
    print(f"[rank-retriever] checkpoint_dir={checkpoint_dir}", flush=True)
    config = OmegaConf.create(
        {
            "recall_retriever": {
                "model_path": args.model_path,
                "device": args.recall_device,
                "top_k": args.recall_top_k,
                "trainable": False,
                "index_refresh": False,
            },
            "ranker": {
                "model_path": args.model_path,
                "encoder_path": str(rank_encoder_path),
                "device": args.device,
                "shared_encoder": True,
                "top_k": args.top_k,
                "trust_remote_code": True,
            },
            "ranker_training": {"device": args.device},
        }
    )
    worker = LocalE5RankerWorker(config)
    worker.init_model()
    recall_retriever = None
    if args.retrieval_service_url:
        print(f"[recall-retriever] using service url={args.retrieval_service_url}", flush=True)
    else:
        recall_retriever = FrozenE5RecallRetriever(
            model_path=args.model_path,
            device=args.recall_device,
            batch_size=args.recall_encode_batch_size,
        )

    val_df = pd.read_parquet(args.val_data)
    docs = load_corpus(Path(args.corpus_jsonl), args.candidate_docs)
    recall_doc_emb = None
    if recall_retriever is not None:
        doc_texts = [(doc.get("title") + "\n" if doc.get("title") else "") + doc.get("contents", "") for doc in docs]
        recall_doc_emb = recall_retriever.encode(doc_texts, max_length=256)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        for step in range(args.max_steps):
            row = val_df.iloc[step % len(val_df)].to_dict()
            question = row_question(row)
            if args.retrieval_service_url:
                recall_top50 = retrieve_topk_from_service(
                    query=question,
                    url=args.retrieval_service_url,
                    top_k=args.recall_top_k,
                )
            else:
                recall_top50 = [
                    normalize_recall_doc(doc, rank=int(doc.get("recall_rank", rank)))
                    for rank, doc in enumerate(
                        recall_retriever.retrieve_topk(
                            query=question,
                            docs=docs,
                            doc_emb=recall_doc_emb,
                            top_k=args.recall_top_k,
                            max_query_length=192,
                        ),
                        start=1,
                    )
                ]
            rank_top50 = worker.rank_topk(query=question, docs=recall_top50, top_k=args.recall_top_k)
            rank_top5 = rank_top50[: args.top_k]
            result_docs = []
            for doc in rank_top5:
                result_docs.append(
                    {
                        "rank_rank": doc["rank_rank"],
                        "doc_id": doc["doc_id"],
                        "rank_score": doc["rank_score"],
                        "recall_rank": doc["recall_rank"],
                        "recall_score": doc["recall_score"],
                        "title": doc.get("title"),
                        "text_preview": str(doc.get("text") or doc.get("contents") or "")[:240],
                    }
                )
            row_out = {
                "step": step + 1,
                "query": question,
                "recall_top50_sample": [
                    {
                        "recall_rank": doc["recall_rank"],
                        "doc_id": doc["doc_id"],
                        "recall_score": doc["recall_score"],
                        "text_preview": doc["text"][:160],
                    }
                    for doc in recall_top50[:5]
                ],
                "rerank_top50_sample": [
                    {
                        "rank_rank": doc["rank_rank"],
                        "doc_id": doc["doc_id"],
                        "rank_score": doc["rank_score"],
                        "recall_rank": doc["recall_rank"],
                        "recall_score": doc["recall_score"],
                        "text_preview": str(doc.get("text") or doc.get("contents") or "")[:160],
                    }
                    for doc in rerank_top50[:5]
                ],
                "rank_top5": result_docs,
            }
            fp.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            print(json.dumps(row_out, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
