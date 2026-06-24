#!/usr/bin/env python3
"""Core CoAgenticRetriever ranker contrastive training smoke.

This script validates the retriever contrastive framework without starting the
full agent LLM rollout stack. The local scheduling script starts the frozen
recall service, this script consumes its top50 results, then uses the trainable
shared-encoder ranker to rescore top50 -> top5.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModel, AutoTokenizer


def add_project_paths(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "verl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, required=True)
    parser.add_argument("--num-groups-per-step", type=int, required=True)
    parser.add_argument("--neg-per-pos", type=int, required=True)
    parser.add_argument("--positive-top-k", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--max-query-length", type=int, required=True)
    parser.add_argument("--max-doc-length", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--recall-device", default="cuda:1")
    parser.add_argument("--recall-top-k", type=int, required=True)
    parser.add_argument("--rank-top-k", type=int, required=True)
    parser.add_argument("--recall-encode-batch-size", type=int, default=64)
    parser.add_argument("--retrieval-service-url", default="")
    parser.add_argument("--construction-log-jsonl", default="")
    parser.add_argument("--metrics-jsonl", default="")
    return parser.parse_args()


def load_corpus(path: Path, limit: int = 4096) -> list[dict]:
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
                    "score": 1.0 / (len(docs) + 1),
                }
            )
            if len(docs) >= limit:
                break
    if len(docs) < 32:
        raise RuntimeError(f"Need at least 32 corpus docs for contrastive smoke, got {len(docs)} from {path}")
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
        print(
            f"[recall-retriever] initialized frozen=true path={model_path} device={self.device}",
            flush=True,
        )

    @torch.no_grad()
    def encode(self, texts: list[str], max_length: int) -> torch.Tensor:
        outputs = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            tokens = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            encoded = self.encoder(**tokens)
            emb = mean_pool(encoded.last_hidden_state, tokens["attention_mask"])
            outputs.append(F.normalize(emb, dim=-1).detach())
        return torch.cat(outputs, dim=0)

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
        results = []
        for recall_rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            doc = dict(docs[idx])
            doc["recall_rank"] = recall_rank
            doc["rank"] = recall_rank
            doc["recall_score"] = float(score)
            doc["retriever_score"] = float(score)
            results.append(doc)
        return results


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


def row_answers(row: dict) -> list[str]:
    reward_model = row.get("reward_model") or {}
    try:
        target = reward_model["ground_truth"]["target"]
    except Exception:
        return []
    if hasattr(target, "tolist"):
        target = target.tolist()
    if isinstance(target, (list, tuple)):
        return [str(x) for x in target]
    return [str(target)]


def normalize_recall_doc(doc: dict, recall_rank: int | None = None) -> dict:
    recall_rank_value = int(doc.get("recall_rank") or recall_rank or doc.get("rank") or 1)
    score = doc.get("recall_score", doc.get("retriever_score", doc.get("score")))
    return {
        "doc_id": str(doc.get("doc_id") or doc.get("id") or doc.get("_id") or recall_rank_value),
        "rank": recall_rank_value,
        "recall_rank": recall_rank_value,
        "title": doc.get("title") or None,
        "text": doc.get("contents") or doc.get("text") or doc.get("passage") or "",
        "retriever_score": float(score or 0.0),
        "recall_score": float(score or 0.0),
        "metadata": {"source": "wiki-18-smoke"},
    }


def normalize_ranked_doc(doc: dict, recall_rank: int | None = None) -> dict:
    recall_rank_value = int(doc.get("recall_rank") or recall_rank or doc.get("rank") or 1)
    if "rank_rank" not in doc or doc["rank_rank"] in (None, ""):
        raise ValueError("ranker smoke document is missing rank_rank")
    if "rank_score" not in doc or doc["rank_score"] in (None, ""):
        raise ValueError("ranker smoke document is missing rank_score")
    rank = int(doc["rank_rank"])
    score = doc.get("recall_score", doc.get("retriever_score", doc.get("score")))
    out = {
        "doc_id": str(doc.get("doc_id") or doc.get("id") or doc.get("_id") or rank),
        "rank": rank,
        "recall_rank": recall_rank_value,
        "title": doc.get("title") or None,
        "text": doc.get("contents") or doc.get("text") or doc.get("passage") or "",
        "retriever_score": float(score or 0.0),
        "recall_score": float(score or 0.0),
        "metadata": {"source": "wiki-18-smoke"},
    }
    out["rank_score"] = float(doc["rank_score"])
    out["metadata"]["rank_score"] = float(doc["rank_score"])
    out["rank_rank"] = int(doc["rank_rank"])
    out["metadata"]["rank_rank"] = int(doc["rank_rank"])
    return out


def build_recall_topk(docs: list[dict], row_idx: int, top_k: int) -> list[dict]:
    start = (row_idx * 37) % max(1, len(docs) - top_k)
    return [normalize_recall_doc(doc, recall_rank=rank) for rank, doc in enumerate(docs[start : start + top_k], start=1)]


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
        docs.append(normalize_recall_doc(doc, recall_rank=recall_rank))
    if len(docs) < top_k:
        raise RuntimeError(f"Recall service returned {len(docs)} docs, expected {top_k}: {url}")
    return docs


def make_trajectory(
    row: dict,
    recall_top50_docs: list[dict],
    rerank_top50_docs: list[dict],
    rerank_top5_docs: list[dict],
    row_idx: int,
    global_step: int,
) -> dict:
    question = row_question(row)
    answers = row_answers(row)
    return {
        "trajectory_id": f"smoke-{global_step}-{row_idx}",
        "origin_query": question,
        "golden_answers": answers,
        "final_answer": "",
        "score": 1.0,
        "score_type": "f1",
        "is_valid": True,
        "messages": [],
        "tool_calls": [
            {
                "tool_call_id": f"smoke-{global_step}-{row_idx}:search:0",
                "turn_idx": 0,
                "tool_name": "search",
                "sub_query": question,
                "recall_top50_docs": recall_top50_docs,
                "rank_top50_docs": rerank_top50_docs,
                "rerank_top50_docs": rerank_top50_docs,
                "ranked_passages": rerank_top50_docs,
                "rank_top5_docs": rerank_top5_docs,
                "rerank_top5_docs": rerank_top5_docs,
            }
        ],
        "metadata": {"dataset": row.get("data_source"), "global_steps": global_step},
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    add_project_paths(project_root)

    from ranker_strategies.config import (
        build_collator,
        build_construction_logger,
        build_replay_buffer,
        build_sample_builder,
        build_selector,
        build_signal_builder,
    )
    from verl.trainer.ppo.ranker_contrastive_step import process_ranker_contrastive_step
    from verl.workers.ranker.e5_ranker_worker import LocalE5RankerWorker

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_parquet(args.train_data)
    docs = load_corpus(Path(args.corpus_jsonl))

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
                "device": args.device,
                "shared_encoder": True,
                "top_k": args.rank_top_k,
                "trust_remote_code": True,
            },
            "ranker_training": {
                "device": args.device,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_query_length": args.max_query_length,
                "max_doc_length": args.max_doc_length,
                "log_every_n_steps": 10,
                "log_first_sample": True,
                "construction_log_jsonl": args.construction_log_jsonl,
                "max_grad_norm": 1.0,
                "trajectory_selector": {
                    "type": "top_f1_trajectories",
                    "max_selected_trajectories": 1,
                    "min_final_reward": 0.0,
                },
                "signal_builder": {
                    "type": "topk_pseudo_rank",
                    "positive_top_k": args.positive_top_k,
                    "allow_all_negative": False,
                },
                "sample_builder": {
                    "type": "random_negative_repeat",
                    "num_groups_per_step": args.num_groups_per_step,
                    "neg_per_pos": args.neg_per_pos,
                    "allow_repeat_negative_sampling": True,
                },
                "loss": {"type": "info_nce", "temperature": args.temperature},
                "replay_buffer": {"enable": True, "max_size": 200000, "fresh_ratio": 0.5},
                "optim": {"lr": 2e-5, "weight_decay": 0.01, "warmup_steps": 0, "total_steps": args.max_steps},
            },
        }
    )

    worker = LocalE5RankerWorker(config)
    worker.init_model()
    recall_retriever = None
    recall_doc_emb = None
    if args.retrieval_service_url:
        print(f"[recall-retriever] using service url={args.retrieval_service_url}", flush=True)
    else:
        recall_retriever = FrozenE5RecallRetriever(
            model_path=args.model_path,
            device=args.recall_device,
            batch_size=args.recall_encode_batch_size,
        )
        doc_texts = [(doc.get("title") + "\n" if doc.get("title") else "") + doc.get("contents", "") for doc in docs]
        recall_doc_emb = recall_retriever.encode(doc_texts, max_length=args.max_doc_length)
    replay_buffer = build_replay_buffer(config)
    selector = build_selector(config)
    signal_builder = build_signal_builder(config)
    sample_builder = build_sample_builder(config)
    collator = build_collator(config, worker.tokenizer)
    construction_logger = build_construction_logger(config)

    metrics_path = Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "ranker_contrastive_smoke_metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    last_metrics = {}
    for step in range(1, args.max_steps + 1):
        row = train_df.iloc[(step - 1) % len(train_df)].to_dict()
        question = row_question(row)
        if args.retrieval_service_url:
            recall_top50_docs = retrieve_topk_from_service(
                query=question,
                url=args.retrieval_service_url,
                top_k=args.recall_top_k,
            )
        else:
            recall_top50_docs = [
                normalize_recall_doc(doc, recall_rank=doc.get("recall_rank"))
                for doc in recall_retriever.retrieve_topk(
                    query=question,
                    docs=docs,
                    doc_emb=recall_doc_emb,
                    top_k=args.recall_top_k,
                    max_query_length=args.max_query_length,
                )
            ]
        rank_top50_docs = [
            normalize_ranked_doc(doc, recall_rank=doc.get("recall_rank"))
            for doc in worker.rank_topk(
                query=question,
                docs=recall_top50_docs,
                top_k=args.recall_top_k,
                max_query_length=args.max_query_length,
                max_doc_length=args.max_doc_length,
            )
        ]
        rank_top5_docs = rank_top50_docs[: args.rank_top_k]
        fresh_trajectories = [
            make_trajectory(
                row,
                recall_top50_docs=recall_top50_docs,
                rerank_top50_docs=rank_top50_docs,
                rerank_top5_docs=rank_top5_docs,
                row_idx=step - 1,
                global_step=step,
            )
        ]
        last_metrics = process_ranker_contrastive_step(
            fresh_trajectories=fresh_trajectories,
            ranker_wg=worker,
            replay_buffer=replay_buffer,
            selector=selector,
            signal_builder=signal_builder,
            sample_builder=sample_builder,
            collator=collator,
            config=config,
            global_steps=step,
            ranker_step_idx=0,
            construction_logger=construction_logger,
        )
        with metrics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"step": step, **last_metrics}, ensure_ascii=False) + "\n")
        print(json.dumps({"step": step, **last_metrics}, ensure_ascii=False), flush=True)

    final_step_dir = output_dir / f"global_step_{args.max_steps}"
    worker.save_checkpoint(str(final_step_dir / "ranker"))
    print(f"saved ranker smoke checkpoint: {final_step_dir / 'ranker'}", flush=True)
    return 0 if last_metrics.get("ranker/skipped", 1) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
