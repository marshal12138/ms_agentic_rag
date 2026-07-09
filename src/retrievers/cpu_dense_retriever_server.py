#!/usr/bin/env python3
"""AIR CPU dense retriever server.

这个服务和现有 GPU/NPU retriever 保持同一个 HTTP 协议：

* POST /retrieve: 输入 queries/topk，输出 {"result": ...}
* GET /health: 给 proxy 和 launcher 做轻量 ready 检查
* GET /gpu_status: 兼容旧 launcher/service_manager 的 ready 检查命名

CPU 版不把 FAISS index 再拷贝成 torch tensor，而是直接使用 FAISS CPU index.search。
这样可以避免每个实例额外构造一份 60G 以上的 torch doc embedding，内存更可控。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_corpus(corpus_path: str):
    """加载 wiki corpus。

    这里沿用 GPU retriever 的 datasets loader，保证返回 document 字段结构完全一致。
    num_proc 保持较小，避免 8 个 CPU backend 同时启动时把 CPU 打满。
    """

    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=2)


def load_docs(corpus, doc_idxs: list[int]):
    return [corpus[int(idx)] for idx in doc_idxs]


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError(f"pooling method not implemented: {pooling_method}")


def load_model(model_path: str):
    """加载 CPU encoder。

    CPU retriever 默认使用 float32，优先保证结果稳定；线程数由启动脚本通过
    OMP_NUM_THREADS/MKL_NUM_THREADS/torch.set_num_threads 控制。
    """

    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


class Encoder:
    def __init__(self, model_name: str, model_path: str, pooling_method: str, max_length: int):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.model, self.tokenizer = load_model(model_path=model_path)

    @torch.no_grad()
    def encode(self, query_list: List[str] | str, is_query: bool = True) -> torch.Tensor:
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            prefix = "query: " if is_query else "passage: "
            query_list = [f"{prefix}{query}" for query in query_list]
        elif "bge" in self.model_name.lower() and is_query:
            query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        return query_emb.contiguous()


class CpuFaissRetriever:
    def __init__(
        self,
        *,
        index_path: str,
        corpus_path: str,
        retriever_name: str,
        retriever_model: str,
        topk: int,
        query_batch_size: int,
        cpu_threads: int,
        doc_dtype: str,
    ):
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
            faiss.omp_set_num_threads(cpu_threads)
        self.topk = topk
        self.batch_size = query_batch_size
        self.cpu_threads = cpu_threads
        self.doc_dtype = doc_dtype
        self.request_count = 0

        t0 = time.time()
        self.index = faiss.read_index(index_path)
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(f"only inner-product indexes are supported, metric_type={self.index.metric_type}")
        print(
            json.dumps(
                {
                    "event": "faiss_cpu_index_loaded",
                    "ntotal": int(self.index.ntotal),
                    "dim": int(self.index.d),
                    "metric_type": int(self.index.metric_type),
                    "elapsed_s": round(time.time() - t0, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        self.corpus = load_corpus(corpus_path)
        self.encoder = Encoder(
            model_name=retriever_name,
            model_path=retriever_model,
            pooling_method="mean",
            max_length=256,
        )

    @torch.no_grad()
    def batch_search(self, query_list: List[str], num: int | None = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        request_id = self.request_count + 1
        self.request_count = request_id
        encode_elapsed_s = 0.0
        faiss_elapsed_s = 0.0
        load_docs_elapsed_s = 0.0
        request_start = time.perf_counter()

        all_scores: list[list[float]] = []
        all_idxs: list[list[int]] = []
        for start in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start : start + self.batch_size]
            encode_start = time.perf_counter()
            query_emb = self.encoder.encode(query_batch)
            query_np = query_emb.detach().cpu().numpy().astype(np.float32, copy=False)
            encode_elapsed_s += time.perf_counter() - encode_start
            faiss_start = time.perf_counter()
            scores, idxs = self.index.search(query_np, int(num))
            faiss_elapsed_s += time.perf_counter() - faiss_start
            all_scores.extend(scores.astype(np.float32, copy=False).tolist())
            all_idxs.extend(idxs.astype(np.int64, copy=False).tolist())

        flat_idxs = [int(idx) for row in all_idxs for idx in row]
        load_docs_start = time.perf_counter()
        docs = load_docs(self.corpus, flat_idxs)
        load_docs_elapsed_s = time.perf_counter() - load_docs_start
        results = [docs[i * num : (i + 1) * num] for i in range(len(all_idxs))]
        print(
            json.dumps(
                {
                    "event": "cpu_retriever_request",
                    "request_id": request_id,
                    "query_count": len(query_list),
                    "topk": int(num),
                    "cpu_threads": self.cpu_threads,
                    "query_batch_size": self.batch_size,
                    "encode_elapsed_s": round(encode_elapsed_s, 6),
                    "faiss_elapsed_s": round(faiss_elapsed_s, 6),
                    "load_docs_elapsed_s": round(load_docs_elapsed_s, 6),
                    "total_elapsed_s": round(time.perf_counter() - request_start, 6),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if return_score:
            return results, all_scores
        return results

    def status(self) -> dict:
        return {
            "status": "ok",
            "backend_type": "cpu",
            "doc_embeddings_shape": [int(self.index.ntotal), int(self.index.d)],
            "doc_embeddings_dtype": self.doc_dtype,
            "doc_embeddings_device": "cpu_faiss",
            "cpu_threads": self.cpu_threads,
            "torch_num_threads": torch.get_num_threads(),
            "faiss_omp_threads": faiss.omp_get_max_threads(),
            "pid": os.getpid(),
        }


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever: CpuFaissRetriever
default_topk: int


@app.get("/health")
def health():
    return retriever.status()


@app.get("/gpu_status")
def gpu_status():
    # 保留旧端点名，避免调用方需要区分 CPU/GPU backend。
    return retriever.status()


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or default_topk
    if request.return_scores:
        results, scores = retriever.batch_search(request.queries, num=topk, return_score=True)
    else:
        results = retriever.batch_search(request.queries, num=topk, return_score=False)
        scores = None
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            assert scores is not None
            resp.append([{"document": doc, "score": score} for doc, score in zip(single_result, scores[i])])
        else:
            resp.append(single_result)
    return {"result": resp}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a CPU FAISS dense retriever.")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--doc_dtype", choices=("float32",), default="float32")
    args = parser.parse_args()

    global retriever, default_topk
    default_topk = args.topk
    retriever = CpuFaissRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retriever_name=args.retriever_name,
        retriever_model=args.retriever_model,
        topk=args.topk,
        query_batch_size=args.query_batch_size,
        cpu_threads=args.cpu_threads,
        doc_dtype=args.doc_dtype,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
