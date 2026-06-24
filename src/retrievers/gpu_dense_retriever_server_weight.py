#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    return datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)


def load_docs(corpus, doc_idxs):
    return [corpus[int(idx)] for idx in doc_idxs]


def get_doc_content(doc) -> str:
    if isinstance(doc, dict):
        for key in ("contents", "text", "content", "passage"):
            if key in doc and doc[key]:
                return str(doc[key])
        return " ".join(str(v) for v in doc.values())
    return str(doc)


def load_model(model_path: str, use_fp16: bool, device: str):
    AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    if pooling_method == "cls":
        return last_hidden_state[:, 0]
    if pooling_method == "pooler":
        return pooler_output
    raise NotImplementedError(f"pooling method not implemented: {pooling_method}")


class Encoder:
    def __init__(self, model_name: str, model_path: str, pooling_method: str, max_length: int, use_fp16: bool, device: str):
        self.model_name = model_name
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.device = device
        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, device=device)

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long, device=inputs["input_ids"].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        return query_emb.contiguous()


class GpuTorchFlatRetriever:
    def __init__(
        self,
        *,
        index_path: str,
        corpus_path: str,
        bm25_path: str,
        retriever_name: str,
        retriever_model: str,
        topk: int,
        device: str,
        query_batch_size: int,
        doc_dtype: str,
    ):
        if not device.startswith("cuda"):
            raise ValueError("gpu_dense_retriever_server requires a cuda device")
        self.device = torch.device(device)
        self.topk = topk
        self.batch_size = query_batch_size
        self.rrf_k=60
        self.bm25_path=bm25_path
        t0 = time.time()

        index = faiss.read_index(index_path)
        if not hasattr(index, "get_xb"):
            raise TypeError(f"only FAISS flat indexes with get_xb are supported, got {type(index)}")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError(f"only inner-product indexes are supported, metric_type={index.metric_type}")
        xb = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
        dtype = torch.float16 if doc_dtype == "float16" else torch.float32
        self.doc_embeddings = torch.empty((index.ntotal, index.d), dtype=dtype, device=self.device)
        chunk_size = 262144
        for start in range(0, index.ntotal, chunk_size):
            end = min(start + chunk_size, index.ntotal)
            chunk = torch.from_numpy(np.asarray(xb[start:end], dtype=np.float32))
            self.doc_embeddings[start:end].copy_(chunk.to(self.device, dtype=dtype, non_blocking=False))
        del index, xb
        torch.cuda.synchronize(self.device)
        print(
            json.dumps(
                {
                    "event": "doc_embeddings_loaded_to_gpu",
                    "shape": list(self.doc_embeddings.shape),
                    "dtype": str(self.doc_embeddings.dtype),
                    "device": str(self.doc_embeddings.device),
                    "elapsed_s": round(time.time() - t0, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        self.corpus = load_corpus(corpus_path)

        t1 = time.time()
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.bm25_path)
        print(
            json.dumps(
                {
                    "event": "char_bm25_index_built",
                    # "num_docs": len(tokenized_corpus),
                    "elapsed_s": round(time.time() - t1, 3),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        
        # TODO: 加入图结构数据

        self.encoder = Encoder(
            model_name=retriever_name,
            model_path=retriever_model,
            pooling_method="mean",
            max_length=256,
            use_fp16=device.startswith("cuda"),
            device=device,
        )
        

    @torch.no_grad()
    def _dense_rank(self, query_list: List[str], num: int):
        all_scores = []
        all_idxs = []
        for start in range(0, len(query_list), self.batch_size):
            query_batch = query_list[start : start + self.batch_size]
            query_emb = self.encoder.encode(query_batch)
            query_emb = query_emb.to(device=self.device, dtype=self.doc_embeddings.dtype)
            scores = query_emb @ self.doc_embeddings.T
            top_scores, top_idxs = torch.topk(scores, k=num, dim=1)
            all_scores.extend(top_scores.float().cpu().tolist())
            all_idxs.extend(top_idxs.cpu().tolist())
            del query_emb, scores, top_scores, top_idxs
        return all_idxs, all_scores

    def _bm25_rank(self, query_list: List[str], num: int):
        all_idxs = []
        all_scores = []
        for query in query_list:
            # 使用searcher进行搜索
            hits = self.searcher.search(query, num)
            
            if len(hits) < 1:
                # 如果没有搜索结果，添加空列表
                all_idxs.append([])
                all_scores.append([])
                continue
            
            # 获取分数列表
            scores = [hit.score for hit in hits]
            
            # 如果结果少于num，发出警告
            if len(hits) > num:
                hits = hits[:num]
                scores = scores[:num]
            
            # 获取文档ID（这里假设searcher有docid属性）
            top_idxs = [hit.docid for hit in hits]
            top_scores = scores
            
            all_idxs.append(top_idxs)
            all_scores.append(top_scores)
        
        return all_idxs, all_scores

    def _rrf_fuse(self, dense_idxs: List[int], bm25_idxs: List[int], weights:List[float], num: int):
        fused = {}
        for rank, idx in enumerate(dense_idxs):
            fused[idx] = fused.get(idx, 0.0) + weights[1] / (self.rrf_k + rank + 1)
        for rank, idx in enumerate(bm25_idxs):
            fused[idx] = fused.get(idx, 0.0) + weights[0] / (self.rrf_k + rank + 1)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:num]
        idxs = [idx for idx, _ in ranked]
        scores = [score for _, score in ranked]
        return idxs, scores

    @torch.no_grad()
    def batch_search(self, query_list: List[str], weights: List[float], num: int | None = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        candidate_num = max(num, self.topk)
        dense_idxs, _ = self._dense_rank(query_list, candidate_num)
        bm25_idxs, _ = self._bm25_rank(query_list, candidate_num)

        all_idxs = []
        all_scores = []
        for i in range(len(query_list)):
            fused_idxs, fused_scores = self._rrf_fuse(dense_idxs[i], bm25_idxs[i], weights, num)
            all_idxs.append(fused_idxs)
            all_scores.append(fused_scores)

        flat_idxs = [idx for row in all_idxs for idx in row]
        docs = load_docs(self.corpus, flat_idxs)
        results = []
        offset = 0
        for row in all_idxs:
            results.append(docs[offset : offset + len(row)])
            offset += len(row)
        if return_score:
            return results, all_scores
        return results


class QueryRequest(BaseModel):
    queries: List[str]
    bm25_weight: float
    dense_weight: float
    graph_weight: float = 0.0
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()
retriever: GpuTorchFlatRetriever
default_topk: int


@app.get("/gpu_status")
def gpu_status():
    emb = retriever.doc_embeddings
    return {
        "doc_embeddings_shape": list(emb.shape),
        "doc_embeddings_dtype": str(emb.dtype),
        "doc_embeddings_device": str(emb.device),
        "cuda_memory_allocated": torch.cuda.memory_allocated(emb.device),
        "cuda_memory_reserved": torch.cuda.memory_reserved(emb.device),
    }


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    topk = request.topk or default_topk
    weights=[]
    weights.append(request.bm25_weight)
    weights.append(request.dense_weight)
    # weights.append(request.graph_weight)
    if request.return_scores:
        results, scores = retriever.batch_search(request.queries, weights, num=topk, return_score=True)
    else:
        results = retriever.batch_search(request.queries, weights, num=topk, return_score=False)
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
    parser = argparse.ArgumentParser(description="Launch a GPU-resident torch dense retriever.")
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--bm25_path", required=True)
    # parser.add_argument("--graph_path", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query_batch_size", type=int, default=32)
    parser.add_argument("--doc_dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    global retriever, default_topk
    default_topk = args.topk
    retriever = GpuTorchFlatRetriever(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        bm25_path=args.bm25_path,
        # graph_path=args.graph_path,
        retriever_name=args.retriever_name,
        retriever_model=args.retriever_model,
        topk=args.topk,
        device=args.device,
        query_batch_size=args.query_batch_size,
        doc_dtype=args.doc_dtype,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
