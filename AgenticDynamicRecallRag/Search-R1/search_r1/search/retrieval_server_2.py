import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False, device: str = "cuda"):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, device):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, device=device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn('Not enough documents retrieved!')
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [
                json.loads(self.searcher.doc(hit.docid).raw())['contents'] 
                for hit in hits
            ]
            results = [
                {
                    'title': content.split("\n")[0].strip("\""),
                    'text': "\n".join(content.split("\n")[1:]),
                    'contents': content
                } 
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16,
            device = config.device
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i*num : (i+1)*num] for i in range(len(batch_idxs))]
            
            results.extend(batch_results)
            scores.extend(batch_scores)
            
            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            if self.config.device.startswith("cuda"):
                torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

class GraphRetriever(BaseRetriever):
    """Graph retriever backed by Neo4j.

    Builds an entity-document bipartite graph: spaCy NER over (title + contents)
    extracts entity nodes, each document becomes a Document node, and an Entity
    is linked via :MENTIONS to every Document it appears in.

    Retrieval combines (a) query-entity matches and (b) FAISS vector seeds, then
    expands k hops over Entity-Document edges to gather candidates and scores
    them by entity-overlap + seed-coverage + connectivity.
    """

    def __init__(self, config, encoder=None, faiss_index=None, corpus=None):
        super().__init__(config)
        if corpus is None:
            corpus = load_corpus(config.corpus_path)
        self.corpus = corpus
        self.encoder = encoder
        self.faiss_index = faiss_index

        self.k_hop = getattr(config, "graph_k_hop", 1)
        self.entity_seed_topk = getattr(config, "graph_entity_seed_topk", 10)
        self.vector_seed_topk = getattr(config, "graph_vector_seed_topk", 10)
        self.expand_limit = getattr(config, "graph_expand_limit", 200)
        self.spacy_batch_size = getattr(config, "graph_spacy_batch_size", 64)
        self.write_batch_size = getattr(config, "graph_write_batch_size", 500)

        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_user, config.neo4j_password),
        )

        import spacy
        try:
            self.nlp = spacy.load(
                config.spacy_model,
                disable=["parser", "lemmatizer", "tagger", "attribute_ruler"],
            )
        except OSError as e:
            raise RuntimeError(
                f"spaCy model '{config.spacy_model}' not installed. "
                f"Run: python -m spacy download {config.spacy_model}"
            ) from e

        graph_dir = os.path.dirname(os.path.abspath(config.corpus_path))
        self.flag_path = os.path.join(graph_dir, "neo4j_graph.flag")
        if os.path.exists(self.flag_path):
            print(f"[GraphRetriever] Reusing existing graph; flag at {self.flag_path}")
            with open(self.flag_path) as f:
                print(f.read())
        else:
            print(f"[GraphRetriever] No graph found at {self.flag_path}; building...")
            self._build_graph()

    def close(self):
        if getattr(self, "driver", None) is not None:
            self.driver.close()
    @staticmethod
    def _normalize_entity(text: str) -> str:
        return " ".join(text.lower().split())

    _ENTITY_LABELS = {
        "PERSON", "NORP", "ORG", "GPE", "LOC", "FAC", "PRODUCT",
        "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE",
    }

    def _extract_entities(self, text: str):
        if not text:
            return []
        doc = self.nlp(text[:5000])
        seen = set()
        out = []
        for ent in doc.ents:
            if ent.label_ not in self._ENTITY_LABELS:
                continue
            norm = self._normalize_entity(ent.text)
            if not norm or len(norm) < 2 or norm in seen:
                continue
            seen.add(norm)
            out.append(norm)
        return out

    def _ensure_schema(self):
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS "
                "FOR (d:Document) REQUIRE d.doc_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.name IS UNIQUE"
            )

    def _build_graph(self):
        self._ensure_schema()

        n_docs = len(self.corpus)
        print(f"[GraphRetriever] Building graph over {n_docs} documents...")

        buffer = []
        flushed = 0
        pbar = tqdm(total=n_docs, desc="NER + graph write")

        def doc_text(i):
            row = self.corpus[i]
            title = row.get("title") or ""
            contents = row.get("contents") or row.get("text") or ""
            doc_id = str(row.get("id", i))
            combined = (title + ". " + contents).strip(". ").strip()
            return doc_id, title, combined

        gen = (doc_text(i) for i in range(n_docs))
        items = ((text, (doc_id, title)) for (doc_id, title, text) in gen)

        for doc, ctx in self.nlp.pipe(
            items, as_tuples=True, batch_size=self.spacy_batch_size
        ):
            doc_id, title = ctx
            seen = set()
            ents = []
            for ent in doc.ents:
                if ent.label_ not in self._ENTITY_LABELS:
                    continue
                norm = self._normalize_entity(ent.text)
                if not norm or len(norm) < 2 or norm in seen:
                    continue
                seen.add(norm)
                ents.append(norm)
            buffer.append({"doc_id": doc_id, "title": title, "entities": ents})
            pbar.update(1)
            if len(buffer) >= self.write_batch_size:
                self._flush_batch(buffer)
                flushed += len(buffer)
                buffer.clear()
        if buffer:
            self._flush_batch(buffer)
            flushed += len(buffer)
            buffer.clear()
        pbar.close()

        with open(self.flag_path, "w") as f:
            f.write(
                f"corpus_path={self.config.corpus_path}\n"
                f"index_path={self.config.index_path}\n"
                f"neo4j_uri={self.config.neo4j_uri}\n"
                f"docs_written={flushed}\n"
                f"spacy_model={self.config.spacy_model}\n"
            )
        print(f"[GraphRetriever] Done. Wrote {flushed} docs. Flag: {self.flag_path}")

    def _flush_batch(self, batch):
        with self.driver.session() as session:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (d:Document {doc_id: row.doc_id})
                SET d.title = row.title
                WITH d, row.entities AS ents
                UNWIND ents AS ename
                MERGE (e:Entity {name: ename})
                MERGE (e)-[:MENTIONS]->(d)
                """,
                rows=batch,
            )

    def _row_to_doc(self, doc_id_or_idx):
        try:
            idx = int(doc_id_or_idx)
            if 0 <= idx < len(self.corpus):
                return self.corpus[idx]
        except (ValueError, TypeError):
            pass
        return None

    def _vector_seeds(self, query: str, k: int):
        if self.encoder is None or self.faiss_index is None:
            return [], []
        emb = self.encoder.encode(query)
        scores, idxs = self.faiss_index.search(emb, k=k)
        return idxs[0].tolist(), scores[0].tolist()

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk

        query_entities = self._extract_entities(query)
        vec_idxs, vec_scores = self._vector_seeds(query, self.vector_seed_topk)
        vec_doc_ids = [str(i) for i in vec_idxs]

        candidates = {}

        with self.driver.session() as session:
            if query_entities:
                rec = session.run(
                    """
                    UNWIND $entities AS ename
                    MATCH (e:Entity {name: ename})-[:MENTIONS]->(d:Document)
                    WITH d, count(DISTINCT e) AS hits
                    ORDER BY hits DESC
                    LIMIT $limit
                    RETURN d.doc_id AS doc_id, hits
                    """,
                    entities=query_entities,
                    limit=self.entity_seed_topk,
                )
                for r in rec:
                    candidates[r["doc_id"]] = (
                        candidates.get(r["doc_id"], 0.0) + 2.0 * float(r["hits"])
                    )

            for did, sc in zip(vec_doc_ids, vec_scores):
                candidates[did] = candidates.get(did, 0.0) + 1.5 + 0.5 * float(sc)

            if candidates and self.k_hop > 0:
                seed_ids = list(candidates.keys())
                rec = session.run(
                    f"""
                    MATCH (seed:Document)
                    WHERE seed.doc_id IN $seed_ids
                    MATCH (seed)<-[:MENTIONS]-(e:Entity)-[:MENTIONS]->(other:Document)
                    WHERE other.doc_id <> seed.doc_id
                    WITH other.doc_id AS doc_id, count(DISTINCT e) AS shared
                    ORDER BY shared DESC
                    LIMIT $limit
                    RETURN doc_id, shared
                    """,
                    seed_ids=seed_ids,
                    limit=self.expand_limit,
                )
                for r in rec:
                    candidates[r["doc_id"]] = (
                        candidates.get(r["doc_id"], 0.0) + 0.3 * float(r["shared"])
                    )

        if not candidates:
            if return_score:
                return [], []
            return []

        ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)[:num]
        results, scores = [], []
        for doc_id, score in ranked:
            doc = self._row_to_doc(doc_id)
            if doc is None:
                continue
            results.append(doc)
            scores.append(score)

        if return_score:
            return results, scores
        return results

    def _batch_search(self, query_list, num=None, return_score=False):
        if isinstance(query_list, str):
            query_list = [query_list]
        results, scores = [], []
        for q in query_list:
            r, s = self._search(q, num, True)
            results.append(r)
            scores.append(s)
        if return_score:
            return results, scores
        return results


def get_retriever(config):
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        bm25_index_path: Optional[str] = None,
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
        device: str = "cuda",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "neo4j",
        spacy_model: str = "en_core_web_sm",
        graph_k_hop: int = 1,
        graph_entity_seed_topk: int = 10,
        graph_vector_seed_topk: int = 10,
        graph_expand_limit: int = 200,
        graph_spacy_batch_size: int = 64,
        graph_write_batch_size: int = 500,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.bm25_index_path = bm25_index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size
        self.device = device
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.spacy_model = spacy_model
        self.graph_k_hop = graph_k_hop
        self.graph_entity_seed_topk = graph_entity_seed_topk
        self.graph_vector_seed_topk = graph_vector_seed_topk
        self.graph_expand_limit = graph_expand_limit
        self.graph_spacy_batch_size = graph_spacy_batch_size
        self.graph_write_batch_size = graph_write_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

retrievers: Dict[str, BaseRetriever] = {}


def _run_retrieval(name: str, request: "QueryRequest"):
    retriever = retrievers.get(name)
    if retriever is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Retriever '{name}' is not available on this server.",
        )
    topk = request.topk if request.topk else config.retrieval_topk

    if request.return_scores:
        results, scores = retriever.batch_search(
            query_list=request.queries, num=topk, return_score=True
        )
        resp = []
        for single_result, single_scores in zip(results, scores):
            resp.append([
                {"document": doc, "score": score}
                for doc, score in zip(single_result, single_scores)
            ])
        return {"result": resp}

    results = retriever.batch_search(
        query_list=request.queries, num=topk, return_score=False
    )
    return {"result": results}


@app.post("/retrieve/dense")
def retrieve_dense(request: QueryRequest):
    """Dense (FAISS) retrieval over the wiki-18 corpus."""
    return _run_retrieval("dense", request)


@app.post("/retrieve/bm25")
def retrieve_bm25(request: QueryRequest):
    """BM25 (Lucene/Pyserini) lexical retrieval."""
    return _run_retrieval("bm25", request)


@app.post("/retrieve/graph")
def retrieve_graph(request: QueryRequest):
    """Graph retrieval over the Neo4j entity-document graph."""
    return _run_retrieval("graph", request)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the local multi-retriever server.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="FAISS index path (e5_Flat.index).")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Corpus jsonl (wiki-18.jsonl).")
    parser.add_argument("--bm25_index_path", type=str, default=None, help="Pyserini Lucene index path. If omitted, /retrieve/bm25 is disabled.")
    parser.add_argument("--topk", type=int, default=3, help="Default top-k.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Dense retriever model name (used by encoder).")
    parser.add_argument("--retriever_model", type=str, default="/data01/ms_wksp/agent_up_to_date/models/retriever/e5-base-v2", help="Path of the dense retriever model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for FAISS.')
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--device", type=str, default="cuda", help="Encoder device.")

    parser.add_argument("--neo4j_uri", type=str, default="bolt://localhost:7687", help="Neo4j Bolt URI.")
    parser.add_argument("--neo4j_user", type=str, default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j_password", type=str, required=True, help="Neo4j password.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_sm", help="spaCy model for NER.")
    parser.add_argument("--graph_k_hop", type=int, default=1)
    parser.add_argument("--graph_entity_seed_topk", type=int, default=10)
    parser.add_argument("--graph_vector_seed_topk", type=int, default=10)
    parser.add_argument("--graph_expand_limit", type=int, default=200)
    parser.add_argument("--disable_graph", action="store_true", help="Skip loading graph retriever.")
    parser.add_argument("--disable_dense", action="store_true", help="Skip loading dense retriever.")

    args = parser.parse_args()

    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        bm25_index_path=args.bm25_index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=args.device.startswith("cuda"),
        retrieval_batch_size=512,
        device=args.device,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        spacy_model=args.spacy_model,
        graph_k_hop=args.graph_k_hop,
        graph_entity_seed_topk=args.graph_entity_seed_topk,
        graph_vector_seed_topk=args.graph_vector_seed_topk,
        graph_expand_limit=args.graph_expand_limit,
    )

    dense_retriever = None
    if not args.disable_dense:
        print("[server] Loading DenseRetriever...")
        dense_retriever = DenseRetriever(config)
        retrievers["dense"] = dense_retriever

    if args.bm25_index_path:
        print("[server] Loading BM25Retriever...")
        bm25_config = Config(
            retrieval_method="bm25",
            index_path=args.bm25_index_path,
            corpus_path=args.corpus_path,
            retrieval_topk=args.topk,
        )
        retrievers["bm25"] = BM25Retriever(bm25_config)
    else:
        print("[server] --bm25_index_path not provided; /retrieve/bm25 disabled.")

    if not args.disable_graph:
        print("[server] Loading GraphRetriever...")
        retrievers["graph"] = GraphRetriever(
            config,
            encoder=dense_retriever.encoder if dense_retriever else None,
            faiss_index=dense_retriever.index if dense_retriever else None,
            corpus=dense_retriever.corpus if dense_retriever else None,
        )

    print(f"[server] Available retrievers: {list(retrievers.keys())}")
    uvicorn.run(app, host=args.host, port=args.port)
