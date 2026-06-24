# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from .base_tool import BaseTool
from .schemas import AgentToolResponse, OpenAIFunctionToolSchema
from .utils.answer_match_reward import compute_average_hit_at_ks, compute_ndcg_at_m, has_answer_in_documents
from .utils.search import call_search_api, format_tool_response

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class CoAgenticRetrieverTool(BaseTool):
    """Recall retriever plus dense ranker search tool.

    The recall retriever service is frozen and returns top-N passages. The
    local dense ranker sorts that same pool and exposes only ranker top-M
    passages to the agent.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema = None):
        self._instance_kwargs = {}
        if tool_schema is None:
            tool_schema = OpenAIFunctionToolSchema(
                type="function",
                function={
                    "name": "search",
                    "description": "Search for relevant documents to answer the user's question with Hybrid Retriever. The sum of three weights should be 1.0.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information.",
                            },
                            "dense_weight": {
                                "type": "float",
                                "description": "The weight of sub Dense Retriever. Value range: [0.0, 1.0], where 0 means disabled and 1 means full contribution."
                            },
                            "bm25_weight": {
                                "type": "float",
                                "description": "The weight of sub BM25 Retriever. Value range: [0.0, 1.0], where 0 means disabled and 1 means full contribution."
                            },
                            "graph_weight": {
                                "type": "float",
                                "description": "The weight of sub Graph Retriever. Value range: [0.0, 1.0], where 0 means disabled and 1 means full contribution."
                            }
                        },
                        "required": ["query", "dense_weight","bm25_weight", "graph_weight"],
                    },
                },
            )

        super().__init__(config, tool_schema)

        self.retrieval_url = config.get("retrieval_service_url")
        self.timeout = config.get("timeout", 30)
        self.default_top_n = int(config.get("default_top_n", config.get("recall_top_k", 50)))
        self.default_top_m = int(config.get("default_top_m", config.get("agent_top_k", 5)))
        self.format_penalty = float(config.get("format_penalty", -0.2))
        self.trivial_answers = set(
            answer.lower().strip()
            for answer in config.get("trivial_answers", ["yes", "no", "true", "false"])
        )

        self.tool_score_metric = config.get("tool_score_metric", "hit")
        if self.tool_score_metric not in ("hit", "ndcg"):
            raise ValueError(f"tool_score_metric must be 'hit' or 'ndcg', got {self.tool_score_metric!r}")

        self.max_retries = int(config.get("max_retries", 3))
        self.retry_delay = float(config.get("retry_delay", 1.0))
        self.retry_backoff = float(config.get("retry_backoff", 2.0))

        ranker_config = dict(config.get("ranker", {}))
        self.ranker_enabled = bool(config.get("ranker_enabled", True))
        self.ranker_top_k = int(ranker_config.get("top_k", self.default_top_m))
        self.ranker_max_query_length = int(ranker_config.get("max_query_length", 192))
        self.ranker_max_doc_length = int(ranker_config.get("max_doc_length", 256))
        self.ranker = None
        if self.ranker_enabled:
            from verl.workers.ranker.e5_ranker_worker import LocalE5RankerWorker

            self.ranker = LocalE5RankerWorker({"ranker": ranker_config})
            self.ranker.init_model()

        max_concurrent_per_worker = int(config.get("max_concurrent_per_worker", 16))
        self._semaphore = asyncio.Semaphore(max_concurrent_per_worker)
        self.search_timing_jsonl = os.getenv("COAGENTIC_RETRIEVER_SEARCH_TIMING_JSONL", "")

        if not self.retrieval_url:
            raise ValueError("retrieval_service_url must be provided in config")

    async def create(self, instance_id=None, create_kwargs=None, **kwargs):
        instance_id, response = await super().create(instance_id, **kwargs)
        self._instance_kwargs[instance_id] = create_kwargs or {}
        return instance_id, response

    async def tool_format(self, params: dict[str, Any])->bool:
        gt=set(self.tool_schema.function["parameters"]["properties"].keys())
        label=set(params.keys())
        # 计算总和
        w_sum=0.0
        for key, val in params.items():
            if "weight" in key:
                try:
                    w_sum += float(val)
                except (TypeError, ValueError):
                    pass
        return gt == label and w_sum == 1.0 
        
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        create_kwargs = self._instance_kwargs.get(instance_id, {})
        query = parameters.get("query")
        
        def to_float_or_none(val):
            try:
                return float(val) if val is not None and not isinstance(val, bool) else None
            except (ValueError, TypeError):
                return None

        dense_weight = to_float_or_none(parameters.get("dense_weight"))
        bm25_weight = to_float_or_none(parameters.get("bm25_weight"))
        graph_weight = to_float_or_none(parameters.get("graph_weight"))
        top_n = int(create_kwargs.get("top_n", self.default_top_n))
        top_m = int(create_kwargs.get("top_m", self.default_top_m))
        answers = create_kwargs.get("answers", [])
        # print("HUIHUHSGUSDFG")
        # print(parameters)
        # print(f"{dense_weight}, {bm25_weight}, {graph_weight}")
        # import pdb;pdb.set_trace()

        if not query:
            logger.error("No query provided to CoAgenticRetrieverTool.execute")
            return AgentToolResponse(text="Error: No query provided"), 0.0, {"ranker_failed": True}

        metrics: dict[str, Any] = {
            "sub_query": query,
            "w_dense":dense_weight,
            "w_bm25": bm25_weight,
            "w_graph": graph_weight,
            "ranker_success": False,
            "ranker_failed": False,
            "ranker_fallback": False,
            "ranker_fallback_reason": None,
            "format_valid": self.tool_format(parameters)
        }

        try:
            recall_docs = await self._call_retrieval_api(query, dense_weight, bm25_weight, graph_weight, top_n)
        except Exception as exc:
            logger.error(f"Recall retriever failed for query {query[:50]!r}: {exc}")
            metrics["ranker_failed"] = True
            metrics["recall_failed"] = True
            return AgentToolResponse(text=f"Recall retriever error: {exc}"), 0.0, metrics

        recall_docs = self._normalize_recall_docs(recall_docs)
        metrics["num_recall_docs"] = len(recall_docs)
        metrics["recall_top50_docs"] = recall_docs[:top_n]

        answers_are_trivial = bool(
            self.trivial_answers
            and answers
            and all(str(answer).lower().strip() in self.trivial_answers for answer in answers)
        )
        metrics["answers_are_trivial"] = answers_are_trivial

        if answers_are_trivial:
            answer_in_docs = False
        else:
            loop = asyncio.get_event_loop()
            answer_in_docs = await loop.run_in_executor(
                None,
                lambda: has_answer_in_documents(answers=answers, documents=recall_docs),
            )
        metrics["answer_in_docs"] = answer_in_docs

        try:
            if not self.ranker_enabled or self.ranker is None:
                ranked_docs = recall_docs
                metrics["ranker_fallback"] = True
                metrics["ranker_fallback_reason"] = "ranker_disabled"
            else:
                ranked_docs = self.ranker.rank_topk(
                    query=query,
                    docs=recall_docs,
                    top_k=len(recall_docs),
                    max_query_length=self.ranker_max_query_length,
                    max_doc_length=self.ranker_max_doc_length,
                )
                metrics["ranker_success"] = True
        except Exception as exc:
            logger.error(f"Dense ranker failed: {type(exc).__name__}: {exc}")
            ranked_docs = recall_docs
            metrics["ranker_failed"] = True
            metrics["ranker_fallback"] = True
            metrics["ranker_fallback_reason"] = "execution_error"
            metrics["ranker_error_type"] = type(exc).__name__
            metrics["ranker_error_message"] = str(exc)

        agent_top_k = min(top_m, self.ranker_top_k, len(ranked_docs))
        final_docs = ranked_docs[:agent_top_k]
        metrics["rank_top50_docs"] = ranked_docs[:top_n]
        metrics["rank_top5_docs"] = final_docs
        metrics["num_ranked_docs"] = len(ranked_docs)

        response_text = format_tool_response(final_docs)
        reward = await self._compute_tool_reward(
            answers=answers,
            recall_docs=recall_docs,
            ranked_docs=ranked_docs,
            final_docs=final_docs,
            top_m=agent_top_k,
            hit_cutoffs=create_kwargs.get("hit_cutoffs", [1, 3, 5]),
            metrics=metrics,
        )

        if metrics["ranker_success"]:
            metrics["tool_score"] = reward if answer_in_docs else 0.0
        elif "tool_score" not in metrics:
            metrics["tool_score"] = 0.0 if not answer_in_docs else reward
            
        if not metrics["format_valid"]:
            metrics["tool_score"] -= 0.2

        return AgentToolResponse(text=response_text), reward, metrics

    async def _compute_tool_reward(
        self,
        *,
        answers: list[str],
        recall_docs: list[dict],
        ranked_docs: list[dict],
        final_docs: list[dict],
        top_m: int,
        hit_cutoffs: list[int],
        metrics: dict[str, Any],
    ) -> float:
        loop = asyncio.get_running_loop()
        if self.tool_score_metric == "ndcg":
            ranked_indices = [int(doc.get("recall_rank", idx + 1)) - 1 for idx, doc in enumerate(ranked_docs[:top_m])]
            reward, num_relevant = await loop.run_in_executor(
                None,
                lambda: compute_ndcg_at_m(
                    answers=answers,
                    all_documents=recall_docs,
                    ranked_indices=ranked_indices,
                    top_m=top_m,
                ),
            )
            metrics["ndcg_at_m"] = reward
            metrics["num_relevant_in_pool"] = num_relevant
            return float(reward)

        reward = await loop.run_in_executor(
            None,
            lambda: compute_average_hit_at_ks(
                answers=answers,
                documents=final_docs,
                hit_cutoffs=hit_cutoffs,
            ),
        )
        metrics["average_hit_at_ks"] = reward
        return float(reward)

    async def _call_retrieval_api(self, query: str, dense_weight: float, bm25_weight: float, graph_weight: float, top_n: int) -> list[dict]:
        last_error = None
        retry_delay = self.retry_delay

        for attempt in range(self.max_retries):
            attempt_start = time.perf_counter()
            try:
                async with self._semaphore:
                    result = await call_search_api(
                        query=query,
                        search_api_url=self.retrieval_url,
                        dense_weight=dense_weight, 
                        bm25_weight=bm25_weight,
                        graph_weight=graph_weight,
                        top_k=top_n,
                        semaphore=None,
                        timeout=self.timeout,
                    )
                    if result["status"] == "error":
                        raise Exception(result["error"])

                    documents = result["documents"]
                    self._write_search_timing(
                        query=query,
                        top_n=top_n,
                        elapsed_s=time.perf_counter() - attempt_start,
                        status="success",
                        attempt=attempt + 1,
                        num_documents=len(documents),
                    )
                    if attempt > 0:
                        logger.info(f"Recall retriever succeeded on attempt {attempt + 1}/{self.max_retries}")
                    return documents
            except Exception as exc:
                last_error = exc
                self._write_search_timing(
                    query=query,
                    top_n=top_n,
                    elapsed_s=time.perf_counter() - attempt_start,
                    status="retry" if attempt < self.max_retries - 1 else "error",
                    attempt=attempt + 1,
                    error=str(exc),
                )
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Recall retriever attempt {attempt + 1}/{self.max_retries} failed: {exc}. "
                        f"Retrying in {retry_delay:.1f}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= self.retry_backoff
                else:
                    logger.error(f"Recall retriever failed after {self.max_retries} attempts. Last error: {exc}")

        raise Exception(f"Recall retriever failed after {self.max_retries} attempts: {last_error}")

    def _write_search_timing(
        self,
        *,
        query: str,
        top_n: int,
        elapsed_s: float,
        status: str,
        attempt: int,
        num_documents: int = 0,
        error: str = "",
    ) -> None:
        if not self.search_timing_jsonl:
            return
        record = {
            "ts": time.time(),
            "pid": os.getpid(),
            "action": "search",
            "elapsed_s": elapsed_s,
            "status": status,
            "attempt": attempt,
            "top_n": top_n,
            "num_documents": num_documents,
            "query_chars": len(query or ""),
            "error": error[:500],
        }
        try:
            with open(self.search_timing_jsonl, "a", encoding="utf-8") as fp:
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning(f"Failed to write search timing record: {exc}")

    @staticmethod
    def _normalize_recall_docs(documents: list[dict]) -> list[dict]:
        normalized = []
        for idx, raw_doc in enumerate(documents, start=1):
            doc = dict(raw_doc)
            doc["recall_rank"] = int(doc.get("recall_rank") or doc.get("rank") or idx)
            if "recall_score" not in doc:
                doc["recall_score"] = doc.get("score")
            normalized.append(doc)
        return normalized

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_kwargs:
            del self._instance_kwargs[instance_id]
