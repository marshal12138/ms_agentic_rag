"""Configuration for the alpha-fusion validation pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List


def _alpha_grid(step: float) -> List[float]:
    n = int(round(1.0 / step))
    return [round(i * step, 4) for i in range(n + 1)]


@dataclass
class Config:
    # data
    dataset_path: str = "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/co_search/raw_sets/nq/test.jsonl"
    corpus_path: str = "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/data/retrieval/wiki-18/wiki-18.jsonl"

    # how many valid samples to collect before stopping
    num_samples: int = 100
    # hard cap on dataset records to scan (None = whole file)
    max_scan: int | None = None

    # llm (OpenAI-compatible chat completions)
    llm_endpoint: str = "http://127.0.0.1:8000/v1/chat/completions"
    llm_model: str = "qwen3-4b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 600.0
    llm_max_retries: int = 2

    # retriever (readme /retrieve: {"queries": [...]} -> {"result": [[bm25_ids, dense_ids], ...]})
    retriever_endpoint: str = "http://localhost:9011/retrieve"
    retriever_timeout_seconds: float = 120.0

    # fusion
    rrf_k: int = 60
    alpha_step: float = 0.1
    topk: int = 5  # final window we report positions within

    # concurrency
    workers: int = 8

    # output
    output_path: str = "results.jsonl"
    report_path: str = "report.png"

    alphas: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.alphas:
            self.alphas = _alpha_grid(self.alpha_step)


def parse_args(argv: List[str] | None = None) -> Config:
    p = argparse.ArgumentParser(description="Validate effect of retriever-fusion alpha on top-k answer placement.")
    p.add_argument("--dataset-path", default=Config.dataset_path)
    p.add_argument("--corpus-path", default=Config.corpus_path)
    p.add_argument("--num-samples", type=int, default=Config.num_samples)
    p.add_argument("--max-scan", type=int, default=None)

    p.add_argument("--llm-endpoint", default=Config.llm_endpoint)
    p.add_argument("--llm-model", default=Config.llm_model)
    p.add_argument("--llm-temperature", type=float, default=Config.llm_temperature)
    p.add_argument("--llm-max-tokens", type=int, default=Config.llm_max_tokens)
    p.add_argument("--llm-timeout-seconds", type=float, default=Config.llm_timeout_seconds)
    p.add_argument("--llm-max-retries", type=int, default=Config.llm_max_retries)

    p.add_argument("--retriever-endpoint", default=Config.retriever_endpoint)
    p.add_argument("--retriever-timeout-seconds", type=float, default=Config.retriever_timeout_seconds)

    p.add_argument("--rrf-k", type=int, default=Config.rrf_k)
    p.add_argument("--alpha-step", type=float, default=Config.alpha_step)
    p.add_argument("--topk", type=int, default=Config.topk)

    p.add_argument("--workers", type=int, default=Config.workers)

    p.add_argument("--output-path", default=Config.output_path)
    p.add_argument("--report-path", default=Config.report_path)

    a = p.parse_args(argv)
    return Config(
        dataset_path=a.dataset_path,
        corpus_path=a.corpus_path,
        num_samples=a.num_samples,
        max_scan=a.max_scan,
        llm_endpoint=a.llm_endpoint,
        llm_model=a.llm_model,
        llm_temperature=a.llm_temperature,
        llm_max_tokens=a.llm_max_tokens,
        llm_timeout_seconds=a.llm_timeout_seconds,
        llm_max_retries=a.llm_max_retries,
        retriever_endpoint=a.retriever_endpoint,
        retriever_timeout_seconds=a.retriever_timeout_seconds,
        rrf_k=a.rrf_k,
        alpha_step=a.alpha_step,
        topk=a.topk,
        workers=a.workers,
        output_path=a.output_path,
        report_path=a.report_path,
    )
