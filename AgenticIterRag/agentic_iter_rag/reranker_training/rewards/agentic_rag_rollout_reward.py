"""AIR LLM reranker stage2 agentic RAG rollout reward。

stage2 才会把 reranker 输出接回 frozen search agent 的历史上下文里继续 rollout。这个文件只是新的
reward 入口命名，底层复用已有 continuation reward 实现，避免复制 agent/retriever 交互逻辑。
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agentic_iter_rag.reranker_training.continuation_reward import (
    compute_air_branch_continuation_reward,
    compute_air_branch_continuation_reward_details,
)

_BATCH_REWARD_EXTRA_DEFAULTS: dict[str, Any] = {
    "score": 0.0,
    "answer_score": 0.0,
    "evidence_hit_score": 0.0,
    "valid": False,
    "format_valid": False,
    "format_error_code": "",
    "continuation_status": "",
    "assistant_turns": 0,
    "user_turns": 0,
    "search_count": 0,
    "elapsed_s": 0.0,
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sanitize_batch_reward_extra(details: dict[str, Any]) -> dict[str, Any]:
    """Keep only fixed-shape fields that VERL can convert into numpy arrays."""

    merged = {**_BATCH_REWARD_EXTRA_DEFAULTS, **details}
    return {
        "score": _as_float(merged.get("score")),
        "answer_score": _as_float(merged.get("answer_score")),
        "evidence_hit_score": _as_float(merged.get("evidence_hit_score")),
        "valid": bool(merged.get("valid")),
        "format_valid": bool(merged.get("format_valid")),
        "format_error_code": str(merged.get("format_error_code") or ""),
        "continuation_status": str(merged.get("continuation_status") or ""),
        "assistant_turns": _as_int(merged.get("assistant_turns")),
        "user_turns": _as_int(merged.get("user_turns")),
        "search_count": _as_int(merged.get("search_count")),
        "elapsed_s": _as_float(merged.get("elapsed_s")),
    }


def compute_agentic_rag_rollout_reward_details(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """返回 stage2 agentic rollout reward 的详细结果。

    这里保留独立函数名，是为了让配置语义和训练计划一致；内部继续复用 continuation reward 的实现。
    """

    return compute_air_branch_continuation_reward_details(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_agentic_rag_rollout_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> float:
    """VERL custom_reward_function 入口，只返回 float 分数。"""

    return compute_air_branch_continuation_reward(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )


def compute_agentic_rag_rollout_reward_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict[str, Any]],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """stage2 batch reward 入口。

    VERL 的 batch reward manager 会一次传入整个 rollout batch。这里用线程池并发执行
    continuation reward，让 64*n 条样本同时打到 frozen-agent proxy；vLLM backend 再做
    continuous batching。格式错误样本仍然会在 continuation reward 内直接返回惩罚，不会触发 agent。
    """

    item_count = len(solution_strs)
    if not (len(data_sources) == len(ground_truths) == len(extra_infos) == item_count):
        raise ValueError("AIR stage2 batch reward inputs must have the same length")
    if item_count == 0:
        return []

    max_workers = int(
        kwargs.get("batch_workers")
        or os.environ.get("AIR_CONTINUATION_BATCH_WORKERS")
        or min(64, item_count)
    )
    max_workers = max(1, min(max_workers, item_count))
    results: list[dict[str, Any]] = [
        _sanitize_batch_reward_extra({"score": 0.0}) for _ in range(item_count)
    ]

    def compute_one(index: int) -> dict[str, Any]:
        return compute_agentic_rag_rollout_reward_details(
            data_source=str(data_sources[index]),
            solution_str=str(solution_strs[index]),
            ground_truth=ground_truths[index],
            extra_info=extra_infos[index],
            **kwargs,
        )

    started = time.perf_counter()
    print(
        f"[AIR stage2 batch reward] start batch_size={item_count} workers={max_workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(compute_one, idx): idx for idx in range(item_count)}
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = _sanitize_batch_reward_extra(dict(future.result()))
    elapsed_s = time.perf_counter() - started
    print(
        f"[AIR stage2 batch reward] done batch_size={item_count} workers={max_workers} elapsed_s={elapsed_s:.3f}",
        flush=True,
    )
    return results
