"""AIR LLM reranker stage1 格式和长度 reward。

stage1 不启动 frozen agent，也不调用 retriever。它只判断模型输出是不是合法 reranker action，
并在格式合法时鼓励输出控制在 512 token 以内。
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from agentic_iter_rag.reranker_training.parser import parse_rerank_response


def _maybe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def _load_tokenizer(tokenizer_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def _response_length_tokens(solution_str: str, extra_info: dict[str, Any], tokenizer_path: str | None) -> int:
    """计算 response token 长度。

    VERL naive reward manager 默认不把 response_mask 直接传给 custom reward，所以这里先读取
    extra_info 中可能存在的长度字段；如果拿不到，再用 reranker tokenizer 对输出重新 tokenize。
    两者都不可用时 fail-fast，避免把未知长度误当成短输出给最高分。
    """

    for key in ("response_length_tokens", "response_token_length", "response_length"):
        value = _maybe_int(extra_info.get(key))
        if value is not None:
            return value
    rollout_scores = extra_info.get("rollout_reward_scores")
    if isinstance(rollout_scores, dict):
        for key in ("response_length_tokens", "response_token_length", "response_length"):
            value = _maybe_int(rollout_scores.get(key))
            if value is not None:
                return value
    if not tokenizer_path:
        raise RuntimeError(
            "reranker_format_reward requires response token length or tokenizer_path for length fallback"
        )
    tokenizer = _load_tokenizer(str(tokenizer_path))
    return len(tokenizer.encode(solution_str, add_special_tokens=False))


def _has_valid_rerank_block(text: str, expected_count: int, max_index: int) -> bool:
    """Check whether a response has one executable <rerank> block despite a missing reason tag."""

    start_tag = "<rerank>"
    end_tag = "</rerank>"
    if text.count(start_tag) != 1 or text.count(end_tag) != 1:
        return False
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or start >= end:
        return False
    body = text[start + len(start_tag) : end].strip()
    if not body:
        return False
    leftovers = re.sub(r"\[\d+\]|\s|>", "", body)
    if leftovers:
        return False
    indices = [int(item) for item in re.findall(r"\[(\d+)\]", body)]
    if len(indices) != expected_count:
        return False
    if any(idx < 1 or idx > max_index for idx in indices):
        return False
    return len(set(indices)) == len(indices)


def compute_reranker_format_reward_details(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | None,
    extra_info: dict[str, Any] | None,
    **kwargs: Any,
) -> dict[str, Any]:
    """返回 stage1 reward 详细信息，方便单测和人工审计。

    结构性错误仍给负分；对空 rerank、index 数量不对、重复/越界 index 这类可恢复错误给分层信号，
    避免 GRPO 在格式开始变坏后整组 reward 全部塌成同一个负分。
    """

    del data_source, ground_truth
    info = extra_info if isinstance(extra_info, dict) else {}
    # expected_count 是模型必须输出的 index 数量；max_index 是候选池最大编号。
    # CoSearch 对齐协议下二者不同：输出 5 个 index，但合法范围是 [1, 50]。
    expected_count = int(kwargs.get("expected_count", kwargs.get("visible_top_m", 5)))
    max_index = int(kwargs.get("max_index", kwargs.get("candidate_top_n", 50)))
    format_invalid_score = float(kwargs.get("format_invalid_score", -0.5))
    short_valid_score = float(kwargs.get("short_valid_score", 1.0))
    long_valid_score = float(kwargs.get("long_valid_score", 0.8))
    length_threshold_tokens = int(kwargs.get("length_threshold_tokens", 512))
    tokenizer_path = kwargs.get("tokenizer_path")
    partial_credit_enabled = bool(kwargs.get("partial_credit_enabled", True))
    empty_rerank_score = float(kwargs.get("empty_rerank_score", -0.2))
    wrong_index_count_base_score = float(kwargs.get("wrong_index_count_base_score", 0.0))
    wrong_index_count_span_score = float(kwargs.get("wrong_index_count_span_score", 0.4))
    wrong_index_count_max_score = float(kwargs.get("wrong_index_count_max_score", 0.4))
    duplicate_or_out_of_range_score = float(kwargs.get("duplicate_or_out_of_range_score", 0.2))
    invalid_rerank_text_score = float(kwargs.get("invalid_rerank_text_score", -0.1))
    missing_reason_with_valid_rerank_score = float(kwargs.get("missing_reason_with_valid_rerank_score", 0.1))

    parsed = parse_rerank_response(solution_str, expected_count=expected_count, max_index=max_index)
    if not parsed.valid:
        score = format_invalid_score
        if partial_credit_enabled:
            if parsed.error_code == "empty_rerank":
                score = empty_rerank_score
            elif parsed.error_code == "wrong_index_count":
                valid_unique = {
                    idx for idx in parsed.ranked_indices
                    if 1 <= int(idx) <= max_index
                }
                ratio = min(1.0, len(valid_unique) / max(1, expected_count))
                score = min(wrong_index_count_max_score, wrong_index_count_base_score + ratio * wrong_index_count_span_score)
            elif parsed.error_code in {"duplicate_index", "index_out_of_range"}:
                score = duplicate_or_out_of_range_score
            elif parsed.error_code == "invalid_rerank_text":
                score = invalid_rerank_text_score
            elif parsed.error_code == "missing_reason_tag" and _has_valid_rerank_block(
                solution_str,
                expected_count=expected_count,
                max_index=max_index,
            ):
                score = missing_reason_with_valid_rerank_score
        return {
            "score": score,
            "valid": False,
            "format_valid": False,
            "format_error_code": parsed.error_code,
            "format_error_message": parsed.error_message,
            "expected_count": expected_count,
            "max_index": max_index,
            "response_length_tokens": None,
            "length_threshold_tokens": length_threshold_tokens,
            "length_penalty_applied": False,
            "parse": parsed.to_dict(),
        }

    response_length = _response_length_tokens(solution_str, info, str(tokenizer_path) if tokenizer_path else None)
    length_penalty_applied = response_length > length_threshold_tokens
    score = long_valid_score if length_penalty_applied else short_valid_score
    return {
        "score": float(score),
        "valid": True,
        "format_valid": True,
        "format_error_code": None,
        "format_error_message": None,
        "expected_count": expected_count,
        "max_index": max_index,
        "ranked_indices": parsed.ranked_indices,
        "response_length_tokens": response_length,
        "length_threshold_tokens": length_threshold_tokens,
        "length_penalty_applied": length_penalty_applied,
        "parse": parsed.to_dict(),
    }


def compute_reranker_format_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any] | None,
    extra_info: dict[str, Any] | None,
    **kwargs: Any,
) -> float:
    """VERL custom_reward_function 入口。

    训练入口只返回 float，避免 VERL naive reward manager 把调试字段写入 non_tensor_batch 后出现尺寸问题。
    """

    result = compute_reranker_format_reward_details(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    return float(result["score"])
