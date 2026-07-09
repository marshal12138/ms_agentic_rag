"""AIR LLM reranker reward 计算工具。"""

from __future__ import annotations

from typing import Any

from agentic_iter_rag.reranker_training.parser import parse_rerank_response
from agentic_iter_rag.reranker_training.rewards.reranker_format_reward import (
    compute_reranker_format_reward_details,
)


def compute_format_only_reward(
    reranker_output_text: str,
    sample_extra_info: dict[str, Any],
    *,
    expected_count: int = 5,
    max_index: int = 50,
    format_penalty: float = -0.5,
    short_valid_score: float = 1.0,
    long_valid_score: float = 0.5,
    length_threshold_tokens: int = 512,
    tokenizer_path: str | None = None,
) -> dict[str, Any]:
    """计算 smoke 阶段使用的格式 reward。

    格式错误直接给 format_penalty，且不触发 continuation。格式正确后再看 response 长度：
    <=512 token 给 1.0，>512 token 给 0.5。这里保持和真实 stage1 reward 完全一致。
    """

    details = compute_reranker_format_reward_details(
        data_source="agentic_iter_rag.llm_reranker.branch_grpo",
        solution_str=reranker_output_text,
        ground_truth=None,
        extra_info=sample_extra_info,
        expected_count=expected_count,
        max_index=max_index,
        format_invalid_score=format_penalty,
        short_valid_score=short_valid_score,
        long_valid_score=long_valid_score,
        length_threshold_tokens=length_threshold_tokens,
        tokenizer_path=tokenizer_path,
    )
    if not details["format_valid"]:
        return {
            "score": float(details["score"]),
            "valid": False,
            "format_valid": False,
            "format_error_code": details["format_error_code"],
            "continuation_status": "skipped_format_error",
            "visible_doc_ids": [],
            "response_length_tokens": details.get("response_length_tokens"),
            "length_penalty_applied": details.get("length_penalty_applied"),
            "parse": details["parse"],
        }
    parsed = parse_rerank_response(reranker_output_text, expected_count=expected_count, max_index=max_index)
    index_to_doc = {str(k): str(v) for k, v in sample_extra_info["candidate_index_to_doc_id"].items()}
    ranked_doc_ids = [index_to_doc[str(idx)] for idx in parsed.ranked_indices]
    return {
        "score": float(details["score"]),
        "valid": True,
        "format_valid": True,
        "format_error_code": None,
        "continuation_status": "smoke_not_run",
        "ranked_doc_ids": ranked_doc_ids,
        "visible_doc_ids": ranked_doc_ids,
        "response_length_tokens": details.get("response_length_tokens"),
        "length_penalty_applied": details.get("length_penalty_applied"),
        "parse": parsed.to_dict(),
    }
