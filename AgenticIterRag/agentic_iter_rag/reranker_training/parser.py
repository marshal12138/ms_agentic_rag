"""解析 AIR LLM reranker 的 <reason>/<rerank> 输出。"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


@dataclass
class RerankParseResult:
    """reranker 输出解析结果；invalid 不抛异常，交给 reward 层直接惩罚。"""

    valid: bool
    reason: str | None
    ranked_indices: list[int]
    error_code: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _tag_span(text: str, start_tag: str, end_tag: str) -> tuple[int, int] | None:
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0:
        return None
    return start, end


def _single_tag_pair(text: str, start_tag: str, end_tag: str, error_prefix: str) -> RerankParseResult | None:
    """校验标签只出现一组。

    reranker 输出是可执行 action，不是普通自然语言；多组标签会让 action 边界不清楚，所以直接判格式错。
    """

    if text.count(start_tag) > 1 or text.count(end_tag) > 1:
        return RerankParseResult(False, None, [], f"duplicate_{error_prefix}_tag", f"duplicate {start_tag}...{end_tag}")
    return None


def parse_rerank_response(
    text: str,
    expected_count: int = 5,
    max_index: int = 50,
) -> RerankParseResult:
    """解析 reranker 输出。

    这里不做自动修复。原因是格式错误本身就是模型动作错误，如果自动补齐或猜测排序，
    后续 continuation reward 就不再对应模型真实输出。
    """

    expected_count = int(expected_count)
    max_index = int(max_index)
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    if max_index < expected_count:
        raise ValueError("max_index must be >= expected_count")

    duplicate_reason = _single_tag_pair(text, "<reason>", "</reason>", "reason")
    if duplicate_reason is not None:
        return duplicate_reason
    duplicate_rerank = _single_tag_pair(text, "<rerank>", "</rerank>", "rerank")
    if duplicate_rerank is not None:
        return duplicate_rerank

    reason_span = _tag_span(text, "<reason>", "</reason>")
    if reason_span is None:
        return RerankParseResult(False, None, [], "missing_reason_tag", "missing <reason>...</reason>")
    rerank_span = _tag_span(text, "<rerank>", "</rerank>")
    if rerank_span is None:
        return RerankParseResult(False, None, [], "missing_rerank_tag", "missing <rerank>...</rerank>")
    reason_start, reason_end = reason_span
    rerank_start, rerank_end = rerank_span
    if not (reason_start < reason_end < rerank_start < rerank_end):
        return RerankParseResult(False, None, [], "tag_order_error", "tags must be <reason> then <rerank>")
    suffix_start = rerank_end + len("</rerank>")
    outside = (text[:reason_start] + text[suffix_start:]).strip()
    if outside:
        return RerankParseResult(False, None, [], "extra_text_outside_tags", "extra text outside <reason>/<rerank> tags")

    reason = text[reason_start + len("<reason>") : reason_end].strip()
    if not reason:
        return RerankParseResult(False, None, [], "empty_reason", "<reason> block is empty")
    rerank_body = text[rerank_start + len("<rerank>") : rerank_end].strip()
    if not rerank_body:
        return RerankParseResult(False, reason, [], "empty_rerank", "<rerank> block is empty")

    # 只允许 [数字]、> 和空白。逗号、JSON、doc_id 都视为格式错误。
    # CoSearch 对齐协议下，模型输出 top5，但 index 仍然可以落在 top50 候选池的任意位置。
    leftovers = re.sub(r"\[\d+\]|\s|>", "", rerank_body)
    if leftovers:
        return RerankParseResult(False, reason, [], "invalid_rerank_text", f"invalid token in rerank: {leftovers[:40]}")

    indices = [int(item) for item in re.findall(r"\[(\d+)\]", rerank_body)]
    if len(indices) != expected_count:
        return RerankParseResult(False, reason, indices, "wrong_index_count", f"expected {expected_count}, got {len(indices)}")
    if any(idx < 1 or idx > max_index for idx in indices):
        return RerankParseResult(False, reason, indices, "index_out_of_range", f"indices must be in [1,{max_index}]")
    if len(set(indices)) != len(indices):
        return RerankParseResult(False, reason, indices, "duplicate_index", "indices must be distinct")
    return RerankParseResult(True, reason, indices, None, None)


def render_identity_rerank_response(expected_count: int = 5, max_index: int | None = None) -> str:
    """生成一个合法的 identity reranker 输出，用于 smoke 测试和流程打通。"""

    del max_index
    order = " > ".join(f"[{idx}]" for idx in range(1, int(expected_count) + 1))
    return f"<reason>Smoke test keeps the retriever order to validate the training pipeline.</reason>\n<rerank>{order}</rerank>"
