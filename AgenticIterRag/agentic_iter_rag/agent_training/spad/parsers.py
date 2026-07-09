"""Parsers and lightweight validators for SPAD-RAG XML actions."""

from __future__ import annotations

import re
import json
from dataclasses import asdict, dataclass


@dataclass
class AnswerParseResult:
    valid: bool
    reason: str | None
    answer: str | None
    error_code: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, object | None]:
        return asdict(self)


@dataclass
class StopAtAnswerParseResult:
    valid: bool
    reason: str | None
    error_code: str | None
    error_message: str | None
    assistant_turn_count: int = 1
    missing_reason_count: int = 0
    warning_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object | None]:
        return asdict(self)


def _single_tag(text: str, start_tag: str, end_tag: str) -> tuple[str, str | None]:
    if text.count(start_tag) != 1 or text.count(end_tag) != 1:
        return "", f"expected exactly one {start_tag}...{end_tag}"
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or start + len(start_tag) > end:
        return "", f"malformed {start_tag}...{end_tag}"
    return text[start + len(start_tag) : end].strip(), None


def _tag_span(text: str, start_tag: str, end_tag: str) -> tuple[int, int, int, int, str] | None:
    if text.count(start_tag) != 1 or text.count(end_tag) != 1:
        return None
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or start + len(start_tag) > end:
        return None
    content_start = start + len(start_tag)
    return start, content_start, end, end + len(end_tag), text[content_start:end].strip()


def _extract_assistant_blocks(text: str) -> list[str]:
    """Split an agent-loop trajectory into assistant turns.

    VERL dumps multi-turn trajectories as generated assistant text interleaved
    with ``user\n<tool_response>...`` blocks. The first assistant turn has no
    leading ``assistant\n`` marker, while subsequent turns do.
    """

    chatml_blocks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.S)
    if chatml_blocks:
        return [block.strip() for block in chatml_blocks if block.strip()]

    remaining = text.strip()
    if not remaining:
        return []
    if remaining.startswith("assistant\n"):
        remaining = remaining[len("assistant\n") :]

    blocks: list[str] = []
    while remaining:
        match = re.search(r"\n?user\n<tool_response>", remaining)
        if match is None:
            final = remaining.strip()
            if final:
                blocks.append(final)
            break
        candidate = remaining[: match.start()].strip()
        if candidate:
            blocks.append(candidate)
        rest = remaining[match.start() + len(match.group(0)) :]
        assistant_pos = rest.find("\nassistant\n")
        if assistant_pos < 0:
            break
        remaining = rest[assistant_pos + len("\nassistant\n") :]
    return blocks or [text.strip()]


def _reason_before_action(block: str, action_start: int) -> tuple[str | None, str | None]:
    reason_span = _tag_span(block, "<reason>", "</reason>")
    think_span = _tag_span(block, "<think>", "</think>")
    if reason_span is not None:
        start, _, end, _, content = reason_span
        if start > action_start or end > action_start:
            return None, "reason_after_action"
        if not content:
            return None, "empty_reason"
        return content, None
    if think_span is not None:
        start, _, end, _, content = think_span
        if start > action_start or end > action_start:
            return None, "think_after_action"
        # Qwen3 non-thinking templates can yield an empty <think></think> block.
        return content, None
    if block.count("<reason>") + block.count("</reason>") > 0:
        return None, "malformed_reason_tag"
    if block.count("<think>") + block.count("</think>") > 0:
        return None, "malformed_think_tag"
    return None, "missing_reason"


def _validate_tool_block(block: str, *, allow_missing_reason: bool) -> tuple[bool, str | None, int]:
    tool_span = _tag_span(block, "<tool_call>", "</tool_call>")
    if tool_span is None:
        return False, "invalid_tool_call_tag", 0
    if "<answer>" in block:
        return False, "answer_in_tool_turn", 0
    tool_start, _, _, _, tool_content = tool_span
    try:
        payload = json.loads(tool_content)
    except Exception:
        return False, "invalid_tool_call_json", 0
    if payload.get("name") != "search":
        return False, "invalid_tool_name", 0
    arguments = payload.get("arguments")
    if not isinstance(arguments, dict) or not isinstance(arguments.get("query"), str) or not arguments["query"].strip():
        return False, "invalid_tool_arguments", 0
    _, reason_error = _reason_before_action(block, tool_start)
    if reason_error == "missing_reason" and allow_missing_reason:
        return True, None, 1
    if reason_error is not None:
        return False, reason_error, 0
    return True, None, 0


def parse_search_policy_trajectory_stop(
    text: str,
    *,
    allow_missing_reason: bool = True,
    allow_answer_body: bool = True,
) -> StopAtAnswerParseResult:
    """Parse a Stage 1 trajectory and validate a final ``<answer>`` stop.

    The teacher answer reward should only run when intermediate search actions
    are structurally valid and the actor eventually emits the answer action. The
    answer body is ignored in Stage 1 because search-policy credit is assigned to
    the stop decision, not to actor answer wording.
    """

    blocks = _extract_assistant_blocks(text)
    if not blocks:
        return StopAtAnswerParseResult(False, None, "empty_trajectory", "no assistant turns found", 0)

    missing_reason_count = 0
    for block in blocks[:-1]:
        ok, error_code, missing_reason = _validate_tool_block(block, allow_missing_reason=allow_missing_reason)
        missing_reason_count += missing_reason
        if not ok:
            return StopAtAnswerParseResult(
                False,
                None,
                error_code,
                f"invalid intermediate assistant turn: {error_code}",
                len(blocks),
                missing_reason_count,
            )

    final = blocks[-1]
    answer_start = final.find("<answer>")
    if answer_start < 0:
        return StopAtAnswerParseResult(
            False,
            None,
            "no_finish",
            "final assistant turn does not contain <answer>",
            len(blocks),
            missing_reason_count,
        )
    if final.count("<answer>") != 1:
        return StopAtAnswerParseResult(
            False,
            None,
            "invalid_answer_open_tag",
            "expected exactly one <answer> in final assistant turn",
            len(blocks),
            missing_reason_count,
        )
    if "<tool_call>" in final:
        return StopAtAnswerParseResult(
            False,
            None,
            "tool_call_in_answer_turn",
            "final assistant turn contains both tool_call and answer",
            len(blocks),
            missing_reason_count,
        )
    reason, reason_error = _reason_before_action(final, answer_start)
    if reason_error == "missing_reason" and allow_missing_reason:
        missing_reason_count += 1
    elif reason_error is not None:
        return StopAtAnswerParseResult(
            False,
            reason,
            reason_error,
            f"invalid final assistant reasoning block: {reason_error}",
            len(blocks),
            missing_reason_count,
        )
    suffix = final[answer_start + len("<answer>") :].strip()
    warnings: list[str] = []
    if suffix:
        if not allow_answer_body:
            return StopAtAnswerParseResult(
                False,
                reason,
                "answer_body_present",
                "Stage 1 expected stop at the opening <answer> tag",
                len(blocks),
                missing_reason_count,
            )
        warnings.append("answer_body_present")
    return StopAtAnswerParseResult(
        True,
        reason,
        None,
        None,
        len(blocks),
        missing_reason_count,
        tuple(warnings),
    )


def parse_reason_answer(text: str) -> AnswerParseResult:
    """Parse strict <reason>...</reason><answer>...</answer> output."""

    reason, reason_error = _single_tag(text, "<reason>", "</reason>")
    if reason_error:
        return AnswerParseResult(False, None, None, "invalid_reason_tag", reason_error)
    answer, answer_error = _single_tag(text, "<answer>", "</answer>")
    if answer_error:
        return AnswerParseResult(False, reason, None, "invalid_answer_tag", answer_error)
    if not reason:
        return AnswerParseResult(False, reason, answer, "empty_reason", "<reason> is empty")
    if not answer:
        return AnswerParseResult(False, reason, answer, "empty_answer", "<answer> is empty")
    reason_end = text.find("</reason>")
    answer_start = text.find("<answer>")
    if reason_end > answer_start:
        return AnswerParseResult(False, reason, answer, "tag_order_error", "reason must appear before answer")
    suffix = text[text.find("</answer>") + len("</answer>") :].strip()
    prefix = text[: text.find("<reason>")].strip()
    middle = text[reason_end + len("</reason>") : answer_start].strip()
    if prefix or middle or suffix:
        return AnswerParseResult(False, reason, answer, "extra_text_outside_tags", "extra text outside tags")
    return AnswerParseResult(True, reason, answer, None, None)


def parse_reason_answer_opening_stop(text: str) -> StopAtAnswerParseResult:
    """Parse Stage 1 stop action: <reason>...</reason><answer>.

    Stage 1 deliberately stops at the opening <answer> tag. The answer body is
    not part of search-policy credit assignment, so a missing </answer> is valid
    here but remains invalid for Stage 2/3 answer training.
    """

    if "user\n<tool_response>" in text or "\nassistant\n" in text:
        return parse_search_policy_trajectory_stop(text)
    if text.count("<reason>") != 1 or text.count("</reason>") != 1:
        return StopAtAnswerParseResult(False, None, "invalid_reason_tag", "expected exactly one <reason>...</reason>")
    reason_start = text.find("<reason>")
    reason_end = text.find("</reason>")
    if reason_start < 0 or reason_end < 0 or reason_start + len("<reason>") > reason_end:
        return StopAtAnswerParseResult(False, None, "malformed_reason_tag", "malformed <reason>...</reason>")
    reason = text[reason_start + len("<reason>") : reason_end].strip()
    if not reason:
        return StopAtAnswerParseResult(False, reason, "empty_reason", "<reason> is empty")
    answer_count = text.count("<answer>")
    if answer_count != 1:
        return StopAtAnswerParseResult(False, reason, "invalid_answer_open_tag", "expected exactly one <answer>")
    answer_start = text.find("<answer>")
    if reason_end > answer_start:
        return StopAtAnswerParseResult(False, reason, "tag_order_error", "<reason> must appear before <answer>")
    if "</answer>" in text:
        return StopAtAnswerParseResult(False, reason, "answer_body_present", "Stage 1 must stop at <answer> opening tag")
    prefix = text[:reason_start].strip()
    middle = text[reason_end + len("</reason>") : answer_start].strip()
    suffix = text[answer_start + len("<answer>") :].strip()
    if prefix or middle or suffix:
        return StopAtAnswerParseResult(False, reason, "extra_text_outside_tags", "extra text outside Stage 1 stop tags")
    return StopAtAnswerParseResult(True, reason, None, None)


def extract_last_answer(text: str) -> str | None:
    matches = list(re.finditer(r"<answer>(.*?)</answer>", text, re.S))
    if not matches:
        return None
    return matches[-1].group(1).strip()
