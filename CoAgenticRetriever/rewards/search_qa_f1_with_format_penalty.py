# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 Search-R1 Contributors
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
# Adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/verl/utils/reward_score/qa_em.py

import random
import re
import string
from collections import Counter


def _scalar_value(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar_value(value[0])
    return value


def _as_bool(value, default=False):
    value = _scalar_value(value)
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
        return default
    return bool(value)


def _as_int(value, default=0):
    value = _scalar_value(value)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _has_min_one_search(solution_str: str, extra_info: dict) -> bool:
    if extra_info is None:
        extra_info = {}

    if extra_info.get("min_one_search") is not None:
        return _as_bool(extra_info.get("min_one_search"))
    if extra_info.get("has_search_tool_call") is not None:
        return _as_bool(extra_info.get("has_search_tool_call"))
    if extra_info.get("tool_call_count") is not None:
        return _as_int(extra_info.get("tool_call_count")) > 0

    return "<tool_call>" in solution_str and "</tool_call>" in solution_str


def _detect_reasoning_tags(processed_str: str):
    candidates = [
        ("<think>", "</think>"),
        ("<reason>", "</reason>"),
    ]
    for start_tag, end_tag in candidates:
        if processed_str.count(start_tag) == 1 and processed_str.count(end_tag) == 1:
            return start_tag, end_tag
    return None, None


def _tag_span(text: str, start_tag: str, end_tag: str):
    if text.count(start_tag) != 1 or text.count(end_tag) != 1:
        return None
    start = text.find(start_tag)
    end = text.find(end_tag)
    if start < 0 or end < 0 or start > end:
        return None
    content_start = start + len(start_tag)
    if content_start > end:
        return None
    return start, content_start, end, end + len(end_tag), text[content_start:end]


def _extract_assistant_blocks(full_text: str) -> list[str]:
    chatml_blocks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", full_text, re.DOTALL)
    if chatml_blocks:
        return [block.strip() for block in chatml_blocks if block.strip()]

    text = full_text.strip()
    if not text:
        return []

    # Agent-loop rollout dumps store the generated trajectory as:
    #   <assistant turn 1> \nuser\n<tool_response>...\nassistant\n<assistant turn 2>
    # The first assistant turn usually has no leading "assistant\n" marker.
    blocks: list[str] = []
    if text.startswith("assistant\n"):
        text = text[len("assistant\n") :]

    while text:
        user_match = re.search(r"\n?user\n<tool_response>", text)
        user_pos = user_match.start() if user_match is not None else -1
        if user_pos < 0:
            candidate = text.strip()
            if candidate:
                blocks.append(candidate)
            break

        candidate = text[:user_pos].strip()
        if candidate:
            blocks.append(candidate)

        rest = text[user_match.start() + len(user_match.group(0)) :] if user_match is not None else ""
        assistant_pos = rest.find("\nassistant\n")
        if assistant_pos < 0:
            break
        text = rest[assistant_pos + len("\nassistant\n") :]

    if blocks:
        return blocks
    return [text]


def validate_response_structure(processed_str: str, do_print: bool, answer_turn=False) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    if do_print:
        print("\n[Structure Validation]")

    if answer_turn:
        action_start_tag, action_end_tag = "<answer>", "</answer>"
        forbidden_start_tag = "<tool_call>"
    else:
        action_start_tag, action_end_tag = "<tool_call>", "</tool_call>"
        forbidden_start_tag = "<answer>"

    action_span = _tag_span(processed_str, action_start_tag, action_end_tag)
    if action_span is None:
        if do_print:
            print(f"  [Error] Expected exactly one {action_start_tag}...{action_end_tag} block")
        return False
    if processed_str.count(forbidden_start_tag) > 0:
        if do_print:
            print(f"  [Error] Unexpected {forbidden_start_tag} block in this assistant turn")
        return False

    action_start, _, action_end, _, action_content = action_span
    if not action_content.strip():
        if do_print:
            print(f"  [Error] {action_start_tag}...{action_end_tag} is empty")
        return False

    reason_span = _tag_span(processed_str, "<reason>", "</reason>")
    think_span = _tag_span(processed_str, "<think>", "</think>")
    if reason_span is None and processed_str.count("<reason>") + processed_str.count("</reason>") > 0:
        if do_print:
            print("  [Error] Malformed <reason>...</reason> block")
        return False
    if think_span is None and processed_str.count("<think>") + processed_str.count("</think>") > 0:
        if do_print:
            print("  [Error] Malformed <think>...</think> block")
        return False

    if reason_span is not None:
        reason_start, _, reason_end, _, reason_content = reason_span
        if reason_start > action_start or reason_end > action_start:
            if do_print:
                print("  [Error] <reason> block must appear before the action block")
            return False
        if not reason_content.strip():
            if do_print:
                print("  [Error] <reason>...</reason> is empty")
            return False
    elif think_span is not None:
        think_start, _, think_end, _, _ = think_span
        if think_start > action_start or think_end > action_start:
            if do_print:
                print("  [Error] <think> block must appear before the action block")
            return False
        # Empty <think></think> is valid in DeepSeek/Qwen non-thinking chat templates.
    else:
        if do_print:
            print("  [Error] Expected a reasoning block before the action block")
        return False

    if do_print:
        print("  Tag sequence validation passed")
        print(f"  {action_start_tag}...{action_end_tag} is valid (length={len(action_content.strip())})")

    if action_end < action_start:
        if do_print:
            print("  [Error] Action tags are out of order")
        return False

    return True

def compute_format_reward(full_text: str) -> bool:
    assistant_blocks = _extract_assistant_blocks(full_text)
    if not assistant_blocks:
        return False

    format_rewards = []
    for i, block in enumerate(assistant_blocks): 
        if i == len(assistant_blocks) - 1: 
            format_r = validate_response_structure(block, do_print=False, answer_turn=True)
        else:
            format_r = validate_response_structure(block, do_print=False, answer_turn=False) 

        format_rewards.append(format_r) 

    return all(format_rewards)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]

    f1_score = 0. 
    for golden_answer in golden_answers:
        f1_score = max(f1_score, f1(prediction, golden_answer))
    return f1_score

def f1(prediction, answer):
    """Compute the F1 score between the prediction and the answer.
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(answer).split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0  matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def count_answer_tags(text):
    opening_tags = text.count("<answer>")
    closing_tags = text.count("</answer>")

    return opening_tags, closing_tags


def search_qa_f1_penalty_compute_score(data_source, solution_str, ground_truth, extra_info, format_penalty=-0.2, **kwargs):
    """The scoring function for F1.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        extra_info: extra information
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_penalty: the penalty for incorrect format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    assert format_penalty <= 0.0, "format_penalty should be non-positive"

    ans_score = 0.0 
    if answer is not None:
        ans_score = compute_f1(answer, ground_truth["target"])

    format_valid = compute_format_reward(solution_str)
    min_one_search = _has_min_one_search(solution_str, extra_info)
    is_format_correct = format_valid and min_one_search

    if is_format_correct:
        total_score = ans_score
    else:
        total_score = format_penalty
    
    result = {
        "score": total_score,
        "valid": is_format_correct,
        "f1": ans_score,
        "format_valid": format_valid,
        "min_one_search": min_one_search,
        "tool_call_count": _as_int(extra_info.get("tool_call_count")) if extra_info else 0,
    }
    if extra_info and extra_info.get("json_correct") is not None:
        result["json_correct"] = extra_info.get("json_correct")

    if extra_info and extra_info.get("one_tool_call_per_assistant") is not None:
        result["one_tool_call_per_assistant"] = extra_info.get("one_tool_call_per_assistant")

    if extra_info and extra_info.get("invalid_direct_answer_before_search") is not None:
        result["invalid_direct_answer_before_search"] = extra_info.get("invalid_direct_answer_before_search")

    # print(f"🔧 [DEBUG] answer: {answer}, ground_truth: {ground_truth['target']}, score: {total_score}, ans_score: {ans_score}, format_correct: {float(is_format_correct)}")
    return result
