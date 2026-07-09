"""AIR branch reranker continuation reward.

这个模块是 LLM reranker 真正做 GRPO 时需要的 reward 语义层：

1. 解析 reranker 的 <reason>/<rerank> 输出。
2. 格式错误直接给 format_penalty，并且不触发 continuation。
3. 格式正确时，把 reranker 排序后的 top5 文档渲染成新的 tool message。
4. 用 messages_before_tool_response + new_tool_message 继续驱动 frozen agent。
5. 后续 search 只走 retriever，不再调用 reranker。
6. 用最终 answer 计算 answer_reward 或 delta_answer_reward。

训练入口目前默认仍使用 smoke 后端。这个模块先把真实 reward 的可测接口落下来，
后续接入 VERL 后端时可以直接作为 custom_reward_function 使用。
"""

from __future__ import annotations

import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from agentic_iter_rag.reranker_training.parser import parse_rerank_response


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def f1_score(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    ans_tokens = normalize_answer(answer).split()
    if not pred_tokens or not ans_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ans_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ans_tokens)
    return 2 * precision * recall / (precision + recall)


def answer_reward(answer: str, targets: Any) -> float:
    if targets is None:
        return 0.0
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = [str(item) for item in targets]
    return max((f1_score(answer, target) for target in target_list), default=0.0)


def evidence_hit_reward(documents: list[dict[str, Any]], targets: Any) -> float:
    """Return 1 when any selected document contains a normalized gold answer substring."""

    if targets is None:
        return 0.0
    if isinstance(targets, str):
        target_list = [targets]
    else:
        target_list = [str(item) for item in targets]
    normalized_docs = "\n".join(normalize_answer(str(doc.get("contents") or doc.get("text") or "")) for doc in documents)
    for target in target_list:
        normalized_target = normalize_answer(str(target))
        if normalized_target and normalized_target in normalized_docs:
            return 1.0
    return 0.0


def extract_answer(text: str) -> str:
    matches = list(ANSWER_RE.finditer(text or ""))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    matches = list(TOOL_CALL_RE.finditer(text or ""))
    if not matches:
        return [], text or ""
    assistant_for_history = (text or "")[: matches[0].end()]
    payloads: list[dict[str, Any]] = []
    for match in matches:
        try:
            payload = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
        if payload.get("name") != "search":
            continue
        arguments = payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        payloads.append({"name": "search", "arguments": arguments})
    return payloads, assistant_for_history


def normalize_doc(doc: dict[str, Any], rank: int | None = None) -> dict[str, Any]:
    """把增强轨迹 doc 统一成 AIR tool response/retriever 使用的字段。"""

    doc_id = str(doc.get("doc_id") or doc.get("id") or "")
    contents = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
    out = dict(doc)
    out["id"] = str(out.get("id") or doc_id)
    out["doc_id"] = doc_id or out["id"]
    out["contents"] = str(contents)
    out["text"] = str(out.get("text") or contents)
    if rank is not None:
        out.setdefault("recall_rank", rank)
    return out


def format_tool_response(documents: list[dict[str, Any]], max_doc_length: int = 2000) -> str:
    """渲染 agent 可见 top5 observation。

    这里保持和 AIR infer 的 tool response 风格一致，只把 reranker 选出的 top5 暴露给 agent。
    """

    if not documents:
        return "No documents found."
    lines = []
    for idx, doc in enumerate(documents, start=1):
        contents = str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")
        if len(contents) > max_doc_length:
            contents = contents[:max_doc_length] + "..."
        title = str(doc.get("title") or "")
        if title:
            lines.append(f"[{idx}] Title: {title}\n{contents}")
        else:
            lines.append(f"[{idx}] {contents}")
    return "\n".join(lines)


def ranked_docs_from_rerank(indices: list[int], extra_info: dict[str, Any]) -> list[dict[str, Any]]:
    candidate_docs = [normalize_doc(doc) for doc in list(extra_info.get("candidate_docs") or [])]
    index_to_doc_id = {str(k): str(v) for k, v in dict(extra_info.get("candidate_index_to_doc_id") or {}).items()}
    doc_by_id = {str(doc["doc_id"]): doc for doc in candidate_docs}
    ranked: list[dict[str, Any]] = []
    for index in indices:
        doc_id = index_to_doc_id.get(str(index))
        if not doc_id or doc_id not in doc_by_id:
            raise ValueError(f"rerank index {index} cannot be mapped to candidate doc")
        ranked.append(doc_by_id[doc_id])
    return ranked


def build_new_tool_message(
    reranker_output_text: str,
    extra_info: dict[str, Any],
    *,
    expected_count: int = 5,
    max_index: int = 50,
    max_doc_length: int = 2000,
) -> dict[str, Any]:
    """从 reranker 输出构造新的 tool message。

    这个函数只做纯转换，方便单测。真正 continuation 在 compute_air_branch_continuation_reward 里执行。
    """

    # stage2 直接消费 reranker 输出的 top5；不再要求 full50 排序再切前 5。
    parsed = parse_rerank_response(reranker_output_text, expected_count=expected_count, max_index=max_index)
    if not parsed.valid:
        return {
            "valid": False,
            "format_error_code": parsed.error_code,
            "format_error_message": parsed.error_message,
            "tool_message": None,
            "visible_doc_ids": [],
            "ranked_doc_ids": [],
            "parse": parsed.to_dict(),
        }
    ranked_docs = ranked_docs_from_rerank(parsed.ranked_indices, extra_info)
    return {
        "valid": True,
        "format_error_code": None,
        "format_error_message": None,
        "tool_message": {"role": "tool", "content": format_tool_response(ranked_docs, max_doc_length=max_doc_length)},
        "visible_docs": ranked_docs,
        "visible_doc_ids": [str(doc["doc_id"]) for doc in ranked_docs],
        "ranked_doc_ids": [str(doc["doc_id"]) for doc in ranked_docs],
        "parse": parsed.to_dict(),
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def retrieve(query: str, *, retrieval_url: str, top_n: int, timeout: float) -> list[dict[str, Any]]:
    payload = {"queries": [query], "topk": top_n, "return_scores": True}
    data = post_json(retrieval_url, payload, timeout)
    raw_candidates = (data.get("result") or [[]])[0]
    docs: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_candidates, start=1):
        doc = item.get("document", item)
        score = item.get("score", doc.get("score", 0.0))
        docs.append(
            normalize_doc(
                {
                    "id": str(doc.get("id", "")),
                    "doc_id": str(doc.get("doc_id") or doc.get("id", "")),
                    "contents": doc.get("contents") or doc.get("text") or doc.get("passage") or "",
                    "title": doc.get("title", ""),
                    "score": float(score or 0.0),
                    "recall_score": float(score or 0.0),
                },
                rank=idx,
            )
        )
    return docs


@lru_cache(maxsize=1)
def load_tokenizer() -> Any:
    tokenizer_path = os.environ.get("AIR_CONTINUATION_TOKENIZER_PATH") or os.environ.get("AIR_CONTINUATION_AGENT_MODEL")
    if not tokenizer_path:
        raise RuntimeError("AIR_CONTINUATION_TOKENIZER_PATH or AIR_CONTINUATION_AGENT_MODEL must be set")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=env_bool("AIR_CONTINUATION_TRUST_REMOTE_CODE", True))


def apply_chat_template(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    tokenizer = load_tokenizer()
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=env_bool("AIR_CONTINUATION_ENABLE_THINKING", False),
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)


def complete_agent(prompt: str) -> str:
    base_url = os.environ.get("AIR_CONTINUATION_AGENT_BASE_URL")
    model = os.environ.get("AIR_CONTINUATION_AGENT_SERVED_MODEL", "agentic-iter-rag-frozen-agent")
    if not base_url:
        raise RuntimeError("AIR_CONTINUATION_AGENT_BASE_URL must be set for continuation reward")
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": env_int("AIR_CONTINUATION_MAX_RESPONSE_LENGTH", 1024),
        "temperature": env_float("AIR_CONTINUATION_TEMPERATURE", 0.0),
        "top_p": env_float("AIR_CONTINUATION_TOP_P", 1.0),
    }
    data = post_json(f"{base_url.rstrip('/')}/v1/completions", payload, env_float("AIR_CONTINUATION_REQUEST_TIMEOUT", 180.0))
    return data["choices"][0].get("text", "")


def run_continuation(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """继续 rollout。

    后续 search 严格 retriever-only：agent 发出的新 search 只走 retrieve()，不再进入 reranker。
    """

    tokenizer = load_tokenizer()
    retrieval_url = os.environ.get("AIR_CONTINUATION_RETRIEVAL_URL")
    if not retrieval_url:
        raise RuntimeError("AIR_CONTINUATION_RETRIEVAL_URL must be set for continuation reward")
    max_assistant_turns = env_int("AIR_CONTINUATION_MAX_ASSISTANT_TURNS", 6)
    max_user_turns = env_int("AIR_CONTINUATION_MAX_USER_TURNS", 6)
    max_prompt_length = env_int("AIR_CONTINUATION_MAX_PROMPT_LENGTH", 11264)
    max_tool_response_length = env_int("AIR_CONTINUATION_MAX_TOOL_RESPONSE_LENGTH", 4096)
    candidate_top_n = env_int("AIR_CONTINUATION_CANDIDATE_TOP_N", 50)
    visible_top_m = env_int("AIR_CONTINUATION_VISIBLE_TOP_M", 5)
    request_timeout = env_float("AIR_CONTINUATION_REQUEST_TIMEOUT", 180.0)

    history = [dict(item) for item in messages]
    user_turns = 1
    assistant_turns = 0
    search_count = 0
    started = time.perf_counter()

    while assistant_turns < max_assistant_turns and user_turns <= max_user_turns:
        prompt_ids = apply_chat_template(history, add_generation_prompt=True)
        if max_prompt_length > 0 and len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
        prompt = tokenizer.decode(prompt_ids)
        assistant_text = complete_agent(prompt)
        assistant_turns += 1
        answer = extract_answer(assistant_text)
        if answer:
            history.append({"role": "assistant", "content": assistant_text})
            return {
                "status": "answered",
                "answer": answer,
                "messages": history,
                "assistant_turns": assistant_turns,
                "user_turns": user_turns,
                "search_count": search_count,
                "elapsed_s": time.perf_counter() - started,
            }

        tool_calls, assistant_for_history = parse_tool_calls(assistant_text)
        if len(tool_calls) != 1:
            history.append({"role": "assistant", "content": assistant_text})
            return {
                "status": "no_valid_single_tool_call",
                "answer": "",
                "messages": history,
                "assistant_turns": assistant_turns,
                "user_turns": user_turns,
                "search_count": search_count,
                "elapsed_s": time.perf_counter() - started,
            }
        query = str((tool_calls[0].get("arguments") or {}).get("query") or "").strip()
        history.append({"role": "assistant", "content": assistant_for_history})
        if not query:
            history.append({"role": "tool", "content": "Error: No query provided"})
        else:
            docs = retrieve(query, retrieval_url=retrieval_url, top_n=candidate_top_n, timeout=request_timeout)
            visible_docs = docs[:visible_top_m]
            history.append(
                {
                    "role": "tool",
                    "content": format_tool_response(visible_docs, max_doc_length=max_tool_response_length),
                }
            )
        search_count += 1
        user_turns += 1

    return {
        "status": "max_turns",
        "answer": "",
        "messages": history,
        "assistant_turns": assistant_turns,
        "user_turns": user_turns,
        "search_count": search_count,
        "elapsed_s": time.perf_counter() - started,
    }


def compute_air_branch_continuation_reward_details(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """计算 AIR reranker continuation reward 的详细结果。

    注意：格式错误不触发 continuation，因为这时 reranker action 不可执行。
    这个函数保留 answer、visible_doc_ids 等调试字段，供 smoke、单测和人工审计使用；
    真正给 VERL 的 custom reward 入口会只返回分数，避免调试字段污染 non_tensor_batch。
    """

    expected_count = int(kwargs.get("expected_count") or os.environ.get("AIR_CONTINUATION_VISIBLE_TOP_M") or 5)
    max_index = int(kwargs.get("max_index") or os.environ.get("AIR_CONTINUATION_CANDIDATE_TOP_N") or 50)
    format_penalty = float(kwargs.get("format_penalty") or os.environ.get("AIR_CONTINUATION_FORMAT_PENALTY") or -0.5)
    reward_strategy = str(kwargs.get("reward_strategy") or os.environ.get("AIR_CONTINUATION_REWARD_STRATEGY") or "answer_reward")
    evidence_hit_weight = float(
        kwargs.get("evidence_hit_weight")
        or os.environ.get("AIR_CONTINUATION_EVIDENCE_HIT_WEIGHT")
        or 0.0
    )

    action = build_new_tool_message(
        solution_str,
        extra_info,
        expected_count=expected_count,
        max_index=max_index,
        max_doc_length=env_int("AIR_CONTINUATION_MAX_TOOL_RESPONSE_LENGTH", 4096),
    )
    if not action["valid"]:
        return {
            "score": format_penalty,
            "answer_score": 0.0,
            "evidence_hit_score": 0.0,
            "valid": False,
            "format_valid": False,
            "format_error_code": action["format_error_code"],
            "continuation_status": "skipped_format_error",
            "answer": "",
            "visible_doc_ids": [],
            "assistant_turns": 0,
            "user_turns": 0,
            "search_count": 0,
            "elapsed_s": 0.0,
        }

    evidence_score = evidence_hit_reward(list(action.get("visible_docs") or []), ground_truth.get("target"))
    if env_bool("AIR_CONTINUATION_SKIP_AGENT", False):
        return {
            "score": 0.0,
            "answer_score": 0.0,
            "evidence_hit_score": float(evidence_score),
            "valid": True,
            "format_valid": True,
            "continuation_status": "skipped_by_env",
            "visible_doc_ids": action["visible_doc_ids"],
            "answer": "",
            "assistant_turns": 0,
            "user_turns": 0,
            "search_count": 0,
            "elapsed_s": 0.0,
        }

    messages_before = list(extra_info.get("messages_before_tool_response") or [])
    if not messages_before:
        raise RuntimeError("extra_info.messages_before_tool_response is required for continuation reward")
    messages = [dict(item) for item in messages_before] + [dict(action["tool_message"])]
    continuation = run_continuation(messages)
    answer_score = answer_reward(continuation["answer"], ground_truth.get("target"))
    score = answer_score
    baseline = extra_info.get("baseline_reward")
    if reward_strategy == "delta_answer_reward":
        if baseline is None:
            raise RuntimeError("baseline_reward is required for delta_answer_reward")
        score = answer_score - float(baseline)
    elif reward_strategy == "evidence_hit_reward":
        score = evidence_score
    elif reward_strategy == "answer_reward_plus_evidence_hit":
        weight = min(1.0, max(0.0, evidence_hit_weight))
        score = (1.0 - weight) * answer_score + weight * evidence_score
    elif reward_strategy != "answer_reward":
        raise RuntimeError(f"unsupported AIR continuation reward_strategy={reward_strategy!r}")
    return {
        "score": float(score),
        "answer_score": float(answer_score),
        "evidence_hit_score": float(evidence_score),
        "valid": True,
        "format_valid": True,
        "continuation_status": continuation["status"],
        "answer": continuation["answer"],
        "visible_doc_ids": action["visible_doc_ids"],
        "assistant_turns": continuation["assistant_turns"],
        "user_turns": continuation["user_turns"],
        "search_count": continuation["search_count"],
        "elapsed_s": continuation["elapsed_s"],
    }


def compute_air_branch_continuation_reward(
    data_source: str,
    solution_str: str,
    ground_truth: dict[str, Any],
    extra_info: dict[str, Any],
    **kwargs: Any,
) -> float:
    """VERL custom_reward_function 入口，只返回可训练的 reward 分数。

    VERL 的 naive reward manager 会把 dict 返回值里的每个字段都写回 ``non_tensor_batch``。
    AIR 的调试字段里有 answer、visible_doc_ids 这类字符串/列表字段，在 rollout n 和 DP 分片下容易
    和 tensor batch 尺寸不一致。因此训练入口只返回 float 分数；详细字段由
    ``compute_air_branch_continuation_reward_details`` 在 smoke/审计场景里使用。
    """

    result = compute_air_branch_continuation_reward_details(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
    return float(result["score"])
