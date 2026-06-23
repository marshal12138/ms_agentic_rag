#!/usr/bin/env python3
"""vLLM-only CoSearch evaluator.

This script deliberately does not import or run VERL. It uses:
- vLLM OpenAI-compatible completion APIs for the agent and reranker models.
- The local dense retriever HTTP API for top-50 retrieval.
- The same dataset prompt, reranker prompt, tool response format, and EM/F1
  normalization used by the CoSearch training path.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import os
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
TRAIN_TOOL_UTILS = ROOT / "CoSearch" / "verl" / "verl" / "tools" / "utils"


def load_training_utils_module(module_name: str, path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"training utility module not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load training utility module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_prompts = load_training_utils_module("cosearch_train_prompts", TRAIN_TOOL_UTILS / "prompts.py")
train_search = load_training_utils_module("cosearch_train_search", TRAIN_TOOL_UTILS / "search.py")
train_validates = load_training_utils_module("cosearch_train_validates", TRAIN_TOOL_UTILS / "validates.py")

TRAIN_RERANK_PROMPT_WITH_INITIAL_QUERY = train_prompts.RERANK_PROMPT_WITH_INITIAL_QUERY
train_format_tool_response = train_search.format_tool_response
train_format_tool_response_with_docid_map = train_search.format_tool_response_with_docid_map
train_validate_rerank_output = train_validates.validate_rerank_output


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)


HF_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


@dataclass
class EvalArgs:
    run_mode: str
    agent_model: Path
    reranker_model: Path | None
    data_path: Path
    max_eval_num: int
    batch_size: int
    keep_trace: str
    trace_dir: Path
    report_path: Path
    strategy_name: str
    retrieval_url: str
    agent_base_url: str
    reranker_base_url: str | None
    agent_served_model: str
    reranker_served_model: str
    top_n: int
    top_m: int
    max_assistant_turns: int
    max_user_turns: int
    max_tool_response_length: int
    max_prompt_length: int
    max_response_length: int
    reranker_max_prompt_length: int
    reranker_max_response_length: int
    temperature: float
    top_p: float
    reranker_temperature: float
    request_timeout: float
    llm_io_jsonl: Path | None


def write_llm_io_trace(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    max_records = int(os.getenv("COSEARCH_LLM_IO_MAX_RECORDS", "0") or 0)
    if max_records > 0 and path.exists():
        with path.open("r", encoding="utf-8") as fp:
            if sum(1 for _ in fp) >= max_records:
                return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps({"ts": time.time(), **record}, ensure_ascii=False, default=str) + "\n")


def is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    if any((path / filename).exists() for filename in HF_WEIGHT_FILES):
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def resolve_model_dir(path: Path, role: str) -> Path:
    path = path.resolve()
    if is_hf_model_dir(path):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {path}")

    role_candidates = {
        "agent": [
            path / "hf_safetensors" / "actor",
            path / "actor" / "hf_safetensors",
            path / "actor",
        ],
        "reranker": [
            path / "hf_safetensors" / "reranker_actor_rollout",
            path / "reranker_actor_rollout" / "hf_safetensors",
            path / "reranker_actor_rollout",
        ],
    }[role]

    for candidate in role_candidates:
        if is_hf_model_dir(candidate):
            return candidate.resolve()

    loadable = [p for p in path.rglob("*") if is_hf_model_dir(p)]
    if role == "agent":
        preferred = [p for p in loadable if "actor" in p.parts and "reranker_actor_rollout" not in p.parts]
    else:
        preferred = [p for p in loadable if any("reranker" in part for part in p.parts)]
    if preferred:
        return sorted(preferred, key=lambda p: (len(p.parts), str(p)))[0].resolve()
    if len(loadable) == 1:
        return loadable[0].resolve()

    raise FileNotFoundError(f"cannot resolve a loadable HF safetensors model for role={role} under {path}")


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc((text or "").lower())).split())


def exact_match(prediction: str, answers: list[str]) -> float:
    pred = normalize_answer(prediction)
    return float(any(pred == normalize_answer(answer) for answer in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    def one_f1(answer: str) -> float:
        pred_tokens = normalize_answer(prediction).split()
        answer_tokens = normalize_answer(answer).split()
        common = Counter(pred_tokens) & Counter(answer_tokens)
        num_same = sum(common.values())
        if num_same == 0 or not pred_tokens or not answer_tokens:
            return 0.0
        precision = num_same / len(pred_tokens)
        recall = num_same / len(answer_tokens)
        return 2 * precision * recall / (precision + recall)

    return max((one_f1(answer) for answer in answers), default=0.0)


def extract_answer(text: str) -> str:
    matches = list(ANSWER_RE.finditer(text or ""))
    if not matches:
        return ""
    return matches[-1].group(1).strip()


def extract_question(prompt: list[dict[str, Any]], extra_info: dict[str, Any]) -> str:
    question = extra_info.get("question")
    if question:
        return str(question)
    content = "\n".join(str(msg.get("content", "")) for msg in prompt)
    match = re.search(r"Question:\s*(.*)", content)
    return match.group(1).strip() if match else content[-500:]


def truncate_tool_response(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def coerce_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value]
    if hasattr(value, "tolist"):
        return [dict(item) for item in value.tolist()]
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"unsupported prompt type: {type(value)}")


def coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def classify_eval_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if (
        "maximum context length" in lowered
        or ("input tokens" in lowered and "please reduce" in lowered)
        or "context length" in lowered
    ):
        return "context_length_exceeded"
    return exc.__class__.__name__


def build_failed_result(row: dict[str, Any], idx: int, exc: Exception, elapsed_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
    error_type = classify_eval_error(exc)
    error_message = str(exc)
    try:
        data_source = str(row.get("data_source", "unknown"))
    except Exception:
        data_source = "unknown"
    try:
        prompt_messages = coerce_messages(row.get("prompt", []))
    except Exception:
        prompt_messages = []
    try:
        reward_model = coerce_dict(row.get("reward_model", {}))
        answers = list(reward_model.get("ground_truth", {}).get("target", []))
    except Exception:
        answers = []

    metrics = {
        "index": idx,
        "data_source": data_source,
        "em": 0.0,
        "f1": 0.0,
        "tool_calls": 0,
        "agent_turns": 0,
        "agent_decision_total_s": 0.0,
        "agent_decision_avg_s": 0.0,
        "retrieve_total_s": 0.0,
        "reranker_total_s": 0.0,
        "recall_total_s": 0.0,
        "recall_avg_s": 0.0,
        "total_s": elapsed_s,
        "status": "failed",
        "error_type": error_type,
        "error_message": error_message[:2000],
    }
    trace = {
        "index": idx,
        "data_source": data_source,
        "prompt": prompt_messages,
        "sub_queries": [],
        "reranked_top5_chunks": [],
        "final_answer": "",
        "ground_truth_answer": answers,
        "status": "failed",
        "error_type": error_type,
        "error_message": error_message,
        "metrics": metrics,
        "stage_records": [],
    }
    return metrics, trace


async def post_json(session: aiohttp.ClientSession, url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
        text = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"POST {url} failed: status={response.status} body={text[:500]}")
        return json.loads(text)


async def complete(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": ["<|im_end|>"],
    }
    data = await post_json(session, f"{base_url.rstrip('/')}/v1/completions", payload, timeout)
    return data["choices"][0].get("text", "")


async def retrieve(session: aiohttp.ClientSession, retrieval_url: str, query: str, top_n: int, timeout: float) -> list[dict[str, Any]]:
    payload = {"queries": [query], "topk": top_n, "return_scores": True}
    data = await post_json(session, retrieval_url, payload, timeout)
    raw_candidates = (data.get("result") or [[]])[0]
    documents = []
    for item in raw_candidates:
        doc = item.get("document", {})
        documents.append(
            {
                "id": str(doc.get("id", "")),
                "contents": doc.get("contents") or doc.get("text") or doc.get("passage") or "",
                "title": doc.get("title", ""),
                "score": float(item.get("score", 0.0)),
            }
        )
    return documents


def parse_tool_call(text: str) -> tuple[dict[str, Any] | None, str]:
    match = TOOL_CALL_RE.search(text or "")
    if not match:
        return None, text or ""
    truncated = (text or "")[: match.end()]
    try:
        payload = json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None, truncated
    if payload.get("name") != "search":
        return None, truncated
    args = payload.get("arguments")
    if not isinstance(args, dict) or not isinstance(args.get("query"), str) or not args["query"].strip():
        return None, truncated
    return payload, truncated


async def run_reranker(
    session: aiohttp.ClientSession,
    args: EvalArgs,
    tokenizer: Any,
    initial_query: str,
    sub_query: str,
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    passages_block, docid_map = train_format_tool_response_with_docid_map(documents)
    prompt_text = TRAIN_RERANK_PROMPT_WITH_INITIAL_QUERY.format(
        N=len(documents),
        M=min(args.top_m, len(documents)),
        initial_query=initial_query,
        sub_query=sub_query,
        passages_block=passages_block,
    )
    reranker_prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    if len(reranker_prompt_ids) > args.reranker_max_prompt_length:
        reranker_prompt_ids = reranker_prompt_ids[: args.reranker_max_prompt_length]
    reranker_prompt = tokenizer.decode(reranker_prompt_ids)

    output = await complete(
        session=session,
        base_url=args.reranker_base_url,
        model=args.reranker_served_model,
        prompt=reranker_prompt,
        max_tokens=args.reranker_max_response_length,
        temperature=args.reranker_temperature,
        top_p=1.0,
        timeout=args.request_timeout,
    )
    write_llm_io_trace(
        args.llm_io_jsonl,
        {
            "source": "eval",
            "role": "reranker",
            "initial_query": initial_query,
            "sub_query": sub_query,
            "top_n": len(documents),
            "top_m": min(args.top_m, len(documents)),
            "prompt_token_count": len(reranker_prompt_ids),
            "output_token_count": len(tokenizer.encode(output, add_special_tokens=False)),
            "prompt_text": reranker_prompt,
            "output_text": output,
            "sampling_params": {"temperature": args.reranker_temperature, "top_p": 1.0},
        },
    )
    result = train_validate_rerank_output(output, len(documents), min(args.top_m, len(documents)), docid_map)
    if result.get("status_message") == "Success.":
        return result["reranked_docs"], {"success": True, "raw_output": output, "indices": result["reranked"], "errors": []}
    return documents[: args.top_m], {"success": False, "raw_output": output, "indices": [], "errors": result.get("errors", [])}


async def evaluate_one(
    session: aiohttp.ClientSession,
    args: EvalArgs,
    agent_tokenizer: Any,
    reranker_tokenizer: Any | None,
    row: dict[str, Any],
    idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_source = str(row.get("data_source", "unknown"))
    prompt_messages = coerce_messages(row["prompt"])
    reward_model = coerce_dict(row.get("reward_model", {}))
    extra_info = coerce_dict(row.get("extra_info", {}))
    answers = list(reward_model.get("ground_truth", {}).get("target", []))
    initial_query = extract_question(prompt_messages, extra_info)

    messages = [dict(item) for item in prompt_messages]
    agent_system_prompt_ids = agent_tokenizer.apply_chat_template(
        [{}],
        add_generation_prompt=False,
        tokenize=True,
        enable_thinking=False,
    )
    prompt_ids = agent_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=False,
    )
    assistant_texts: list[str] = []
    sub_queries: list[str] = []
    top5_by_call: list[list[dict[str, Any]]] = []
    top50_by_call: list[list[dict[str, Any]]] = []
    stage_records: list[dict[str, Any]] = []

    agent_total_s = 0.0
    retrieve_total_s = 0.0
    reranker_total_s = 0.0
    total_start = time.perf_counter()
    final_answer = ""
    status = "max_turns"

    user_turns = 0
    for _turn in range(args.max_assistant_turns):
        active_prompt_ids = prompt_ids
        if len(active_prompt_ids) > args.max_prompt_length:
            active_prompt_ids = active_prompt_ids[-args.max_prompt_length :]
        prompt_text = agent_tokenizer.decode(active_prompt_ids)

        t0 = time.perf_counter()
        assistant_text = await complete(
            session=session,
            base_url=args.agent_base_url,
            model=args.agent_served_model,
            prompt=prompt_text,
            max_tokens=args.max_response_length,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.request_timeout,
        )
        write_llm_io_trace(
            args.llm_io_jsonl,
            {
                "source": "eval",
                "role": "agent",
                "index": idx,
                "data_source": data_source,
                "initial_query": initial_query,
                "assistant_turn": len(assistant_texts) + 1,
                "user_turn": user_turns,
                "prompt_token_count": len(active_prompt_ids),
                "output_token_count": len(agent_tokenizer.encode(assistant_text, add_special_tokens=False)),
                "prompt_text": prompt_text,
                "output_text": assistant_text,
                "sampling_params": {"temperature": args.temperature, "top_p": args.top_p},
            },
        )
        agent_elapsed = time.perf_counter() - t0
        agent_total_s += agent_elapsed

        tool_payload, assistant_for_history = parse_tool_call(assistant_text)
        if tool_payload is not None and user_turns < args.max_user_turns:
            messages.append({"role": "assistant", "content": assistant_for_history})
            assistant_texts.append(assistant_for_history)
            prompt_ids.extend(agent_tokenizer.encode(assistant_for_history, add_special_tokens=False))
            sub_query = tool_payload["arguments"]["query"].strip()
            sub_queries.append(sub_query)

            t_retrieve = time.perf_counter()
            documents = await retrieve(session, args.retrieval_url, sub_query, args.top_n, args.request_timeout)
            retrieve_elapsed = time.perf_counter() - t_retrieve
            retrieve_total_s += retrieve_elapsed

            if args.run_mode == "no-ranker":
                final_documents = documents[: args.top_m]
                rerank_meta = {
                    "success": False,
                    "skipped": True,
                    "raw_output": "",
                    "indices": [],
                    "errors": [],
                }
                reranker_elapsed = 0.0
            else:
                if reranker_tokenizer is None:
                    raise ValueError("reranker_tokenizer is required outside no-ranker mode")
                t_rerank = time.perf_counter()
                final_documents, rerank_meta = await run_reranker(
                    session=session,
                    args=args,
                    tokenizer=reranker_tokenizer,
                    initial_query=initial_query,
                    sub_query=sub_query,
                    documents=documents,
                )
                reranker_elapsed = time.perf_counter() - t_rerank
                reranker_total_s += reranker_elapsed

            top5_by_call.append(final_documents[: args.top_m])
            if args.keep_trace == "full":
                top50_by_call.append(documents)

            tool_response_text = truncate_tool_response(
                train_format_tool_response(final_documents[: args.top_m]),
                args.max_tool_response_length,
            )
            tool_message = {"role": "tool", "content": tool_response_text}
            messages.append(tool_message)
            tool_response_ids = agent_tokenizer.apply_chat_template(
                [tool_message],
                add_generation_prompt=True,
                tokenize=True,
            )
            prompt_ids.extend(tool_response_ids[len(agent_system_prompt_ids) :])
            user_turns += 1
            stage_records.append(
                {
                    "sub_query": sub_query,
                    "agent_decision_s": agent_elapsed,
                    "retrieve_s": retrieve_elapsed,
                    "reranker_s": reranker_elapsed,
                    "recall_s": retrieve_elapsed + reranker_elapsed,
                    "reranker_success": rerank_meta["success"],
                    "reranker_skipped": bool(rerank_meta.get("skipped", False)),
                }
            )
            continue

        assistant_texts.append(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})
        prompt_ids.extend(agent_tokenizer.encode(assistant_text, add_special_tokens=False))
        final_answer = extract_answer(assistant_text)
        status = "answered" if final_answer else "no_valid_answer"
        break

    total_s = time.perf_counter() - total_start
    em = exact_match(final_answer, answers)
    f1 = token_f1(final_answer, answers)
    tool_calls = len(sub_queries)
    agent_turns = len(assistant_texts)

    metrics = {
        "index": idx,
        "data_source": data_source,
        "em": em,
        "f1": f1,
        "tool_calls": tool_calls,
        "agent_turns": agent_turns,
        "agent_decision_total_s": agent_total_s,
        "agent_decision_avg_s": agent_total_s / agent_turns if agent_turns else 0.0,
        "retrieve_total_s": retrieve_total_s,
        "reranker_total_s": reranker_total_s,
        "recall_total_s": retrieve_total_s + reranker_total_s,
        "recall_avg_s": (retrieve_total_s + reranker_total_s) / tool_calls if tool_calls else 0.0,
        "total_s": total_s,
        "status": status,
    }
    trace = {
        "index": idx,
        "data_source": data_source,
        "prompt": prompt_messages,
        "sub_queries": sub_queries,
        "reranked_top5_chunks": top5_by_call,
        "final_answer": final_answer,
        "ground_truth_answer": answers,
        "status": status,
        "metrics": metrics,
        "stage_records": stage_records,
    }
    if args.keep_trace == "full":
        trace["retrieved_top50_chunks"] = top50_by_call
    return metrics, trace


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "em",
        "f1",
        "tool_calls",
        "agent_decision_avg_s",
        "agent_decision_total_s",
        "recall_avg_s",
        "recall_total_s",
        "retrieve_total_s",
        "reranker_total_s",
        "total_s",
    ]
    result = {"n": float(len(rows))}
    for key in keys:
        result[key] = avg([float(row.get(key, 0.0)) for row in rows])
    return result


def build_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_source[row["data_source"]].append(row)
    source_summary = {source: summarize_group(rows) for source, rows in sorted(by_source.items())}
    micro = summarize_group(metrics)
    macro = {"n": float(len(source_summary))}
    if source_summary:
        for key in next(iter(source_summary.values())).keys():
            if key == "n":
                continue
            macro[key] = avg([summary[key] for summary in source_summary.values()])
    else:
        macro.update({key: 0.0 for key in micro if key != "n"})
    status_counts = dict(Counter(row.get("status", "unknown") for row in metrics))
    failure_count = int(status_counts.get("failed", 0))
    success_count = len(metrics) - failure_count
    return {
        "micro": micro,
        "macro": macro,
        "by_data_source": source_summary,
        "status_counts": status_counts,
        "success_count": success_count,
        "failure_count": failure_count,
    }


def fmt(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "0.0000"
    return f"{value:.4f}"


def effect_table(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Scope | N | EM | F1 |",
        "|---|---:|---:|---:|",
    ]
    for name, row in summary.items():
        lines.append(
            f"| {name} | {int(row.get('n', 0))} | {fmt(row.get('em', 0.0))} | {fmt(row.get('f1', 0.0))} |"
        )
    return "\n".join(lines)


def performance_table(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Scope | N | Tool Calls | Agent Turn Avg s | Agent Total Avg s | Retrieve Total Avg s | Reranker Total Avg s | Recall Call Avg s | Recall Total Avg s | Total Avg s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        lines.append(
            f"| {name} | {int(row.get('n', 0))} | {fmt(row.get('tool_calls', 0.0))} | "
            f"{fmt(row.get('agent_decision_avg_s', 0.0))} | {fmt(row.get('agent_decision_total_s', 0.0))} | "
            f"{fmt(row.get('retrieve_total_s', 0.0))} | {fmt(row.get('reranker_total_s', 0.0))} | "
            f"{fmt(row.get('recall_avg_s', 0.0))} | {fmt(row.get('recall_total_s', 0.0))} | "
            f"{fmt(row.get('total_s', 0.0))} |"
        )
    return "\n".join(lines)


def write_report(args: EvalArgs, summary: dict[str, Any], metrics: list[dict[str, Any]], elapsed_s: float) -> None:
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    overview = {
        "micro-average": summary["micro"],
        "macro-average": summary["macro"],
    }
    per_source = summary["by_data_source"]
    status_counts = summary.get("status_counts") or dict(Counter(row.get("status", "unknown") for row in metrics))
    lines = [
        f"# CoSearch vLLM Evaluation Report",
        "",
        f"- Strategy: `{args.strategy_name}`",
        f"- Run mode: `{args.run_mode}`",
        f"- Dataset: `{args.data_path}`",
        f"- Examples: `{len(metrics)}`",
        f"- Success count: `{summary.get('success_count', 0)}`",
        f"- Failure count: `{summary.get('failure_count', 0)}`",
        f"- Agent model: `{args.agent_model}`",
        f"- Reranker model: `{args.reranker_model if args.reranker_model is not None else 'disabled'}`",
        f"- Trace dir: `{args.trace_dir}`",
        f"- Wall time: `{fmt(elapsed_s)}s`",
        f"- Status counts: `{status_counts}`",
        "",
        "## Effect Metrics",
        "",
        effect_table(overview),
        "",
        "## Effect Metrics By Dataset",
        "",
        effect_table(per_source),
        "",
        "## Performance Metrics",
        "",
        "Agent Avg s is the per-query average agent generation time per assistant turn. "
        "Recall Avg s is the per-query average retrieve+rerank time per tool call. "
        "Total Avg s is the end-to-end per-query latency average.",
        "",
        performance_table(overview),
        "",
        "## Performance Metrics By Dataset",
        "",
        performance_table(per_source),
        "",
    ]
    args.report_path.write_text("\n".join(lines), encoding="utf-8")


async def run_eval(args: EvalArgs) -> None:
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model, trust_remote_code=True)
    reranker_tokenizer = (
        AutoTokenizer.from_pretrained(args.reranker_model, trust_remote_code=True)
        if args.reranker_model is not None
        else None
    )

    df = pd.read_parquet(args.data_path)
    if args.max_eval_num >= 0:
        df = df.head(args.max_eval_num)
    records = df.to_dict(orient="records")
    start = time.perf_counter()
    semaphore = asyncio.Semaphore(args.batch_size)

    async with aiohttp.ClientSession() as session:
        async def guarded(i: int, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
            async with semaphore:
                sample_start = time.perf_counter()
                try:
                    return await evaluate_one(session, args, agent_tokenizer, reranker_tokenizer, row, i)
                except Exception as exc:
                    elapsed_s = time.perf_counter() - sample_start
                    return build_failed_result(row, i, exc, elapsed_s)

        tasks = [guarded(i, row) for i, row in enumerate(records)]
        results = await asyncio.gather(*tasks)

    metrics = [item[0] for item in results]
    traces = [item[1] for item in results]
    with (args.trace_dir / "traces.jsonl").open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    with (args.trace_dir / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in metrics:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = build_summary(metrics)
    elapsed = time.perf_counter() - start
    (args.trace_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.trace_dir / "run_config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args, summary, metrics, elapsed)


def parse_run_args(ns: argparse.Namespace) -> EvalArgs:
    reranker_model = None
    if ns.run_mode != "no-ranker":
        if ns.reranker_model is None:
            raise ValueError("--reranker-model is required unless --run-mode no-ranker")
        if ns.reranker_base_url is None:
            raise ValueError("--reranker-base-url is required unless --run-mode no-ranker")
        reranker_model = resolve_model_dir(ns.reranker_model, "reranker")

    return EvalArgs(
        run_mode=ns.run_mode,
        agent_model=resolve_model_dir(ns.agent_model, "agent"),
        reranker_model=reranker_model,
        data_path=ns.data_path,
        max_eval_num=ns.max_eval_num,
        batch_size=ns.batch_size,
        keep_trace=ns.keep_trace,
        trace_dir=ns.trace_dir,
        report_path=ns.report_path,
        strategy_name=ns.strategy_name,
        retrieval_url=ns.retrieval_url,
        agent_base_url=ns.agent_base_url,
        reranker_base_url=ns.reranker_base_url,
        agent_served_model=ns.agent_served_model,
        reranker_served_model=ns.reranker_served_model,
        top_n=ns.top_n,
        top_m=ns.top_m,
        max_assistant_turns=ns.max_assistant_turns,
        max_user_turns=ns.max_user_turns,
        max_tool_response_length=ns.max_tool_response_length,
        max_prompt_length=ns.max_prompt_length,
        max_response_length=ns.max_response_length,
        reranker_max_prompt_length=ns.reranker_max_prompt_length,
        reranker_max_response_length=ns.reranker_max_response_length,
        temperature=ns.temperature,
        top_p=ns.top_p,
        reranker_temperature=ns.reranker_temperature,
        request_timeout=ns.request_timeout,
        llm_io_jsonl=ns.llm_io_jsonl,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-model")
    resolve_parser.add_argument("--path", type=Path, required=True)
    resolve_parser.add_argument("--role", choices=["agent", "reranker"], required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-mode", choices=["full", "no-ranker"], default="full")
    run_parser.add_argument("--agent-model", type=Path, required=True)
    run_parser.add_argument("--reranker-model", type=Path, default=None)
    run_parser.add_argument("--data-path", type=Path, required=True)
    run_parser.add_argument("--max-eval-num", type=int, default=-1)
    run_parser.add_argument("--batch-size", type=int, default=32)
    run_parser.add_argument("--keep-trace", choices=["partial", "full"], default="partial")
    run_parser.add_argument("--trace-dir", type=Path, required=True)
    run_parser.add_argument("--report-path", type=Path, required=True)
    run_parser.add_argument("--strategy-name", default="default")
    run_parser.add_argument("--retrieval-url", required=True)
    run_parser.add_argument("--agent-base-url", required=True)
    run_parser.add_argument("--reranker-base-url", default=None)
    run_parser.add_argument("--agent-served-model", default="cosearch-agent")
    run_parser.add_argument("--reranker-served-model", default="cosearch-reranker")
    run_parser.add_argument("--top-n", type=int, default=50)
    run_parser.add_argument("--top-m", type=int, default=5)
    run_parser.add_argument("--max-assistant-turns", type=int, default=6)
    run_parser.add_argument("--max-user-turns", type=int, default=6)
    run_parser.add_argument("--max-tool-response-length", type=int, default=4096)
    run_parser.add_argument("--max-prompt-length", type=int, default=11264)
    run_parser.add_argument("--max-response-length", type=int, default=1024)
    run_parser.add_argument("--reranker-max-prompt-length", type=int, default=16384)
    run_parser.add_argument("--reranker-max-response-length", type=int, default=1024)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--reranker-temperature", type=float, default=0.0)
    run_parser.add_argument("--request-timeout", type=float, default=180.0)
    run_parser.add_argument("--llm-io-jsonl", type=Path, default=None)

    ns = parser.parse_args()
    if ns.command == "resolve-model":
        print(resolve_model_dir(ns.path, ns.role))
        return 0

    args = parse_run_args(ns)
    asyncio.run(run_eval(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
