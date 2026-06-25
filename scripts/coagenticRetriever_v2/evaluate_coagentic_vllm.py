#!/usr/bin/env python3
"""vLLM-only CoAgenticRetriever evaluator.

This evaluator intentionally avoids VERL model loading and checkpoint resume.
It keeps the CoAgenticRetriever inference path explicit:

agent LLM -> recall retriever top-N -> dense ranker reorder ->
top-M tool response -> agent LLM.
"""

from __future__ import annotations

import argparse
import asyncio
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

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
TRACE_TOP5 = 5
TRACE_TOP50 = 50

HF_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


@dataclass
class EvalArgs:
    run_mode: str
    reranker: str
    agent_model: Path | None
    ranker_model: Path | None
    ranker_encoder: Path | None
    llm_judge_endpoint: str | None
    llm_judge_model: str | None
    llm_judge_prompt_path: Path | None
    llm_judge_max_chunk_chars: int
    llm_judge_max_tokens: int
    llm_judge_temperature: float
    llm_judge_request_timeout: float
    llm_judge_max_retries: int
    llm_judge_retry_delay: float
    llm_judge_retry_backoff: float
    data_path: Path
    max_eval_num: int
    max_ranker_steps: int
    batch_size: int
    keep_trace: str
    trace_dir: Path
    report_path: Path
    strategy_name: str
    retrieval_url: str
    agent_base_url: str | None
    agent_served_model: str
    top_n: int
    top_m: int
    ranker_top_k: int
    max_assistant_turns: int
    max_user_turns: int
    max_tool_response_length: int
    max_prompt_length: int
    max_response_length: int
    max_model_len: int
    temperature: float
    top_p: float
    request_timeout: float
    max_retries: int
    retry_delay: float
    retry_backoff: float
    llm_io_jsonl: Path | None
    metrics_jsonl: Path | None
    search_timing_jsonl: Path | None
    ranker_output_jsonl: Path | None
    validation_data_dir: Path | None
    rollout_data_dir: Path | None
    ranker_device: str
    ranker_max_query_length: int
    ranker_max_doc_length: int
    trust_remote_code: bool
    enable_thinking: bool
    tool_config_path: Path | None
    inject_tool_schema: bool
    tool_schemas: list[dict[str, Any]] | None
    stop_sequences: list[str] | None


def append_jsonl(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def write_llm_io_trace(path: Path | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    max_records = int(os.getenv("COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS", "0") or 0)
    if max_records > 0 and path.exists():
        with path.open("r", encoding="utf-8") as fp:
            if sum(1 for _ in fp) >= max_records:
                return
    append_jsonl(path, {"ts": time.time(), **record})


def is_hf_model_dir(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").exists():
        return False
    if any((path / filename).exists() for filename in HF_WEIGHT_FILES):
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def global_step_number(path: Path) -> int:
    match = re.match(r"global_step_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def resolve_global_step_dir(path: Path) -> Path | None:
    if not path.exists() or not path.is_dir():
        return None
    if path.name.startswith("global_step_"):
        return path.resolve()
    marker = path / "latest_checkpointed_iteration.txt"
    if marker.exists():
        text = marker.read_text(encoding="utf-8").strip()
        if text.isdigit():
            candidate = path / f"global_step_{text}"
            if candidate.is_dir():
                return candidate.resolve()
    candidates = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("global_step_")]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (global_step_number(p), str(p)))[-1].resolve()


def resolve_model_dir(path: Path, role: str) -> Path:
    path = path.resolve()
    if is_hf_model_dir(path):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"model path does not exist: {path}")

    search_root = resolve_global_step_dir(path) or path
    role_candidates = {
        "agent": [
            search_root / "hf_safetensors" / "actor",
            search_root / "actor" / "hf_safetensors",
            search_root / "actor",
        ],
    }[role]
    for candidate in role_candidates:
        if is_hf_model_dir(candidate):
            return candidate.resolve()

    loadable = [p for p in search_root.rglob("*") if is_hf_model_dir(p)]
    if role == "agent":
        preferred = [p for p in loadable if "actor" in p.parts and "reranker_actor_rollout" not in p.parts]
    else:
        preferred = loadable
    if preferred:
        return sorted(preferred, key=lambda p: (len(p.parts), str(p)))[0].resolve()
    if len(loadable) == 1:
        return loadable[0].resolve()
    raise FileNotFoundError(f"cannot resolve a loadable HF model for role={role} under {path}")


def resolve_ranker_encoder_dir(path: Path) -> Path:
    path = path.resolve()
    if is_hf_model_dir(path):
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"ranker path does not exist: {path}")

    search_root = resolve_global_step_dir(path) or path
    candidates = [
        search_root / "rank_encoder",
        search_root / "ranker" / "rank_encoder",
        search_root / "retriever" / "rank_encoder",
    ]
    for candidate in candidates:
        if is_hf_model_dir(candidate):
            return candidate.resolve()

    rank_encoder_dirs = [
        p for p in search_root.rglob("rank_encoder") if p.is_dir() and is_hf_model_dir(p)
    ]
    if rank_encoder_dirs:
        return sorted(rank_encoder_dirs, key=lambda p: (global_step_number(p.parent.parent), str(p)))[-1].resolve()

    loadable = [p for p in search_root.rglob("*") if is_hf_model_dir(p)]
    if len(loadable) == 1:
        return loadable[0].resolve()
    raise FileNotFoundError(f"cannot resolve ranker encoder under {path}")


def resolve_ranker_paths(
    ranker_model: Path | None,
    ranker_encoder: Path | None,
    ranker_base_model: Path | None,
) -> tuple[Path, Path]:
    if ranker_model is None and ranker_encoder is None:
        raise ValueError("ranker_model or ranker_encoder is required; ranker path fallback is not allowed")

    encoder_source = ranker_encoder or ranker_model
    assert encoder_source is not None
    encoder_path = resolve_ranker_encoder_dir(encoder_source)

    if ranker_base_model is not None:
        model_path = ranker_base_model.resolve()
    elif ranker_model is not None and is_hf_model_dir(ranker_model.resolve()):
        model_path = ranker_model.resolve()
    else:
        model_path = encoder_path

    if not model_path.exists():
        raise FileNotFoundError(f"ranker tokenizer/base model path does not exist: {model_path}")
    return model_path, encoder_path


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


def coerce_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value]
    if hasattr(value, "tolist"):
        return [dict(item) for item in value.tolist()]
    if isinstance(value, str):
        return json.loads(value)
    raise TypeError(f"unsupported prompt type: {type(value)}")


def coerce_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, float) and math.isnan(value):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    return dict(value)


def coerce_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    if hasattr(value, "tolist"):
        return [str(item) for item in value.tolist()]
    return [str(value)]


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


def format_tool_response(documents: list[dict[str, Any]], max_doc_length: int = 2000) -> str:
    if not documents:
        return "No documents found."
    lines = []
    for idx, doc in enumerate(documents, start=1):
        contents = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
        if len(contents) > max_doc_length:
            contents = contents[:max_doc_length] + "..."
        title = doc.get("title", "")
        if title:
            lines.append(f"[{idx}] Title: {title}\n{contents}")
        else:
            lines.append(f"[{idx}] {contents}")
    return "\n".join(lines)


def train_format_tool_response(documents: list[dict[str, Any]], max_doc_length: int = 2000) -> str:
    try:
        from verl.tools.utils.search import format_tool_response as train_impl
    except Exception:
        train_impl = format_tool_response
    return train_impl(documents, max_doc_length=max_doc_length)


def trace_doc_id(doc: dict[str, Any]) -> str:
    for key in ("id", "doc_id", "chunk_id", "passage_id", "_id"):
        value = doc.get(key)
        if value not in (None, ""):
            return str(value)
    title = str(doc.get("title") or "")
    contents = str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")
    return f"{title}|{contents[:80]}"


def trace_doc_ids(documents: list[dict[str, Any]]) -> list[str]:
    return [trace_doc_id(doc) for doc in documents]


def apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
    tool_schemas: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Any:
    if tool_schemas is not None:
        kwargs["tools"] = tool_schemas
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data or {}
    except ModuleNotFoundError:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}


def default_tool_schema_for_class(class_name: str) -> dict[str, Any]:
    if class_name != "verl.tools.coagentic_retriever_tool.CoAgenticRetrieverTool":
        raise ValueError(
            "tool_config has no explicit tool_schema; only CoAgenticRetrieverTool has a local default "
            f"schema in this evaluator, got class_name={class_name!r}"
        )
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search for relevant documents to answer the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information.",
                    }
                },
                "required": ["query"],
            },
        },
    }


def load_tool_schemas(tool_config_path: Path | None) -> list[dict[str, Any]] | None:
    if tool_config_path is None:
        return None
    config = load_yaml_mapping(tool_config_path)
    schemas: list[dict[str, Any]] = []
    for tool_config in config.get("tools") or []:
        schema = tool_config.get("tool_schema")
        if schema is None:
            schema = default_tool_schema_for_class(str(tool_config.get("class_name", "")))
        schemas.append(schema)
    return schemas or None


def classify_eval_error(exc: Exception) -> str:
    text = str(exc).lower()
    if (
        "maximum context length" in text
        or ("input tokens" in text and "please reduce" in text)
        or "context length" in text
    ):
        return "context_length_exceeded"
    return exc.__class__.__name__


def build_failed_result(row: dict[str, Any], idx: int, exc: Exception, elapsed_s: float) -> tuple[dict[str, Any], dict[str, Any]]:
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
        answers = coerce_answers((reward_model.get("ground_truth") or {}).get("target"))
    except Exception:
        answers = []
    error_type = classify_eval_error(exc)
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
        "ranker_total_s": 0.0,
        "recall_total_s": 0.0,
        "recall_avg_s": 0.0,
        "total_s": elapsed_s,
        "status": "failed",
        "error_type": error_type,
        "error_message": str(exc)[:2000],
    }
    trace = {
        "index": idx,
        "data_source": data_source,
        "prompt": prompt_messages,
        "sub_queries": [],
        "recall_top5_chunks": [],
        "ranked_top50_chunks": [],
        "final_top5_chunks": [],
        "ranked_top5_chunks": [],
        "reranked_top5_chunks": [],
        "final_answer": "",
        "ground_truth_answer": answers,
        "status": "failed",
        "error_type": error_type,
        "error_message": str(exc),
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
    stop_sequences: list[str] | None,
) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    if stop_sequences:
        payload["stop"] = stop_sequences
    data = await post_json(session, f"{base_url.rstrip('/')}/v1/completions", payload, timeout)
    return data["choices"][0].get("text", "")


async def retrieve(session: aiohttp.ClientSession, args: EvalArgs, query: str) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    delay = args.retry_delay
    for attempt in range(1, args.max_retries + 1):
        started = time.perf_counter()
        try:
            payload = {"queries": [query], "topk": args.top_n, "return_scores": True}
            data = await post_json(session, args.retrieval_url, payload, args.request_timeout)
            raw_candidates = (data.get("result") or [[]])[0]
            documents: list[dict[str, Any]] = []
            for idx, item in enumerate(raw_candidates, start=1):
                doc = item.get("document", item)
                score = item.get("score", doc.get("score", 0.0))
                documents.append(
                    {
                        "id": str(doc.get("id", "")),
                        "contents": doc.get("contents") or doc.get("text") or doc.get("passage") or "",
                        "title": doc.get("title", ""),
                        "score": float(score or 0.0),
                        "recall_score": float(score or 0.0),
                        "recall_rank": idx,
                    }
                )
            append_jsonl(
                args.search_timing_jsonl,
                {
                    "ts": time.time(),
                    "action": "search",
                    "elapsed_s": time.perf_counter() - started,
                    "status": "success",
                    "attempt": attempt,
                    "top_n": args.top_n,
                    "num_documents": len(documents),
                    "query_chars": len(query or ""),
                    "error": "",
                },
            )
            return documents
        except Exception as exc:
            last_error = exc
            append_jsonl(
                args.search_timing_jsonl,
                {
                    "ts": time.time(),
                    "action": "search",
                    "elapsed_s": time.perf_counter() - started,
                    "status": "retry" if attempt < args.max_retries else "error",
                    "attempt": attempt,
                    "top_n": args.top_n,
                    "num_documents": 0,
                    "query_chars": len(query or ""),
                    "error": str(exc)[:500],
                },
            )
            if attempt < args.max_retries:
                await asyncio.sleep(delay)
                delay *= args.retry_backoff
    raise RuntimeError(f"recall retriever failed after {args.max_retries} attempts: {last_error}")


def parse_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    matches = list(TOOL_CALL_RE.finditer(text or ""))
    if not matches:
        return [], text or ""
    truncated = (text or "")[: matches[0].end()]
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
    return payloads, truncated


def truncate_after_first_tool_call_ids(tokenizer: Any, response_ids: list[int]) -> list[int]:
    end_ids = tokenizer.encode("</tool_call>", add_special_tokens=False)
    if not end_ids:
        return response_ids
    for i in range(0, len(response_ids) - len(end_ids) + 1):
        if response_ids[i : i + len(end_ids)] == end_ids:
            return response_ids[: i + len(end_ids)]
    return response_ids


def mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class LocalE5Ranker:
    def __init__(
        self,
        *,
        model_path: Path,
        encoder_path: Path,
        device: str,
        max_query_length: int,
        max_doc_length: int,
        trust_remote_code: bool,
    ) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"ranker device {device} requires CUDA, but torch.cuda.is_available() is False")
        self.torch = torch
        self.functional = F
        self.device = torch.device(device)
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.use_e5_prefix = "e5" in str(model_path).lower() or "e5" in str(encoder_path).lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.encoder = AutoModel.from_pretrained(encoder_path, trust_remote_code=trust_remote_code)
        self.encoder.to(self.device)
        self.encoder.eval()

    def _format_query(self, text: str) -> str:
        return f"query: {text}" if self.use_e5_prefix else text

    def _format_doc(self, text: str) -> str:
        return f"passage: {text}" if self.use_e5_prefix else text

    def _encode(self, texts: list[str], max_length: int) -> Any:
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with self.torch.no_grad():
            outputs = self.encoder(**tokens)
            hidden = outputs.last_hidden_state
            embeddings = mean_pool(hidden, tokens["attention_mask"])
            return self.functional.normalize(embeddings, dim=-1)

    def rank_topk(self, query: str, docs: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        if not docs:
            return []
        top_k = len(docs) if top_k is None else min(int(top_k), len(docs))
        doc_texts = [
            (str(doc.get("title") or "") + "\n" if doc.get("title") else "")
            + str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")
            for doc in docs
        ]
        query_emb = self._encode([self._format_query(query)], self.max_query_length)
        doc_emb = self._encode([self._format_doc(text) for text in doc_texts], self.max_doc_length)
        scores = self.torch.matmul(query_emb, doc_emb.T).squeeze(0)
        top_scores, top_indices = self.torch.topk(scores, k=top_k)
        ranked = []
        for rank_position, (score, idx) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            doc = dict(docs[idx])
            doc["recall_rank"] = int(doc.get("recall_rank") or doc.get("rank") or idx + 1)
            doc["recall_score"] = doc.get("recall_score", doc.get("retriever_score", doc.get("score")))
            doc["rank_score"] = float(score)
            doc["rank_rank"] = rank_position
            ranked.append(doc)
        return ranked


class LLMJudgeRanker:
    """OpenAI-compatible listwise reranker using the async-labeling judge prompt."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        prompt_path: Path,
        max_chunk_chars: int,
        max_tokens: int,
        temperature: float,
        request_timeout: float,
        max_retries: int,
        retry_delay: float,
        retry_backoff: float,
        llm_io_jsonl: Path | None,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.max_chunk_chars = max(1, int(max_chunk_chars))
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.request_timeout = float(request_timeout)
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = float(retry_delay)
        self.retry_backoff = float(retry_backoff)
        self.llm_io_jsonl = llm_io_jsonl
        text = prompt_path.read_text(encoding="utf-8")
        self.system_template, self.user_template = self._parse_prompt(text)

    @staticmethod
    def _parse_prompt(text: str) -> tuple[str, str]:
        system_marker = "## system:"
        user_marker = "## user:"
        system_pos = text.find(system_marker)
        user_pos = text.find(user_marker)
        if system_pos < 0 or user_pos < 0 or user_pos <= system_pos:
            raise ValueError("LLM judge prompt must contain '## system:' before '## user:'")
        return text[system_pos + len(system_marker):user_pos].strip(), text[user_pos + len(user_marker):].strip()

    def _render_doc(self, doc: dict[str, Any], rank: int) -> str:
        doc_id = str(doc.get("id") or "")
        title = str(doc.get("title") or "")
        text = str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")[: self.max_chunk_chars]
        recall_rank = int(doc.get("recall_rank") or doc.get("rank") or rank)
        recall_score = doc.get("recall_score", doc.get("score", 0.0))
        return "\n".join(
            [
                f"[id: {doc_id}]",
                f"title: {title}",
                f"retriever_rank_for_tie_break_only: {recall_rank}",
                f"retriever_score_for_tie_break_only: {recall_score}",
                "snippet:",
                text,
            ]
        )

    def _render_messages(self, query: str, docs: list[dict[str, Any]]) -> list[dict[str, str]]:
        allowed_ids = [str(doc.get("id") or "") for doc in docs]
        user = self.user_template
        user = user.replace("{{原始查询问题}}", query)
        user = user.replace("{{规范化后的查询问题}}", query)
        user = user.replace("{{允许的所有段落ID列表}}", ", ".join(allowed_ids))

        marker = "[id: {{段落ID}}]"
        start = user.find(marker)
        if start < 0:
            raise ValueError("candidate template marker not found in LLM judge prompt")
        text_marker = "{{段落文本片段}}"
        text_pos = user.find(text_marker, start)
        if text_pos < 0:
            raise ValueError("candidate text placeholder not found in LLM judge prompt")
        line_end = user.find("\n", text_pos + len(text_marker))
        if line_end < 0:
            line_end = text_pos + len(text_marker)
        rendered = "\n\n".join(self._render_doc(doc, rank) for rank, doc in enumerate(docs, start=1))
        user = user[:start] + rendered + user[line_end:]
        user = user.replace("\n\n[... 其他段落候选者以此类推 ...]", "")
        user = user.replace("\n[... 其他段落候选者以此类推 ...]", "")
        if "{{" in user or "}}" in user:
            raise ValueError("unrendered LLM judge prompt placeholder remains")
        return [
            {"role": "system", "content": self.system_template},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
        matches = re.findall(r"\{.*?\}", stripped, flags=re.DOTALL)
        if not matches:
            raise ValueError("judge response does not contain a JSON object")
        return json.loads(matches[-1])

    @classmethod
    def _parse_ranked_ids(cls, text: str) -> list[str]:
        obj = cls._extract_json_object(text)
        ranked_ids = obj.get("ranked_ids")
        if not isinstance(ranked_ids, list):
            raise ValueError("judge response missing list field: ranked_ids")
        return [str(item) for item in ranked_ids]

    @staticmethod
    def _validate_ranked_ids(ranked_ids: list[str], docs: list[dict[str, Any]]) -> None:
        expected_ids = [str(doc.get("id") or "") for doc in docs]
        expected_set = set(expected_ids)
        if len(ranked_ids) != len(expected_ids):
            raise ValueError(f"ranked_ids must contain exactly {len(expected_ids)} ids, got {len(ranked_ids)}")
        if len(set(ranked_ids)) != len(ranked_ids):
            raise ValueError("ranked_ids contains duplicated ids")
        unknown = [doc_id for doc_id in ranked_ids if doc_id not in expected_set]
        if unknown:
            raise ValueError(f"ranked_ids contains unknown ids: {unknown[:5]}")
        missing = [doc_id for doc_id in expected_ids if doc_id not in set(ranked_ids)]
        if missing:
            raise ValueError(f"ranked_ids missing request ids: {missing[:5]}")

    async def rank_topk(
        self,
        *,
        session: aiohttp.ClientSession,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int | None = None,
        index: int | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not docs:
            return [], {"ranker_success": False, "search_tool_error": False, "reranker": "llm_as_judge"}
        top_k = len(docs) if top_k is None else min(int(top_k), len(docs))
        messages = self._render_messages(query, docs)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "chat_template_kwargs": {
                "thinking": False,
                "enable_thinking": False,
                "reasoning_effort": "none",
            },
        }
        last_error: Exception | None = None
        delay = self.retry_delay
        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                data = await post_json(session, self.endpoint, payload, self.request_timeout)
                content = data["choices"][0]["message"].get("content", "")
                ranked_ids = self._parse_ranked_ids(content)
                self._validate_ranked_ids(ranked_ids, docs)
                by_id = {str(doc.get("id") or ""): doc for doc in docs}
                ranked = []
                for rank_position, doc_id in enumerate(ranked_ids[:top_k], start=1):
                    source = by_id[doc_id]
                    doc = dict(source)
                    doc["recall_rank"] = int(doc.get("recall_rank") or doc.get("rank") or 0)
                    doc["recall_score"] = doc.get("recall_score", doc.get("score"))
                    doc["rank_rank"] = rank_position
                    doc["rank_score"] = float((len(ranked_ids) - rank_position + 1) / len(ranked_ids))
                    doc["ranker"] = "llm_as_judge"
                    ranked.append(doc)
                write_llm_io_trace(
                    self.llm_io_jsonl,
                    {
                        "source": "coagentic_eval",
                        "role": "llm_as_judge_reranker",
                        "index": index,
                        "query": query,
                        "attempt": attempt,
                        "prompt_text": json.dumps(messages, ensure_ascii=False),
                        "output_text": content,
                        "usage": data.get("usage") or {},
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
                return ranked, {
                    "ranker_success": True,
                    "search_tool_error": False,
                    "reranker": "llm_as_judge",
                    "ranker_attempts": attempt,
                    "ranker_error": "",
                }
            except Exception as exc:
                last_error = exc
                write_llm_io_trace(
                    self.llm_io_jsonl,
                    {
                        "source": "coagentic_eval",
                        "role": "llm_as_judge_reranker",
                        "index": index,
                        "query": query,
                        "attempt": attempt,
                        "error": str(exc)[:500],
                        "elapsed_s": time.perf_counter() - started,
                    },
                )
                if attempt < self.max_retries:
                    await asyncio.sleep(delay)
                    delay *= self.retry_backoff
        fallback = []
        for rank_position, source in enumerate(docs[:top_k], start=1):
            doc = dict(source)
            doc["rank_rank"] = rank_position
            doc["rank_score"] = float(doc.get("recall_score", doc.get("score", 0.0)) or 0.0)
            doc["ranker"] = "llm_as_judge_fallback_recall"
            fallback.append(doc)
        return fallback, {
            "ranker_success": False,
            "search_tool_error": True,
            "reranker": "llm_as_judge",
            "ranker_attempts": self.max_retries,
            "ranker_error": str(last_error)[:500] if last_error else "unknown error",
            "ranker_fallback": "recall_order",
        }


Ranker = LocalE5Ranker | LLMJudgeRanker


async def select_final_docs(
    session: aiohttp.ClientSession,
    args: EvalArgs,
    ranker: Ranker | None,
    query: str,
    documents: list[dict[str, Any]],
    index: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    meta: dict[str, Any] = {
        "ranker_success": False,
        "search_tool_error": False,
        "reranker": args.reranker,
    }
    if args.run_mode == "no-ranker":
        raise RuntimeError("select_final_docs must not be called in no-ranker mode")
    if ranker is None:
        raise RuntimeError("ranker is required; ranker fallback is not allowed")
    if isinstance(ranker, LLMJudgeRanker):
        ranked_docs, llm_meta = await ranker.rank_topk(
            session=session,
            query=query,
            docs=documents,
            top_k=len(documents),
            index=index,
        )
        meta.update(llm_meta)
    else:
        ranked_docs = ranker.rank_topk(query=query, docs=documents, top_k=len(documents))
        meta["ranker_success"] = True
    agent_top_k = min(args.top_m, args.ranker_top_k, len(ranked_docs))
    return ranked_docs, ranked_docs[:agent_top_k], meta


async def evaluate_one(
    session: aiohttp.ClientSession,
    args: EvalArgs,
    agent_tokenizer: Any,
    ranker: Ranker | None,
    row: dict[str, Any],
    idx: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    data_source = str(row.get("data_source", "unknown"))
    prompt_messages = coerce_messages(row["prompt"])
    reward_model = coerce_dict(row.get("reward_model", {}))
    extra_info = coerce_dict(row.get("extra_info", {}))
    answers = coerce_answers((reward_model.get("ground_truth") or {}).get("target"))
    initial_query = extract_question(prompt_messages, extra_info)

    messages = [dict(item) for item in prompt_messages]
    agent_tool_schemas = args.tool_schemas if args.inject_tool_schema else None
    agent_system_prompt_ids = apply_chat_template(
        agent_tokenizer,
        [{}],
        enable_thinking=args.enable_thinking,
        tool_schemas=agent_tool_schemas,
        add_generation_prompt=False,
        tokenize=True,
    )
    prompt_ids = apply_chat_template(
        agent_tokenizer,
        messages,
        enable_thinking=args.enable_thinking,
        tool_schemas=agent_tool_schemas,
        add_generation_prompt=True,
        tokenize=True,
    )
    if args.max_prompt_length > 0 and len(prompt_ids) > args.max_prompt_length:
        prompt_ids = prompt_ids[-args.max_prompt_length :]
    assistant_texts: list[str] = []
    sub_queries: list[str] = []
    recall_top5_by_call: list[list[dict[str, Any]]] = []
    ranked_top50_by_call: list[list[dict[str, Any]]] = []
    ranked_top5_by_call: list[list[dict[str, Any]]] = []
    final_top5_by_call: list[list[dict[str, Any]]] = []
    recall_top50_by_call: list[list[dict[str, Any]]] = []
    stage_records: list[dict[str, Any]] = []

    agent_total_s = 0.0
    retrieve_total_s = 0.0
    ranker_total_s = 0.0
    total_start = time.perf_counter()
    final_answer = ""
    status = "max_turns"
    user_turns = 0

    if args.agent_base_url is None:
        raise ValueError("agent_base_url is required outside ranker-only mode")

    for _turn in range(args.max_assistant_turns):
        active_prompt_ids = prompt_ids
        if args.max_model_len > 0 and len(active_prompt_ids) >= args.max_model_len:
            raise RuntimeError(
                f"prompt length {len(active_prompt_ids)} leaves no generation budget under "
                f"max_model_len={args.max_model_len}"
            )
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
            stop_sequences=args.stop_sequences,
        )
        agent_elapsed = time.perf_counter() - t0
        agent_total_s += agent_elapsed
        assistant_ids = agent_tokenizer.encode(assistant_text, add_special_tokens=False)
        write_llm_io_trace(
            args.llm_io_jsonl,
            {
                "source": "coagentic_eval",
                "role": "agent",
                "index": idx,
                "data_source": data_source,
                "initial_query": initial_query,
                "assistant_turn": len(assistant_texts) + 1,
                "user_turn": user_turns,
                "prompt_token_count": len(active_prompt_ids),
                "output_token_count": len(assistant_ids),
                "prompt_text": prompt_text,
                "output_text": assistant_text,
                "sampling_params": {
                    "max_tokens": args.max_response_length,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stop": args.stop_sequences,
                },
            },
        )

        prompt_ids.extend(assistant_ids)
        assistant_texts.append(assistant_text)
        answer_in_current_turn = extract_answer(assistant_text)

        if args.max_assistant_turns and len(assistant_texts) >= args.max_assistant_turns:
            final_answer = answer_in_current_turn
            status = "answered" if final_answer else "max_turns"
            break
        if args.max_user_turns and user_turns >= args.max_user_turns:
            final_answer = answer_in_current_turn
            status = "answered" if final_answer else "max_user_turns"
            break

        tool_payloads, assistant_for_history = parse_tool_calls(assistant_text)
        if len(tool_payloads) == 1:
            tool_payload = tool_payloads[0]
            assistant_history_ids = agent_tokenizer.encode(assistant_for_history, add_special_tokens=False)
            trim_tokens = len(assistant_ids) - len(assistant_history_ids)
            if trim_tokens > 0:
                prompt_ids = prompt_ids[:-trim_tokens]
            assistant_texts[-1] = assistant_for_history
            messages.append({"role": "assistant", "content": assistant_for_history})
            raw_sub_query = (tool_payload.get("arguments") or {}).get("query")
            sub_query = str(raw_sub_query).strip() if raw_sub_query is not None else ""
            sub_queries.append(sub_query)

            if sub_query:
                t_retrieve = time.perf_counter()
                documents = await retrieve(session, args, sub_query)
                retrieve_elapsed = time.perf_counter() - t_retrieve
                retrieve_total_s += retrieve_elapsed

                t_ranker = time.perf_counter()
                if args.run_mode == "no-ranker":
                    ranked_docs = documents
                    final_docs = documents[: args.top_m]
                    ranker_meta = {
                        "ranker_success": False,
                        "search_tool_error": False,
                    }
                else:
                    ranked_docs, final_docs, ranker_meta = await select_final_docs(
                        session,
                        args,
                        ranker,
                        sub_query,
                        documents,
                        index=idx,
                    )
                ranker_elapsed = time.perf_counter() - t_ranker
                ranker_total_s += ranker_elapsed
                tool_response_text = truncate_tool_response(
                    train_format_tool_response(final_docs),
                    args.max_tool_response_length,
                )
            else:
                retrieve_elapsed = 0.0
                ranker_elapsed = 0.0
                documents = []
                ranked_docs = []
                final_docs = []
                ranker_meta = {
                    "ranker_success": False,
                    "search_tool_error": True,
                }
                tool_response_text = "Error: No query provided"

            recall_top5_docs = documents[:TRACE_TOP5]
            ranked_top50_docs = ranked_docs[:TRACE_TOP50]
            ranked_top5_docs = ranked_docs[:TRACE_TOP5]
            final_top5_docs = final_docs[:TRACE_TOP5]
            recall_top5_by_call.append(recall_top5_docs)
            ranked_top50_by_call.append(ranked_top50_docs)
            ranked_top5_by_call.append(ranked_top5_docs)
            final_top5_by_call.append(final_top5_docs)
            if args.keep_trace == "full":
                recall_top50_by_call.append(documents[:TRACE_TOP50])

            tool_message = {"role": "tool", "content": tool_response_text}
            messages.append(tool_message)
            tool_response_ids = apply_chat_template(
                agent_tokenizer,
                [tool_message],
                enable_thinking=args.enable_thinking,
                tool_schemas=agent_tool_schemas,
                add_generation_prompt=True,
                tokenize=True,
            )
            tool_response_delta_ids = tool_response_ids[len(agent_system_prompt_ids) :]
            prompt_ids.extend(tool_response_delta_ids)
            user_turns += 1
            recall_top5_doc_ids = trace_doc_ids(recall_top5_docs)
            ranked_top5_doc_ids = trace_doc_ids(ranked_top5_docs)
            ranked_top50_doc_ids = trace_doc_ids(ranked_top50_docs)
            final_top5_doc_ids = trace_doc_ids(final_top5_docs)
            stage_records.append(
                {
                    "sub_query": sub_query,
                    "agent_decision_s": agent_elapsed,
                    "retrieve_s": retrieve_elapsed,
                    "ranker_s": ranker_elapsed,
                    "recall_s": retrieve_elapsed + ranker_elapsed,
                    "num_recall_docs": len(documents),
                    "num_ranked_docs": len(ranked_docs),
                    "num_agent_visible_docs": len(final_docs),
                    "recall_top5_doc_ids": recall_top5_doc_ids,
                    "ranked_top5_doc_ids": ranked_top5_doc_ids,
                    "ranked_top50_doc_ids": ranked_top50_doc_ids,
                    "final_top5_doc_ids": final_top5_doc_ids,
                    "ranker_changed_top5": recall_top5_doc_ids != final_top5_doc_ids,
                    "ranker_promoted_doc_ids": [
                        doc_id for doc_id in final_top5_doc_ids if doc_id not in recall_top5_doc_ids
                    ],
                    "ranker_dropped_doc_ids": [
                        doc_id for doc_id in recall_top5_doc_ids if doc_id not in final_top5_doc_ids
                    ],
                    **ranker_meta,
                }
            )
            continue

        messages.append({"role": "assistant", "content": assistant_text})
        final_answer = answer_in_current_turn
        if final_answer:
            status = "answered" if sub_queries else "direct_answer_before_search"
        elif tool_payloads:
            status = "multiple_tool_calls"
        else:
            status = "no_valid_answer"
        break

    total_s = time.perf_counter() - total_start
    tool_calls = len(sub_queries)
    agent_turns = len(assistant_texts)
    recall_doc_counts = [float(record.get("num_recall_docs", 0.0)) for record in stage_records]
    ranked_doc_counts = [float(record.get("num_ranked_docs", 0.0)) for record in stage_records]
    visible_doc_counts = [float(record.get("num_agent_visible_docs", 0.0)) for record in stage_records]
    metrics = {
        "index": idx,
        "data_source": data_source,
        "em": exact_match(final_answer, answers),
        "f1": token_f1(final_answer, answers),
        "tool_calls": tool_calls,
        "agent_turns": agent_turns,
        "agent_decision_total_s": agent_total_s,
        "agent_decision_avg_s": agent_total_s / agent_turns if agent_turns else 0.0,
        "retrieve_total_s": retrieve_total_s,
        "ranker_total_s": ranker_total_s,
        "recall_total_s": retrieve_total_s + ranker_total_s,
        "recall_avg_s": (retrieve_total_s + ranker_total_s) / tool_calls if tool_calls else 0.0,
        "total_s": total_s,
        "num_recall_docs": avg(recall_doc_counts),
        "num_ranked_docs": avg(ranked_doc_counts),
        "num_agent_visible_docs": avg(visible_doc_counts),
        "ranker_enabled": args.run_mode != "no-ranker",
        "status": status,
    }
    trace = {
        "index": idx,
        "data_source": data_source,
        "prompt": prompt_messages,
        "sub_queries": sub_queries,
        "recall_top5_chunks": recall_top5_by_call,
        "ranked_top50_chunks": ranked_top50_by_call,
        "final_top5_chunks": final_top5_by_call,
        "ranked_top5_chunks": ranked_top5_by_call,
        "reranked_top5_chunks": final_top5_by_call,
        "final_answer": final_answer,
        "ground_truth_answer": answers,
        "status": status,
        "metrics": metrics,
        "stage_records": stage_records,
    }
    if args.keep_trace == "full":
        trace["retrieved_top50_chunks"] = recall_top50_by_call
    return metrics, trace


async def run_ranker_only(args: EvalArgs, ranker: Ranker) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    df = pd.read_parquet(args.data_path)
    if args.max_eval_num >= 0:
        df = df.head(args.max_eval_num)
    if args.max_ranker_steps >= 0:
        df = df.head(args.max_ranker_steps)
    records = df.to_dict(orient="records")

    metrics: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    async with aiohttp.ClientSession() as session:
        for idx, row in enumerate(records):
            started = time.perf_counter()
            prompt_messages = coerce_messages(row["prompt"])
            reward_model = coerce_dict(row.get("reward_model", {}))
            extra_info = coerce_dict(row.get("extra_info", {}))
            answers = coerce_answers((reward_model.get("ground_truth") or {}).get("target"))
            query = extract_question(prompt_messages, extra_info)
            data_source = str(row.get("data_source", "unknown"))
            documents = await retrieve(session, args, query)
            t_ranker = time.perf_counter()
            ranked_docs, final_docs, ranker_meta = await select_final_docs(
                session,
                args,
                ranker,
                query,
                documents,
                index=idx,
            )
            ranker_elapsed = time.perf_counter() - t_ranker
            total_s = time.perf_counter() - started
            recall_top5_docs = documents[:TRACE_TOP5]
            ranked_top50_docs = ranked_docs[:TRACE_TOP50]
            ranked_top5_docs = ranked_docs[:TRACE_TOP5]
            final_top5_docs = final_docs[:TRACE_TOP5]
            metric = {
                "index": idx,
                "data_source": data_source,
                "query": query,
                "em": 0.0,
                "f1": 0.0,
                "tool_calls": 1,
                "agent_turns": 0,
                "agent_decision_total_s": 0.0,
                "agent_decision_avg_s": 0.0,
                "retrieve_total_s": total_s - ranker_elapsed,
                "ranker_total_s": ranker_elapsed,
                "recall_total_s": total_s,
                "recall_avg_s": total_s,
                "total_s": total_s,
                "num_recall_docs": len(documents),
                "num_ranked_docs": len(ranked_docs),
                "num_agent_visible_docs": len(final_docs),
                "ranker_enabled": True,
                "status": "ranked",
                **ranker_meta,
            }
            trace = {
                "index": idx,
                "data_source": data_source,
                "query": query,
                "ground_truth_answer": answers,
                "recall_top5_chunks": [recall_top5_docs],
                "ranked_top50_chunks": [ranked_top50_docs],
                "final_top5_chunks": [final_top5_docs],
                "ranked_top5_chunks": [ranked_top5_docs],
                "reranked_top5_chunks": [final_top5_docs],
                "recall_top_docs": documents if args.keep_trace == "full" else documents[: args.top_m],
                "ranked_top_docs": ranked_docs if args.keep_trace == "full" else final_docs,
                "final_docs": final_docs,
                "stage_records": [
                    {
                        "sub_query": query,
                        "num_recall_docs": len(documents),
                        "num_ranked_docs": len(ranked_docs),
                        "num_agent_visible_docs": len(final_docs),
                        "recall_top5_doc_ids": trace_doc_ids(recall_top5_docs),
                        "ranked_top5_doc_ids": trace_doc_ids(ranked_top5_docs),
                        "ranked_top50_doc_ids": trace_doc_ids(ranked_top50_docs),
                        "final_top5_doc_ids": trace_doc_ids(final_top5_docs),
                    }
                ],
                "metrics": metric,
                "status": "ranked",
            }
            metrics.append(metric)
            traces.append(trace)
    return metrics, traces


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
        "ranker_total_s",
        "total_s",
        "num_recall_docs",
        "num_ranked_docs",
        "num_agent_visible_docs",
    ]
    result = {"n": float(len(rows))}
    for key in keys:
        result[key] = avg([float(row.get(key, 0.0)) for row in rows])
    return result


def build_summary(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        by_source[row.get("data_source", "unknown")].append(row)
    source_summary = {source: summarize_group(rows) for source, rows in sorted(by_source.items())}
    micro = summarize_group(metrics)
    macro = {"n": float(len(source_summary))}
    if source_summary:
        for key in next(iter(source_summary.values())).keys():
            if key != "n":
                macro[key] = avg([summary[key] for summary in source_summary.values()])
    else:
        macro.update({key: 0.0 for key in micro if key != "n"})
    status_counts = dict(Counter(row.get("status", "unknown") for row in metrics))
    failure_count = int(status_counts.get("failed", 0))
    return {
        "micro": micro,
        "macro": macro,
        "by_data_source": source_summary,
        "status_counts": status_counts,
        "success_count": len(metrics) - failure_count,
        "failure_count": failure_count,
    }


def fmt(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "0.0000"
    return f"{value:.4f}"


def effect_table(summary: dict[str, dict[str, float]]) -> str:
    lines = ["| Scope | N | EM | F1 |", "|---|---:|---:|---:|"]
    for name, row in summary.items():
        lines.append(f"| {name} | {int(row.get('n', 0))} | {fmt(row.get('em', 0.0))} | {fmt(row.get('f1', 0.0))} |")
    return "\n".join(lines)


def performance_table(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "| Scope | N | Tool Calls | Agent Avg s | Retrieve Avg s | Ranker Avg s | Recall Avg s | Total Avg s | Visible Docs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary.items():
        lines.append(
            f"| {name} | {int(row.get('n', 0))} | {fmt(row.get('tool_calls', 0.0))} | "
            f"{fmt(row.get('agent_decision_total_s', 0.0))} | {fmt(row.get('retrieve_total_s', 0.0))} | "
            f"{fmt(row.get('ranker_total_s', 0.0))} | {fmt(row.get('recall_total_s', 0.0))} | "
            f"{fmt(row.get('total_s', 0.0))} | {fmt(row.get('num_agent_visible_docs', 0.0))} |"
        )
    return "\n".join(lines)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def mirror_jsonl(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is not None:
        write_jsonl(path, rows)


def write_outputs(args: EvalArgs, metrics: list[dict[str, Any]], traces: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.trace_dir / "metrics.jsonl", metrics)
    write_jsonl(args.trace_dir / "traces.jsonl", traces)
    mirror_jsonl(args.metrics_jsonl, metrics)
    if args.run_mode == "ranker-only":
        mirror_jsonl(args.ranker_output_jsonl, traces)
    if args.validation_data_dir is not None:
        args.validation_data_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.validation_data_dir / "metrics.jsonl", metrics)
        write_jsonl(args.validation_data_dir / "traces.jsonl", traces)
    if args.rollout_data_dir is not None:
        args.rollout_data_dir.mkdir(parents=True, exist_ok=True)
    (args.trace_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.trace_dir / "run_config.json").write_text(
        json.dumps({key: str(value) for key, value in vars(args).items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_report(args: EvalArgs, summary: dict[str, Any], metrics: list[dict[str, Any]], elapsed_s: float) -> None:
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    overview = {"micro-average": summary["micro"], "macro-average": summary["macro"]}
    per_source = summary["by_data_source"]
    status_counts = summary.get("status_counts") or dict(Counter(row.get("status", "unknown") for row in metrics))
    ranker_enabled = args.run_mode != "no-ranker"
    lines = [
        "# CoAgenticRetriever vLLM Evaluation Report",
        "",
        f"- Strategy: `{args.strategy_name}`",
        f"- Run mode: `{args.run_mode}`",
        f"- Reranker: `{args.reranker}`",
        f"- Enable thinking: `{str(args.enable_thinking).lower()}`",
        f"- Ranker enabled: `{str(ranker_enabled).lower()}`",
        f"- Dataset: `{args.data_path}`",
        f"- Examples: `{len(metrics)}`",
        f"- Success count: `{summary.get('success_count', 0)}`",
        f"- Failure count: `{summary.get('failure_count', 0)}`",
        f"- Agent model: `{args.agent_model if args.agent_model else 'not used'}`",
        f"- Ranker tokenizer/base model: `{args.ranker_model if args.ranker_model else 'not used'}`",
        f"- Ranker encoder: `{args.ranker_encoder if args.ranker_encoder else 'not used'}`",
        f"- LLM judge endpoint: `{args.llm_judge_endpoint if args.llm_judge_endpoint else 'not used'}`",
        f"- LLM judge model: `{args.llm_judge_model if args.llm_judge_model else 'not used'}`",
        f"- Recall service: `{args.retrieval_url}`",
        f"- Trace dir: `{args.trace_dir}`",
        f"- Runtime metrics JSONL: `{args.metrics_jsonl if args.metrics_jsonl else ''}`",
        f"- Search timing JSONL: `{args.search_timing_jsonl if args.search_timing_jsonl else ''}`",
        f"- LLM IO JSONL: `{args.llm_io_jsonl if args.llm_io_jsonl else ''}`",
        f"- Validation data dir: `{args.validation_data_dir if args.validation_data_dir else ''}`",
        f"- Wall time: `{fmt(elapsed_s)}s`",
        f"- Status counts: `{status_counts}`",
        "",
        "## Eval Path",
        "",
    ]
    if args.run_mode == "no-ranker":
        lines.extend(
            [
                f"- Search path: `agent LLM -> recall retriever top-{args.top_n} -> recall top-{args.top_m} tool response -> agent LLM`",
                "- Dense ranker participation: `disabled`",
            ]
        )
    elif args.run_mode == "full":
        lines.extend(
            [
                f"- Search path: `agent LLM -> recall retriever top-{args.top_n} -> {args.reranker} reorder -> top-{min(args.top_m, args.ranker_top_k)} tool response -> agent LLM`",
                f"- Ranker participation: `{args.reranker}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Search path: `recall retriever top-{args.top_n} -> {args.reranker} reorder -> top-{min(args.top_m, args.ranker_top_k)} output`",
                "- Agent LLM participation: `disabled`",
            ]
        )
    lines.extend(
        [
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
            performance_table(overview),
            "",
            "## Performance Metrics By Dataset",
            "",
            performance_table(per_source),
            "",
            "## Artifacts",
            "",
            "- `metrics.jsonl`: per-example metrics under trace dir and runtime log path.",
            "- `traces.jsonl`: per-example conversation/search traces.",
            "- `summary.json`: aggregate metrics.",
            "- `run_config.json`: resolved runtime configuration.",
            "- `validation_data/`: mirrored eval metrics/traces for compatibility with previous full eval artifacts.",
        ]
    )
    if args.run_mode == "ranker-only":
        lines.append("- `ranker_infer_smoke.jsonl`: ranker-only retrieval and ranking output.")
    args.report_path.write_text("\n".join(lines), encoding="utf-8")


def make_ranker(args: EvalArgs) -> Ranker | None:
    if args.run_mode == "no-ranker":
        return None
    if args.reranker == "llm_as_judge":
        if args.llm_judge_endpoint is None or args.llm_judge_model is None or args.llm_judge_prompt_path is None:
            raise ValueError("llm_judge_endpoint, llm_judge_model and llm_judge_prompt_path are required")
        return LLMJudgeRanker(
            endpoint=args.llm_judge_endpoint,
            model=args.llm_judge_model,
            prompt_path=args.llm_judge_prompt_path,
            max_chunk_chars=args.llm_judge_max_chunk_chars,
            max_tokens=args.llm_judge_max_tokens,
            temperature=args.llm_judge_temperature,
            request_timeout=args.llm_judge_request_timeout,
            max_retries=args.llm_judge_max_retries,
            retry_delay=args.llm_judge_retry_delay,
            retry_backoff=args.llm_judge_retry_backoff,
            llm_io_jsonl=args.llm_io_jsonl,
        )
    if args.ranker_model is None or args.ranker_encoder is None:
        raise ValueError("ranker_model and ranker_encoder are required when dense ranker is enabled")
    return LocalE5Ranker(
        model_path=args.ranker_model,
        encoder_path=args.ranker_encoder,
        device=args.ranker_device,
        max_query_length=args.ranker_max_query_length,
        max_doc_length=args.ranker_max_doc_length,
        trust_remote_code=args.trust_remote_code,
    )


async def run_eval(args: EvalArgs) -> None:
    start = time.perf_counter()
    ranker = make_ranker(args)

    if args.run_mode == "ranker-only":
        assert ranker is not None
        metrics, traces = await run_ranker_only(args, ranker)
    else:
        if args.agent_model is None:
            raise ValueError("agent_model is required outside ranker-only mode")
        agent_tokenizer = AutoTokenizer.from_pretrained(args.agent_model, trust_remote_code=args.trust_remote_code)
        df = pd.read_parquet(args.data_path)
        if args.max_eval_num >= 0:
            df = df.head(args.max_eval_num)
        records = df.to_dict(orient="records")
        semaphore = asyncio.Semaphore(args.batch_size)

        async with aiohttp.ClientSession() as session:
            async def guarded(i: int, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
                async with semaphore:
                    sample_start = time.perf_counter()
                    try:
                        return await evaluate_one(session, args, agent_tokenizer, ranker, row, i)
                    except Exception as exc:
                        return build_failed_result(row, i, exc, time.perf_counter() - sample_start)

            results = await asyncio.gather(*[guarded(i, row) for i, row in enumerate(records)])
        metrics = [item[0] for item in results]
        traces = [item[1] for item in results]

    summary = build_summary(metrics)
    elapsed = time.perf_counter() - start
    write_outputs(args, metrics, traces, summary)
    write_report(args, summary, metrics, elapsed)


def parse_run_args(ns: argparse.Namespace) -> EvalArgs:
    agent_model = None
    if ns.run_mode != "ranker-only":
        agent_model = resolve_model_dir(ns.agent_model, "agent")

    ranker_model = None
    ranker_encoder = None
    if ns.run_mode != "no-ranker" and ns.reranker == "dense_e5":
        ranker_model, ranker_encoder = resolve_ranker_paths(
            ranker_model=ns.ranker_model,
            ranker_encoder=ns.ranker_encoder,
            ranker_base_model=ns.ranker_base_model,
        )
        missing_ranker_args = [
            name
            for name in ("ranker_top_k", "ranker_device", "ranker_max_query_length", "ranker_max_doc_length")
            if getattr(ns, name) in (None, "")
        ]
        if missing_ranker_args:
            raise ValueError(
                "ranker arguments are required when ranker is enabled; missing "
                + ", ".join(missing_ranker_args)
            )
    if ns.run_mode != "no-ranker" and ns.reranker == "llm_as_judge":
        missing_llm_judge_args = [
            name
            for name in ("llm_judge_endpoint", "llm_judge_model", "llm_judge_prompt_path", "ranker_top_k")
            if getattr(ns, name) in (None, "")
        ]
        if missing_llm_judge_args:
            raise ValueError(
                "LLM judge reranker arguments are required; missing "
                + ", ".join(missing_llm_judge_args)
            )

    return EvalArgs(
        run_mode=ns.run_mode,
        reranker="none" if ns.run_mode == "no-ranker" else ns.reranker,
        agent_model=agent_model,
        ranker_model=ranker_model,
        ranker_encoder=ranker_encoder,
        llm_judge_endpoint=ns.llm_judge_endpoint,
        llm_judge_model=ns.llm_judge_model,
        llm_judge_prompt_path=ns.llm_judge_prompt_path,
        llm_judge_max_chunk_chars=ns.llm_judge_max_chunk_chars,
        llm_judge_max_tokens=ns.llm_judge_max_tokens,
        llm_judge_temperature=ns.llm_judge_temperature,
        llm_judge_request_timeout=ns.llm_judge_request_timeout,
        llm_judge_max_retries=ns.llm_judge_max_retries,
        llm_judge_retry_delay=ns.llm_judge_retry_delay,
        llm_judge_retry_backoff=ns.llm_judge_retry_backoff,
        data_path=ns.data_path,
        max_eval_num=ns.max_eval_num,
        max_ranker_steps=ns.max_ranker_steps,
        batch_size=ns.batch_size,
        keep_trace=ns.keep_trace,
        trace_dir=ns.trace_dir,
        report_path=ns.report_path,
        strategy_name=ns.strategy_name,
        retrieval_url=ns.retrieval_url,
        agent_base_url=ns.agent_base_url,
        agent_served_model=ns.agent_served_model,
        top_n=ns.top_n,
        top_m=ns.top_m,
        ranker_top_k=ns.ranker_top_k,
        max_assistant_turns=ns.max_assistant_turns,
        max_user_turns=ns.max_user_turns,
        max_tool_response_length=ns.max_tool_response_length,
        max_prompt_length=ns.max_prompt_length,
        max_response_length=ns.max_response_length,
        max_model_len=ns.max_model_len,
        temperature=ns.temperature,
        top_p=ns.top_p,
        request_timeout=ns.request_timeout,
        max_retries=ns.max_retries,
        retry_delay=ns.retry_delay,
        retry_backoff=ns.retry_backoff,
        llm_io_jsonl=ns.llm_io_jsonl,
        metrics_jsonl=ns.metrics_jsonl,
        search_timing_jsonl=ns.search_timing_jsonl,
        ranker_output_jsonl=ns.ranker_output_jsonl,
        validation_data_dir=ns.validation_data_dir,
        rollout_data_dir=ns.rollout_data_dir,
        ranker_device=ns.ranker_device,
        ranker_max_query_length=ns.ranker_max_query_length,
        ranker_max_doc_length=ns.ranker_max_doc_length,
        trust_remote_code=ns.trust_remote_code,
        enable_thinking=ns.enable_thinking,
        tool_config_path=ns.tool_config_path,
        inject_tool_schema=ns.inject_tool_schema,
        tool_schemas=load_tool_schemas(ns.tool_config_path),
        stop_sequences=ns.stop_sequences,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    resolve_parser = subparsers.add_parser("resolve-model")
    resolve_parser.add_argument("--path", type=Path, required=True)
    resolve_parser.add_argument("--role", choices=["agent", "ranker-encoder"], required=True)

    resolve_ranker_parser = subparsers.add_parser("resolve-ranker")
    resolve_ranker_parser.add_argument("--ranker-model", type=Path, default=None)
    resolve_ranker_parser.add_argument("--ranker-base-model", type=Path, default=None)
    resolve_ranker_parser.add_argument("--ranker-encoder", type=Path, default=None)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-mode", choices=["ranker-only", "full", "no-ranker"], required=True)
    run_parser.add_argument("--reranker", choices=["dense_e5", "llm_as_judge"], default="dense_e5")
    run_parser.add_argument("--agent-model", type=Path, default=None)
    run_parser.add_argument("--ranker-model", type=Path, default=None)
    run_parser.add_argument("--ranker-base-model", type=Path, default=None)
    run_parser.add_argument("--ranker-encoder", type=Path, default=None)
    run_parser.add_argument("--llm-judge-endpoint", default=None)
    run_parser.add_argument("--llm-judge-model", default=None)
    run_parser.add_argument("--llm-judge-prompt-path", type=Path, default=None)
    run_parser.add_argument("--llm-judge-max-chunk-chars", type=int, default=512)
    run_parser.add_argument("--llm-judge-max-tokens", type=int, default=1024)
    run_parser.add_argument("--llm-judge-temperature", type=float, default=0.0)
    run_parser.add_argument("--llm-judge-request-timeout", type=float, default=600.0)
    run_parser.add_argument("--llm-judge-max-retries", type=int, default=3)
    run_parser.add_argument("--llm-judge-retry-delay", type=float, default=2.0)
    run_parser.add_argument("--llm-judge-retry-backoff", type=float, default=2.0)
    run_parser.add_argument("--data-path", type=Path, required=True)
    run_parser.add_argument("--max-eval-num", type=int, default=-1)
    run_parser.add_argument("--max-ranker-steps", type=int, default=-1)
    run_parser.add_argument("--batch-size", type=int, default=8)
    run_parser.add_argument("--keep-trace", choices=["partial", "full"], default="partial")
    run_parser.add_argument("--trace-dir", type=Path, required=True)
    run_parser.add_argument("--report-path", type=Path, required=True)
    run_parser.add_argument("--strategy-name", default="default")
    run_parser.add_argument("--retrieval-url", required=True)
    run_parser.add_argument("--agent-base-url", default=None)
    run_parser.add_argument("--agent-served-model", default="coagentic-agent")
    run_parser.add_argument("--top-n", type=int, default=50)
    run_parser.add_argument("--top-m", type=int, default=5)
    run_parser.add_argument("--ranker-top-k", type=int, default=None)
    run_parser.add_argument("--max-assistant-turns", type=int, default=6)
    run_parser.add_argument("--max-user-turns", type=int, default=6)
    run_parser.add_argument("--max-tool-response-length", type=int, default=4096)
    run_parser.add_argument("--max-prompt-length", type=int, default=11264)
    run_parser.add_argument("--max-response-length", type=int, default=1024)
    run_parser.add_argument("--max-model-len", type=int, default=8192)
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--top-p", type=float, default=1.0)
    run_parser.add_argument("--request-timeout", type=float, default=180.0)
    run_parser.add_argument("--max-retries", type=int, default=3)
    run_parser.add_argument("--retry-delay", type=float, default=1.0)
    run_parser.add_argument("--retry-backoff", type=float, default=2.0)
    run_parser.add_argument("--llm-io-jsonl", type=Path, default=None)
    run_parser.add_argument("--metrics-jsonl", type=Path, default=None)
    run_parser.add_argument("--search-timing-jsonl", type=Path, default=None)
    run_parser.add_argument("--ranker-output-jsonl", type=Path, default=None)
    run_parser.add_argument("--validation-data-dir", type=Path, default=None)
    run_parser.add_argument("--rollout-data-dir", type=Path, default=None)
    run_parser.add_argument("--ranker-device", default=None)
    run_parser.add_argument("--ranker-max-query-length", type=int, default=None)
    run_parser.add_argument("--ranker-max-doc-length", type=int, default=None)
    run_parser.add_argument("--tool-config-path", type=Path, default=None)
    run_parser.add_argument("--inject-tool-schema", action=argparse.BooleanOptionalAction, default=False)
    run_parser.add_argument("--stop-sequence", action="append", dest="stop_sequences", default=None)
    run_parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)

    ns = parser.parse_args()
    if ns.command == "resolve-model":
        if ns.role == "agent":
            print(resolve_model_dir(ns.path, "agent"))
        else:
            print(resolve_ranker_encoder_dir(ns.path))
        return 0
    if ns.command == "resolve-ranker":
        model, encoder = resolve_ranker_paths(ns.ranker_model, ns.ranker_encoder, ns.ranker_base_model)
        print(json.dumps({"ranker_model": str(model), "ranker_encoder": str(encoder)}, ensure_ascii=False))
        return 0

    args = parse_run_args(ns)
    asyncio.run(run_eval(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
