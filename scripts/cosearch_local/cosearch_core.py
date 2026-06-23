"""Shared utilities for the local single-GPU CoSearch reproduction."""

from __future__ import annotations

import json
import math
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import requests


RERANK_PROMPT_WITH_INITIAL_QUERY = """
You are a professional document reranker specialized in multi-step search and reasoning tasks.

You will be given:
- An Initial Query: the user's ultimate question and final goal.
- A Current Sub-Query: a focused query generated to retrieve information for the current step.
- A list of {N} candidate passages.

Your goal is:
Rank EXACTLY {M} passages that are MOST USEFUL at this step.

Primary principle:
Ranking is based on the Current Sub-Query,
but the Sub-Query MUST be interpreted and constrained by the Initial Query.

# === STRICT OUTPUT FORMAT (must match EXACTLY) ===
<reason> ... </reason>
<rerank> ... </rerank>

Anything outside these two tags or in a different order is invalid.

# === BLOCK 1: <reason> ... </reason>
Explain your ranking decisions clearly and concretely.
Do NOT include passage indices here.

# === BLOCK 2: <rerank> ... </rerank>
Output EXACTLY {M} distinct indices from [1] to [{N}] chained with ' > '.
Example for M=5:
<rerank>[27] > [233] > [105] > [729] > [688]</rerank>

# === INPUT BEGINS ===
Initial Query:
{initial_query}

Current Sub-Query:
{sub_query}

Passages ({N} total):
{passages_block}
# === INPUT ENDS ===
"""


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
TOOL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)
RERANK_RE = re.compile(r"<rerank>(.*?)</rerank>", re.DOTALL)
IDX_RE = re.compile(r"\[(\d+)\]")


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def remove_punc(s: str) -> str:
        return "".join(ch for ch in s if ch not in string.punctuation)

    return " ".join(remove_articles(remove_punc(text.lower())).split())


def token_f1(prediction: str, answers: list[str] | str) -> float:
    if isinstance(answers, str):
        answers = [answers]
    pred_tokens = normalize_answer(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        gold_tokens = normalize_answer(answer).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def extract_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def parse_search_query(text: str) -> tuple[str | None, bool]:
    matches = TOOL_RE.findall(text or "")
    if len(matches) != 1 or len(REASON_RE.findall(text or "")) != 1:
        return None, False
    try:
        payload = json.loads(matches[0].strip())
    except Exception:
        return None, False
    if payload.get("name") != "search":
        return None, False
    args = payload.get("arguments")
    if not isinstance(args, dict) or not isinstance(args.get("query"), str):
        return None, False
    return args["query"].strip(), bool(args["query"].strip())


def final_answer_format_ok(text: str) -> bool:
    return len(REASON_RE.findall(text or "")) == 1 and extract_answer(text) is not None


def format_docs_for_prompt(docs: list[dict], max_doc_length: int = 2000) -> str:
    lines = []
    for i, doc in enumerate(docs, 1):
        contents = doc.get("contents") or doc.get("text") or doc.get("passage") or ""
        title = doc.get("title", "")
        if len(contents) > max_doc_length:
            contents = contents[:max_doc_length] + "..."
        if title:
            lines.append(f"[{i}] Title: {title}\n{contents}")
        else:
            lines.append(f"[{i}] {contents}")
    return "\n".join(lines)


def retrieve(retrieval_url: str, query: str, top_n: int, timeout: int = 60) -> list[dict]:
    resp = requests.post(
        retrieval_url,
        json={"queries": [query], "topk": top_n, "return_scores": True},
        timeout=timeout,
    )
    resp.raise_for_status()
    raw = resp.json().get("result", [[]])[0]
    docs = []
    for item in raw:
        doc = item.get("document", item)
        out = {
            "id": str(doc.get("id", "")),
            "title": doc.get("title", ""),
            "contents": doc.get("contents") or doc.get("text") or doc.get("passage") or "",
            "score": float(item.get("score", doc.get("score", 0.0))),
        }
        docs.append(out)
    return docs


def validate_rerank_output(output: str, n_docs: int, top_m: int) -> tuple[list[int], list[str]]:
    errors = []
    reason_blocks = REASON_RE.findall(output or "")
    if len(reason_blocks) != 1:
        errors.append(f"expected one reason block, got {len(reason_blocks)}")
    rerank_blocks = RERANK_RE.findall(output or "")
    if len(rerank_blocks) != 1:
        errors.append(f"expected one rerank block, got {len(rerank_blocks)}")
        return [], errors
    rank_text = rerank_blocks[0].strip()
    expected = re.compile(r"^\s*\[\d+\](\s*>\s*\[\d+\]){" + str(top_m - 1) + r"}\s*$")
    if not expected.match(rank_text):
        errors.append("bad rerank chain format")
    ids = [int(x) for x in IDX_RE.findall(rank_text)]
    if len(ids) != top_m:
        errors.append(f"expected {top_m} ids, got {len(ids)}")
    if len(set(ids)) != len(ids):
        errors.append("duplicate ids")
    if any(i < 1 or i > n_docs for i in ids):
        errors.append("id out of range")
    if errors:
        return [], errors
    return [i - 1 for i in ids], []


def answer_in_text(answers: list[str], text: str) -> bool:
    norm_text = f" {normalize_answer(text)} "
    return any(f" {normalize_answer(a)} " in norm_text for a in answers if normalize_answer(a))


def answer_in_docs(answers: list[str], docs: list[dict]) -> bool:
    return any(answer_in_text(answers, d.get("contents", "")) for d in docs)


def average_hit_at_ks(answers: list[str], docs: list[dict], cutoffs: tuple[int, ...] = (1, 3, 5)) -> float:
    hits = [1 if answer_in_text(answers, d.get("contents", "")) else 0 for d in docs]
    scores = [1 if sum(hits[: min(k, len(hits))]) > 0 else 0 for k in cutoffs]
    return sum(scores) / len(scores)


def rouge1_f1(a: str, b: str) -> float:
    ta = normalize_answer(a).split()
    tb = normalize_answer(b).split()
    if not ta or not tb:
        return 0.0
    ca = Counter(ta)
    cb = Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def semantic_group_indices(items: list[dict], threshold: float, min_size: int) -> dict[str, list[int]]:
    base_groups: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(items):
        base_groups[f"{item['question_id']}_{'easy' if item.get('answer_in_docs') else 'hard'}"].append(idx)

    final_groups: dict[str, list[int]] = {}
    for base_key, idxs in base_groups.items():
        reps: list[str] = []
        clusters: list[list[int]] = []
        for idx in idxs:
            subq = items[idx].get("sub_query", "")
            placed = False
            for cluster_id, rep in enumerate(reps):
                if rouge1_f1(subq, rep) >= threshold:
                    clusters[cluster_id].append(idx)
                    placed = True
                    break
            if not placed:
                reps.append(subq)
                clusters.append([idx])
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) >= min_size:
                final_groups[f"{base_key}_cluster_{cluster_id}"] = cluster
    return final_groups


def group_advantages(rewards: list[float], groups: dict[str, list[int]]) -> dict[int, float]:
    advantages = {}
    for idxs in groups.values():
        vals = [rewards[i] for i in idxs]
        mean = sum(vals) / len(vals)
        variance = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = math.sqrt(variance) if variance > 1e-8 else 1.0
        for i in idxs:
            advantages[i] = (rewards[i] - mean) / std
    return advantages


def composite_ranker_reward(
    tool_score: float,
    agent_score: float,
    docs_have_answer: bool,
    valid_format: bool,
    agent_threshold: float = 0.8,
    cond_threshold: float = 0.5,
    format_penalty: float = -0.2,
) -> float:
    if not valid_format:
        return format_penalty
    agent_bin = 1.0 if agent_score >= agent_threshold else 0.0
    if docs_have_answer and tool_score < cond_threshold:
        return tool_score
    return tool_score + agent_bin


@dataclass
class GenerationSample:
    prompt_ids: list[int]
    completion_ids: list[int]
    reward: float
    advantage: float = 0.0
    meta: dict[str, Any] | None = None
