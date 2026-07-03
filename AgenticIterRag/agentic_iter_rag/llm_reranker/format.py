"""Prompt rendering and ranked-id parsing for LLM rerankers."""

from __future__ import annotations

import json
import re
from typing import Any


AIR_RERANK_PROMPT_WITH_INITIAL_QUERY = """
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

In particular:
- Prefer passages that can directly help answer the Initial Query.
- If none can directly answer it, prefer passages that best match the Sub-Query
  WHILE staying strictly within the scope and intent of the Initial Query.

# === STRICT OUTPUT FORMAT (must match EXACTLY) ===
<reason> ... </reason>
<rerank> ... </rerank>

Anything outside these two tags or in a different order is invalid.

# === BLOCK 1: <reason> ... </reason>
Explain your ranking decisions clearly and concretely.

Follow these steps:
1. Identify what the Initial Query is ultimately asking.
2. Identify what specific information the Current Sub-Query is seeking.
3. Explain how the selected passages either:
   - directly help answer the Initial Query, or
   - provide the most relevant information for the Sub-Query
     without drifting away from the Initial Query.
4. If a passage matches the Sub-Query but is off-topic or irrelevant
   to the Initial Query, explain why it is ranked lower.
5. When multiple passages are similar, break ties using factuality,
   entity specificity, and usefulness for later steps.

Write 5-8 short sentences.
Do NOT include passage indices here.

# === BLOCK 2: <rerank> ... </rerank>
Purpose: output the final ranking of EXACTLY {M} passages you judge are MOST USEFUL at this step.

Format requirements:
- Use ONLY indices from the input (between [1] and [{N}]).
- Include EXACTLY {M} distinct indices (no repeats).
- Chain them with ' > ' (spaces around '>').
- No commas, no scores, no extra text.
- Order by usefulness: the FIRST passage listed is the MOST useful, and usefulness decreases left to right.

Example (structure only, M=5):
<rerank>[27] > [233] > [105] > [729] > [688]</rerank>

# === DECISION GUIDELINES
0. Final-Answer Priority:
   If a passage directly helps answer the Initial Query,
   rank it higher even if it only partially matches the Sub-Query.
1. Sub-Query Relevance:
   Among remaining passages, prefer those that best match the Sub-Query.
2. Initial-Query Constraint:
   Any passage that drifts away from the Initial Query's topic
   should be ranked lower, even if it matches the Sub-Query well.
3. Information Gain:
   Prefer concrete facts, entities, relations, or dates over vague descriptions.
4. Specificity:
   Rank specific, clearly grounded passages above generic or background content.

# === INPUT BEGINS ===
Initial Query:
{initial_query}

Current Sub-Query:
{sub_query}

Passages ({N} total):
{passages_block}
# === INPUT ENDS ===
"""


def format_air_passages(
    docs: list[dict[str, Any]],
    *,
    max_doc_chars: int,
) -> tuple[str, dict[str, str]]:
    """按 AIR reranker 候选序号格式渲染文档，并返回序号到真实 doc_id 的映射。"""

    passages: list[str] = []
    index_to_doc_id: dict[str, str] = {}
    for idx, doc in enumerate(docs, start=1):
        doc_id = str(doc.get("doc_id") or doc.get("id") or idx)
        title = str(doc.get("title") or "")
        text = str(doc.get("text") or doc.get("contents") or doc.get("passage") or "")[:max_doc_chars]
        if title:
            passages.append(f"[{idx}] Title: {title}\n{text}")
        else:
            passages.append(f"[{idx}] {text}")
        index_to_doc_id[str(idx)] = doc_id
    return "\n".join(passages), index_to_doc_id


def render_air_rerank_tags_prompt(
    *,
    initial_query: str,
    sub_query: str,
    docs: list[dict[str, Any]],
    top_m: int,
    max_doc_chars: int,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    """渲染 AIR reranker prompt，输出 VERL chat message 和 doc_id 映射。"""

    passages_block, index_to_doc_id = format_air_passages(docs, max_doc_chars=max_doc_chars)
    prompt_text = AIR_RERANK_PROMPT_WITH_INITIAL_QUERY.format(
        N=len(docs),
        M=min(top_m, len(docs)),
        initial_query=initial_query,
        sub_query=sub_query,
        passages_block=passages_block,
    )
    return [{"role": "user", "content": prompt_text}], index_to_doc_id


def render_reranker_prompt(query: str, docs: list[dict[str, Any]], max_doc_chars: int = 512) -> list[dict[str, str]]:
    passages = []
    for idx, doc in enumerate(docs, start=1):
        doc_id = str(doc.get("doc_id") or doc.get("id") or idx)
        title = str(doc.get("title") or "")
        text = str(doc.get("text") or doc.get("contents") or "")[:max_doc_chars]
        passages.append(f"[{idx}] doc_id={doc_id}\ntitle={title}\ntext={text}")
    user = (
        "Rank the candidate documents by relevance to the search query. "
        "Return JSON only: {\"ranked_doc_ids\": [\"...\"]}.\n\n"
        f"query: {query}\n\n" + "\n\n".join(passages)
    )
    return [
        {"role": "system", "content": "You are a precise document reranker."},
        {"role": "user", "content": user},
    ]


def parse_ranked_doc_ids(text: str, valid_doc_ids: list[str]) -> list[str]:
    valid = [str(x) for x in valid_doc_ids]
    try:
        data = json.loads(text)
        ids = data.get("ranked_doc_ids") if isinstance(data, dict) else data
        if isinstance(ids, list):
            parsed = [str(x) for x in ids if str(x) in valid]
            if parsed:
                return parsed + [doc_id for doc_id in valid if doc_id not in parsed]
    except json.JSONDecodeError:
        pass
    found = [match for match in re.findall(r"[A-Za-z0-9_.:/-]+", text) if match in valid]
    deduped: list[str] = []
    for doc_id in found:
        if doc_id not in deduped:
            deduped.append(doc_id)
    return deduped + [doc_id for doc_id in valid if doc_id not in deduped]
