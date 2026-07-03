"""OpenAI-compatible LLM reranker client."""

from __future__ import annotations

import json
import urllib.request
from typing import Any

from agentic_iter_rag.llm_reranker.format import parse_ranked_doc_ids, render_reranker_prompt


def rerank_with_openai_compatible_endpoint(
    *,
    endpoint: str,
    model: str,
    query: str,
    docs: list[dict[str, Any]],
    timeout: int = 180,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> list[dict[str, Any]]:
    messages = render_reranker_prompt(query, docs)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    content = data["choices"][0]["message"]["content"]
    ordered_ids = parse_ranked_doc_ids(content, [str(doc.get("doc_id")) for doc in docs])
    by_id = {str(doc.get("doc_id")): doc for doc in docs}
    return [by_id[doc_id] for doc_id in ordered_ids if doc_id in by_id]

