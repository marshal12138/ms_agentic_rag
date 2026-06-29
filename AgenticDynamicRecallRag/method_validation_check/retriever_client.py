"""Client for the readme /retrieve endpoint.

Request body:  {"queries": ["...", "..."]}
Response body: {"result": [[bm25_ids, dense_ids], ...]} -- one length-2 list per query,
each inner list holding the per-retriever ranked document ids.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import List

from .config import Config


class RetrieverClient:
    def __init__(self, config: Config):
        self.endpoint = config.retriever_endpoint
        self.timeout = config.retriever_timeout_seconds
        self.max_retries = config.llm_max_retries

    def retrieve(self, queries: List[str]) -> List[List[List]]:
        """Return result[q] = [bm25_ids, dense_ids] for each query."""
        payload = {"queries": list(queries)}
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for _attempt in range(max(1, self.max_retries + 1)):
            req = urllib.request.Request(
                self.endpoint,
                data=encoded,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["result"]
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"retriever request failed after retries: {last_error}")
