"""OpenAI-compatible chat-completions client for query generation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional

from .config import Config
from .prompt import build_prompt, extract_query


class LLMClient:
    def __init__(self, config: Config):
        self.endpoint = config.llm_endpoint
        self.model = config.llm_model
        self.temperature = config.llm_temperature
        self.max_tokens = config.llm_max_tokens
        self.timeout = config.llm_timeout_seconds
        self.max_retries = config.llm_max_retries

    def _post_json(self, payload: dict) -> dict:
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
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
                time.sleep(0.5)
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def generate_query(self, question: str) -> Optional[str]:
        """Return the search query the model emits, or None if it does not call the tool."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_prompt(question)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = self._post_json(payload)
        try:
            content = data["choices"][0]["message"].get("content", "")
        except (KeyError, IndexError, TypeError):
            return None
        return extract_query(content)

if __name__ == "__main__":
    config=Config()
    llm=LLMClient(config)
    question="英国的首都是哪个？"
    print(build_prompt(question))
    print(llm.generate_query(question))