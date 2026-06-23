"""LLM-as-judge rank-50 scoring stage."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schemas import AsyncLabelRequest, JudgeChunkScore
from ..prompt import MarkdownSystemUserPrompt, stage_prompt_config

OUTPUT_MODE_CHAT_TEMPLATE_KWARGS = {
    "no_think": {"thinking": False, "enable_thinking": False, "reasoning_effort": "none"},
    "think_high": {"thinking": True, "enable_thinking": True, "reasoning_effort": "high"},
    "think_max": {"thinking": True, "enable_thinking": True, "reasoning_effort": "max"},
}


def _require_stage_config(config: dict[str, Any], key: str):
    if key not in config or config[key] in (None, ""):
        raise KeyError(f"missing required async labeling stage config: {key}")
    return config[key]


@dataclass(slots=True)
class StageResult:
    ok: bool
    scores: list[JudgeChunkScore] = field(default_factory=list)
    raw_response: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    error_type: str | None = None
    error_message: str | None = None


class LLMJudgeRank50Stage:
    """Validate and convert LLM ranked_ids output to rank scores.

    The stage calls an OpenAI-compatible chat-completions endpoint and only
    returns ranking scores. Positive/negative labels are created by sample
    builders downstream.
    """

    def __init__(self, config: dict[str, Any], *, project_root: str | Path | None = None):
        self.config = dict(config)
        prompt_cfg = stage_prompt_config(self.config)
        self.endpoint = str(_require_stage_config(self.config, "endpoint"))
        self.model = str(_require_stage_config(self.config, "model"))
        self.temperature = float(_require_stage_config(self.config, "temperature"))
        self.max_tokens = int(_require_stage_config(self.config, "max_tokens"))
        self.request_timeout_seconds = float(_require_stage_config(self.config, "request_timeout_seconds"))
        self.max_retries = int(_require_stage_config(self.config, "max_retries"))
        self.prompt_version = str(prompt_cfg["version"])
        self.output_mode = self._normalize_output_mode(str(prompt_cfg["output_mode"]))
        self.chat_template_kwargs = self._chat_template_kwargs_for_output_mode(self.output_mode)
        self.prompt = MarkdownSystemUserPrompt(
            str(prompt_cfg["path"]),
            project_root=project_root,
            max_chunk_chars=int(prompt_cfg["max_chunk_chars"]),
        )

    def score(self, request: AsyncLabelRequest) -> StageResult:
        started = time.perf_counter()
        try:
            messages = self.prompt.render_messages(request)
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "chat_template_kwargs": dict(self.chat_template_kwargs),
            }
            data = self._post_json(payload)
            content = data["choices"][0]["message"].get("content", "")
            ranked_ids = self.parse_ranked_ids(content)
            scores = self.score_ranked_ids(request, ranked_ids)
            return StageResult(
                ok=True,
                scores=scores,
                raw_response=content,
                usage=dict(data.get("usage") or {}),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            return StageResult(
                ok=False,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    @staticmethod
    def _normalize_output_mode(output_mode: str) -> str:
        aliases = {
            "no_think_json": "no_think",
            "non_think": "no_think",
            "non-think": "no_think",
            "think-high": "think_high",
            "think-max": "think_max",
        }
        normalized = aliases.get(output_mode, output_mode)
        if normalized not in OUTPUT_MODE_CHAT_TEMPLATE_KWARGS:
            allowed = ", ".join(sorted(OUTPUT_MODE_CHAT_TEMPLATE_KWARGS))
            raise ValueError(f"unsupported LLM judge output_mode: {output_mode}; allowed: {allowed}")
        return normalized

    @staticmethod
    def _chat_template_kwargs_for_output_mode(output_mode: str) -> dict[str, Any]:
        return dict(OUTPUT_MODE_CHAT_TEMPLATE_KWARGS[output_mode])

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
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
                with urllib.request.urlopen(req, timeout=self.request_timeout_seconds) as resp:
                    body = resp.read().decode("utf-8")
                return json.loads(body)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError(f"LLM judge request failed after retries: {last_error}")

    def parse_ranked_ids(self, text: str) -> list[str]:
        obj = self._extract_json_object(text)
        ranked_ids = obj.get("ranked_ids")
        if not isinstance(ranked_ids, list):
            raise ValueError("judge response missing list field: ranked_ids")
        return [str(item) for item in ranked_ids]

    def score_ranked_ids(self, request: AsyncLabelRequest, ranked_ids: list[str]) -> list[JudgeChunkScore]:
        request.validate_rank50()
        request_ids = [chunk.doc_id for chunk in request.ranked_chunk_list]
        request_id_set = set(request_ids)
        if len(ranked_ids) != 50:
            raise ValueError(f"ranked_ids must contain exactly 50 ids, got {len(ranked_ids)}")
        if len(set(ranked_ids)) != 50:
            raise ValueError("ranked_ids contains duplicated ids")
        unknown = [doc_id for doc_id in ranked_ids if doc_id not in request_id_set]
        if unknown:
            raise ValueError(f"ranked_ids contains unknown ids: {unknown[:5]}")
        missing = [doc_id for doc_id in request_ids if doc_id not in set(ranked_ids)]
        if missing:
            raise ValueError(f"ranked_ids missing request ids: {missing[:5]}")
        return [
            JudgeChunkScore(doc_id=doc_id, judge_rank=rank, judge_score=(50 - rank + 1) / 50)
            for rank, doc_id in enumerate(ranked_ids, start=1)
        ]

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return json.loads(stripped)
        matches = re.findall(r"\{.*?\}", stripped, flags=re.DOTALL)
        if not matches:
            raise ValueError("judge response does not contain a JSON object")
        return json.loads(matches[-1])
