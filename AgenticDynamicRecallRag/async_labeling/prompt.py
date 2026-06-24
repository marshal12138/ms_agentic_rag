"""Prompt loading and rendering for LLM judge stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import validate_prompt_path
from .schemas import AsyncLabelRequest, CandidateChunk


class MarkdownSystemUserPrompt:
    def __init__(self, path: str, *, project_root: str | Path | None = None, max_chunk_chars: int = 512):
        self.path = validate_prompt_path(path, project_root=project_root)
        self.max_chunk_chars = max(1, int(max_chunk_chars))
        text = self.path.read_text(encoding="utf-8")
        self.system_template, self.user_template = self._parse(text)

    def render_messages(self, request: AsyncLabelRequest) -> list[dict[str, str]]:
        request.validate_rank50()
        allowed_ids = [chunk.doc_id for chunk in request.ranked_chunk_list]
        user = self.user_template
        user = user.replace("{{原始查询问题}}", request.origin_query)
        user = user.replace("{{规范化后的查询问题}}", request.sub_query)
        user = user.replace("{{允许的所有段落ID列表}}", ", ".join(allowed_ids))
        user = self._replace_candidate_template(user, request.ranked_chunk_list)
        if "{{" in user or "}}" in user:
            raise ValueError(f"unrendered prompt placeholder remains in {self.path}")
        return [
            {"role": "system", "content": self.system_template.strip()},
            {"role": "user", "content": user.strip()},
        ]

    def _replace_candidate_template(self, user: str, chunks: list[CandidateChunk]) -> str:
        marker = "[id: {{段落ID}}]"
        start = user.find(marker)
        if start < 0:
            raise ValueError(f"candidate template marker not found in prompt: {self.path}")
        text_marker = "{{段落文本片段}}"
        text_pos = user.find(text_marker, start)
        if text_pos < 0:
            raise ValueError(f"candidate text placeholder not found in prompt: {self.path}")
        line_end = user.find("\n", text_pos + len(text_marker))
        if line_end < 0:
            line_end = text_pos + len(text_marker)
        block = user[start:line_end]
        rendered = "\n\n".join(self._render_chunk(chunk, idx) for idx, chunk in enumerate(chunks, start=1))
        output = user[:start] + rendered + user[line_end:]
        output = output.replace("\n\n[... 其他段落候选者以此类推 ...]", "")
        output = output.replace("\n[... 其他段落候选者以此类推 ...]", "")
        return output

    def _render_chunk(self, chunk: CandidateChunk, rank: int) -> str:
        title = chunk.title or ""
        retriever_rank = chunk.rank_rank or chunk.recall_rank or rank
        retriever_score = chunk.rank_score
        if retriever_score is None:
            retriever_score = chunk.recall_score
        if retriever_score is None:
            retriever_score = ""
        text = (chunk.text or "")[: self.max_chunk_chars]
        return "\n".join(
            [
                f"[id: {chunk.doc_id}]",
                f"title: {title}",
                f"retriever_rank_for_tie_break_only: {retriever_rank}",
                f"retriever_score_for_tie_break_only: {retriever_score}",
                "snippet:",
                text,
            ]
        )

    @staticmethod
    def _parse(text: str) -> tuple[str, str]:
        system_marker = "## system:"
        user_marker = "## user:"
        system_pos = text.find(system_marker)
        user_pos = text.find(user_marker)
        if system_pos < 0 or user_pos < 0 or user_pos <= system_pos:
            raise ValueError("prompt must contain '## system:' before '## user:'")
        system = text[system_pos + len(system_marker):user_pos]
        user = text[user_pos + len(user_marker):]
        return system, user


def stage_prompt_config(stage_config: dict[str, Any]) -> dict[str, Any]:
    prompt = dict(stage_config.get("prompt") or {})
    return {
        "path": prompt.get("path", "CoAgenticRetriever/async_labeling/prompts/llm_judge_rank50_v1.md"),
        "version": prompt.get("version", "llm_judge_rank50_v1"),
        "max_chunk_chars": prompt.get("max_chunk_chars", 512),
        "output_mode": prompt.get("output_mode", "no_think"),
    }
