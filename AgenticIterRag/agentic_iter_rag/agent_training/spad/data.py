"""Dataset helpers for SPAD-RAG smoke and dataset preparation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):
        return value.tolist()
    return [value]


def load_rl_rows(paths: list[str], *, max_samples: int = -1) -> list[dict[str, Any]]:
    """Load standard VERL RL parquet rows with pandas."""

    import pandas as pd

    rows: list[dict[str, Any]] = []
    for path in paths:
        frame = pd.read_parquet(Path(path))
        rows.extend(frame.to_dict("records"))
        if max_samples >= 0 and len(rows) >= max_samples:
            return rows[:max_samples]
    return rows


def row_question(row: dict[str, Any]) -> str:
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("question"):
        return str(extra["question"])
    prompt = _to_list(row.get("prompt"))
    if prompt and isinstance(prompt[0], dict):
        content = str(prompt[0].get("content") or "")
        marker = "Question:"
        if marker in content:
            return content.rsplit(marker, 1)[-1].strip()
    return ""


def row_gold_answers(row: dict[str, Any]) -> list[str]:
    reward_model = row.get("reward_model") if isinstance(row.get("reward_model"), dict) else {}
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model.get("ground_truth"), dict) else {}
    target = ground_truth.get("target")
    return [str(item) for item in _to_list(target)]


def row_prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = _to_list(row.get("prompt"))
    out: list[dict[str, str]] = []
    for item in prompt:
        if isinstance(item, dict):
            out.append({"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")})
    return out
