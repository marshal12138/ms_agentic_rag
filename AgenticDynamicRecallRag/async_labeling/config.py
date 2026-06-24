"""Configuration helpers for async labeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - omegaconf is available in training env
    OmegaConf = None


@dataclass(slots=True)
class PromptConfig:
    path: str = "CoAgenticRetriever/async_labeling/prompts/llm_judge_rank50_v1.md"
    version: str = "llm_judge_rank50_v1"
    format: str = "markdown_system_user_template"
    max_chunk_chars: int = 512
    include_title: bool = True
    include_scores: bool = True
    shuffle_passages: bool = False
    output_mode: str = "no_think"


@dataclass(slots=True)
class LLMJudgeStageConfig:
    type: str = "llm_as_judge"
    endpoint: str = "http://127.0.0.1:8067/v1/chat/completions"
    model: str = "DeepSeek-V4-Flash"
    score_schema: str = "ranked_ids_top50"
    max_docs_per_request: int = 50
    temperature: float = 0.0
    max_tokens: int = 1024
    request_timeout_seconds: int = 600
    max_retries: int = 2
    prompt: PromptConfig = field(default_factory=PromptConfig)


@dataclass(slots=True)
class SampleBuilderConfig:
    type: str = "random_negative_repeat_from_signal"
    num_groups_per_step: int = 32
    neg_per_pos: int = 15
    allow_repeat_negative_sampling: bool = True
    seed: int = 42
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrajectorySelectorConfig:
    type: str | None = "best_and_worst_f1"
    max_selected_trajectories: int | None = None
    min_final_reward: float | None = None
    top_k: int | None = None
    bottom_n: int | None = None


@dataclass(slots=True)
class AsyncLabelingConfig:
    enable: bool = False
    max_sub_query: int = 10
    max_glb_step_lag: int = 3
    request_queue_size: int = 2048
    completed_buffer_size: int = 4096
    sample_builder_request_batch: int = 1
    drop_policy: str = "drop_oldest"
    num_workers: int = 4
    request_timeout_seconds: int = 60
    max_retries: int = 2
    sub_query_selection_policy: str = "random"
    selection_seed: int = 42
    stages: list[dict[str, Any]] = field(default_factory=list)
    trajectory_selector: TrajectorySelectorConfig = field(default_factory=TrajectorySelectorConfig)
    sample_builder: SampleBuilderConfig = field(default_factory=SampleBuilderConfig)
    logging: dict[str, Any] = field(default_factory=dict)


def _to_plain(config: Any) -> dict[str, Any]:
    if OmegaConf is not None:
        return OmegaConf.to_container(config, resolve=True) or {}
    if isinstance(config, dict):
        return config
    raise TypeError("config must be a dict or OmegaConf object")


def _merge_dataclass(default: Any, values: dict[str, Any]) -> Any:
    kwargs = {}
    for field_name in getattr(default, "__dataclass_fields__", {}):
        if field_name in values:
            kwargs[field_name] = values[field_name]
        else:
            kwargs[field_name] = getattr(default, field_name)
    return type(default)(**kwargs)


def load_async_labeling_config(config: Any) -> AsyncLabelingConfig:
    plain = _to_plain(config)
    root = plain.get("ranker_training", {}).get("async_labeling", plain.get("async_labeling", plain))
    if root is None:
        root = {}
    root = dict(root)
    trajectory_selector = _merge_dataclass(
        TrajectorySelectorConfig(),
        dict(root.pop("trajectory_selector", {}) or {}),
    )
    sample_builder = _merge_dataclass(SampleBuilderConfig(), dict(root.pop("sample_builder", {}) or {}))
    cfg = _merge_dataclass(AsyncLabelingConfig(), root)
    cfg.trajectory_selector = trajectory_selector
    cfg.sample_builder = sample_builder
    return cfg


def validate_prompt_path(path: str, project_root: str | Path | None = None) -> Path:
    prompt_path = Path(path)
    candidates: list[Path]
    if prompt_path.is_absolute():
        candidates = [prompt_path]
    else:
        roots = [Path.cwd()]
        if project_root is not None:
            root = Path(project_root)
            roots = [root, root.parent, *roots]
        candidates = [root / prompt_path for root in roots]
    resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        tried = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"async labeling prompt file not found: {path}; tried: {tried}")
    text = resolved.read_text(encoding="utf-8")
    if "## system:" not in text or "## user:" not in text:
        raise ValueError(f"prompt file must contain '## system:' and '## user:' sections: {resolved}")
    return resolved
