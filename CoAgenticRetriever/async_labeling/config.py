"""Configuration helpers for async labeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - omegaconf is available in training env
    OmegaConf = None


@dataclass(slots=True)
class PromptConfig:
    path: str
    version: str
    format: str
    max_chunk_chars: int
    include_title: bool
    include_scores: bool
    shuffle_passages: bool
    output_mode: str


@dataclass(slots=True)
class LLMJudgeStageConfig:
    type: str
    endpoint: str
    model: str
    score_schema: str
    max_docs_per_request: int
    temperature: float
    max_tokens: int
    request_timeout_seconds: int
    max_retries: int
    prompt: PromptConfig


@dataclass(slots=True)
class SampleBuilderConfig:
    type: str
    num_groups_per_step: int
    neg_per_pos: int
    allow_repeat_negative_sampling: bool
    seed: int
    strategy_kwargs: dict[str, Any]


@dataclass(slots=True)
class TrajectorySelectorConfig:
    type: str
    max_selected_trajectories: int | None
    min_final_reward: float | None
    top_k: int | None
    bottom_n: int | None


@dataclass(slots=True)
class AsyncLabelingConfig:
    enable: bool
    max_sub_query: int
    max_glb_step_lag: int
    request_queue_size: int
    completed_buffer_size: int
    sample_builder_request_batch: int
    background_ranker_thread: bool
    ranker_updates_per_global_step: int
    drop_policy: str
    num_workers: int
    request_timeout_seconds: int
    max_retries: int
    sub_query_selection_policy: str
    selection_seed: int
    label_policy: str
    label_source: str
    score_version: str
    stages: list[dict[str, Any]]
    trajectory_selector: TrajectorySelectorConfig
    sample_builder: SampleBuilderConfig
    logging: dict[str, Any]


def _to_plain(config: Any) -> dict[str, Any]:
    if OmegaConf is not None:
        return OmegaConf.to_container(config, resolve=True) or {}
    if isinstance(config, dict):
        return config
    raise TypeError("config must be a dict or OmegaConf object")


def _require_mapping(config: dict[str, Any], path: str) -> dict[str, Any]:
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"missing required async labeling config: {path}")
        cur = cur[part]
    if not isinstance(cur, dict):
        raise TypeError(f"async labeling config must be a mapping: {path}")
    return dict(cur)


def _require_value(config: dict[str, Any], path: str):
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"missing required async labeling config: {path}")
        cur = cur[part]
    if cur is None or cur == "":
        raise KeyError(f"missing required async labeling config: {path}")
    return cur


def _require_present(config: dict[str, Any], path: str):
    cur: Any = config
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"missing required async labeling config: {path}")
        cur = cur[part]
    return cur


def _require_bool(config: dict[str, Any], path: str) -> bool:
    value = _require_value(config, path)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"async labeling config must be a bool: {path}")


def load_async_labeling_config(config: Any) -> AsyncLabelingConfig:
    plain = _to_plain(config)
    if "ranker_training" not in plain or "async_labeling" not in dict(plain["ranker_training"]):
        raise KeyError("missing required async labeling config: ranker_training.async_labeling")
    root = plain["ranker_training"]["async_labeling"]
    if root is None:
        raise KeyError("missing required async labeling config: ranker_training.async_labeling")
    root = dict(root)
    trajectory_selector_root = _require_mapping(root, "trajectory_selector")
    sample_builder_root = _require_mapping(root, "sample_builder")
    stages = _require_value(root, "stages")
    logging_cfg = _require_mapping(root, "logging")

    trajectory_selector = TrajectorySelectorConfig(
        type=str(_require_value(trajectory_selector_root, "type")),
        max_selected_trajectories=_require_present(trajectory_selector_root, "max_selected_trajectories")
        if "max_selected_trajectories" in trajectory_selector_root
        else None,
        min_final_reward=_require_present(trajectory_selector_root, "min_final_reward")
        if "min_final_reward" in trajectory_selector_root
        else None,
        top_k=_require_present(trajectory_selector_root, "top_k") if "top_k" in trajectory_selector_root else None,
        bottom_n=_require_present(trajectory_selector_root, "bottom_n")
        if "bottom_n" in trajectory_selector_root
        else None,
    )
    sample_builder = SampleBuilderConfig(
        type=str(_require_value(sample_builder_root, "type")),
        num_groups_per_step=int(_require_value(sample_builder_root, "num_groups_per_step")),
        neg_per_pos=int(_require_value(sample_builder_root, "neg_per_pos")),
        allow_repeat_negative_sampling=_require_bool(sample_builder_root, "allow_repeat_negative_sampling"),
        seed=int(_require_value(sample_builder_root, "seed")),
        strategy_kwargs=dict(_require_mapping(sample_builder_root, "strategy_kwargs")),
    )
    return AsyncLabelingConfig(
        enable=_require_bool(root, "enable"),
        max_sub_query=int(_require_value(root, "max_sub_query")),
        max_glb_step_lag=int(_require_value(root, "max_glb_step_lag")),
        request_queue_size=int(_require_value(root, "request_queue_size")),
        completed_buffer_size=int(_require_value(root, "completed_buffer_size")),
        sample_builder_request_batch=int(_require_value(root, "sample_builder_request_batch")),
        background_ranker_thread=_require_bool(root, "background_ranker_thread"),
        ranker_updates_per_global_step=int(_require_value(root, "ranker_updates_per_global_step")),
        drop_policy=str(_require_value(root, "drop_policy")),
        num_workers=int(_require_value(root, "num_workers")),
        request_timeout_seconds=int(_require_value(root, "request_timeout_seconds")),
        max_retries=int(_require_value(root, "max_retries")),
        sub_query_selection_policy=str(_require_value(root, "sub_query_selection_policy")),
        selection_seed=int(_require_value(root, "selection_seed")),
        label_policy=str(_require_value(root, "label_policy")),
        label_source=str(_require_value(root, "label_source")),
        score_version=str(_require_value(root, "score_version")),
        stages=list(stages),
        trajectory_selector=trajectory_selector,
        sample_builder=sample_builder,
        logging=logging_cfg,
    )


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
