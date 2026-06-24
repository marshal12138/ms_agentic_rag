"""Factory helpers for async ranker strategy adapters."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from async_labeling.config import SampleBuilderConfig, TrajectorySelectorConfig
from ranker_strategies.config import get_nested, to_plain_dict
from ranker_strategies.trajectory_selector import TopF1TrajectorySelector

from .sample_builder import RandomNegativeRepeatSampleBuilder
from .signal_builder import LLMJudgeTopKSignalBuilder
from .trajectory_selector import BestAndWorstTrajectorySelector


def _to_plain_config(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    return to_plain_dict(config)


def build_async_trajectory_selector(
    async_config: TrajectorySelectorConfig | None = None,
    trainer_config: Any = None,
):
    cfg = {}
    if trainer_config is not None:
        cfg = to_plain_dict(get_nested(trainer_config, "ranker_training.trajectory_selector", {}))
    if async_config is not None:
        override_cfg = {k: v for k, v in _to_plain_config(async_config).items() if v is not None}
        if override_cfg:
            cfg = {**cfg, **override_cfg}

    trajectory_selector_type = cfg.get("type", "best_and_worst_f1")
    if trajectory_selector_type == "top_f1_trajectories":
        return TopF1TrajectorySelector(
            max_selected_trajectories=cfg.get("max_selected_trajectories", 1),
            min_final_reward=cfg.get("min_final_reward", 0.0),
        )
    if trajectory_selector_type == "best_and_worst_f1":
        return BestAndWorstTrajectorySelector(
            top_k=cfg.get("top_k", 1),
            bottom_n=cfg.get("bottom_n", 2),
            min_final_reward=cfg.get("min_final_reward", 0.0),
        )
    raise ValueError(f"unsupported async trajectory_selector.type: {trajectory_selector_type!r}")


def build_async_signal_builder(config: SampleBuilderConfig):
    strategy_kwargs = dict(config.strategy_kwargs or {})
    signal_type = strategy_kwargs.get("signal_builder_type", "llm_judge_topk")
    if signal_type in {"llm_judge_topk", "llm_judge_top1"}:
        default_top_k = 1 if signal_type == "llm_judge_top1" else 5
        positive_top_k = int(strategy_kwargs.get("positive_top_k", default_top_k))
        return LLMJudgeTopKSignalBuilder(
            positive_top_k=positive_top_k,
            label_source=strategy_kwargs.get("label_source", f"llm_judge_top{positive_top_k}"),
        )
    raise ValueError(f"unsupported async signal_builder_type: {signal_type}")


def build_async_sample_builder(async_config: SampleBuilderConfig, trainer_config=None):
    sample_type = async_config.type
    if sample_type == "random_negative_repeat_from_signal":
        sample_type = "random_negative_repeat"
    if sample_type != "random_negative_repeat":
        raise ValueError(f"unsupported async sample_builder type: {async_config.type}")

    num_groups_per_step = async_config.num_groups_per_step
    neg_per_pos = async_config.neg_per_pos
    allow_repeat_negative_sampling = async_config.allow_repeat_negative_sampling
    seed = async_config.seed

    if trainer_config is not None:
        num_groups_per_step = get_nested(
            trainer_config,
            "ranker_training.sample_builder.num_groups_per_step",
            num_groups_per_step,
        )
        neg_per_pos = get_nested(
            trainer_config,
            "ranker_training.sample_builder.neg_per_pos",
            neg_per_pos,
        )
        allow_repeat_negative_sampling = get_nested(
            trainer_config,
            "ranker_training.sample_builder.allow_repeat_negative_sampling",
            allow_repeat_negative_sampling,
        )
        seed = get_nested(trainer_config, "ranker_training.seed", seed)

    return RandomNegativeRepeatSampleBuilder(
        num_groups_per_step=num_groups_per_step,
        neg_per_pos=neg_per_pos,
        allow_repeat_negative_sampling=allow_repeat_negative_sampling,
        seed=seed,
    )
