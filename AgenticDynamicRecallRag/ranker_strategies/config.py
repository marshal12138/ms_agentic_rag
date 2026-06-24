"""Factory helpers for ranker contrastive strategies."""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from .collator import RankerContrastiveCollator
from .logging_utils import ContrastiveConstructionLogger
from .replay_buffer import RankerContrastiveReplayBuffer
from .sample_builder import RandomNegativeRepeatSampleBuilder
from .signal_builder import TopKPseudoRankSignalBuilder
from .trajectory_selector import BestAndWorstTrajectorySelector, TopF1TrajectorySelector


def to_plain_dict(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if OmegaConf.is_config(config):
        return OmegaConf.to_container(config, resolve=True) or {}
    if isinstance(config, dict):
        return config
    return {}


def get_nested(config: Any, path: str, default=None):
    cur = config
    for part in path.split("."):
        if OmegaConf.is_config(cur):
            cur = cur.get(part, default)
        elif isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
        if cur is default:
            return default
    return cur


def get_first_nested(config: Any, paths: list[str], default=None):
    for path in paths:
        value = get_nested(config, path, default)
        if value is not default:
            return value
    return default


def build_selector(config: Any) -> TopF1TrajectorySelector | BestAndWorstTrajectorySelector:
    cfg = to_plain_dict(
        get_nested(config, "ranker_training.trajectory_selector", {})
    )
    selector_type = cfg.get("type", "top_f1_trajectories")
    if selector_type == "top_f1_trajectories":
        return TopF1TrajectorySelector(
            max_selected_trajectories=cfg.get("max_selected_trajectories", 1),
            min_final_reward=cfg.get("min_final_reward", 0.0),
        )
    if selector_type == "best_and_worst_f1":
        return BestAndWorstTrajectorySelector(
            top_k=cfg.get("top_k", 1),
            bottom_n=cfg.get("bottom_n", 2),
            min_final_reward=cfg.get("min_final_reward", 0.0),
        )
    raise ValueError(f"Unsupported trajectory_selector.type={selector_type!r}")


def build_signal_builder(config: Any) -> TopKPseudoRankSignalBuilder:
    cfg = to_plain_dict(
        get_nested(config, "ranker_training.signal_builder", {})
    )
    signal_type = cfg.get("type", "topk_pseudo_rank")
    if signal_type != "topk_pseudo_rank":
        raise ValueError(f"Unsupported signal_builder.type={signal_type!r}")
    return TopKPseudoRankSignalBuilder(
        positive_top_k=cfg.get("positive_top_k", 5),
        allow_all_negative=cfg.get("allow_all_negative", False),
    )


def build_sample_builder(config: Any) -> RandomNegativeRepeatSampleBuilder:
    cfg = to_plain_dict(
        get_nested(config, "ranker_training.sample_builder", {})
    )
    sample_type = cfg.get("type", "random_negative_repeat")
    if sample_type != "random_negative_repeat":
        raise ValueError(f"Unsupported sample_builder.type={sample_type!r}")
    return RandomNegativeRepeatSampleBuilder(
        num_groups_per_step=cfg.get(
            "num_groups_per_step",
            get_nested(config, "ranker_training.batch_size", 32),
        ),
        neg_per_pos=cfg.get("neg_per_pos", 15),
        allow_repeat_negative_sampling=cfg.get("allow_repeat_negative_sampling", True),
        seed=get_nested(config, "ranker_training.seed", 42),
    )


def build_replay_buffer(config: Any) -> RankerContrastiveReplayBuffer:
    cfg = to_plain_dict(
        get_nested(config, "ranker_training.replay_buffer", {})
    )
    return RankerContrastiveReplayBuffer(
        max_size=cfg.get("max_size", 200000),
        seed=get_nested(config, "ranker_training.seed", 42),
    )


def build_collator(config: Any, tokenizer) -> RankerContrastiveCollator:
    return RankerContrastiveCollator(
        tokenizer=tokenizer,
        max_query_length=get_first_nested(config, ["ranker.max_query_length", "ranker_training.max_query_length"], 192),
        max_doc_length=get_first_nested(config, ["ranker.max_doc_length", "ranker_training.max_doc_length"], 256),
    )


def build_construction_logger(config: Any) -> ContrastiveConstructionLogger:
    return ContrastiveConstructionLogger(
        every_n_steps=get_first_nested(
            config,
            ["ranker_training.log_every_n_steps"],
            10,
        ),
        force_first=get_first_nested(
            config,
            ["ranker_training.log_first_sample"],
            True,
        ),
        jsonl_path=get_first_nested(
            config,
            ["ranker_training.construction_log_jsonl"],
            None,
        ),
    )
