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


def require_nested(config: Any, path: str):
    sentinel = object()
    value = get_nested(config, path, sentinel)
    if value is sentinel or value is None or value == "":
        raise KeyError(f"missing required ranker config: {path}")
    return value


def require_present_nested(config: Any, path: str):
    sentinel = object()
    value = get_nested(config, path, sentinel)
    if value is sentinel:
        raise KeyError(f"missing required ranker config: {path}")
    return value


def build_selector(config: Any) -> TopF1TrajectorySelector | BestAndWorstTrajectorySelector:
    cfg = to_plain_dict(
        require_nested(config, "ranker_training.trajectory_selector")
    )
    selector_type = cfg["type"]
    if selector_type == "top_f1_trajectories":
        return TopF1TrajectorySelector(
            max_selected_trajectories=cfg["max_selected_trajectories"],
            min_final_reward=cfg["min_final_reward"],
        )
    if selector_type == "best_and_worst_f1":
        return BestAndWorstTrajectorySelector(
            top_k=cfg["top_k"],
            bottom_n=cfg["bottom_n"],
            min_final_reward=cfg["min_final_reward"],
        )
    raise ValueError(f"Unsupported trajectory_selector.type={selector_type!r}")


def build_signal_builder(config: Any) -> TopKPseudoRankSignalBuilder:
    cfg = to_plain_dict(
        require_nested(config, "ranker_training.signal_builder")
    )
    signal_type = cfg["type"]
    if signal_type != "topk_pseudo_rank":
        raise ValueError(f"Unsupported signal_builder.type={signal_type!r}")
    return TopKPseudoRankSignalBuilder(
        positive_top_k=cfg["positive_top_k"],
        allow_all_negative=cfg["allow_all_negative"],
    )


def build_sample_builder(config: Any) -> RandomNegativeRepeatSampleBuilder:
    cfg = to_plain_dict(
        require_nested(config, "ranker_training.sample_builder")
    )
    sample_type = cfg["type"]
    if sample_type != "random_negative_repeat":
        raise ValueError(f"Unsupported sample_builder.type={sample_type!r}")
    return RandomNegativeRepeatSampleBuilder(
        num_groups_per_step=cfg["num_groups_per_step"],
        neg_per_pos=cfg["neg_per_pos"],
        allow_repeat_negative_sampling=cfg["allow_repeat_negative_sampling"],
        seed=require_nested(config, "ranker_training.seed"),
    )


def build_replay_buffer(config: Any) -> RankerContrastiveReplayBuffer:
    cfg = to_plain_dict(
        require_nested(config, "ranker_training.replay_buffer")
    )
    return RankerContrastiveReplayBuffer(
        max_size=cfg["max_size"],
        seed=require_nested(config, "ranker_training.seed"),
    )


def build_collator(config: Any, tokenizer) -> RankerContrastiveCollator:
    model_path = str(require_nested(config, "ranker.model_path"))
    tokenizer_name = str(getattr(tokenizer, "name_or_path", "") or "")
    return RankerContrastiveCollator(
        tokenizer=tokenizer,
        max_query_length=require_nested(config, "ranker.max_query_length"),
        max_doc_length=require_nested(config, "ranker.max_doc_length"),
        use_e5_prefix=("e5" in model_path.lower() or "e5" in tokenizer_name.lower()),
    )


def build_construction_logger(config: Any) -> ContrastiveConstructionLogger:
    return ContrastiveConstructionLogger(
        every_n_steps=require_nested(config, "ranker_training.log_every_n_steps"),
        force_first=require_nested(config, "ranker_training.log_first_sample"),
        jsonl_path=require_present_nested(config, "ranker_training.construction_log_jsonl"),
    )
