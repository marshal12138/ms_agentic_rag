"""Factory helpers for async-labeling sample builders."""

from __future__ import annotations

from ..config import SampleBuilderConfig
from .random_negative_repeat_from_signal import RandomNegativeRepeatFromSignalBuilder


def build_sample_builder(config: SampleBuilderConfig):
    if config.type != "random_negative_repeat_from_signal":
        raise ValueError(f"unsupported async sample_builder type: {config.type}")
    return RandomNegativeRepeatFromSignalBuilder(
        num_groups_per_step=config.num_groups_per_step,
        neg_per_pos=config.neg_per_pos,
        allow_repeat_negative_sampling=config.allow_repeat_negative_sampling,
        seed=config.seed,
    )
