"""Ranker contrastive training step orchestration.

This module is deliberately independent from PPO/GRPO internals. It consumes
fresh rollout trajectories and calls a ranker worker method that implements
InfoNCE optimization.
"""

from __future__ import annotations

import time
from typing import Any

from verl.utils.metric import reduce_metrics


def _cfg_get(config: Any, path: str, default=None):
    cur = config
    for part in path.split("."):
        if hasattr(cur, "get"):
            cur = cur.get(part, default)
        elif isinstance(cur, dict):
            cur = cur.get(part, default)
        else:
            return default
        if cur is default:
            return default
    return cur


def _cfg_get_first(config: Any, paths: list[str], default=None):
    for path in paths:
        value = _cfg_get(config, path, default)
        if value is not default:
            return value
    return default


def _call_update(ranker_wg, batch):
    output = ranker_wg.update_ranker_contrastive(batch)
    metrics = output.meta_info.get("metrics", {}) if output is not None else {}
    return reduce_metrics(metrics) if isinstance(metrics, dict) else metrics


def process_ranker_contrastive_step(
    fresh_trajectories,
    ranker_wg,
    replay_buffer,
    selector,
    signal_builder,
    sample_builder,
    collator,
    config,
    global_steps: int,
    ranker_step_idx: int,
    construction_logger=None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    timing: dict[str, float] = {}
    start_total = time.perf_counter()

    t0 = time.perf_counter()
    selected_contexts = selector.select(fresh_trajectories or [])
    timing["timing/ranker_select"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    labeled_docs = signal_builder.build(selected_contexts)
    timing["timing/ranker_signal"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    contrastive_samples = sample_builder.build(labeled_docs)
    timing["timing/ranker_sample_build"] = time.perf_counter() - t0

    added_samples = 0
    if ranker_step_idx == 0:
        added_samples = replay_buffer.add(contrastive_samples, source_step=global_steps)

    batch_size = int(_cfg_get(config, "ranker_training.batch_size", 32))
    fresh_ratio = float(
        _cfg_get(config, "ranker_training.replay_buffer.fresh_ratio", 0.5)
    )
    train_samples = replay_buffer.sample(batch_size=batch_size, fresh_ratio=fresh_ratio)

    metrics.update(
        {
            "ranker/selected_contexts": len(selected_contexts),
            "ranker/labeled_contexts": len(labeled_docs),
            "ranker/fresh_samples": len(contrastive_samples),
            "ranker/added_samples": added_samples,
            "ranker/buffer_size": len(replay_buffer),
            "ranker/train_samples": len(train_samples),
            "ranker/step_idx_in_global": ranker_step_idx,
        }
    )

    if construction_logger is not None:
        construction_logger.log(
            samples=contrastive_samples or train_samples,
            global_steps=global_steps,
            ranker_step_idx=ranker_step_idx,
            metrics=metrics,
        )

    if not train_samples:
        metrics["ranker/skipped"] = 1
        metrics["ranker/skip_no_contrastive_samples"] = 1
        metrics["timing/ranker_contrastive_total"] = time.perf_counter() - start_total
        metrics.update(timing)
        return metrics

    t0 = time.perf_counter()
    batch = collator(train_samples)
    timing["timing/ranker_collate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    worker_metrics = _call_update(ranker_wg, batch)
    timing["timing/ranker_update"] = time.perf_counter() - t0
    metrics.update(worker_metrics)
    metrics["ranker/skipped"] = 0
    metrics["ranker/update_step"] = int(_cfg_get(config, "ranker_training.update_step", 0))
    metrics["timing/ranker_contrastive_total"] = time.perf_counter() - start_total
    metrics.update(timing)
    return metrics
