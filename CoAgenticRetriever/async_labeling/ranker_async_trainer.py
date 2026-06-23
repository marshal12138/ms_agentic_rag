"""Background ranker trainer consuming async candidate signals."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from async_ranker_strategies.config import build_async_sample_builder, build_async_signal_builder
from ranker_strategies.config import require_nested

from .config import AsyncLabelingConfig
from .labeler import AsyncLabeler


logger = logging.getLogger(__name__)


class RankerAsyncTrainer:
    def __init__(
        self,
        *,
        config: Any,
        async_config: AsyncLabelingConfig,
        async_labeler: AsyncLabeler,
        ranker_wg,
        replay_buffer,
        collator,
        construction_logger=None,
        ranker_lock: threading.RLock | None = None,
        after_update_callback=None,
    ):
        self.config = config
        self.async_config = async_config
        self.async_labeler = async_labeler
        self.ranker_wg = ranker_wg
        self.replay_buffer = replay_buffer
        self.collator = collator
        self.construction_logger = construction_logger
        self.after_update_callback = after_update_callback
        self.signal_builder = build_async_signal_builder(async_config.sample_builder)
        self.sample_builder = build_async_sample_builder(async_config.sample_builder, trainer_config=config)
        self.sample_builder_request_batch = max(1, int(async_config.sample_builder_request_batch))
        self.ranker_lock = ranker_lock or threading.RLock()
        self._closed = threading.Event()
        self._thread: threading.Thread | None = None
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            "ranker/async_updates": 0,
            "ranker/async_skipped": 0,
            "ranker/async_consumed_signals": 0,
            "ranker/async_built_samples": 0,
        }
        self._latest_metrics: dict[str, Any] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="ranker-async-trainer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._closed.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def save_checkpoint(self, path: str) -> None:
        with self.ranker_lock:
            self.ranker_wg.save_checkpoint(path)

    def get_metrics(self) -> dict[str, Any]:
        with self._metrics_lock:
            out = dict(self._metrics)
            latest = dict(self._latest_metrics)
            for key in ("ranker/async_updates", "ranker/async_skipped", "ranker/async_consumed_signals", "ranker/async_built_samples"):
                latest.pop(key, None)
            out.update(latest)
        out.update({f"async_labeler/{k}": v for k, v in self.async_labeler.get_metrics().items()})
        return out

    def _loop(self) -> None:
        while not self._closed.is_set():
            try:
                updated = self.try_train_once(wait=True, timeout=1.0)
            except Exception:
                logger.exception("ranker async trainer loop failed")
                self._inc("ranker/async_errors", 1)
                time.sleep(1.0)
                continue
            if not updated:
                self._inc("ranker/async_wait_empty", 1)

    def try_train_once(self, *, wait: bool = False, timeout: float | None = 0.0) -> bool:
        signals = self.async_labeler.completed.pop_latest(
            n=self.sample_builder_request_batch,
            wait=wait,
            timeout=timeout,
        )
        if not signals:
            return False
        self._train_once(signals)
        return True

    def _train_once(self, signals) -> None:
        start = time.perf_counter()
        metrics = {
            "ranker/async_sample_builder_request_batch": self.sample_builder_request_batch,
            "ranker/async_consumed_signals": len(signals),
        }
        self._merge_latest(metrics)
        labeled_contexts = self.signal_builder.build(signals)
        metrics["ranker/async_labeled_contexts"] = len(labeled_contexts)
        self._merge_latest(metrics)
        samples = self.sample_builder.build(labeled_contexts)
        metrics["ranker/async_built_samples"] = len(samples)
        self._merge_latest(metrics)
        added = self.replay_buffer.add(samples, source_step=max(signal.created_global_step for signal in signals))
        batch_size = int(require_nested(self.config, "ranker_training.batch_size"))
        fresh_ratio = float(require_nested(self.config, "ranker_training.replay_buffer.fresh_ratio"))
        train_samples = self.replay_buffer.sample(batch_size=batch_size, fresh_ratio=fresh_ratio)
        metrics["ranker/async_added_samples"] = added
        metrics["ranker/async_train_samples"] = len(train_samples)
        metrics["ranker/async_buffer_size"] = len(self.replay_buffer)
        self._merge_latest(metrics)
        if self.construction_logger is not None:
            self.construction_logger.log(
                samples=samples or train_samples,
                global_steps=max(signal.created_global_step for signal in signals),
                ranker_step_idx=0,
                metrics=metrics,
            )
        if not train_samples:
            metrics["ranker/async_skipped"] = 1
            self._merge_metrics(metrics)
            return
        batch = self.collator(train_samples)
        metrics["ranker/async_collated_batches"] = 1
        self._merge_latest(metrics)
        with self.ranker_lock:
            output = self.ranker_wg.update_ranker_contrastive(batch)
        worker_metrics = output.meta_info.get("metrics", {}) if output is not None else {}
        metrics.update(worker_metrics)
        if self.after_update_callback is not None:
            callback_metrics = self.after_update_callback() or {}
            metrics.update(callback_metrics)
        metrics["ranker/async_updates"] = 1
        metrics["timing/ranker_async_update_total"] = time.perf_counter() - start
        self._merge_metrics(metrics)

    def _inc(self, key: str, value: int) -> None:
        with self._metrics_lock:
            self._metrics[key] = int(self._metrics.get(key, 0)) + value

    def _merge_metrics(self, metrics: dict[str, Any]) -> None:
        with self._metrics_lock:
            for key in ("ranker/async_updates", "ranker/async_skipped", "ranker/async_consumed_signals", "ranker/async_built_samples"):
                if key in metrics:
                    self._metrics[key] = int(self._metrics.get(key, 0)) + int(metrics[key])
            self._latest_metrics = metrics

    def _merge_latest(self, metrics: dict[str, Any]) -> None:
        with self._metrics_lock:
            self._latest_metrics = dict(metrics)
