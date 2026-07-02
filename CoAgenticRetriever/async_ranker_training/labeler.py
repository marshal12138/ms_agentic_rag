"""Async ranker training labeler facade and worker runtime."""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Any

from .buffer import CompletedSignalBuffer
from .config import AsyncRankerTrainingConfig
from .logging import AsyncRankerTrainingLogger
from .schemas import AsyncLabelFailure, AsyncLabelRequest, CandidateSignalData
from .stages.llm_judge_rank50 import LLMJudgeRank50Stage


class AsyncLabeler:
    def __init__(self, config: AsyncRankerTrainingConfig, *, project_root: str | None = None, log_dir: str | None = None):
        self.config = config
        self.requests: Queue[AsyncLabelRequest] = Queue(maxsize=config.request_queue_size)
        self.completed = CompletedSignalBuffer(config.completed_buffer_size, config.drop_policy)
        self.submitted_count = 0
        self.dropped_count = 0
        self.completed_count = 0
        self.failed_count = 0
        self.expired_count = 0
        self.project_root = project_root
        self.log_dir = log_dir
        self.logger = AsyncRankerTrainingLogger(log_dir) if log_dir else None
        stage_cfg = next((dict(stage) for stage in config.stages if stage.get("type") == "llm_as_judge"), None)
        if stage_cfg is None:
            raise KeyError("missing required async ranker training config: stages[type=llm_as_judge]")
        self.stage = LLMJudgeRank50Stage(stage_cfg, project_root=project_root)
        self.current_global_step = 0
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []

    def submit(self, requests: list[AsyncLabelRequest]) -> int:
        accepted = 0
        for request in requests:
            try:
                self.requests.put_nowait(request)
            except Exception:
                with self._lock:
                    self.dropped_count += 1
                continue
            accepted += 1
            with self._lock:
                self.submitted_count += 1
            self._write("requests.jsonl", self._request_log(request))
        return accepted

    def start(self) -> None:
        if self._workers:
            return
        for idx in range(max(1, int(self.config.num_workers))):
            thread = threading.Thread(target=self._worker_loop, name=f"async-ranker-training-{idx}", daemon=True)
            thread.start()
            self._workers.append(thread)

    def update_global_step(self, global_step: int) -> None:
        with self._lock:
            self.current_global_step = max(self.current_global_step, int(global_step))

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            metrics = {
                "submitted_count": self.submitted_count,
                "dropped_count": self.dropped_count,
                "completed_count": self.completed_count,
                "failed_count": self.failed_count,
                "expired_count": self.expired_count,
                "request_queue_size": self.requests.qsize(),
                "completed_buffer_size": len(self.completed),
                "completed_buffer_dropped_count": self.completed.dropped_count,
            }
        self._write("metrics.jsonl", {"ts": time.time(), **metrics})
        return metrics

    def close(self) -> None:
        self._closed.set()
        for thread in self._workers:
            thread.join(timeout=2.0)

    def _worker_loop(self) -> None:
        while not self._closed.is_set():
            try:
                request = self.requests.get(timeout=0.5)
            except Exception:
                continue
            try:
                if self._is_expired(request):
                    self._record_failure(request, "expired_request", "request exceeded max_glb_step_lag before judge")
                    continue
                result = self.stage.score(request)
                if not result.ok:
                    self._record_failure(request, result.error_type or "judge_failed", result.error_message or "")
                    continue
                if self._is_expired(request):
                    self._record_failure(request, "expired_result", "request exceeded max_glb_step_lag before buffer write")
                    continue
                signal = CandidateSignalData(
                    signal_id=f"{request.request_id}:{int(time.time() * 1000)}",
                    created_global_step=request.created_global_step,
                    completed_at=time.time(),
                    origin_query=request.origin_query,
                    sub_query=request.sub_query,
                    trajectory_id=request.trajectory_id,
                    tool_call_id=request.tool_call_id,
                    ranked_chunk_list=request.ranked_chunk_list,
                    judge_scores=result.scores,
                    extra_scores=None,
                    final_scores=result.scores,
                    label_source=self.config.label_source,
                    score_version=self.config.score_version,
                    prompt_version=request.prompt_version or self.stage.prompt_version,
                    metadata={
                        **dict(request.trace_metadata or {}),
                        "turn_idx": request.turn_idx,
                        "trajectory_score": request.trajectory_score,
                        "score_type": request.score_type,
                        "label_policy": request.label_policy,
                        "usage": result.usage,
                        "latency_ms": result.latency_ms,
                    },
                )
                self.completed.push(signal)
                with self._lock:
                    self.completed_count += 1
                self._write("completed_signals.jsonl", signal.to_dict())
            finally:
                self.requests.task_done()

    def _is_expired(self, request: AsyncLabelRequest) -> bool:
        with self._lock:
            lag = self.current_global_step - int(request.created_global_step)
        return lag > int(self.config.max_glb_step_lag)

    def _record_failure(self, request: AsyncLabelRequest, error_type: str, message: str) -> None:
        if error_type.startswith("expired"):
            with self._lock:
                self.expired_count += 1
        else:
            with self._lock:
                self.failed_count += 1
        failure = AsyncLabelFailure(
            request_id=request.request_id,
            created_global_step=request.created_global_step,
            failed_at=time.time(),
            error_type=error_type,
            error_message=message,
            trajectory_id=request.trajectory_id,
            tool_call_id=request.tool_call_id,
        )
        self._write("failures.jsonl", {
            "request_id": failure.request_id,
            "created_global_step": failure.created_global_step,
            "failed_at": failure.failed_at,
            "error_type": failure.error_type,
            "error_message": failure.error_message,
            "trajectory_id": failure.trajectory_id,
            "tool_call_id": failure.tool_call_id,
        })

    def _write(self, name: str, record: dict[str, Any]) -> None:
        if self.logger is not None:
            self.logger.write(name, record)

    def _request_log(self, request: AsyncLabelRequest) -> dict[str, Any]:
        return {
            "request_id": request.request_id,
            "created_global_step": request.created_global_step,
            "origin_query": request.origin_query,
            "sub_query": request.sub_query,
            "trajectory_id": request.trajectory_id,
            "tool_call_id": request.tool_call_id,
            "chunk_count": len(request.ranked_chunk_list),
            "prompt_version": request.prompt_version,
        }
