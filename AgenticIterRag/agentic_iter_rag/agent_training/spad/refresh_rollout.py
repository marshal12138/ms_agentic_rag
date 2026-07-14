"""Stage 2 answer refresh data preparation for SPAD-RAG."""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.config import input_train_files
from agentic_iter_rag.agent_training.spad.data import load_rl_rows, row_gold_answers, row_prompt_messages, row_question
from agentic_iter_rag.agent_training.spad.manifest import write_records, write_sub_stage_manifest
from agentic_iter_rag.agent_training.spad.parsers import extract_last_answer
from agentic_iter_rag.agent_training.spad.prompts import (
    DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
    build_teacher_messages,
    resolve_teacher_prompt,
    smoke_actor_answer,
)
from agentic_iter_rag.agent_training.spad.reward import compute_f1
from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import _extract_teacher_result
from agentic_iter_rag.agent_training.spad.service_manager import (
    ManagedProcess,
    SpadServiceManager,
    as_int_list,
    csv_ids,
    post_json,
    project_root,
    quote,
    repo_root,
    tail_text,
    validate_replica_config,
    write_script,
)
from agentic_iter_rag.utils.io import write_json, write_jsonl


def _post_chat(endpoint: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def _call_chat_content(
    *,
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    request_cfg: dict[str, Any],
    timeout_s: float,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(request_cfg.get("temperature", 0.0)),
        "top_p": float(request_cfg.get("top_p", 1.0)),
        "max_tokens": int(request_cfg.get("max_tokens", 512)),
    }
    chat_template_kwargs = request_cfg.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    data = _post_chat(endpoint, payload, timeout_s)
    return str(data.get("choices", [{}])[0].get("message", {}).get("content", ""))


def _extract_tool_call(text: str) -> tuple[str | None, str | None]:
    matches = list(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S))
    if not matches:
        return None, "missing_tool_call"
    raw = matches[-1].group(1).strip()
    try:
        payload = json.loads(raw)
    except Exception:
        return None, "invalid_tool_call_json"
    if payload.get("name") != "search":
        return None, "invalid_tool_name"
    args = payload.get("arguments")
    if not isinstance(args, dict) or not str(args.get("query") or "").strip():
        return None, "invalid_tool_arguments"
    return str(args["query"]).strip(), None


def _normalize_docs(payload: dict[str, Any], *, top_m: int) -> list[dict[str, Any]]:
    result = payload.get("result") or []
    candidates = result[0] if result else []
    return _normalize_candidate_docs(candidates, top_m=top_m)


def _normalize_candidate_docs(candidates: Any, *, top_m: int) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for idx, item in enumerate(candidates[:top_m], start=1):
        raw_doc = item.get("document") if isinstance(item, dict) else {}
        raw_doc = raw_doc if isinstance(raw_doc, dict) else {}
        contents = raw_doc.get("contents") or raw_doc.get("text") or raw_doc.get("passage") or ""
        docs.append(
            {
                "id": str(raw_doc.get("id", "")),
                "title": str(raw_doc.get("title") or ""),
                "contents": str(contents),
                "score": float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0,
                "rank": idx,
            }
        )
    return docs


def _normalize_docs_at(payload: dict[str, Any], *, query_index: int, top_m: int) -> list[dict[str, Any]]:
    result = payload.get("result") or []
    candidates = result[query_index] if query_index < len(result) else []
    return _normalize_candidate_docs(candidates, top_m=top_m)


def _format_tool_response(docs: list[dict[str, Any]], *, max_doc_chars: int) -> str:
    if not docs:
        return "No documents found."
    lines: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        contents = str(doc.get("contents") or "")
        if len(contents) > max_doc_chars:
            contents = contents[:max_doc_chars] + "..."
        title = str(doc.get("title") or "")
        if title:
            lines.append(f"[{idx}] Title: {title}\n{contents}")
        else:
            lines.append(f"[{idx}] {contents}")
    return "\n".join(lines)


def _search(
    *,
    retrieval_url: str,
    query: str,
    top_n: int,
    top_m: int,
    max_doc_chars: int,
    timeout_s: float,
) -> tuple[str, list[dict[str, Any]], float]:
    started = time.perf_counter()
    payload = post_json(
        retrieval_url,
        {"queries": [query], "topk": top_n, "return_scores": True},
        timeout=timeout_s,
    )
    docs = _normalize_docs(payload, top_m=top_m)
    return _format_tool_response(docs, max_doc_chars=max_doc_chars), docs, time.perf_counter() - started


class JsonlAppendWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


class CounterBag:
    def __init__(self, initial: dict[str, Any]) -> None:
        self._values = dict(initial)
        self._lock = threading.Lock()

    def inc(self, key: str, amount: int = 1) -> None:
        with self._lock:
            self._values[key] = int(self._values.get(key, 0)) + amount

    def add_float(self, key: str, value: float) -> None:
        with self._lock:
            self._values[key] = float(self._values.get(key, 0.0)) + float(value)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)


class ReplicaLease:
    def __init__(self, pool: "ReplicaPool", index: int) -> None:
        self.pool = pool
        self.index = index
        self.replica = pool.replicas[index]

    def __enter__(self) -> dict[str, Any]:
        return self.replica

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.pool.release(self.index)


class ReplicaPool:
    def __init__(self, replicas: list[dict[str, Any]], *, max_inflight_per_replica: int) -> None:
        if not replicas:
            raise ValueError("replica pool must not be empty")
        self.replicas = replicas
        self._semaphore = threading.Semaphore(max(1, len(replicas) * max_inflight_per_replica))
        self._inflight = [0 for _ in replicas]
        self._lock = threading.Lock()

    def acquire(self) -> ReplicaLease:
        self._semaphore.acquire()
        with self._lock:
            index = min(range(len(self.replicas)), key=lambda item: self._inflight[item])
            self._inflight[index] += 1
        return ReplicaLease(self, index)

    def release(self, index: int) -> None:
        with self._lock:
            self._inflight[index] = max(0, self._inflight[index] - 1)
        self._semaphore.release()

    def inflight(self) -> list[int]:
        with self._lock:
            return list(self._inflight)


class RetrievalBatcher:
    """Batch concurrent search calls into multi-query retrieval requests."""

    def __init__(
        self,
        *,
        retrieval_url: str,
        top_n: int,
        top_m: int,
        max_doc_chars: int,
        timeout_s: float,
        batch_size: int,
        flush_interval_ms: int,
    ) -> None:
        self.retrieval_url = retrieval_url
        self.top_n = top_n
        self.top_m = top_m
        self.max_doc_chars = max_doc_chars
        self.timeout_s = timeout_s
        self.batch_size = max(1, int(batch_size))
        self.flush_interval_s = max(0.001, float(flush_interval_ms) / 1000.0)
        self._queue: queue.Queue[tuple[str | None, Future[tuple[str, list[dict[str, Any]], float]] | None]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="spad-retrieval-batcher", daemon=True)
        self._thread.start()

    def search(self, query_text: str) -> tuple[str, list[dict[str, Any]], float]:
        future: Future[tuple[str, list[dict[str, Any]], float]] = Future()
        self._queue.put((query_text, future))
        return future.result(timeout=self.timeout_s + 30)

    def close(self) -> None:
        self._queue.put((None, None))
        self._thread.join(timeout=10)

    def _run(self) -> None:
        pending: list[tuple[str, Future[tuple[str, list[dict[str, Any]], float]]]] = []
        while True:
            try:
                item = self._queue.get(timeout=self.flush_interval_s if pending else None)
            except queue.Empty:
                item = None
            if item is None:
                if pending:
                    self._flush(pending)
                    pending = []
                continue
            query_text, future = item
            if query_text is None or future is None:
                if pending:
                    self._flush(pending)
                return
            pending.append((query_text, future))
            if len(pending) >= self.batch_size:
                self._flush(pending)
                pending = []

    def _flush(self, pending: list[tuple[str, Future[tuple[str, list[dict[str, Any]], float]]]]) -> None:
        started = time.perf_counter()
        queries = [item[0] for item in pending]
        try:
            payload = post_json(
                self.retrieval_url,
                {"queries": queries, "topk": self.top_n, "return_scores": True},
                timeout=self.timeout_s,
            )
            elapsed = time.perf_counter() - started
            for query_index, (_, future) in enumerate(pending):
                docs = _normalize_docs_at(payload, query_index=query_index, top_m=self.top_m)
                future.set_result((_format_tool_response(docs, max_doc_chars=self.max_doc_chars), docs, elapsed))
        except Exception as exc:
            for _, future in pending:
                future.set_exception(exc)


def _existing_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    indices: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise ValueError(f"corrupted JSONL at {path}:{line_no}: {exc}") from exc
            if "index" in payload:
                indices.add(int(payload["index"]))
            else:
                metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
                if "index" in metadata:
                    indices.add(int(metadata["index"]))
    return indices


def _reset_outputs(paths: list[Path], *, resume_existing: bool) -> None:
    if resume_existing:
        return
    for path in paths:
        if path.exists():
            path.unlink()


def _parse_npu_smi() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["npu-smi", "info"],
            cwd=str(repo_root()),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    lines = result.stdout.splitlines()
    current_id: int | None = None
    for line in lines:
        match_id = re.match(r"\|\s*(\d+)\s+910", line)
        if match_id:
            current_id = int(match_id.group(1))
            continue
        if current_id is None:
            continue
        match_stats = re.match(r"\|\s*\d+\s+\|\s+[0-9A-Fa-f:.]+\s+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", line)
        if not match_stats:
            match_stats = re.match(r"\|\s*\d+\s+\S+\s+\|\s*(\d+)\s+\d+\s*/\s*\d+\s+(\d+)\s*/\s*(\d+)", line)
        if match_stats:
            rows.append(
                {
                    "id": current_id,
                    "aicore_util": int(match_stats.group(1)),
                    "hbm_mb": int(match_stats.group(2)),
                    "hbm_total_mb": int(match_stats.group(3)),
                }
            )
            current_id = None
    return rows


class Stage2ResourceMonitor:
    def __init__(
        self,
        *,
        jsonl_path: Path,
        report_path: Path,
        interval_s: float,
        get_phase: Any,
        get_progress: Any,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.report_path = report_path
        self.interval_s = max(5.0, float(interval_s))
        self.get_phase = get_phase
        self.get_progress = get_progress
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="spad-stage2-resource-monitor", daemon=True)
        self._samples: list[dict[str, Any]] = []

    def start(self) -> None:
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self._write_report()

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = {
                "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
                "phase": self.get_phase(),
                **self.get_progress(),
                "npu": _parse_npu_smi(),
            }
            self._samples.append(sample)
            with self.jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
            self._stop.wait(self.interval_s)

    def _write_report(self) -> None:
        lines = [
            "# SPAD Stage2 Resource Monitor",
            "",
            f"Generated at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            "",
            "| Time | Phase | Progress | NPU HBM/AICore |",
            "| --- | --- | --- | --- |",
        ]
        for sample in self._samples:
            progress = ", ".join(
                f"{key}={value}"
                for key, value in sample.items()
                if key not in {"ts", "phase", "npu"}
            )
            npu = "; ".join(
                f"NPU{item.get('id')} {item.get('hbm_mb')}MB/{item.get('aicore_util')}%"
                for item in sample.get("npu", [])
            )
            lines.append(f"| {sample['ts']} | {sample['phase']} | {progress} | {npu} |")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_dpo_pair(record: dict[str, Any]) -> tuple[bool, str]:
    messages = record.get("messages_before_final_answer")
    if not isinstance(messages, list) or not messages:
        return False, "missing_messages_before_final_answer"
    chosen = str(record.get("chosen") or "").strip()
    rejected = str(record.get("rejected") or "").strip()
    if not chosen:
        return False, "empty_chosen"
    if not rejected:
        return False, "empty_rejected"
    if "<status>" in chosen:
        return False, "chosen_has_status"
    if chosen == rejected:
        return False, "chosen_equals_rejected"
    return True, ""


def _messages_before_final_answer(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    if messages and messages[-1]["role"] == "assistant":
        return messages[:-1]
    return list(messages)


def _normalize_actor_rejected(text: str, answer: str) -> str:
    """Ensure rejected responses train the answer role with the SPAD XML shape."""

    raw = (text or "").strip()
    if "<reason>" in raw and "</reason>" in raw and "<answer>" in raw and "</answer>" in raw:
        return raw
    answer_text = (answer or extract_last_answer(raw) or raw).strip()
    if raw.startswith("<answer>") and raw.endswith("</answer>"):
        answer_text = raw[len("<answer>") : -len("</answer>")].strip()
    return (
        "<reason>Actor generated a final answer during refresh rollout.</reason>\n"
        f"<answer>{answer_text}</answer>"
    )


def _strip_teacher_status_block(text: str) -> str:
    """Remove teacher-only evidence status before writing answer-distill data."""

    cleaned = re.sub(r"\s*<status>.*?</status>\s*", "\n", text.strip(), flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _ensure_hf_actor_checkpoint(actor_checkpoint: str, checkpoint_dir: Path, log_dir: Path) -> str:
    """Return an HF checkpoint path, merging VERL FSDP actor shards when needed."""

    if not actor_checkpoint:
        raise ValueError("Stage 2 requires actor_checkpoint from Stage 1")
    checkpoint_path = Path(actor_checkpoint)
    if (checkpoint_path / "config.json").exists() and (checkpoint_path / "model.safetensors").exists():
        return str(checkpoint_path)

    actor_dir = checkpoint_path / "actor" if checkpoint_path.name.startswith("global_step_") else checkpoint_path
    if not (actor_dir / "fsdp_config.json").exists():
        raise ValueError(f"unsupported actor checkpoint layout for Stage 2 refresh: {actor_checkpoint}")

    target_dir = checkpoint_dir / "actor_model_hf" / checkpoint_path.name
    if (target_dir / "config.json").exists() and (target_dir / "model.safetensors").exists():
        return str(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(repo_root() / ".venvs" / "ms_agt_rag_overlay" / "bin" / "python"),
        str(project_root() / "verl" / "scripts" / "legacy_model_merger.py"),
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_dir),
        "--target_dir",
        str(target_dir),
    ]
    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = f"{project_root() / 'verl'}:{project_root()}:{env.get('PYTHONPATH', '')}"
    log_path = log_dir / "merge_actor_checkpoint.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, cwd=str(repo_root()), env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"failed to merge actor checkpoint; log={log_path}\n{tail_text(log_path)}")
    return str(target_dir)


def _run_smoke_backend(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    sub_cfg: dict[str, Any],
    stage_dir: Path,
    log_dir: Path,
    dataset_jsonl: Path,
    resource_plan: dict[str, Any],
    actor_checkpoint: str | None,
) -> dict[str, Any]:
    max_samples = int(sub_cfg.get("smoke", {}).get("max_samples", 8))
    rows = load_rl_rows(input_train_files(config, spad_cfg), max_samples=max_samples)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        question = row_question(row)
        gold_answers = row_gold_answers(row)
        teacher_answer = gold_answers[0] if gold_answers else spad_cfg["teacher_answerer"].get("insufficient_answer", "证据不足无法作答")
        chosen = f"<reason>Smoke teacher uses the gold answer as evidence-grounded placeholder.</reason>\n<answer>{teacher_answer}</answer>"
        rejected = smoke_actor_answer(question)
        prompt_messages = row_prompt_messages(row)
        records.append(
            {
                "prompt": prompt_messages,
                "chosen": chosen,
                "rejected": rejected,
                "messages_before_final_answer": prompt_messages,
                "metadata": {
                    "index": index,
                    "question": question,
                    "gold_answers": gold_answers,
                    "actor_checkpoint": actor_checkpoint,
                    "teacher_answer": teacher_answer,
                    "search_count": 1,
                    "format_status": "smoke",
                },
            }
        )
    write_records(dataset_jsonl, records)
    dataset_manifest = stage_dir / "answer_distill_dataset_manifest.json"
    outputs = {
        "status": "completed",
        "backend": "smoke",
        "actor_checkpoint": actor_checkpoint,
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_manifest": str(dataset_manifest),
        "sample_count": len(records),
        "runtime_dir": str(log_dir),
        "resource_plan": resource_plan,
    }
    write_sub_stage_manifest(dataset_manifest, sub_stage="answer_distill_dataset", outputs=outputs)
    return outputs


def _run_trajectory_rollout_phase(
    *,
    rows: list[dict[str, Any]],
    actor_outputs: list[dict[str, Any]],
    recall_output: dict[str, Any],
    trajectory_jsonl: Path,
    actor_checkpoint: str,
    hf_actor_checkpoint: str,
    rollout_cfg: dict[str, Any],
    request_cfg: dict[str, Any],
    top_n: int,
    top_m: int,
    max_doc_chars: int,
    search_timeout: float,
    max_assistant_turns: int,
    scheduler_cfg: dict[str, Any],
    resume_existing: bool,
    inc_progress: Any,
    set_progress: Any,
) -> dict[str, Any]:
    existing = _existing_indices(trajectory_jsonl) if resume_existing else set()
    writer = JsonlAppendWriter(trajectory_jsonl)
    counters = CounterBag(
        {
            "phase": "trajectory_rollout",
            "total": len(rows),
            "trajectory_completed": 0,
            "skipped_actor_no_finish": 0,
            "skipped_actor_missing_tool_call": 0,
            "skipped_actor_invalid_tool_call": 0,
            "skipped_no_search_evidence": 0,
            "actor_errors": 0,
            "search_errors": 0,
            "timeout_errors": 0,
            "written_trajectories": len(existing),
            "_sum_search_count": 0.0,
            "_sum_actor_elapsed_s": 0.0,
        }
    )
    inflight_per_actor = int(scheduler_cfg.get("inflight_per_actor", 16))
    max_inflight_per_actor = int(scheduler_cfg.get("max_inflight_per_actor", max(inflight_per_actor, 24)))
    submit_batch_size = int(scheduler_cfg.get("trajectory_submit_batch_size", len(actor_outputs) * inflight_per_actor))
    progress_log_interval = max(1, int(scheduler_cfg.get("progress_log_interval", 20)))
    actor_timeout = float(scheduler_cfg.get("request_timeout_s") or rollout_cfg.get("actor_timeout_seconds", 180))
    actor_pool = ReplicaPool(actor_outputs, max_inflight_per_replica=max_inflight_per_actor)
    retrieval = RetrievalBatcher(
        retrieval_url=str(recall_output["retrieval_url"]),
        top_n=top_n,
        top_m=top_m,
        max_doc_chars=max_doc_chars,
        timeout_s=search_timeout,
        batch_size=int(scheduler_cfg.get("retrieval_query_batch_size", 32)),
        flush_interval_ms=int(scheduler_cfg.get("retrieval_flush_interval_ms", 50)),
    )

    def process_one(index: int, row: dict[str, Any]) -> dict[str, Any]:
        question = row_question(row)
        gold_answers = row_gold_answers(row)
        messages = row_prompt_messages(row)
        evidence_steps: list[dict[str, Any]] = []
        actor_final = ""
        actor_answer = ""
        skip_reason = ""
        errors: list[str] = []
        actor_elapsed = 0.0
        status = "failed"
        try:
            with actor_pool.acquire() as actor_output:
                for turn in range(max_assistant_turns):
                    actor_started = time.perf_counter()
                    actor_text = _call_chat_content(
                        endpoint=str(actor_output["endpoint"]),
                        model=str(actor_output["model"]),
                        messages=messages,
                        request_cfg=request_cfg,
                        timeout_s=actor_timeout,
                    )
                    actor_elapsed += time.perf_counter() - actor_started
                    messages.append({"role": "assistant", "content": actor_text})
                    if "<answer>" in actor_text:
                        actor_final = actor_text
                        actor_answer = extract_last_answer(actor_text) or ""
                        break
                    query, error_code = _extract_tool_call(actor_text)
                    if error_code:
                        skip_reason = error_code
                        break
                    try:
                        tool_text, docs, search_elapsed = retrieval.search(str(query))
                    except Exception as exc:
                        counters.inc("search_errors")
                        if "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower():
                            counters.inc("timeout_errors")
                            skip_reason = "search_timeout"
                        else:
                            skip_reason = "search_error"
                        errors.append(f"{skip_reason}:{type(exc).__name__}:{exc}")
                        break
                    evidence_steps.append(
                        {
                            "turn": turn + 1,
                            "sub_query": query,
                            "docs": docs,
                            "search_elapsed_s": search_elapsed,
                        }
                    )
                    messages.append({"role": "tool", "content": tool_text})
                else:
                    skip_reason = "max_turns_exceeded"
        except Exception as exc:
            counters.inc("actor_errors")
            if "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower():
                counters.inc("timeout_errors")
                skip_reason = "actor_timeout"
            else:
                skip_reason = f"actor_error:{type(exc).__name__}"
            errors.append(f"{skip_reason}:{exc}")

        if actor_final and evidence_steps:
            status = "completed"
            counters.inc("trajectory_completed")
        else:
            status = "skipped"
            if not actor_final:
                counters.inc("skipped_actor_no_finish")
                skip_reason = skip_reason or "actor_no_finish"
            elif not evidence_steps:
                counters.inc("skipped_no_search_evidence")
                skip_reason = skip_reason or "no_search_evidence"
            if skip_reason == "missing_tool_call":
                counters.inc("skipped_actor_missing_tool_call")
            if skip_reason.startswith("invalid_tool"):
                counters.inc("skipped_actor_invalid_tool_call")

        messages_before = _messages_before_final_answer(messages)
        rejected = _normalize_actor_rejected(actor_final, actor_answer) if actor_final else ""
        record = {
            "index": index,
            "status": status,
            "skip_reason": skip_reason or None,
            "question": question,
            "gold_answers": gold_answers,
            "messages_before_final_answer": messages_before,
            "actor_final": actor_final,
            "actor_answer": actor_answer,
            "rejected": rejected,
            "evidence_steps": evidence_steps,
            "search_count": len(evidence_steps),
            "sub_queries": [item["sub_query"] for item in evidence_steps],
            "actor_elapsed_s": actor_elapsed,
            "actor_checkpoint": actor_checkpoint,
            "hf_actor_checkpoint": hf_actor_checkpoint,
            "errors": errors,
        }
        writer.write(record)
        counters.inc("written_trajectories")
        counters.add_float("_sum_search_count", float(len(evidence_steps)))
        counters.add_float("_sum_actor_elapsed_s", actor_elapsed)
        inc_progress("phase_a_written")
        written = counters.snapshot()["written_trajectories"]
        if int(written) % progress_log_interval == 0:
            print(
                "SPAD Stage2 PhaseA progress: "
                f"written={written}/{len(rows)} completed={counters.snapshot().get('trajectory_completed')} "
                f"actor_inflight={actor_pool.inflight()}",
                flush=True,
            )
        return record

    pending_rows = [(index, row) for index, row in enumerate(rows) if index not in existing]
    set_progress("phase_a_total", len(rows))
    set_progress("phase_a_written", len(existing))
    if pending_rows:
        max_workers = max(1, min(int(submit_batch_size), len(actor_outputs) * max_inflight_per_actor, len(pending_rows)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spad-stage2-actor") as executor:
            futures = [executor.submit(process_one, index, row) for index, row in pending_rows]
            for future in as_completed(futures):
                future.result()
    retrieval.close()

    snapshot = counters.snapshot()
    written = max(1, int(snapshot.get("written_trajectories", 0)))
    snapshot["avg_search_count"] = float(snapshot.pop("_sum_search_count", 0.0)) / written
    snapshot["avg_actor_elapsed_s"] = float(snapshot.pop("_sum_actor_elapsed_s", 0.0)) / written
    return snapshot


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"corrupted JSONL at {path}:{line_no}: {exc}") from exc
    return records


def _teacher_label_result_from_raw(
    *,
    trajectory: dict[str, Any],
    teacher_raw: str,
    teacher_elapsed: float,
    spad_cfg: dict[str, Any],
    filter_cfg: dict[str, Any],
    errors: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
    index = int(trajectory.get("index", -1))
    question = str(trajectory.get("question") or "")
    gold_answers = trajectory.get("gold_answers") or []
    evidence_steps = trajectory.get("evidence_steps") or []
    rejected = str(trajectory.get("rejected") or "")
    messages_before = trajectory.get("messages_before_final_answer") or []
    actor_answer = str(trajectory.get("actor_answer") or "")
    actor_f1 = compute_f1(actor_answer, gold_answers)
    insufficient = str(spad_cfg["teacher_answerer"].get("insufficient_answer", "证据不足无法作答"))
    teacher_prompt_version = str(
        spad_cfg["teacher_answerer"].get("prompt_version")
        or DEFAULT_TEACHER_STATUS_PROMPT_VERSION
    )
    min_teacher_f1 = float(filter_cfg.get("min_teacher_f1", 0.0))
    require_teacher_format = bool(filter_cfg.get("require_teacher_format_valid", True))
    require_evidence_sufficient = bool(filter_cfg.get("require_evidence_sufficient", True))

    teacher_answer = ""
    teacher_evidence_status = ""
    teacher_parse_status = ""
    teacher_format_error = False
    teacher_f1 = 0.0
    evidence_sufficient = False
    chosen = ""
    keep = True
    skip_reason = ""
    counter_updates: dict[str, Any] = {
        "teacher_completed": 1,
        "_sum_teacher_elapsed_s": float(teacher_elapsed),
    }
    errors = list(errors or [])
    if errors:
        keep = False
        skip_reason = errors[0].split(":", 1)[0]
    else:
        (
            teacher_answer,
            teacher_evidence_status,
            teacher_parse_status,
            teacher_format_error,
        ) = _extract_teacher_result(teacher_raw)
        teacher_f1 = compute_f1(teacher_answer, gold_answers)
        evidence_sufficient = bool(
            teacher_evidence_status == "supported_answer"
            and teacher_answer
            and teacher_answer != insufficient
            and not teacher_format_error
        )
        chosen = _strip_teacher_status_block(teacher_raw)

    if keep and require_teacher_format and teacher_parse_status != "parsed":
        counter_updates["skipped_teacher_format"] = 1
        keep = False
        skip_reason = f"teacher_{teacher_parse_status or 'unparsed'}"
    if keep and require_evidence_sufficient and not evidence_sufficient:
        counter_updates["skipped_evidence_insufficient"] = 1
        keep = False
        skip_reason = "evidence_insufficient"
    if keep and teacher_f1 < min_teacher_f1:
        counter_updates["skipped_teacher_f1"] = 1
        keep = False
        skip_reason = "teacher_f1_below_threshold"

    pair_record = {
        "index": index,
        "question": question,
        "gold_answers": gold_answers,
        "messages_before_final_answer": messages_before,
        "chosen": chosen,
        "rejected": rejected,
        "evidence_steps": evidence_steps,
        "teacher_reason": teacher_raw,
        "teacher_f1": teacher_f1,
        "teacher_prompt_version": teacher_prompt_version,
        "filter_status": "kept" if keep else "skipped",
    }
    valid, invalid_reason = _validate_dpo_pair(pair_record)
    if keep and not valid:
        keep = False
        skip_reason = invalid_reason
        if invalid_reason == "chosen_equals_rejected":
            counter_updates["skipped_chosen_equals_rejected"] = 1
        else:
            counter_updates["schema_invalid_pairs"] = 1

    metadata = {
        "index": index,
        "question": question,
        "gold_answers": gold_answers,
        "actor_answer": actor_answer,
        "actor_f1": actor_f1,
        "teacher_answer": teacher_answer,
        "teacher_f1": teacher_f1,
        "teacher_parse_status": teacher_parse_status,
        "teacher_evidence_status": teacher_evidence_status,
        "teacher_format_error": teacher_format_error,
        "teacher_prompt_version": teacher_prompt_version,
        "evidence_sufficient": evidence_sufficient,
        "search_count": int(trajectory.get("search_count") or len(evidence_steps)),
        "sub_queries": trajectory.get("sub_queries") or [],
        "actor_elapsed_s": trajectory.get("actor_elapsed_s", 0.0),
        "teacher_elapsed_s": teacher_elapsed,
        "keep": keep,
        "skip_reason": skip_reason,
        "errors": errors,
    }
    refresh_record = {
        "index": index,
        "status": "kept" if keep else "skipped",
        "skip_reason": None if keep else skip_reason,
        "prompt": messages_before,
        "chosen": chosen,
        "rejected": rejected,
        "messages_before_final_answer": messages_before,
        "evidence_steps": evidence_steps,
        "teacher_raw_content": teacher_raw,
        "metadata": metadata,
    }
    if keep:
        pair_record["filter_status"] = "kept"
        counter_updates["kept"] = 1
        return refresh_record, pair_record, counter_updates
    return refresh_record, None, counter_updates


def _run_teacher_labeling_phase(
    *,
    trajectory_jsonl: Path,
    refresh_jsonl: Path,
    dataset_jsonl: Path,
    teacher_outputs: list[dict[str, Any]],
    spad_cfg: dict[str, Any],
    filter_cfg: dict[str, Any],
    teacher_request: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    resume_existing: bool,
    inc_progress: Any,
    set_progress: Any,
) -> dict[str, Any]:
    if not trajectory_jsonl.exists():
        raise RuntimeError(f"Stage 2 Phase B requires Phase A output: {trajectory_jsonl}")
    trajectories = _iter_jsonl(trajectory_jsonl)
    existing_refresh = _existing_indices(refresh_jsonl) if resume_existing else set()
    existing_pairs = _existing_indices(dataset_jsonl) if resume_existing else set()
    refresh_writer = JsonlAppendWriter(refresh_jsonl)
    pair_writer = JsonlAppendWriter(dataset_jsonl)
    counters = CounterBag(
        {
            "phase": "teacher_labeling",
            "total_trajectories": len(trajectories),
            "eligible_for_teacher": 0,
            "teacher_completed": 0,
            "kept": len(existing_pairs),
            "skipped_no_actor_final": 0,
            "skipped_no_evidence": 0,
            "skipped_teacher_format": 0,
            "skipped_evidence_insufficient": 0,
            "skipped_teacher_f1": 0,
            "skipped_chosen_equals_rejected": 0,
            "teacher_errors": 0,
            "timeout_errors": 0,
            "schema_invalid_pairs": 0,
            "_sum_teacher_elapsed_s": 0.0,
        }
    )
    insufficient = str(spad_cfg["teacher_answerer"].get("insufficient_answer", "证据不足无法作答"))
    min_teacher_f1 = float(filter_cfg.get("min_teacher_f1", 0.0))
    require_teacher_format = bool(filter_cfg.get("require_teacher_format_valid", True))
    require_evidence_sufficient = bool(filter_cfg.get("require_evidence_sufficient", True))
    inflight_per_teacher = int(scheduler_cfg.get("inflight_per_teacher", 4))
    max_inflight_per_teacher = int(scheduler_cfg.get("max_inflight_per_teacher", max(inflight_per_teacher, 6)))
    submit_batch_size = int(scheduler_cfg.get("teacher_submit_batch_size", len(teacher_outputs) * inflight_per_teacher))
    progress_log_interval = max(1, int(scheduler_cfg.get("progress_log_interval", 20)))
    teacher_timeout = float(scheduler_cfg.get("request_timeout_s") or teacher_request.get("timeout_seconds", 180))
    teacher_prompt_version = str(
        teacher_request.get("prompt_version") or DEFAULT_TEACHER_STATUS_PROMPT_VERSION
    )
    teacher_pool = ReplicaPool(teacher_outputs, max_inflight_per_replica=max_inflight_per_teacher)

    def skipped_record(trajectory: dict[str, Any], reason: str) -> dict[str, Any]:
        index = int(trajectory.get("index", -1))
        record = {
            "index": index,
            "status": "skipped",
            "skip_reason": reason,
            "prompt": trajectory.get("messages_before_final_answer") or [],
            "chosen": "",
            "rejected": trajectory.get("rejected") or "",
            "messages_before_final_answer": trajectory.get("messages_before_final_answer") or [],
            "evidence_steps": trajectory.get("evidence_steps") or [],
            "teacher_raw_content": "",
            "metadata": {
                "index": index,
                "question": trajectory.get("question"),
                "gold_answers": trajectory.get("gold_answers") or [],
                "keep": False,
                "skip_reason": reason,
            },
        }
        refresh_writer.write(record)
        inc_progress("phase_b_seen")
        return record

    def process_one(trajectory: dict[str, Any]) -> dict[str, Any]:
        index = int(trajectory.get("index", -1))
        question = str(trajectory.get("question") or "")
        gold_answers = trajectory.get("gold_answers") or []
        evidence_steps = trajectory.get("evidence_steps") or []
        rejected = str(trajectory.get("rejected") or "")
        messages_before = trajectory.get("messages_before_final_answer") or []
        actor_answer = str(trajectory.get("actor_answer") or "")
        actor_f1 = compute_f1(actor_answer, gold_answers)
        counters.inc("eligible_for_teacher")
        teacher_answer = ""
        teacher_raw = ""
        teacher_parse_status = ""
        teacher_evidence_status = ""
        teacher_format_error = False
        teacher_elapsed = 0.0
        teacher_f1 = 0.0
        evidence_sufficient = False
        chosen = ""
        keep = True
        skip_reason = ""
        errors: list[str] = []
        try:
            with teacher_pool.acquire() as teacher_output:
                started = time.perf_counter()
                teacher_raw = _call_chat_content(
                    endpoint=str(teacher_output["endpoint"]),
                    model=str(teacher_output["model"]),
                    messages=build_teacher_messages(
                        question=question,
                        evidence_steps=evidence_steps,
                        include_status=True,
                        prompt_version=teacher_prompt_version,
                    ),
                    request_cfg=teacher_request,
                    timeout_s=teacher_timeout,
                )
                teacher_elapsed = time.perf_counter() - started
            (
                teacher_answer,
                teacher_evidence_status,
                teacher_parse_status,
                teacher_format_error,
            ) = _extract_teacher_result(teacher_raw)
            teacher_f1 = compute_f1(teacher_answer, gold_answers)
            evidence_sufficient = bool(
                teacher_evidence_status == "supported_answer"
                and teacher_answer
                and teacher_answer != insufficient
                and not teacher_format_error
            )
            chosen = _strip_teacher_status_block(teacher_raw)
            counters.inc("teacher_completed")
            counters.add_float("_sum_teacher_elapsed_s", teacher_elapsed)
        except Exception as exc:
            counters.inc("teacher_errors")
            if "timed out" in str(exc).lower() or "timeout" in type(exc).__name__.lower():
                counters.inc("timeout_errors")
                skip_reason = "teacher_timeout"
            else:
                skip_reason = f"teacher_error:{type(exc).__name__}"
            errors.append(f"{skip_reason}:{exc}")
            keep = False

        if keep and require_teacher_format and teacher_parse_status != "parsed":
            counters.inc("skipped_teacher_format")
            keep = False
            skip_reason = f"teacher_{teacher_parse_status or 'unparsed'}"
        if keep and require_evidence_sufficient and not evidence_sufficient:
            counters.inc("skipped_evidence_insufficient")
            keep = False
            skip_reason = "evidence_insufficient"
        if keep and teacher_f1 < min_teacher_f1:
            counters.inc("skipped_teacher_f1")
            keep = False
            skip_reason = "teacher_f1_below_threshold"
        pair_record = {
            "index": index,
            "question": question,
            "gold_answers": gold_answers,
            "messages_before_final_answer": messages_before,
            "chosen": chosen,
            "rejected": rejected,
            "evidence_steps": evidence_steps,
            "teacher_reason": teacher_raw,
            "teacher_f1": teacher_f1,
            "teacher_prompt_version": teacher_prompt_version,
            "filter_status": "kept" if keep else "skipped",
        }
        valid, invalid_reason = _validate_dpo_pair(pair_record)
        if keep and not valid:
            keep = False
            skip_reason = invalid_reason
            if invalid_reason == "chosen_equals_rejected":
                counters.inc("skipped_chosen_equals_rejected")
            else:
                counters.inc("schema_invalid_pairs")
        metadata = {
            "index": index,
            "question": question,
            "gold_answers": gold_answers,
            "actor_answer": actor_answer,
            "actor_f1": actor_f1,
            "teacher_answer": teacher_answer,
            "teacher_f1": teacher_f1,
            "teacher_parse_status": teacher_parse_status,
            "teacher_evidence_status": teacher_evidence_status,
            "teacher_format_error": teacher_format_error,
            "teacher_prompt_version": teacher_prompt_version,
            "evidence_sufficient": evidence_sufficient,
            "search_count": int(trajectory.get("search_count") or len(evidence_steps)),
            "sub_queries": trajectory.get("sub_queries") or [],
            "actor_elapsed_s": trajectory.get("actor_elapsed_s", 0.0),
            "teacher_elapsed_s": teacher_elapsed,
            "keep": keep,
            "skip_reason": skip_reason,
            "errors": errors,
        }
        refresh_record = {
            "index": index,
            "status": "kept" if keep else "skipped",
            "skip_reason": None if keep else skip_reason,
            "prompt": messages_before,
            "chosen": chosen,
            "rejected": rejected,
            "messages_before_final_answer": messages_before,
            "evidence_steps": evidence_steps,
            "teacher_raw_content": teacher_raw,
            "metadata": metadata,
        }
        refresh_writer.write(refresh_record)
        inc_progress("phase_b_seen")
        if keep:
            counters.inc("kept")
            pair_record["filter_status"] = "kept"
            pair_writer.write(pair_record)
            inc_progress("dpo_pairs")
        seen = counters.snapshot().get("teacher_completed", 0)
        if int(seen) > 0 and int(seen) % progress_log_interval == 0:
            print(
                "SPAD Stage2 PhaseB progress: "
                f"teacher_completed={seen} kept={counters.snapshot().get('kept')} "
                f"teacher_inflight={teacher_pool.inflight()}",
                flush=True,
            )
        return refresh_record

    set_progress("phase_b_seen", len(existing_refresh))
    set_progress("dpo_pairs", len(existing_pairs))
    eligible: list[dict[str, Any]] = []
    for trajectory in trajectories:
        index = int(trajectory.get("index", -1))
        if index in existing_refresh:
            continue
        if not trajectory.get("actor_final") or not trajectory.get("rejected"):
            counters.inc("skipped_no_actor_final")
            skipped_record(trajectory, "no_actor_final")
            continue
        if not trajectory.get("evidence_steps"):
            counters.inc("skipped_no_evidence")
            skipped_record(trajectory, "no_evidence")
            continue
        eligible.append(trajectory)

    if eligible:
        max_workers = max(1, min(int(submit_batch_size), len(teacher_outputs) * max_inflight_per_teacher, len(eligible)))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="spad-stage2-teacher") as executor:
            futures = [executor.submit(process_one, trajectory) for trajectory in eligible]
            for future in as_completed(futures):
                future.result()

    snapshot = counters.snapshot()
    completed = max(1, int(snapshot.get("teacher_completed", 0)))
    snapshot["avg_teacher_elapsed_s"] = float(snapshot.pop("_sum_teacher_elapsed_s", 0.0)) / completed
    snapshot["teacher_prompt_version"] = teacher_prompt_version
    return snapshot


def _write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _merge_teacher_resource(spad_cfg: dict[str, Any], resource_cfg: dict[str, Any], replica: dict[str, Any], index: int) -> dict[str, Any]:
    profile_name = str(resource_cfg.get("profile") or spad_cfg["teacher_answerer"]["default_service_profile"])
    profile = dict(spad_cfg["teacher_answerer"]["service_profiles"][profile_name])
    common = dict(resource_cfg.get("common") or {})
    merged = {
        key: value
        for key, value in resource_cfg.items()
        if key not in {"replicas", "common", "gpu_ids", "port", "endpoint"}
    }
    merged.update(profile)
    merged.update(common)
    merged.update(replica)
    merged.setdefault("profile", profile_name)
    merged.setdefault("container_name", f"spad_teacher_batch_{index}")
    return merged


def _start_offline_teacher_worker(
    *,
    runtime_dir: Path,
    shard_id: int,
    input_jsonl: Path,
    output_jsonl: Path,
    merged: dict[str, Any],
    request_cfg: dict[str, Any],
    scheduler_cfg: dict[str, Any],
) -> ManagedProcess:
    container_name = str(merged.get("container_name") or f"spad_teacher_batch_{shard_id}")
    image = str(merged["container_image"])
    gpu_ids = csv_ids(merged["gpu_ids"])
    repo = repo_root()
    script_path = runtime_dir / f"start_teacher_offline_batch_shard{shard_id}.sh"
    log_path = runtime_dir / f"teacher_offline_batch_shard{shard_id}.log"
    worker = repo / "scripts" / "agenticIterRag_v1" / "spad_teacher_offline_batch_worker.py"
    batch_size = int(scheduler_cfg.get("offline_batch_size", scheduler_cfg.get("batch_size", 64)))
    max_num_seqs = int(scheduler_cfg.get("max_num_seqs", merged.get("max_num_seqs", 128)))
    max_num_batched_tokens = int(scheduler_cfg.get("max_num_batched_tokens", merged.get("max_num_batched_tokens", 65536)))
    max_model_len = int(scheduler_cfg.get("max_model_len", merged.get("max_model_len", 32000)))
    gpu_memory_utilization = float(scheduler_cfg.get("gpu_memory_utilization", merged.get("gpu_memory_utilization", 0.95)))
    tensor_parallel_size = int(merged.get("tensor_parallel_size", len(as_int_list(merged.get("gpu_ids")))))
    dtype = str(merged.get("dtype", scheduler_cfg.get("dtype", "bfloat16")))
    cmd_parts = [
        "python",
        quote(worker),
        "--input-jsonl",
        quote(input_jsonl),
        "--output-jsonl",
        quote(output_jsonl),
        "--model-path",
        quote(merged["model_path"]),
        "--tensor-parallel-size",
        quote(tensor_parallel_size),
        "--dtype",
        quote(dtype),
        "--max-model-len",
        quote(max_model_len),
        "--gpu-memory-utilization",
        quote(gpu_memory_utilization),
        "--max-num-seqs",
        quote(max_num_seqs),
        "--max-num-batched-tokens",
        quote(max_num_batched_tokens),
        "--batch-size",
        quote(batch_size),
        "--temperature",
        quote(float(request_cfg.get("temperature", 0.0))),
        "--top-p",
        quote(float(request_cfg.get("top_p", 1.0))),
        "--max-tokens",
        quote(int(request_cfg.get("max_tokens", 512))),
        "--prompt-version",
        quote(request_cfg.get("prompt_version") or DEFAULT_TEACHER_STATUS_PROMPT_VERSION),
        "--shard-id",
        quote(shard_id),
    ]
    if bool(scheduler_cfg.get("enable_prefix_caching", merged.get("enable_prefix_caching", True))):
        cmd_parts.append("--enable-prefix-caching")
    if bool(scheduler_cfg.get("enable_chunked_prefill", merged.get("enable_chunked_prefill", True))):
        cmd_parts.append("--enable-chunked-prefill")
    if bool(scheduler_cfg.get("enforce_eager", merged.get("enforce_eager", False))):
        cmd_parts.append("--enforce-eager")
    if bool(merged.get("disable_custom_all_reduce", True)):
        cmd_parts.append("--disable-custom-all-reduce")
    chat_template_kwargs = request_cfg.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and bool(chat_template_kwargs.get("enable_thinking", False)):
        cmd_parts.append("--enable-thinking")
    docker_cmd = " ".join(cmd_parts)
    write_script(
        script_path,
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"docker rm -f {quote(container_name)} >/dev/null 2>&1 || true",
            f"docker image inspect {quote(image)} >/dev/null 2>&1 || docker pull {quote(image)}",
            "docker run --rm \\",
            f"  --name {quote(container_name)} \\",
            "  --privileged --net=host --ipc=host \\",
            f"  -e ASCEND_RT_VISIBLE_DEVICES={quote(gpu_ids)} \\",
            f"  -e CUDA_VISIBLE_DEVICES={quote(gpu_ids)} \\",
            "  -e HF_HUB_OFFLINE=1 \\",
            "  -e TRANSFORMERS_OFFLINE=1 \\",
            "  -e VLLM_DISABLE_FLASHINFER=1 \\",
            "  -e VLLM_USE_FLASHINFER_SAMPLER=0 \\",
            "  -e VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN} \\",
            "  -e VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0} \\",
            "  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \\",
            f"  -e PYTHONPATH={quote(str(project_root()))} \\",
            "  -v /data01:/data01 \\",
            "  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \\",
            "  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \\",
            f"  {quote(image)} \\",
            f"  bash -lc {quote(docker_cmd)}",
        ],
    )
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(["bash", str(script_path)], cwd=str(repo), stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    log_file.close()
    return ManagedProcess(
        name=f"spad-teacher-offline-batch-{shard_id}",
        process=process,
        log_path=log_path,
        script_path=script_path,
    )


def _run_teacher_labeling_offline_batch_phase(
    *,
    trajectory_jsonl: Path,
    refresh_jsonl: Path,
    dataset_jsonl: Path,
    teacher_resource: dict[str, Any],
    runtime_dir: Path,
    spad_cfg: dict[str, Any],
    filter_cfg: dict[str, Any],
    teacher_request: dict[str, Any],
    scheduler_cfg: dict[str, Any],
    resume_existing: bool,
    inc_progress: Any,
    set_progress: Any,
) -> dict[str, Any]:
    if not trajectory_jsonl.exists():
        raise RuntimeError(f"Stage 2 Phase B requires Phase A output: {trajectory_jsonl}")
    trajectories = _iter_jsonl(trajectory_jsonl)
    existing_refresh = _existing_indices(refresh_jsonl) if resume_existing else set()
    existing_pairs = _existing_indices(dataset_jsonl) if resume_existing else set()
    refresh_writer = JsonlAppendWriter(refresh_jsonl)
    pair_writer = JsonlAppendWriter(dataset_jsonl)
    counters = CounterBag(
        {
            "phase": "teacher_labeling",
            "backend": "offline_vllm_batch",
            "total_trajectories": len(trajectories),
            "eligible_for_teacher": 0,
            "teacher_completed": 0,
            "kept": len(existing_pairs),
            "skipped_no_actor_final": 0,
            "skipped_no_evidence": 0,
            "skipped_teacher_format": 0,
            "skipped_evidence_insufficient": 0,
            "skipped_teacher_f1": 0,
            "skipped_chosen_equals_rejected": 0,
            "teacher_errors": 0,
            "timeout_errors": 0,
            "schema_invalid_pairs": 0,
            "_sum_teacher_elapsed_s": 0.0,
        }
    )

    def skipped_record(trajectory: dict[str, Any], reason: str) -> None:
        index = int(trajectory.get("index", -1))
        refresh_writer.write(
            {
                "index": index,
                "status": "skipped",
                "skip_reason": reason,
                "prompt": trajectory.get("messages_before_final_answer") or [],
                "chosen": "",
                "rejected": trajectory.get("rejected") or "",
                "messages_before_final_answer": trajectory.get("messages_before_final_answer") or [],
                "evidence_steps": trajectory.get("evidence_steps") or [],
                "teacher_raw_content": "",
                "metadata": {
                    "index": index,
                    "question": trajectory.get("question"),
                    "gold_answers": trajectory.get("gold_answers") or [],
                    "keep": False,
                    "skip_reason": reason,
                },
            }
        )
        inc_progress("phase_b_seen")

    set_progress("phase_b_seen", len(existing_refresh))
    set_progress("dpo_pairs", len(existing_pairs))
    eligible: list[dict[str, Any]] = []
    trajectory_by_index: dict[int, dict[str, Any]] = {}
    for trajectory in trajectories:
        index = int(trajectory.get("index", -1))
        trajectory_by_index[index] = trajectory
        if index in existing_refresh:
            continue
        if not trajectory.get("actor_final") or not trajectory.get("rejected"):
            counters.inc("skipped_no_actor_final")
            skipped_record(trajectory, "no_actor_final")
            continue
        if not trajectory.get("evidence_steps"):
            counters.inc("skipped_no_evidence")
            skipped_record(trajectory, "no_evidence")
            continue
        counters.inc("eligible_for_teacher")
        eligible.append(trajectory)

    if not eligible:
        snapshot = counters.snapshot()
        snapshot["avg_teacher_elapsed_s"] = 0.0
        snapshot["offline_shard_count"] = 0
        snapshot["offline_batch_size"] = int(scheduler_cfg.get("offline_batch_size", scheduler_cfg.get("batch_size", 64)))
        return snapshot

    replicas = validate_replica_config(service_name="teacher_answerer", service_cfg=teacher_resource)
    shard_count = max(1, min(int(scheduler_cfg.get("offline_shard_count", len(replicas))), len(replicas), max(1, len(eligible))))
    runtime_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = runtime_dir / "offline_teacher_batch_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_inputs: list[Path] = []
    shard_outputs: list[Path] = []
    processes: list[ManagedProcess] = []
    container_names: list[str] = []
    try:
        for shard_id in range(shard_count):
            shard_rows = eligible[shard_id::shard_count]
            input_jsonl = shard_dir / f"shard_{shard_id}.input.jsonl"
            output_jsonl = shard_dir / f"shard_{shard_id}.raw.jsonl"
            output_jsonl.unlink(missing_ok=True)
            _write_jsonl_records(input_jsonl, shard_rows)
            shard_inputs.append(input_jsonl)
            shard_outputs.append(output_jsonl)
            merged = _merge_teacher_resource(spad_cfg, teacher_resource, replicas[shard_id], shard_id)
            if merged.get("container_name") is None:
                merged["container_name"] = f"spad_teacher_offline_batch_{shard_id}"
            container_names.append(str(merged["container_name"]))
            proc = _start_offline_teacher_worker(
                runtime_dir=runtime_dir,
                shard_id=shard_id,
                input_jsonl=input_jsonl,
                output_jsonl=output_jsonl,
                merged=merged,
                request_cfg=teacher_request,
                scheduler_cfg=scheduler_cfg,
            )
            processes.append(proc)
        progress_log_interval = max(1, int(scheduler_cfg.get("progress_log_interval", 20)))
        last_seen = -1
        while any(proc.poll() is None for proc in processes):
            generated = sum(_count_jsonl(path) for path in shard_outputs)
            set_progress("phase_b_seen", len(existing_refresh) + generated)
            if generated != last_seen and generated > 0 and generated % progress_log_interval == 0:
                print(f"SPAD Stage2 PhaseB offline progress: teacher_generated={generated}/{len(eligible)}", flush=True)
                last_seen = generated
            time.sleep(float(scheduler_cfg.get("offline_poll_interval_s", 5)))
        failures = [proc for proc in processes if proc.poll() != 0]
        if failures:
            details = "\n\n".join(f"--- {proc.name}: {proc.log_path}\n{tail_text(proc.log_path)}" for proc in failures)
            raise RuntimeError(f"offline teacher batch worker failed\n{details}")

        raw_by_index: dict[int, dict[str, Any]] = {}
        for output_jsonl in shard_outputs:
            for record in _iter_jsonl(output_jsonl):
                raw_by_index[int(record["index"])] = record
        for trajectory in eligible:
            index = int(trajectory.get("index", -1))
            raw_record = raw_by_index.get(index)
            if raw_record is None:
                counters.inc("teacher_errors")
                refresh_record, pair_record, updates = _teacher_label_result_from_raw(
                    trajectory=trajectory,
                    teacher_raw="",
                    teacher_elapsed=0.0,
                    spad_cfg=spad_cfg,
                    filter_cfg=filter_cfg,
                    errors=["teacher_missing_output"],
                )
            else:
                refresh_record, pair_record, updates = _teacher_label_result_from_raw(
                    trajectory=trajectory,
                    teacher_raw=str(raw_record.get("teacher_raw_content") or ""),
                    teacher_elapsed=float(raw_record.get("teacher_elapsed_s") or 0.0),
                    spad_cfg=spad_cfg,
                    filter_cfg=filter_cfg,
                )
            for key, value in updates.items():
                if key == "_sum_teacher_elapsed_s":
                    counters.add_float(key, float(value))
                else:
                    counters.inc(key, int(value))
            refresh_writer.write(refresh_record)
            inc_progress("phase_b_seen")
            if pair_record is not None:
                pair_writer.write(pair_record)
                inc_progress("dpo_pairs")
            seen = int(counters.snapshot().get("teacher_completed", 0))
            if seen > 0 and seen % progress_log_interval == 0:
                print(
                    f"SPAD Stage2 PhaseB offline postprocess: teacher_completed={seen} kept={counters.snapshot().get('kept')}",
                    flush=True,
                )
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate(timeout_s=1)
        for name in container_names:
            subprocess.run(["docker", "rm", "-f", name], cwd=str(repo_root()), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    snapshot = counters.snapshot()
    completed = max(1, int(snapshot.get("teacher_completed", 0)))
    snapshot["avg_teacher_elapsed_s"] = float(snapshot.pop("_sum_teacher_elapsed_s", 0.0)) / completed
    snapshot["offline_shard_count"] = shard_count
    snapshot["offline_batch_size"] = int(scheduler_cfg.get("offline_batch_size", scheduler_cfg.get("batch_size", 64)))
    snapshot["teacher_prompt_version"] = str(
        teacher_request.get("prompt_version") or DEFAULT_TEACHER_STATUS_PROMPT_VERSION
    )
    return snapshot


def _run_rollout_backend(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    sub_cfg: dict[str, Any],
    stage_dir: Path,
    log_dir: Path,
    checkpoint_dir: Path,
    dataset_jsonl: Path,
    resource_plan: dict[str, Any],
    actor_checkpoint: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    rollout_cfg = dict(sub_cfg.get("rollout") or {})
    filter_cfg = dict(sub_cfg.get("filter") or {})
    phase_cfgs = sub_cfg.get("phases") if isinstance(sub_cfg.get("phases"), dict) else {}
    phase_order = [str(item) for item in sub_cfg.get("phase_order", ["trajectory_rollout", "teacher_labeling"])]
    resume_from_phase = sub_cfg.get("resume_from_phase")
    stop_after_phase = sub_cfg.get("stop_after_phase")
    if resume_from_phase:
        if str(resume_from_phase) not in phase_order:
            raise ValueError(f"answer_refresh_data.resume_from_phase is not in phase_order: {resume_from_phase}")
        phase_order = phase_order[phase_order.index(str(resume_from_phase)) :]
    if stop_after_phase:
        if str(stop_after_phase) not in phase_order:
            raise ValueError(f"answer_refresh_data.stop_after_phase is not in selected phase_order: {stop_after_phase}")
        phase_order = phase_order[: phase_order.index(str(stop_after_phase)) + 1]

    inputs_cfg = sub_cfg.get("inputs") if isinstance(sub_cfg.get("inputs"), dict) else {}
    max_samples = int(inputs_cfg.get("max_samples") or sub_cfg.get("smoke", {}).get("max_samples") or 8)
    top_n = int(rollout_cfg.get("recall_top_n") or config.get("infer_runtime", {}).get("retriever", {}).get("recall_final_top_n") or 50)
    top_m = int(rollout_cfg.get("visible_top_m", 5))
    max_doc_chars = int(rollout_cfg.get("max_doc_chars", 2000))
    search_timeout = float(rollout_cfg.get("search_timeout", 60))
    max_assistant_turns = int(rollout_cfg.get("max_assistant_turns", 6))
    max_tokens = int(rollout_cfg.get("max_response_tokens", rollout_cfg.get("max_response_length", 1024)))
    request_cfg = {
        "temperature": float(rollout_cfg.get("temperature", 1.0)),
        "top_p": float(rollout_cfg.get("top_p", 1.0)),
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    teacher_request = dict(spad_cfg["teacher_answerer"].get("request") or {})
    teacher_prompt_version, _ = resolve_teacher_prompt(
        str(
            spad_cfg["teacher_answerer"].get("prompt_version")
            or DEFAULT_TEACHER_STATUS_PROMPT_VERSION
        ),
        include_status=True,
    )
    teacher_request["prompt_version"] = teacher_prompt_version

    configured_actor_checkpoint = inputs_cfg.get("actor_checkpoint")
    actor_checkpoint = str(actor_checkpoint or configured_actor_checkpoint or "")
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    hf_actor_checkpoint = actor_checkpoint if dry_run else _ensure_hf_actor_checkpoint(actor_checkpoint, checkpoint_dir, log_dir)
    runtime_dir = log_dir

    services = resource_plan.get("services", {})
    actor_resource = dict(services.get("actor_vllm") or {})
    actor_common = dict(actor_resource.get("common") or {})
    actor_common.setdefault("max_model_len", int(rollout_cfg.get("max_model_len", 16096)))
    actor_common.setdefault("gpu_memory_utilization", float(rollout_cfg.get("gpu_memory_utilization", 0.6)))
    for key in (
        "max_num_seqs",
        "max_num_batched_tokens",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "enforce_eager",
    ):
        if key in rollout_cfg:
            actor_common.setdefault(key, rollout_cfg[key])
    actor_resource["common"] = actor_common
    teacher_resource = dict(services.get("teacher_answerer") or {})
    recall_resource = dict(services.get("recall") or {})

    if "replicas" not in actor_resource:
        raise ValueError("Stage 2 rollout backend requires services.actor_vllm.replicas; legacy single actor is disabled")
    if "replicas" not in teacher_resource:
        raise ValueError("Stage 2 rollout backend requires services.teacher_answerer.replicas; legacy single teacher is disabled")

    trajectory_jsonl = stage_dir / "answer_refresh_actor_trajectories.jsonl"
    refresh_jsonl = stage_dir / "refresh_rollouts.jsonl"
    stats_json = stage_dir / "stage2_stats.json"
    resource_jsonl = stage_dir / "stage2_resource_monitor.jsonl"
    dataset_manifest = stage_dir / "answer_distill_dataset_manifest.json"
    report_path = repo_root() / "docs" / "AgenticIterRag_v1" / "work_report" / "260710_spad_stage2_resource_monitor.md"
    resume_existing = bool(sub_cfg.get("resume_existing", True))

    if dry_run:
        outputs = {
            "status": "planned",
            "backend": "rollout",
            "phase_order": phase_order,
            "actor_checkpoint": actor_checkpoint,
            "hf_actor_checkpoint": hf_actor_checkpoint,
            "trajectory_jsonl": str(trajectory_jsonl),
            "refresh_jsonl": str(refresh_jsonl),
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_manifest": str(dataset_manifest),
            "runtime_dir": str(runtime_dir),
            "resource_plan": resource_plan,
        }
        write_sub_stage_manifest(dataset_manifest, sub_stage="answer_distill_dataset", outputs=outputs)
        return outputs

    rows = load_rl_rows(input_train_files(config, spad_cfg), max_samples=max_samples)
    _reset_outputs([trajectory_jsonl, refresh_jsonl, dataset_jsonl, stats_json, resource_jsonl], resume_existing=resume_existing)
    stage_state = {"phase": "init"}
    progress_state: dict[str, int] = {
        "phase_a_total": len(rows),
        "phase_a_written": len(_existing_indices(trajectory_jsonl)) if resume_existing else 0,
        "phase_b_seen": len(_existing_indices(refresh_jsonl)) if resume_existing else 0,
        "dpo_pairs": len(_existing_indices(dataset_jsonl)) if resume_existing else 0,
    }
    progress_lock = threading.Lock()

    def set_progress(key: str, value: int) -> None:
        with progress_lock:
            progress_state[key] = value

    def inc_progress(key: str, amount: int = 1) -> None:
        with progress_lock:
            progress_state[key] = int(progress_state.get(key, 0)) + amount

    def get_progress() -> dict[str, int]:
        with progress_lock:
            return dict(progress_state)

    monitor = Stage2ResourceMonitor(
        jsonl_path=resource_jsonl,
        report_path=report_path,
        interval_s=float(sub_cfg.get("resource_monitor", {}).get("interval_seconds", 30) if isinstance(sub_cfg.get("resource_monitor"), dict) else 30),
        get_phase=lambda: str(stage_state["phase"]),
        get_progress=get_progress,
    )
    monitor.start()
    service_outputs: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    try:
        if "trajectory_rollout" in phase_order:
            stage_state["phase"] = "trajectory_rollout"
            manager_a = SpadServiceManager(runtime_dir=runtime_dir / "services" / "trajectory_rollout", verl_root=project_root() / "verl")
            try:
                service_outputs["actor_vllm"] = manager_a.start_actor_vllm_replicas(actor_cfg=actor_resource, model_path=hf_actor_checkpoint)
                service_outputs["recall"] = manager_a.start_recall(
                    recall_cfg=recall_resource,
                    final_top_n=top_n,
                    recall_model=str(config["infer_runtime"]["models"]["recall_model_path"]),
                )
                phase_a_stats = _run_trajectory_rollout_phase(
                    rows=rows,
                    actor_outputs=service_outputs["actor_vllm"],
                    recall_output=service_outputs["recall"],
                    trajectory_jsonl=trajectory_jsonl,
                    actor_checkpoint=actor_checkpoint,
                    hf_actor_checkpoint=hf_actor_checkpoint,
                    rollout_cfg=rollout_cfg,
                    request_cfg=request_cfg,
                    top_n=top_n,
                    top_m=top_m,
                    max_doc_chars=max_doc_chars,
                    search_timeout=search_timeout,
                    max_assistant_turns=max_assistant_turns,
                    scheduler_cfg=dict(phase_cfgs.get("trajectory_rollout", {}).get("scheduler") or {}),
                    resume_existing=resume_existing,
                    inc_progress=inc_progress,
                    set_progress=set_progress,
                )
                stats["trajectory_rollout"] = phase_a_stats
            finally:
                if bool(sub_cfg.get("auto_stop_services", True)):
                    manager_a.stop_all()

        if "teacher_labeling" in phase_order:
            stage_state["phase"] = "teacher_labeling"
            teacher_phase_cfg = dict(phase_cfgs.get("teacher_labeling") or {})
            teacher_scheduler_cfg = dict(teacher_phase_cfg.get("scheduler") or {})
            teacher_backend = str(teacher_phase_cfg.get("backend") or teacher_scheduler_cfg.get("backend") or "http")
            if teacher_backend == "offline_vllm_batch":
                phase_b_stats = _run_teacher_labeling_offline_batch_phase(
                    trajectory_jsonl=trajectory_jsonl,
                    refresh_jsonl=refresh_jsonl,
                    dataset_jsonl=dataset_jsonl,
                    teacher_resource=teacher_resource,
                    runtime_dir=runtime_dir / "services" / "teacher_labeling",
                    spad_cfg=spad_cfg,
                    filter_cfg=filter_cfg,
                    teacher_request=teacher_request,
                    scheduler_cfg=teacher_scheduler_cfg,
                    resume_existing=resume_existing,
                    inc_progress=inc_progress,
                    set_progress=set_progress,
                )
                service_outputs["teacher"] = {
                    "backend": "offline_vllm_batch",
                    "runtime_dir": str(runtime_dir / "services" / "teacher_labeling"),
                }
                stats["teacher_labeling"] = phase_b_stats
            elif teacher_backend == "http":
                manager_b = SpadServiceManager(runtime_dir=runtime_dir / "services" / "teacher_labeling", verl_root=project_root() / "verl")
                try:
                    service_outputs["teacher"] = manager_b.start_teacher_replicas(
                        teacher_cfg=spad_cfg["teacher_answerer"],
                        resource_cfg=teacher_resource,
                    )
                    phase_b_stats = _run_teacher_labeling_phase(
                        trajectory_jsonl=trajectory_jsonl,
                        refresh_jsonl=refresh_jsonl,
                        dataset_jsonl=dataset_jsonl,
                        teacher_outputs=service_outputs["teacher"],
                        spad_cfg=spad_cfg,
                        filter_cfg=filter_cfg,
                        teacher_request=teacher_request,
                        scheduler_cfg=teacher_scheduler_cfg,
                        resume_existing=resume_existing,
                        inc_progress=inc_progress,
                        set_progress=set_progress,
                    )
                    stats["teacher_labeling"] = phase_b_stats
                finally:
                    if bool(sub_cfg.get("auto_stop_services", True)):
                        manager_b.stop_all()
            else:
                raise ValueError(f"unsupported Stage2 teacher_labeling backend: {teacher_backend!r}")

        stage_state["phase"] = "finalize"
        kept = int(stats.get("teacher_labeling", {}).get("kept", 0)) if "teacher_labeling" in stats else progress_state.get("dpo_pairs", 0)
        if "teacher_labeling" in phase_order and kept <= 0:
            raise RuntimeError(f"Stage 2 teacher_labeling produced no kept DPO pairs; stats={stats.get('teacher_labeling')}")
        write_json(stats_json, stats)
        outputs = {
            "status": "completed",
            "backend": "rollout",
            "phase_order": phase_order,
            "actor_checkpoint": actor_checkpoint,
            "hf_actor_checkpoint": hf_actor_checkpoint,
            "trajectory_jsonl": str(trajectory_jsonl),
            "refresh_jsonl": str(refresh_jsonl),
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_manifest": str(dataset_manifest),
            "sample_count": kept,
            "refresh_count": int(stats.get("teacher_labeling", {}).get("total_trajectories", 0)),
            "stats": stats,
            "stats_json": str(stats_json),
            "resource_monitor_jsonl": str(resource_jsonl),
            "resource_monitor_report": str(report_path),
            "runtime_dir": str(runtime_dir),
            "service_outputs": service_outputs,
            "resource_plan": resource_plan,
        }
        write_sub_stage_manifest(dataset_manifest, sub_stage="answer_distill_dataset", outputs=outputs)
        return outputs
    finally:
        monitor.stop()


def run_answer_refresh_data(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    stage_dir: Path,
    log_dir: Path,
    checkpoint_dir: Path,
    resource_plan: dict[str, Any],
    dry_run: bool,
    actor_checkpoint: str | None,
) -> dict[str, Any]:
    """Run Stage 2. The rollout backend materializes chosen/rejected pairs."""

    sub_cfg = spad_cfg["sub_stages"]["answer_refresh_data"]
    backend = str(sub_cfg.get("backend") or spad_cfg.get("default_backend") or "smoke")
    actor_checkpoint = str(actor_checkpoint or sub_cfg.get("inputs", {}).get("actor_checkpoint") or "")
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = stage_dir / "answer_distill_pairs.jsonl"
    dataset_manifest = stage_dir / "answer_distill_dataset_manifest.json"
    if dry_run:
        outputs = {
            "status": "planned",
            "backend": backend,
            "actor_checkpoint": actor_checkpoint,
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_manifest": str(dataset_manifest),
            "runtime_dir": str(log_dir),
            "resource_plan": resource_plan,
        }
        write_sub_stage_manifest(dataset_manifest, sub_stage="answer_distill_dataset", outputs=outputs)
        outputs["manifest"] = str(stage_dir / "manifest.json")
        write_sub_stage_manifest(outputs["manifest"], sub_stage="answer_refresh_data", outputs=outputs)
        return outputs

    if backend == "smoke":
        outputs = _run_smoke_backend(
            config=config,
            spad_cfg=spad_cfg,
            sub_cfg=sub_cfg,
            stage_dir=stage_dir,
            log_dir=log_dir,
            dataset_jsonl=dataset_jsonl,
            resource_plan=resource_plan,
            actor_checkpoint=actor_checkpoint,
        )
    elif backend == "rollout":
        outputs = _run_rollout_backend(
            config=config,
            spad_cfg=spad_cfg,
            sub_cfg=sub_cfg,
            stage_dir=stage_dir,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            dataset_jsonl=dataset_jsonl,
            resource_plan=resource_plan,
            actor_checkpoint=actor_checkpoint,
            dry_run=dry_run,
        )
    else:
        raise NotImplementedError(f"Stage 2 backend={backend!r} is not supported")

    outputs["manifest"] = str(stage_dir / "manifest.json")
    write_sub_stage_manifest(outputs["manifest"], sub_stage="answer_refresh_data", outputs=outputs)
    write_json(stage_dir / "summary.json", outputs)
    return outputs
