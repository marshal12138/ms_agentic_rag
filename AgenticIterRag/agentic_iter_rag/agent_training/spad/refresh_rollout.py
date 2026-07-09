"""Stage 2 answer refresh data preparation for SPAD-RAG."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

from agentic_iter_rag.agent_training.spad.config import input_train_files
from agentic_iter_rag.agent_training.spad.data import load_rl_rows, row_gold_answers, row_prompt_messages, row_question
from agentic_iter_rag.agent_training.spad.manifest import write_records, write_sub_stage_manifest
from agentic_iter_rag.agent_training.spad.parsers import extract_last_answer
from agentic_iter_rag.agent_training.spad.prompts import build_teacher_messages, smoke_actor_answer
from agentic_iter_rag.agent_training.spad.reward import compute_f1
from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import _extract_teacher_answer
from agentic_iter_rag.agent_training.spad.service_manager import (
    SpadServiceManager,
    post_json,
    project_root,
    repo_root,
    tail_text,
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


def _ensure_hf_actor_checkpoint(actor_checkpoint: str, stage_dir: Path) -> str:
    """Return an HF checkpoint path, merging VERL FSDP actor shards when needed."""

    if not actor_checkpoint:
        raise ValueError("Stage 2 requires actor_checkpoint from Stage 1")
    checkpoint_path = Path(actor_checkpoint)
    if (checkpoint_path / "config.json").exists() and (checkpoint_path / "model.safetensors").exists():
        return str(checkpoint_path)

    actor_dir = checkpoint_path / "actor" if checkpoint_path.name.startswith("global_step_") else checkpoint_path
    if not (actor_dir / "fsdp_config.json").exists():
        raise ValueError(f"unsupported actor checkpoint layout for Stage 2 refresh: {actor_checkpoint}")

    target_dir = stage_dir / "actor_model_hf" / checkpoint_path.name
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
    log_path = stage_dir / "runtime" / "merge_actor_checkpoint.log"
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
        "resource_plan": resource_plan,
    }
    write_sub_stage_manifest(dataset_manifest, sub_stage="answer_distill_dataset", outputs=outputs)
    return outputs


def _run_rollout_backend(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    sub_cfg: dict[str, Any],
    stage_dir: Path,
    dataset_jsonl: Path,
    resource_plan: dict[str, Any],
    actor_checkpoint: str | None,
    dry_run: bool,
) -> dict[str, Any]:
    rollout_cfg = dict(sub_cfg.get("rollout") or {})
    filter_cfg = dict(sub_cfg.get("filter") or {})
    max_samples = int(sub_cfg.get("inputs", {}).get("max_samples") or sub_cfg.get("smoke", {}).get("max_samples") or 8)
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

    configured_actor_checkpoint = sub_cfg.get("inputs", {}).get("actor_checkpoint")
    actor_checkpoint = str(actor_checkpoint or configured_actor_checkpoint or "")
    hf_actor_checkpoint = _ensure_hf_actor_checkpoint(actor_checkpoint, stage_dir)
    runtime_dir = stage_dir / "runtime"
    manager = SpadServiceManager(runtime_dir=runtime_dir / "services", verl_root=project_root() / "verl")
    service_outputs: dict[str, Any] = {}
    try:
        services = resource_plan.get("services", {})
        actor_resource = dict(services.get("actor_vllm") or {})
        actor_resource.setdefault("max_model_len", int(rollout_cfg.get("max_model_len", 16096)))
        actor_resource.setdefault("gpu_memory_utilization", float(rollout_cfg.get("gpu_memory_utilization", 0.6)))
        teacher_resource = dict(services.get("teacher_answerer") or {})
        recall_resource = dict(services.get("recall") or {})
        if dry_run:
            actor_output = {"status": "planned", "endpoint": actor_resource.get("endpoint", "http://127.0.0.1:8340/v1/chat/completions"), "model": actor_resource.get("served_model_name", "spad-refresh-actor")}
            teacher_output = {"status": "planned", "endpoint": teacher_resource.get("endpoint", "http://127.0.0.1:8067/v1/chat/completions"), "model": teacher_resource.get("served_model_name", "GLM-4.7-Flash")}
            recall_output = {"status": "planned", "retrieval_url": recall_resource.get("retrieval_service_url", "http://127.0.0.1:8130/retrieve")}
        else:
            service_outputs["actor_vllm"] = manager.start_actor_vllm(actor_cfg=actor_resource, model_path=hf_actor_checkpoint)
            service_outputs["teacher"] = manager.start_teacher(teacher_cfg=spad_cfg["teacher_answerer"], resource_cfg=teacher_resource)
            service_outputs["recall"] = manager.start_recall(
                recall_cfg=recall_resource,
                final_top_n=top_n,
                recall_model=str(config["infer_runtime"]["models"]["recall_model_path"]),
            )
            actor_output = service_outputs["actor_vllm"]
            teacher_output = service_outputs["teacher"]
            recall_output = service_outputs["recall"]

        if dry_run:
            return {
                "status": "planned",
                "backend": "rollout",
                "actor_checkpoint": actor_checkpoint,
                "hf_actor_checkpoint": hf_actor_checkpoint,
                "dataset_jsonl": str(dataset_jsonl),
                "service_outputs": service_outputs,
                "resource_plan": resource_plan,
            }

        rows = load_rl_rows(input_train_files(config, spad_cfg), max_samples=max_samples)
        refresh_jsonl = stage_dir / "refresh_rollouts.jsonl"
        records: list[dict[str, Any]] = []
        refresh_records: list[dict[str, Any]] = []
        counters = {
            "total": 0,
            "kept": 0,
            "skipped_no_finish": 0,
            "skipped_teacher_format": 0,
            "skipped_evidence_insufficient": 0,
            "skipped_teacher_f1": 0,
            "actor_errors": 0,
            "teacher_errors": 0,
        }
        insufficient = str(spad_cfg["teacher_answerer"].get("insufficient_answer", "证据不足无法作答"))
        min_teacher_f1 = float(filter_cfg.get("min_teacher_f1", 0.0))
        require_teacher_format = bool(filter_cfg.get("require_teacher_format_valid", True))
        require_evidence_sufficient = bool(filter_cfg.get("require_evidence_sufficient", True))

        for index, row in enumerate(rows):
            counters["total"] += 1
            question = row_question(row)
            gold_answers = row_gold_answers(row)
            messages = row_prompt_messages(row)
            evidence_steps: list[dict[str, Any]] = []
            actor_final = ""
            actor_answer = ""
            skip_reason = ""
            actor_elapsed = 0.0
            try:
                for turn in range(max_assistant_turns):
                    actor_started = time.perf_counter()
                    actor_text = _call_chat_content(
                        endpoint=str(actor_output["endpoint"]),
                        model=str(actor_output["model"]),
                        messages=messages,
                        request_cfg=request_cfg,
                        timeout_s=float(rollout_cfg.get("actor_timeout_seconds", 180)),
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
                    tool_text, docs, search_elapsed = _search(
                        retrieval_url=str(recall_output["retrieval_url"]),
                        query=str(query),
                        top_n=top_n,
                        top_m=top_m,
                        max_doc_chars=max_doc_chars,
                        timeout_s=search_timeout,
                    )
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
                    skip_reason = "max_assistant_turns"
            except Exception as exc:
                counters["actor_errors"] += 1
                skip_reason = f"actor_error:{type(exc).__name__}"

            messages_before = _messages_before_final_answer(messages)
            teacher_answer = ""
            teacher_raw = ""
            teacher_parse_status = ""
            teacher_elapsed = 0.0
            teacher_f1 = 0.0
            actor_f1 = compute_f1(actor_answer, gold_answers)
            evidence_sufficient = False
            chosen = ""
            rejected = _normalize_actor_rejected(actor_final, actor_answer) if actor_final else ""
            if not actor_final:
                counters["skipped_no_finish"] += 1
                skip_reason = skip_reason or "no_actor_answer"
            elif evidence_steps:
                try:
                    teacher_started = time.perf_counter()
                    teacher_raw = _call_chat_content(
                        endpoint=str(teacher_output["endpoint"]),
                        model=str(teacher_output["model"]),
                        messages=build_teacher_messages(question=question, evidence_steps=evidence_steps),
                        request_cfg=teacher_request,
                        timeout_s=float(teacher_request.get("timeout_seconds", 180)),
                    )
                    teacher_elapsed = time.perf_counter() - teacher_started
                    teacher_answer, teacher_parse_status = _extract_teacher_answer(teacher_raw)
                    teacher_f1 = compute_f1(teacher_answer, gold_answers)
                    evidence_sufficient = bool(teacher_answer and teacher_answer != insufficient)
                    chosen = teacher_raw
                except Exception as exc:
                    counters["teacher_errors"] += 1
                    skip_reason = f"teacher_error:{type(exc).__name__}"
            else:
                counters["skipped_no_finish"] += 1
                skip_reason = skip_reason or "no_search_evidence"

            keep = True
            if not actor_final or not evidence_steps:
                keep = False
            if keep and require_teacher_format and teacher_parse_status != "parsed":
                counters["skipped_teacher_format"] += 1
                keep = False
                skip_reason = skip_reason or f"teacher_{teacher_parse_status or 'unparsed'}"
            if keep and require_evidence_sufficient and not evidence_sufficient:
                counters["skipped_evidence_insufficient"] += 1
                keep = False
                skip_reason = skip_reason or "evidence_insufficient"
            if keep and teacher_f1 < min_teacher_f1:
                counters["skipped_teacher_f1"] += 1
                keep = False
                skip_reason = skip_reason or "teacher_f1_below_threshold"

            metadata = {
                "index": index,
                "question": question,
                "gold_answers": gold_answers,
                "actor_checkpoint": actor_checkpoint,
                "hf_actor_checkpoint": hf_actor_checkpoint,
                "actor_answer": actor_answer,
                "actor_f1": actor_f1,
                "teacher_answer": teacher_answer,
                "teacher_f1": teacher_f1,
                "teacher_parse_status": teacher_parse_status,
                "evidence_sufficient": evidence_sufficient,
                "search_count": len(evidence_steps),
                "sub_queries": [item["sub_query"] for item in evidence_steps],
                "actor_elapsed_s": actor_elapsed,
                "teacher_elapsed_s": teacher_elapsed,
                "keep": keep,
                "skip_reason": skip_reason,
            }
            refresh_record = {
                "prompt": messages_before,
                "chosen": chosen,
                "rejected": rejected,
                "messages_before_final_answer": messages_before,
                "evidence_steps": evidence_steps,
                "teacher_raw_content": teacher_raw,
                "metadata": metadata,
            }
            refresh_records.append(refresh_record)
            if keep:
                counters["kept"] += 1
                records.append(refresh_record)

        write_jsonl(refresh_jsonl, refresh_records)
        write_records(dataset_jsonl, records)
        dataset_manifest = stage_dir / "answer_distill_dataset_manifest.json"
        outputs = {
            "status": "completed",
            "backend": "rollout",
            "actor_checkpoint": actor_checkpoint,
            "hf_actor_checkpoint": hf_actor_checkpoint,
            "refresh_jsonl": str(refresh_jsonl),
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_manifest": str(dataset_manifest),
            "sample_count": len(records),
            "refresh_count": len(refresh_records),
            "counters": counters,
            "service_outputs": service_outputs,
            "resource_plan": resource_plan,
        }
        write_sub_stage_manifest(dataset_manifest, sub_stage="answer_distill_dataset", outputs=outputs)
        return outputs
    finally:
        if bool(sub_cfg.get("auto_stop_services", True)):
            manager.stop_all()


def run_answer_refresh_data(
    *,
    config: dict[str, Any],
    spad_cfg: dict[str, Any],
    stage_dir: Path,
    resource_plan: dict[str, Any],
    dry_run: bool,
    actor_checkpoint: str | None,
) -> dict[str, Any]:
    """Run Stage 2. The rollout backend materializes chosen/rejected pairs."""

    sub_cfg = spad_cfg["sub_stages"]["answer_refresh_data"]
    backend = str(sub_cfg.get("backend") or spad_cfg.get("default_backend") or "smoke")
    actor_checkpoint = str(actor_checkpoint or sub_cfg.get("inputs", {}).get("actor_checkpoint") or "")
    stage_dir.mkdir(parents=True, exist_ok=True)
    dataset_jsonl = stage_dir / "answer_distill_pairs.jsonl"
    if dry_run:
        outputs = {
            "status": "planned",
            "backend": backend,
            "actor_checkpoint": actor_checkpoint,
            "dataset_jsonl": str(dataset_jsonl),
            "resource_plan": resource_plan,
        }
        outputs["manifest"] = str(stage_dir / "manifest.json")
        write_sub_stage_manifest(outputs["manifest"], sub_stage="answer_refresh_data", outputs=outputs)
        return outputs

    if backend == "smoke":
        outputs = _run_smoke_backend(
            config=config,
            spad_cfg=spad_cfg,
            sub_cfg=sub_cfg,
            stage_dir=stage_dir,
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
