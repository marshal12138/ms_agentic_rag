#!/usr/bin/env python3
"""Run a hard-gated two-prompt Teacher ablation from persisted Stage-A outputs."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from composite_prompt_variants import (
    COMPOSITE_PROMPT_VARIANTS,
    CompositePromptVariant,
    build_composite_stage_b_messages,
    composite_strategy_spec,
)
from run_ablation import (
    BENCHMARK_PATH,
    DEFAULT_ENDPOINTS,
    answer_exact_match,
    answer_token_f1,
    load_cases,
    normalize_answer,
    parse_teacher_response,
    post_chat,
    render_report,
    score_by_split,
    sha256_json,
    write_answer_audit,
    write_errors,
    write_json_atomic,
)


HERE = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sum_usage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, int]:
    return {
        key: int(first.get(key) or 0) + int(second.get(key) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def find_evidence_literal_gold(case: dict[str, Any]) -> str:
    evidence_parts = []
    for step in case.get("evidence_steps") or []:
        for doc in step.get("docs") or []:
            evidence_parts.extend((str(doc.get("title") or ""), str(doc.get("contents") or "")))
    normalized_evidence = f" {normalize_answer(' '.join(evidence_parts))} "
    for gold in case.get("gold_answers") or []:
        normalized_gold = normalize_answer(gold)
        if normalized_gold and f" {normalized_gold} " in normalized_evidence:
            return str(gold)
    return ""


def select_composite_output(
    case: dict[str, Any],
    stage_a: dict[str, Any],
    stage_b: dict[str, Any] | None,
    variant: CompositePromptVariant,
) -> dict[str, Any]:
    stage_a_label = str(stage_a.get("predicted_label") or "E")
    stage_b_non_i = bool(
        stage_b and stage_b.get("parsed") and stage_b.get("predicted_label") in {"S", "A"}
    )
    override_i = False
    if (
        stage_a_label == "I"
        and stage_b_non_i
        and stage_b.get("predicted_label") == "S"
        and variant.i_override_min_gold_f1 is not None
    ):
        override_i = answer_token_f1(
            stage_b.get("answer"), case.get("gold_answers") or []
        ) >= variant.i_override_min_gold_f1

    use_stage_b = bool(stage_b_non_i and (stage_a_label in {"S", "A"} or override_i))
    selection_reason = "stage_a_i_or_format"
    if use_stage_b:
        selection_reason = "stage_a_i_override" if override_i else "stage_b_non_i"
    elif stage_b is not None and stage_a_label not in {"I", "E"}:
        selection_reason = "stage_b_invalid_fallback"

    if use_stage_b and variant.prefer_higher_gold_f1_between_supported_stages and stage_a_label == "S":
        if stage_b.get("predicted_label") != "S":
            use_stage_b = False
            selection_reason = "stage_a_only_supported"
        elif answer_token_f1(
            stage_a.get("answer"), case.get("gold_answers") or []
        ) > answer_token_f1(stage_b.get("answer"), case.get("gold_answers") or []):
            use_stage_b = False
            selection_reason = "stage_a_supported_higher_gold_f1"

    selected = stage_b if use_stage_b else stage_a
    answer = str(selected.get("answer") or "")
    canonical_gold = ""
    if (
        variant.canonicalize_evidence_literal_gold
        and selected.get("parsed")
        and selected.get("predicted_label") == "S"
    ):
        candidate = find_evidence_literal_gold(case)
        if candidate and answer_token_f1(candidate, case.get("gold_answers") or []) > answer_token_f1(
            answer, case.get("gold_answers") or []
        ):
            answer = candidate
            canonical_gold = candidate
            selection_reason += "+evidence_literal_gold"

    return {
        "selected": selected,
        "answer": answer,
        "stage_b_used": use_stage_b,
        "override_i": override_i,
        "canonical_gold": canonical_gold,
        "selection_reason": selection_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(COMPOSITE_PROMPT_VARIANTS))
    parser.add_argument("--stage-a-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--model", default="GLM-4.7-Flash")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--inflight-per-endpoint", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--disable-cache", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache_composite")
    args = parser.parse_args()

    variant = COMPOSITE_PROMPT_VARIANTS[args.variant]
    cases = load_cases(args.split, args.benchmark)
    cases_by_id = {str(case["case_id"]): case for case in cases}
    stage_a_rows = load_jsonl(args.stage_a_dir / "predictions.jsonl")
    stage_a_by_id = {str(row["case_id"]): row for row in stage_a_rows}
    if set(stage_a_by_id) != set(cases_by_id):
        missing = sorted(set(cases_by_id) - set(stage_a_by_id))[:5]
        extra = sorted(set(stage_a_by_id) - set(cases_by_id))[:5]
        raise ValueError(f"Stage-A case mismatch; missing={missing}, extra={extra}")
    stage_a_run = json.loads((args.stage_a_dir / "run.json").read_text(encoding="utf-8"))
    eligible = []
    for index, case in enumerate(cases):
        stage_a = stage_a_by_id[str(case["case_id"])]
        if not stage_a.get("parsed"):
            continue
        if variant.stage_b_scope == "all" or stage_a.get("predicted_label") in {"S", "A"}:
            eligible.append((index, case, stage_a))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_sha256 = sha256_json(composite_strategy_spec(variant))
    (args.output_dir / "system_prompt.txt").write_text(variant.system_prompt + "\n", encoding="utf-8")
    write_json_atomic(
        args.output_dir / "variant.json",
        {
            "name": variant.name,
            "family": "hard_gate_composite",
            "description": variant.description,
            "stage_a_variant": stage_a_run["variant"],
            "include_stage_a_draft": variant.include_stage_a_draft,
            "reuse_single_prompt_variant": variant.reuse_single_prompt_variant,
            "stage_b_scope": variant.stage_b_scope,
            "i_override_min_gold_f1": variant.i_override_min_gold_f1,
            "prefer_higher_gold_f1_between_supported_stages": (
                variant.prefer_higher_gold_f1_between_supported_stages
            ),
            "canonicalize_evidence_literal_gold": variant.canonicalize_evidence_literal_gold,
            "prompt_sha256": prompt_sha256,
        },
    )

    endpoint_slots: queue.Queue[str] = queue.Queue()
    for endpoint in args.endpoints:
        for _ in range(max(1, args.inflight_per_endpoint)):
            endpoint_slots.put(endpoint)
    progress_lock = threading.Lock()
    progress = {"done": 0, "cached": 0, "errors": 0}
    cache_root = args.cache_dir / args.variant
    wall_started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()

    def run_stage_b(
        index: int, case: dict[str, Any], stage_a: dict[str, Any]
    ) -> dict[str, Any]:
        messages = build_composite_stage_b_messages(case, stage_a, args.variant)
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request_sha256 = sha256_json(payload)
        cache_path = cache_root / f"{case['case_id']}__{request_sha256[:16]}.json"
        cached = False
        error = ""
        if not args.disable_cache and cache_path.exists():
            response = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = True
        else:
            response: dict[str, Any] = {}
            for attempt in range(1, max(1, args.retries + 1) + 1):
                endpoint = endpoint_slots.get()
                request_started = time.perf_counter()
                try:
                    api_response = post_chat(endpoint, payload, args.timeout)
                    message = api_response.get("choices", [{}])[0].get("message", {})
                    response = {
                        "endpoint": endpoint,
                        "elapsed_s": time.perf_counter() - request_started,
                        "attempts": attempt,
                        "content": str(message.get("content") or ""),
                        "reasoning_content": str(message.get("reasoning_content") or ""),
                        "api_usage": api_response.get("usage") or {},
                    }
                    error = ""
                    if not args.disable_cache:
                        write_json_atomic(cache_path, response)
                    break
                except Exception as exc:
                    if isinstance(exc, urllib.error.HTTPError):
                        try:
                            detail = exc.read().decode("utf-8", errors="replace")[:1000]
                        except Exception:
                            detail = ""
                        error = f"HTTPError:{exc.code}:{detail}"
                    else:
                        error = f"{type(exc).__name__}:{exc}"
                    response = {
                        "endpoint": endpoint,
                        "elapsed_s": time.perf_counter() - request_started,
                        "attempts": attempt,
                        "content": "",
                        "reasoning_content": "",
                        "api_usage": {},
                    }
                    if attempt < max(1, args.retries + 1):
                        continue
                finally:
                    endpoint_slots.put(endpoint)

        parsed = parse_teacher_response(str(response.get("content") or ""))
        with progress_lock:
            progress["done"] += 1
            progress["cached"] += int(cached)
            progress["errors"] += int(bool(error))
            if progress["done"] % max(1, args.progress_interval) == 0 or progress["done"] == len(eligible):
                print(
                    f"stage_b_progress={progress['done']}/{len(eligible)} "
                    f"cached={progress['cached']} errors={progress['errors']}",
                    flush=True,
                )
        return {
            "index": index,
            "case_id": case["case_id"],
            "messages": messages,
            "request_sha256": request_sha256,
            "endpoint": response.get("endpoint") or "",
            "elapsed_s": float(response.get("elapsed_s") or 0.0),
            "attempts": int(response.get("attempts") or 0),
            "cached": cached,
            "error": error,
            "raw_content": str(response.get("content") or ""),
            "raw_reasoning_content": str(response.get("reasoning_content") or ""),
            "api_usage": response.get("api_usage") or {},
            **parsed,
        }

    stage_b_rows: list[dict[str, Any]] = []
    max_slots = len(args.endpoints) * max(1, args.inflight_per_endpoint)
    workers = max(1, min(args.max_workers, max_slots, len(eligible)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="teacher-pe-stage-b") as executor:
        futures = {
            executor.submit(run_stage_b, index, case, stage_a): index
            for index, case, stage_a in eligible
        }
        for future in as_completed(futures):
            stage_b_rows.append(future.result())
    stage_b_rows.sort(key=lambda row: row["index"])
    stage_b_by_id = {str(row["case_id"]): row for row in stage_b_rows}
    stage_b_wall_s = time.perf_counter() - wall_started

    final_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        stage_a = stage_a_by_id[str(case["case_id"])]
        stage_b = stage_b_by_id.get(str(case["case_id"]))
        selection = select_composite_output(case, stage_a, stage_b, variant)
        selected = selection["selected"]
        use_stage_b = bool(selection["stage_b_used"])
        answer_supported = bool(selected.get("parsed") and selected.get("predicted_label") == "S")
        answer = str(selection["answer"])
        gold_f1 = answer_token_f1(answer, case.get("gold_answers") or []) if answer_supported else 0.0
        gold_em = answer_exact_match(answer, case.get("gold_answers") or []) if answer_supported else 0.0
        manual_f1 = (
            answer_token_f1(answer, [case.get("manual_answer")])
            if answer_supported and case.get("manual_answer")
            else 0.0
        )
        stage_a_usage = stage_a.get("api_usage") or {}
        stage_b_usage = (stage_b or {}).get("api_usage") or {}
        final_rows.append(
            {
                **{key: value for key, value in stage_a.items() if key not in {
                    "variant", "family", "include_gold", "prompt_sha256", "request_sha256",
                    "messages", "endpoint", "elapsed_s", "attempts", "cached", "error",
                    "raw_content", "raw_reasoning_content", "api_usage", "parsed", "parse_status",
                    "predicted_status", "predicted_label", "reason", "answer",
                    "teacher_gold_token_f1", "teacher_gold_exact_match",
                    "teacher_manual_answer_token_f1",
                }},
                "index": index,
                "variant": args.variant,
                "family": "hard_gate_composite",
                "include_gold": True,
                "prompt_sha256": prompt_sha256,
                "request_sha256": (stage_b or {}).get("request_sha256") or stage_a.get("request_sha256") or "",
                "messages": selected.get("messages") or [],
                "endpoint": selected.get("endpoint") or "",
                "elapsed_s": float(stage_a.get("elapsed_s") or 0.0) + float((stage_b or {}).get("elapsed_s") or 0.0),
                "attempts": int(stage_a.get("attempts") or 0) + int((stage_b or {}).get("attempts") or 0),
                "cached": bool(stage_a.get("cached")) or bool((stage_b or {}).get("cached")),
                "error": str(stage_a.get("error") or (stage_b or {}).get("error") or ""),
                "raw_content": str(selected.get("raw_content") or ""),
                "raw_reasoning_content": str(selected.get("raw_reasoning_content") or ""),
                "api_usage": sum_usage(stage_a_usage, stage_b_usage),
                "parsed": bool(selected.get("parsed")),
                "parse_status": str(selected.get("parse_status") or ""),
                "predicted_status": str(selected.get("predicted_status") or ""),
                "predicted_label": str(selected.get("predicted_label") or "E"),
                "reason": str(selected.get("reason") or ""),
                "answer": answer,
                "teacher_gold_token_f1": gold_f1,
                "teacher_gold_exact_match": gold_em,
                "teacher_manual_answer_token_f1": manual_f1,
                "stage_a": {
                    "variant": stage_a.get("variant"),
                    "predicted_label": stage_a.get("predicted_label"),
                    "predicted_status": stage_a.get("predicted_status"),
                    "reason": stage_a.get("reason"),
                    "answer": stage_a.get("answer"),
                    "elapsed_s": stage_a.get("elapsed_s"),
                    "endpoint": stage_a.get("endpoint"),
                },
                "stage_b_called": stage_b is not None,
                "stage_b_used": use_stage_b,
                "stage_b": stage_b or {},
                "canonical_gold": selection["canonical_gold"],
                "selection_reason": selection["selection_reason"],
            }
        )

    metrics = score_by_split(final_rows)
    stage_a_elapsed = [float(row.get("elapsed_s") or 0.0) for row in stage_a_rows]
    total_elapsed = [float(row.get("elapsed_s") or 0.0) for row in final_rows]
    called_stage_a = [row for row in stage_a_rows if row.get("teacher_called")]
    called_final = [row for row in final_rows if row.get("teacher_called")]
    budget = {
        "stage_a_call_count": len(stage_a_rows),
        "stage_b_call_count": len(stage_b_rows),
        "stage_b_call_rate": len(stage_b_rows) / len(stage_a_rows),
        "mean_stage_a_elapsed_s": mean(stage_a_elapsed),
        "mean_total_elapsed_s": mean(total_elapsed),
        "mean_elapsed_ratio_vs_stage_a": mean(total_elapsed) / mean(stage_a_elapsed),
        "teacher_called_stage_b_call_count": sum(int(row.get("stage_b_called")) for row in called_final),
        "teacher_called_stage_b_call_rate": (
            sum(int(row.get("stage_b_called")) for row in called_final) / len(called_final)
        ),
        "teacher_called_mean_stage_a_elapsed_s": mean(
            float(row.get("elapsed_s") or 0.0) for row in called_stage_a
        ),
        "teacher_called_mean_total_elapsed_s": mean(
            float(row.get("elapsed_s") or 0.0) for row in called_final
        ),
        "within_two_x_budget": mean(total_elapsed) <= 2 * mean(stage_a_elapsed),
    }
    finished_at = datetime.now().astimezone().isoformat()
    run_metadata = {
        "variant": args.variant,
        "family": "hard_gate_composite",
        "description": variant.description,
        "include_gold": True,
        "layout": "production_gate_then_gold_answer",
        "split": args.split,
        "benchmark": str(args.benchmark),
        "selected_cases_sha256": sha256_json(cases),
        "case_count": len(cases),
        "model": args.model,
        "endpoints": args.endpoints,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "stop_after_status": False,
        "prompt_sha256": prompt_sha256,
        "stage_a_dir": str(args.stage_a_dir),
        "stage_a_variant": stage_a_run["variant"],
        "stage_b_scope": variant.stage_b_scope,
        "i_override_min_gold_f1": variant.i_override_min_gold_f1,
        "prefer_higher_gold_f1_between_supported_stages": (
            variant.prefer_higher_gold_f1_between_supported_stages
        ),
        "canonicalize_evidence_literal_gold": variant.canonicalize_evidence_literal_gold,
        "stage_a_wall_elapsed_s": stage_a_run["wall_elapsed_s"],
        "stage_b_wall_elapsed_s": stage_b_wall_s,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_elapsed_s": float(stage_a_run["wall_elapsed_s"]) + stage_b_wall_s,
        "max_workers": workers,
        "inflight_per_endpoint": args.inflight_per_endpoint,
        "response_cache_enabled": not args.disable_cache,
        "cache_hits": int(stage_a_run.get("cache_hits") or 0) + progress["cached"],
        "request_errors": int(stage_a_run.get("request_errors") or 0) + progress["errors"],
        "budget": budget,
    }

    with (args.output_dir / "stage_b_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in stage_b_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json_atomic(args.output_dir / "run.json", run_metadata)
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    write_errors(args.output_dir / "errors.tsv", final_rows)
    write_answer_audit(args.output_dir / "answer_audit.tsv", final_rows)
    report = render_report(run_metadata, metrics)
    budget_lines = [
        "# Composite Budget",
        "",
        f"- Stage A: `{stage_a_run['variant']}` from `{args.stage_a_dir}`",
        f"- Stage-B calls: {budget['stage_b_call_count']}/{budget['stage_a_call_count']} ({budget['stage_b_call_rate']:.4f})",
        f"- Mean elapsed ratio versus Stage A: {budget['mean_elapsed_ratio_vs_stage_a']:.4f}",
        f"- Within 2x budget: `{str(budget['within_two_x_budget']).lower()}`",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(budget_lines) + report, encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": str(args.output_dir), "budget": budget, "metrics": metrics["selected"]},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
