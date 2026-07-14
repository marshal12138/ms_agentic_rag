#!/usr/bin/env python3
"""Replay Search-R1 rollouts through the best SPAD evidence judge."""

from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import statistics
import string
import threading
import time
import unicodedata
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from build_benchmark import DOC_HEADER_RE, parse_question
from prompt_variants import PROMPT_VARIANTS, build_messages
from run_ablation import (
    DEFAULT_ENDPOINTS,
    parse_teacher_response,
    post_chat,
    sha256_json,
    write_json_atomic,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_ROLLOUT_DIR = (
    REPO_ROOT
    / "log/agenticIterRag"
    / "260710-113003-543853-pipeline-agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal"
    / "outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data"
)
DEFAULT_MANUAL_AUDIT = (
    REPO_ROOT
    / "docs/AgenticIterRag_v1/work_report"
    / "260710-17a_Search-R1零奖励人工审查240样本明细.tsv"
)
VARIANT = "baseline_question_tail_evidence_only_v2"
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
LITERAL_TOKEN_RE = re.compile(r"\w+|[^\s]", re.UNICODE)


def variant_prompt_sha256() -> str:
    variant = PROMPT_VARIANTS[VARIANT]
    return sha256_json(
        {
            "name": variant.name,
            "system_prompt": variant.system_prompt,
            "include_gold": variant.include_gold,
            "layout": variant.layout,
        }
    )


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction: str, answer: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not pred_tokens or not answer_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(answer_tokens)
    same = sum(common.values())
    if not same:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_f1(prediction: str, gold_answers: list[str]) -> float:
    return max((answer_f1(prediction, answer) for answer in gold_answers), default=0.0)


def literal_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in LITERAL_TOKEN_RE.findall(unicodedata.normalize("NFD", str(text)))
    ]


def gold_literal_hit(gold_answers: list[str], evidence_steps: list[dict[str, Any]]) -> bool:
    evidence = "\n".join(
        f"{doc.get('title') or ''}\n{doc.get('contents') or ''}"
        for step in evidence_steps
        for doc in step.get("docs") or []
    )
    evidence_tokens = literal_tokens(evidence)
    for answer in gold_answers:
        answer_tokens = literal_tokens(answer)
        if not answer_tokens:
            continue
        for index in range(len(evidence_tokens) - len(answer_tokens) + 1):
            if evidence_tokens[index : index + len(answer_tokens)] == answer_tokens:
                return True
    return False


def parse_docs_tolerant(raw_response: str) -> tuple[list[dict[str, str]], list[str]]:
    """Parse visible docs while ignoring malformed text before the first header."""

    docs: list[dict[str, str]] = []
    warnings: list[str] = []
    current_number: int | None = None
    current_title = ""
    current_lines: list[str] = []
    ignored_prefix: list[str] = []

    def flush() -> None:
        nonlocal current_number, current_title, current_lines
        if current_number is None:
            return
        docs.append({"title": current_title, "contents": "\n".join(current_lines).strip()})
        current_number = None
        current_title = ""
        current_lines = []

    for line in raw_response.splitlines():
        match = DOC_HEADER_RE.match(line.strip())
        if match:
            flush()
            current_number = int(match.group(1))
            current_title = str(match.group(2) if match.group(2) is not None else match.group(3) or "").strip()
        elif current_number is not None:
            current_lines.append(line)
        elif line.strip():
            ignored_prefix.append(line.strip())
    flush()
    if ignored_prefix:
        warnings.append("ignored_text_before_first_doc")
    if not docs:
        warnings.append("no_docs_in_tool_response")
    if len(docs) > 5:
        warnings.append("more_than_five_visible_docs")
        docs = docs[:5]
    return docs, warnings


def parse_visible_evidence(raw_output: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The selected prompt hides sub-queries, so only persisted tool responses are needed."""

    evidence_steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    for round_index, raw_response in enumerate(TOOL_RESPONSE_RE.findall(raw_output), start=1):
        docs, doc_warnings = parse_docs_tolerant(raw_response)
        warnings.extend(f"round_{round_index}:{warning}" for warning in doc_warnings)
        if docs:
            evidence_steps.append(
                {"round_index": round_index, "sub_query": "", "docs": docs}
            )
    if not evidence_steps:
        warnings.append("no_visible_evidence")
    return evidence_steps, warnings


def load_cases(rollout_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(rollout_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    if [path.stem for path in paths] != [str(index) for index in range(1, 9)]:
        raise ValueError(f"Expected rollout steps 1..8 in {rollout_dir}")

    cases: list[dict[str, Any]] = []
    for path in paths:
        step = int(path.stem)
        with path.open("r", encoding="utf-8") as handle:
            for source_line, line in enumerate(handle, start=1):
                source = json.loads(line)
                evidence_steps, evidence_warnings = parse_visible_evidence(
                    str(source.get("output") or "")
                )
                gold_answers = [
                    str(answer) for answer in (source.get("gts") or {}).get("target") or []
                ]
                actor_answer = str(source.get("search_r1_extracted_answer") or "")
                format_valid = source.get("format_status") == "valid" and bool(actor_answer.strip())
                literal_hit = gold_literal_hit(gold_answers, evidence_steps)
                auto_stratum = (
                    ("hit" if literal_hit else "miss")
                    + ("_valid" if source.get("format_status") == "valid" else "_invalid")
                )
                cases.append(
                    {
                        "index": len(cases),
                        "case_id": f"step{step:02d}-line{source_line:04d}",
                        "step": step,
                        "source_file": str(path.relative_to(REPO_ROOT)),
                        "source_line": source_line,
                        "uid": str(source.get("uid") or ""),
                        "question": parse_question(str(source.get("input") or "")),
                        "gold_answers": gold_answers,
                        "actor_answer": actor_answer,
                        "actor_f1": compute_f1(actor_answer, gold_answers),
                        "actor_format_status": str(source.get("format_status") or ""),
                        "actor_has_valid_answer": format_valid,
                        "original_reward": float(source.get("score") or 0.0),
                        "search_count": int(source.get("search_count") or 0),
                        "duplicate_query_count": int(source.get("duplicate_query_count") or 0),
                        "evidence_steps": evidence_steps,
                        "evidence_sha256": sha256_json(evidence_steps),
                        "evidence_parse_warnings": evidence_warnings,
                        "gold_literal_hit": literal_hit,
                        "auto_stratum": auto_stratum,
                    }
                )
    if len(cases) != 4096:
        raise ValueError(f"Expected 4096 rollouts, got {len(cases)}")
    return cases


def make_result(
    case: dict[str, Any],
    response_record: dict[str, Any],
    messages: list[dict[str, str]],
    request_sha256: str,
) -> dict[str, Any]:
    parsed = parse_teacher_response(str(response_record.get("content") or ""))
    gold_answers = case["gold_answers"]
    teacher_f1 = compute_f1(str(parsed.get("answer") or ""), gold_answers)
    evidence_answer_score = (
        teacher_f1 if parsed.get("predicted_label") == "S" and parsed.get("parsed") else 0.0
    )

    # Stage1 stops at the answer opening, so this score rewards the judge's
    # evidence-grounded short answer rather than actor answer wording.
    if parsed.get("parsed"):
        if parsed.get("predicted_label") == "S":
            search_policy_judge_reward = 0.25 + 0.75 * teacher_f1
        else:
            search_policy_judge_reward = 0.0
    else:
        search_policy_judge_reward = -0.1

    # This diagnostic score also evaluates the complete Search-R1 trajectory.
    if case["actor_has_valid_answer"]:
        trajectory_reward = 0.75 * case["actor_f1"] + 0.25 * evidence_answer_score
    else:
        trajectory_reward = 0.10 * evidence_answer_score

    duplicate_penalty = 0.10 * case["duplicate_query_count"]
    result = {
        key: value for key, value in case.items() if key != "evidence_steps"
    }
    result.update(
        {
            "variant": VARIANT,
            "prompt_sha256": variant_prompt_sha256(),
            "user_prompt_sha256": sha256_json(messages[1]["content"]),
            "request_sha256": request_sha256,
            "endpoint": str(response_record.get("endpoint") or ""),
            "elapsed_s": float(response_record.get("elapsed_s") or 0.0),
            "attempts": int(response_record.get("attempts") or 0),
            "error": str(response_record.get("error") or ""),
            "raw_content": str(response_record.get("content") or ""),
            "raw_reasoning_content": str(response_record.get("reasoning_content") or ""),
            "api_usage": response_record.get("api_usage") or {},
            **parsed,
            "teacher_f1": teacher_f1,
            "evidence_answer_score": evidence_answer_score,
            "search_policy_judge_reward": search_policy_judge_reward,
            "trajectory_reward": trajectory_reward,
            "duplicate_penalty": duplicate_penalty,
            "search_policy_reward_with_duplicate_penalty": max(
                -0.5, search_policy_judge_reward - duplicate_penalty
            ),
            "trajectory_reward_with_duplicate_penalty": max(
                -0.5, trajectory_reward - duplicate_penalty
            ),
        }
    )
    return result


def is_nonconstant(values: list[float]) -> bool:
    return bool(values) and max(values) - min(values) > 1e-9


def distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def group_metrics(predictions: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in predictions:
        groups[(row["step"], row["uid"])].append(float(row[field]))
    if len(groups) != 512 or any(len(values) != 8 for values in groups.values()):
        raise ValueError(f"Unexpected GRPO groups for {field}: {len(groups)}")
    by_step = {}
    for step in range(1, 9):
        step_groups = [values for (row_step, _), values in groups.items() if row_step == step]
        by_step[str(step)] = {
            "group_count": len(step_groups),
            "nonconstant_groups": sum(is_nonconstant(values) for values in step_groups),
        }
    return {
        "group_count": len(groups),
        "nonconstant_groups": sum(is_nonconstant(values) for values in groups.values()),
        "uniform_groups": sum(not is_nonconstant(values) for values in groups.values()),
        "by_step": by_step,
    }


def load_manual_labels(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["审查ID"]] = row
    if len(rows) != 240:
        raise ValueError(f"Expected 240 manual labels, got {len(rows)}")
    return rows


def match_manual_rows(
    predictions: list[dict[str, Any]], manual_path: Path
) -> list[dict[str, Any]]:
    manual = load_manual_labels(manual_path)
    prediction_pool: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        prediction_pool[(row["step"], row["auto_stratum"], row["uid"])].append(row)
    for pool in prediction_pool.values():
        pool.sort(key=lambda row: row["source_line"])

    matched = []
    for audit_id, label in manual.items():
        key = (int(label["step"]), label["自动分层"], label["uid"])
        candidates = prediction_pool.get(key) or []
        if not candidates:
            raise ValueError(f"No replay prediction for manual audit {audit_id}")
        # The manual sampler used setdefault(uid, row), i.e. the first source row.
        row = dict(candidates[0])
        row.update(
            {
                "audit_id": audit_id,
                "manual_evidence_label": label["证据标签"],
                "manual_answer_label": label["答案标签"],
                "manual_note": label["人工审查说明"],
            }
        )
        matched.append(row)
    if len({row["case_id"] for row in matched}) != 240:
        raise ValueError("Manual audit replay matches are not unique")
    return matched


def manual_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_confusion = {
        manual: {predicted: 0 for predicted in ("S", "I", "A", "E")}
        for manual in ("S", "I", "A")
    }
    for row in rows:
        evidence_confusion[row["manual_evidence_label"]][row["predicted_label"]] += 1

    i_tp = evidence_confusion["I"]["I"]
    i_fp = evidence_confusion["S"]["I"] + evidence_confusion["A"]["I"]
    i_fn = sum(evidence_confusion["I"][label] for label in ("S", "A", "E"))
    i_precision = i_tp / (i_tp + i_fp) if i_tp + i_fp else 0.0
    i_recall = i_tp / (i_tp + i_fn) if i_tp + i_fn else 0.0
    i_f1 = 2 * i_precision * i_recall / (i_precision + i_recall) if i_precision + i_recall else 0.0

    answer_by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        answer_by_label[row["manual_answer_label"]].append(row)
    answer_summary = {}
    for label in ("C", "P", "W", "N"):
        subset = answer_by_label[label]
        answer_summary[label] = {
            "count": len(subset),
            "actor_f1": distribution([float(row["actor_f1"]) for row in subset]),
            "trajectory_reward": distribution(
                [float(row["trajectory_reward"]) for row in subset]
            ),
            "search_policy_judge_reward": distribution(
                [float(row["search_policy_judge_reward"]) for row in subset]
            ),
        }
    return {
        "case_count": len(rows),
        "judge_parse_rate": sum(bool(row["parsed"]) for row in rows) / len(rows),
        "evidence_confusion": evidence_confusion,
        "evidence_accuracy": sum(
            row["manual_evidence_label"] == row["predicted_label"] for row in rows
        )
        / len(rows),
        "i_binary": {
            "tp": i_tp,
            "fp": i_fp,
            "fn": i_fn,
            "precision": i_precision,
            "recall": i_recall,
            "f1": i_f1,
        },
        "answer_by_manual_label": answer_summary,
    }


def aggregate_metrics(
    predictions: list[dict[str, Any]], manual_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    fields = [
        "original_reward",
        "actor_f1",
        "teacher_f1",
        "search_policy_judge_reward",
        "trajectory_reward",
        "search_policy_reward_with_duplicate_penalty",
        "trajectory_reward_with_duplicate_penalty",
    ]
    score_metrics = {
        field: {
            "distribution": distribution([float(row[field]) for row in predictions]),
            "groups": group_metrics(predictions, field),
        }
        for field in fields
    }
    by_step = {}
    for step in range(1, 9):
        subset = [row for row in predictions if row["step"] == step]
        by_step[str(step)] = {
            "count": len(subset),
            "judge_labels": dict(Counter(row["predicted_label"] for row in subset)),
            "parse_rate": sum(bool(row["parsed"]) for row in subset) / len(subset),
            "mean_teacher_f1": statistics.mean(float(row["teacher_f1"]) for row in subset),
            "mean_search_policy_judge_reward": statistics.mean(
                float(row["search_policy_judge_reward"]) for row in subset
            ),
            "mean_trajectory_reward": statistics.mean(
                float(row["trajectory_reward"]) for row in subset
            ),
        }
    return {
        "case_count": len(predictions),
        "judge_parse_rate": sum(bool(row["parsed"]) for row in predictions) / len(predictions),
        "judge_labels": dict(Counter(row["predicted_label"] for row in predictions)),
        "request_errors": sum(bool(row["error"]) for row in predictions),
        "evidence_parse_warning_cases": sum(
            bool(row["evidence_parse_warnings"]) for row in predictions
        ),
        "scores": score_metrics,
        "by_step": by_step,
        "manual_240": manual_metrics(manual_rows),
    }


def render_report(run: dict[str, Any], metrics: dict[str, Any]) -> str:
    manual = metrics["manual_240"]
    i_metrics = manual["i_binary"]
    lines = [
        "# Search-R1 Question-tail Evidence-only Reward Replay",
        "",
        f"- Cases: {metrics['case_count']}",
        f"- Variant: `{run['variant']}`",
        f"- Cache hits: {run['cache_hits']}",
        f"- Request errors: {metrics['request_errors']}",
        f"- Judge parse rate: {metrics['judge_parse_rate']:.4f}",
        "",
        "## GRPO Group Signal",
        "",
        "| Score | Nonconstant groups | Uniform groups |",
        "| --- | ---: | ---: |",
    ]
    for field in (
        "original_reward",
        "actor_f1",
        "teacher_f1",
        "search_policy_judge_reward",
        "trajectory_reward",
    ):
        groups = metrics["scores"][field]["groups"]
        lines.append(
            f"| `{field}` | {groups['nonconstant_groups']} | {groups['uniform_groups']} |"
        )
    lines.extend(
        [
            "",
            "## Manual 240 Evidence Validation",
            "",
            f"- Evidence accuracy: {manual['evidence_accuracy']:.4f}",
            f"- I precision: {i_metrics['precision']:.4f}",
            f"- I recall: {i_metrics['recall']:.4f}",
            f"- I F1: {i_metrics['f1']:.4f}",
            "",
            "## Manual Answer-label Score Means",
            "",
            "| Manual answer label | N | Actor F1 | Trajectory reward | Search-policy judge reward |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label in ("C", "P", "W", "N"):
        item = manual["answer_by_manual_label"][label]
        lines.append(
            f"| {label} | {item['count']} | {item['actor_f1']['mean']:.4f} | "
            f"{item['trajectory_reward']['mean']:.4f} | "
            f"{item['search_policy_judge_reward']['mean']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize_existing_result(
    output_dir: Path,
    cases: list[dict[str, Any]],
    manual_audit: Path,
) -> None:
    predictions_path = output_dir / "predictions.jsonl"
    run_path = output_dir / "run.json"
    if not predictions_path.is_file() or not run_path.is_file():
        raise FileNotFoundError(f"Incomplete replay result directory: {output_dir}")
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(predictions) != 4096:
        raise ValueError(f"Expected 4096 persisted predictions, got {len(predictions)}")
    cases_by_id = {case["case_id"]: case for case in cases}
    for row in predictions:
        case = cases_by_id.get(row["case_id"])
        if case is None:
            raise ValueError(f"Persisted prediction has unknown case_id={row['case_id']}")
        for field in (
            "gold_literal_hit",
            "auto_stratum",
            "evidence_parse_warnings",
            "evidence_sha256",
        ):
            row[field] = case[field]
    predictions.sort(key=lambda row: row["index"])
    write_predictions(predictions_path, predictions)

    run_metadata = json.loads(run_path.read_text(encoding="utf-8"))
    manual_rows = match_manual_rows(predictions, manual_audit)
    metrics = aggregate_metrics(predictions, manual_rows)
    write_predictions(output_dir / "manual_240_predictions.jsonl", manual_rows)
    write_json_atomic(output_dir / "metrics.json", metrics)
    (output_dir / "report.md").write_text(
        render_report(run_metadata, metrics), encoding="utf-8"
    )
    print(
        json.dumps(
            {"output_dir": str(output_dir), "run": run_metadata, "metrics": metrics},
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--manual-audit", type=Path, default=DEFAULT_MANUAL_AUDIT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--model", default="GLM-4.7-Flash")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--inflight-per-endpoint", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Rebuild metrics from 4096 persisted predictions without calling the model.",
    )
    args = parser.parse_args()

    if args.postprocess_only:
        if args.limit:
            parser.error("--postprocess-only cannot be combined with --limit")
        cases = load_cases(args.rollout_dir)
        finalize_existing_result(args.output_dir, cases, args.manual_audit)
        return

    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    cases = load_cases(args.rollout_dir)
    if args.limit > 0:
        cases = cases[: args.limit]
    variant = PROMPT_VARIANTS[VARIANT]
    (args.output_dir / "system_prompt.txt").write_text(
        variant.system_prompt + "\n", encoding="utf-8"
    )
    write_json_atomic(
        args.output_dir / "variant.json",
        {
            "name": variant.name,
            "family": variant.family,
            "description": variant.description,
            "include_gold": variant.include_gold,
            "layout": variant.layout,
            "prompt_sha256": variant_prompt_sha256(),
        },
    )

    endpoint_slots: queue.Queue[str] = queue.Queue()
    for endpoint in args.endpoints:
        for _ in range(max(1, args.inflight_per_endpoint)):
            endpoint_slots.put(endpoint)
    progress_lock = threading.Lock()
    progress = {"done": 0, "errors": 0}
    started_at = datetime.now().astimezone().isoformat()
    wall_started = time.perf_counter()

    def run_case(case: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(case, VARIANT)
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request_sha256 = sha256_json(payload)
        response_record: dict[str, Any] = {}
        last_error = ""
        for attempt in range(1, max(1, args.retries + 1) + 1):
            endpoint = endpoint_slots.get()
            request_started = time.perf_counter()
            try:
                raw = post_chat(endpoint, payload, args.timeout)
                response_record = {
                    "request_sha256": request_sha256,
                    "endpoint": endpoint,
                    "elapsed_s": time.perf_counter() - request_started,
                    "attempts": attempt,
                    "content": str(
                        raw.get("choices", [{}])[0].get("message", {}).get("content", "")
                    ),
                    "reasoning_content": str(
                        raw.get("choices", [{}])[0]
                        .get("message", {})
                        .get("reasoning_content", "")
                        or ""
                    ),
                    "api_usage": raw.get("usage") or {},
                    "error": "",
                }
                break
            except Exception as exc:
                if isinstance(exc, urllib.error.HTTPError):
                    try:
                        detail = exc.read().decode("utf-8", errors="replace")[:1000]
                    except Exception:
                        detail = ""
                    last_error = f"HTTPError:{exc.code}:{detail}"
                else:
                    last_error = f"{type(exc).__name__}:{exc}"
                if attempt >= max(1, args.retries + 1):
                    response_record = {
                        "request_sha256": request_sha256,
                        "endpoint": endpoint,
                        "elapsed_s": time.perf_counter() - request_started,
                        "attempts": attempt,
                        "content": "",
                        "reasoning_content": "",
                        "api_usage": {},
                        "error": last_error,
                    }
            finally:
                endpoint_slots.put(endpoint)

        result = make_result(case, response_record, messages, request_sha256)
        with progress_lock:
            progress["done"] += 1
            progress["errors"] += int(bool(result["error"]))
            if progress["done"] % max(1, args.progress_interval) == 0 or progress["done"] == len(cases):
                print(
                    f"progress={progress['done']}/{len(cases)} errors={progress['errors']}",
                    flush=True,
                )
        return result

    max_slots = len(args.endpoints) * max(1, args.inflight_per_endpoint)
    max_workers = max(1, min(args.max_workers, max_slots, len(cases)))
    predictions = []
    partial_path = args.output_dir / "predictions.partial.jsonl"
    with partial_path.open("w", encoding="utf-8") as partial_handle:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="search-r1-replay") as executor:
            futures = {executor.submit(run_case, case): case["index"] for case in cases}
            for future in as_completed(futures):
                row = future.result()
                predictions.append(row)
                partial_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                partial_handle.flush()
    predictions.sort(key=lambda row: row["index"])
    write_predictions(args.output_dir / "predictions.jsonl", predictions)
    partial_path.unlink()

    finished_at = datetime.now().astimezone().isoformat()
    run_metadata = {
        "variant": VARIANT,
        "rollout_dir": str(args.rollout_dir.resolve()),
        "manual_audit": str(args.manual_audit.resolve()),
        "case_count": len(cases),
        "model": args.model,
        "endpoints": args.endpoints,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_elapsed_s": time.perf_counter() - wall_started,
        "max_workers": max_workers,
        "inflight_per_endpoint": args.inflight_per_endpoint,
        "response_cache_enabled": False,
        "cache_hits": 0,
        "request_errors": progress["errors"],
    }
    write_json_atomic(args.output_dir / "run.json", run_metadata)

    if len(cases) == 4096:
        manual_rows = match_manual_rows(predictions, args.manual_audit)
        metrics = aggregate_metrics(predictions, manual_rows)
        write_predictions(args.output_dir / "manual_240_predictions.jsonl", manual_rows)
        write_json_atomic(args.output_dir / "metrics.json", metrics)
        (args.output_dir / "report.md").write_text(
            render_report(run_metadata, metrics), encoding="utf-8"
        )
    else:
        metrics = {
            "case_count": len(cases),
            "judge_parse_rate": sum(bool(row["parsed"]) for row in predictions) / len(predictions),
            "request_errors": sum(bool(row["error"]) for row in predictions),
        }
        write_json_atomic(args.output_dir / "metrics.json", metrics)

    print(json.dumps({"output_dir": str(args.output_dir), "run": run_metadata, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
