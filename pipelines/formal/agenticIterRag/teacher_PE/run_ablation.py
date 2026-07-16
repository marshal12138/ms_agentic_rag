#!/usr/bin/env python3
"""Run and score a GLM-4.7 SPAD teacher prompt ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import queue
import re
import string
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_variants import PROMPT_VARIANTS, build_messages


HERE = Path(__file__).resolve().parent
BENCHMARK_PATH = HERE / "benchmark_237.jsonl"
DEFAULT_ENDPOINTS = [f"http://127.0.0.1:{port}/v1/chat/completions" for port in range(8067, 8071)]
STATUS_TO_LABEL = {
    "supported_answer": "S",
    "insufficient_evidence": "I",
    "ambiguous_evidence": "A",
}
LABELS = ("S", "I", "A")
FULL_PARSE_RE = re.compile(
    r"<reason>(.*?)</reason>\s*<status>(.*?)</status>\s*<answer>(.*?)</answer>",
    re.DOTALL,
)
STATUS_PARSE_RE = re.compile(r"<reason>(.*?)</reason>\s*<status>(.*?)</status>", re.DOTALL)
STATUS_STOPPED_RE = re.compile(r"<reason>(.*?)</reason>\s*<status>([^<\s]+)\s*$", re.DOTALL)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_cases(split: str, benchmark_path: Path = BENCHMARK_PATH) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with benchmark_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            case = json.loads(line)
            if split == "all" or case["split"] == split:
                cases.append(case)
    if not cases:
        raise ValueError(f"No benchmark cases selected for split={split!r}")
    return cases


def parse_teacher_response(content: str) -> dict[str, Any]:
    full_matches = list(FULL_PARSE_RE.finditer(content))
    if full_matches:
        reason, status, answer = (item.strip() for item in full_matches[-1].groups())
        parse_status = "parsed_full"
    else:
        status_matches = list(STATUS_PARSE_RE.finditer(content))
        if status_matches:
            reason, status = (item.strip() for item in status_matches[-1].groups())
            answer = ""
            parse_status = "parsed_through_status"
        else:
            stopped_matches = list(STATUS_STOPPED_RE.finditer(content))
            if stopped_matches:
                reason, status = (item.strip() for item in stopped_matches[-1].groups())
                answer = ""
                parse_status = "parsed_status_stop_without_close"
            else:
                return {
                    "parsed": False,
                    "parse_status": "missing_reason_status_tags",
                    "predicted_status": "",
                    "predicted_label": "E",
                    "reason": "",
                    "answer": "",
                }
    if status not in STATUS_TO_LABEL:
        return {
            "parsed": False,
            "parse_status": "invalid_status",
            "predicted_status": status,
            "predicted_label": "E",
            "reason": reason,
            "answer": answer,
        }
    if full_matches and (not answer or answer in {"...", "…"}):
        return {
            "parsed": False,
            "parse_status": "empty_or_placeholder_answer",
            "predicted_status": status,
            "predicted_label": "E",
            "reason": reason,
            "answer": answer,
        }
    return {
        "parsed": True,
        "parse_status": parse_status,
        "predicted_status": status,
        "predicted_label": STATUS_TO_LABEL[status],
        "reason": reason,
        "answer": answer,
    }


def normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_exact_match(prediction: Any, gold_answers: list[Any]) -> float:
    normalized = normalize_answer(prediction)
    if not normalized:
        return 0.0
    return float(any(normalized == normalize_answer(answer) for answer in gold_answers))


def answer_token_f1(prediction: Any, gold_answers: list[Any]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0
    best = 0.0
    for answer in gold_answers:
        answer_tokens = normalize_answer(answer).split()
        if not answer_tokens:
            continue
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        same = sum(common.values())
        if not same:
            continue
        precision = same / len(prediction_tokens)
        recall = same / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def safe_ratio(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def score_predictions(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {label: {pred: 0 for pred in (*LABELS, "E")} for label in LABELS}
    for row in predictions:
        confusion[row["manual_label"]][row["predicted_label"]] += 1

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[manual][label] for manual in LABELS if manual != label)
        fn = sum(confusion[label][pred] for pred in (*LABELS, "E") if pred != label)
        precision = safe_ratio(tp, tp + fp)
        recall = safe_ratio(tp, tp + fn)
        f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total = len(predictions)
    exact = sum(confusion[label][label] for label in LABELS)
    parsed = sum(int(row["parsed"]) for row in predictions)
    i_tp = confusion["I"]["I"]
    i_fp = confusion["S"]["I"] + confusion["A"]["I"]
    i_fn_as_s = confusion["I"]["S"]
    i_fn_as_a = confusion["I"]["A"]
    i_fn_format = confusion["I"]["E"]
    i_fn = i_fn_as_s + i_fn_as_a + i_fn_format
    i_tn = confusion["S"]["S"] + confusion["S"]["A"] + confusion["A"]["S"] + confusion["A"]["A"]
    tolerated_sa = confusion["S"]["A"] + confusion["A"]["S"]
    involved_i_errors = i_fp + i_fn
    reason_lengths = [len(row.get("reason") or "") for row in predictions if row.get("parsed")]
    supported_answer_lengths = [
        len(row.get("answer") or "")
        for row in predictions
        if row.get("parsed") and row.get("predicted_label") == "S"
    ]
    completion_tokens = [
        int((row.get("api_usage") or {}).get("completion_tokens") or 0)
        for row in predictions
        if int((row.get("api_usage") or {}).get("completion_tokens") or 0) > 0
    ]
    manual_supported = [row for row in predictions if row["manual_label"] == "S"]
    answered_manual_supported = [
        row
        for row in manual_supported
        if row.get("parsed") and row.get("predicted_label") == "S"
    ]
    supported_gold_f1 = [
        answer_token_f1(row.get("answer"), row.get("gold_answers") or [])
        if row.get("parsed") and row.get("predicted_label") == "S"
        else 0.0
        for row in manual_supported
    ]
    supported_gold_em = [
        answer_exact_match(row.get("answer"), row.get("gold_answers") or [])
        if row.get("parsed") and row.get("predicted_label") == "S"
        else 0.0
        for row in manual_supported
    ]
    answered_gold_f1 = [
        answer_token_f1(row.get("answer"), row.get("gold_answers") or [])
        for row in answered_manual_supported
    ]
    answered_gold_em = [
        answer_exact_match(row.get("answer"), row.get("gold_answers") or [])
        for row in answered_manual_supported
    ]
    manual_answer_f1 = [
        answer_token_f1(row.get("answer"), [row.get("manual_answer")])
        if row.get("parsed") and row.get("predicted_label") == "S"
        else 0.0
        for row in manual_supported
    ]
    i_f1 = safe_ratio(2 * i_tp, 2 * i_tp + i_fp + i_fn)
    gold_f1_coverage = safe_ratio(sum(supported_gold_f1), len(manual_supported))

    return {
        "case_count": total,
        "parse_count": parsed,
        "parse_rate": safe_ratio(parsed, total),
        "accuracy": safe_ratio(exact, total),
        "macro_f1": sum(f1_values) / len(f1_values),
        "confusion_matrix": confusion,
        "per_class": per_class,
        "i_binary": {
            "tp": i_tp,
            "fp": i_fp,
            "fn": i_fn,
            "tn": i_tn,
            "missed_i_as_s": i_fn_as_s,
            "missed_i_as_a": i_fn_as_a,
            "missed_i_format_error": i_fn_format,
            "precision": safe_ratio(i_tp, i_tp + i_fp),
            "recall": safe_ratio(i_tp, i_tp + i_fn),
            "f1": i_f1,
            "accuracy": safe_ratio(i_tp + i_tn, total),
        },
        "tolerated_sa_confusion": tolerated_sa,
        "involved_i_errors": involved_i_errors,
        "format_error_count": total - parsed,
        "answer_gold": {
            "manual_supported_count": len(manual_supported),
            "answered_manual_supported_count": len(answered_manual_supported),
            "answered_manual_supported_rate": safe_ratio(
                len(answered_manual_supported), len(manual_supported)
            ),
            "token_f1_coverage": gold_f1_coverage,
            "exact_match_coverage": safe_ratio(sum(supported_gold_em), len(manual_supported)),
            "conditional_token_f1": safe_ratio(
                sum(answered_gold_f1), len(answered_manual_supported)
            ),
            "conditional_exact_match": safe_ratio(
                sum(answered_gold_em), len(answered_manual_supported)
            ),
            "manual_answer_token_f1_coverage": safe_ratio(
                sum(manual_answer_f1), len(manual_supported)
            ),
        },
        "equal_weight_objective": 0.5 * i_f1 + 0.5 * gold_f1_coverage,
        "output_length": {
            "avg_reason_chars": safe_ratio(sum(reason_lengths), len(reason_lengths)),
            "avg_supported_answer_chars": safe_ratio(
                sum(supported_answer_lengths), len(supported_answer_lengths)
            ),
            "max_supported_answer_chars": max(supported_answer_lengths, default=0),
            "avg_completion_tokens": safe_ratio(sum(completion_tokens), len(completion_tokens)),
        },
    }


def score_by_split(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"selected": score_predictions(predictions)}
    called = [row for row in predictions if row.get("teacher_called")]
    controls = [row for row in predictions if not row.get("teacher_called")]
    if called:
        result["teacher_called"] = score_predictions(called)
    if controls:
        result["teacher_not_called_control"] = score_predictions(controls)
    result["by_step_layer"] = {
        layer: score_predictions([row for row in predictions if row.get("step_layer") == layer])
        for layer in sorted({str(row.get("step_layer") or "") for row in predictions})
        if layer
    }
    for split in ("dev", "holdout"):
        subset = [row for row in predictions if row["split"] == split]
        if subset:
            result[split] = score_predictions(subset)
    return result


def post_chat(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def render_report(run_metadata: dict[str, Any], metrics: dict[str, Any]) -> str:
    selected = metrics["selected"]
    i_metrics = selected["i_binary"]
    lengths = selected["output_length"]
    answer_metrics = selected["answer_gold"]
    lines = [
        f"# Teacher Prompt Ablation: {run_metadata['variant']}",
        "",
        f"- Family: `{run_metadata['family']}`",
        f"- Evaluated split: `{run_metadata['split']}`",
        f"- Cases: {selected['case_count']}",
        f"- Prompt SHA256: `{run_metadata['prompt_sha256']}`",
        f"- Started: `{run_metadata['started_at']}`",
        f"- Finished: `{run_metadata['finished_at']}`",
        "",
        "## Main Metrics",
        "",
        "| Accuracy | Macro-F1 | Parse rate | I precision | I recall | I F1 | I binary accuracy | I-related errors | S/A confusion |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {selected['accuracy']:.4f} | {selected['macro_f1']:.4f} | {selected['parse_rate']:.4f} | "
            f"{i_metrics['precision']:.4f} | {i_metrics['recall']:.4f} | {i_metrics['f1']:.4f} | "
            f"{i_metrics['accuracy']:.4f} | {selected['involved_i_errors']} | "
            f"{selected['tolerated_sa_confusion']} |"
        ),
        "",
        "## Equal-Weight Objective",
        "",
        "The selection objective is `0.5 * I F1 + 0.5 * gold token-F1 coverage on manual-S cases`. "
        "A manual-S case predicted as non-S or failing parse contributes zero answer score.",
        "",
        "| Equal objective | I F1 | Gold token-F1 coverage | Gold EM coverage | Answered manual-S | Conditional gold token-F1 | Manual-answer token-F1 coverage |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| {selected['equal_weight_objective']:.4f} | {i_metrics['f1']:.4f} | "
            f"{answer_metrics['token_f1_coverage']:.4f} | "
            f"{answer_metrics['exact_match_coverage']:.4f} | "
            f"{answer_metrics['answered_manual_supported_count']}/{answer_metrics['manual_supported_count']} | "
            f"{answer_metrics['conditional_token_f1']:.4f} | "
            f"{answer_metrics['manual_answer_token_f1_coverage']:.4f} |"
        ),
        "",
        "## Confusion Matrix",
        "",
        "Rows are manual labels; columns are model predictions.",
        "",
        "| Manual \\ Pred | S | I | A | Format error |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    confusion = selected["confusion_matrix"]
    for label in LABELS:
        lines.append(
            f"| {label} | {confusion[label]['S']} | {confusion[label]['I']} | "
            f"{confusion[label]['A']} | {confusion[label]['E']} |"
        )
    lines.extend(
        [
            "",
        "## I Error Breakdown",
            "",
            f"- False I (manual S/A -> I): {i_metrics['fp']}",
            f"- Missed I as S: {i_metrics['missed_i_as_s']}",
            f"- Missed I as A: {i_metrics['missed_i_as_a']}",
            f"- Missed I due to format error: {i_metrics['missed_i_format_error']}",
            "",
            "## Output Length",
            "",
            f"- Average reason characters: {lengths['avg_reason_chars']:.1f}",
            f"- Average supported-answer characters: {lengths['avg_supported_answer_chars']:.1f}",
            f"- Maximum supported-answer characters: {lengths['max_supported_answer_chars']}",
            f"- Average completion tokens: {lengths['avg_completion_tokens']:.1f}",
            "",
        ]
    )
    slice_rows = []
    for slice_name in ("teacher_called", "teacher_not_called_control"):
        if slice_name not in metrics:
            continue
        slice_metrics = metrics[slice_name]
        slice_rows.append(
            f"| {slice_name} | {slice_metrics['case_count']} | "
            f"{slice_metrics['i_binary']['precision']:.4f} | "
            f"{slice_metrics['i_binary']['recall']:.4f} | "
            f"{slice_metrics['i_binary']['f1']:.4f} | "
            f"{slice_metrics['answer_gold']['token_f1_coverage']:.4f} | "
            f"{slice_metrics['equal_weight_objective']:.4f} |"
        )
    for layer, slice_metrics in metrics.get("by_step_layer", {}).items():
        slice_rows.append(
            f"| {layer} | {slice_metrics['case_count']} | "
            f"{slice_metrics['i_binary']['precision']:.4f} | "
            f"{slice_metrics['i_binary']['recall']:.4f} | "
            f"{slice_metrics['i_binary']['f1']:.4f} | "
            f"{slice_metrics['answer_gold']['token_f1_coverage']:.4f} | "
            f"{slice_metrics['equal_weight_objective']:.4f} |"
        )
    if slice_rows:
        lines.extend(
            [
                "## Operational Slices",
                "",
                "The actual `teacher_called` slice is the primary operational diagnostic; controls and step layers detect distribution shifts.",
                "",
                "| Slice | Cases | I precision | I recall | I F1 | Gold token-F1 coverage | Equal objective |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *slice_rows,
                "",
            ]
        )
    return "\n".join(lines)


def write_errors(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "split",
        "manual_label",
        "predicted_label",
        "error_kind",
        "question",
        "manual_reason",
        "manual_answer",
        "gold_answers",
        "model_reason",
        "model_answer",
        "model_gold_token_f1",
        "endpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in predictions:
            if row["manual_label"] == row["predicted_label"]:
                continue
            if row["manual_label"] in {"S", "A"} and row["predicted_label"] in {"S", "A"}:
                kind = "tolerated_SA_confusion"
            elif row["manual_label"] != "I" and row["predicted_label"] == "I":
                kind = "false_I"
            elif row["manual_label"] == "I":
                kind = f"missed_I_as_{row['predicted_label']}"
            else:
                kind = "other"
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "split": row["split"],
                    "manual_label": row["manual_label"],
                    "predicted_label": row["predicted_label"],
                    "error_kind": kind,
                    "question": row["question"],
                    "manual_reason": row["manual_reason"],
                    "manual_answer": row.get("manual_answer") or "",
                    "gold_answers": json.dumps(row.get("gold_answers") or [], ensure_ascii=False),
                    "model_reason": row["reason"],
                    "model_answer": row["answer"],
                    "model_gold_token_f1": answer_token_f1(
                        row.get("answer"), row.get("gold_answers") or []
                    ),
                    "endpoint": row["endpoint"],
                }
            )


def write_answer_audit(path: Path, predictions: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "split",
        "step_layer",
        "teacher_called",
        "predicted_label",
        "question",
        "gold_answers",
        "manual_answer",
        "model_answer",
        "gold_exact_match",
        "gold_token_f1",
        "manual_answer_token_f1",
        "endpoint",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in predictions:
            if row["manual_label"] != "S":
                continue
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "split": row["split"],
                    "step_layer": row.get("step_layer") or "",
                    "teacher_called": str(bool(row.get("teacher_called"))).lower(),
                    "predicted_label": row["predicted_label"],
                    "question": row["question"],
                    "gold_answers": json.dumps(row.get("gold_answers") or [], ensure_ascii=False),
                    "manual_answer": row.get("manual_answer") or "",
                    "model_answer": row.get("answer") or "",
                    "gold_exact_match": row["teacher_gold_exact_match"],
                    "gold_token_f1": row["teacher_gold_token_f1"],
                    "manual_answer_token_f1": row["teacher_manual_answer_token_f1"],
                    "endpoint": row["endpoint"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(PROMPT_VARIANTS))
    parser.add_argument("--split", choices=("all", "dev", "holdout"), default="dev")
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--model", default="GLM-4.7-Flash")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--inflight-per-endpoint", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache")
    parser.add_argument(
        "--disable-cache",
        action="store_true",
        help="Always call the model and do not read or write the response cache.",
    )
    parser.add_argument(
        "--draft-predictions",
        nargs="*",
        type=Path,
        default=[],
        help="Prediction JSONL files whose untrusted outputs are appended to a meta-arbiter system prompt.",
    )
    args = parser.parse_args()

    variant = PROMPT_VARIANTS[args.variant]
    draft_sources: list[dict[str, Any]] = []
    for draft_path in args.draft_predictions:
        rows_by_case: dict[str, dict[str, Any]] = {}
        with draft_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows_by_case[str(row["case_id"])] = row
        draft_sources.append({"path": str(draft_path), "rows": rows_by_case})
    if variant.family == "multi_draft_meta" and not draft_sources:
        parser.error("multi_draft_meta variants require --draft-predictions")
    cases = load_cases(args.split, args.benchmark)
    if args.limit > 0:
        cases = cases[: args.limit]
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = args.output_dir or HERE / "results" / f"{timestamp}_{args.variant}_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_sha256 = sha256_json(
        {
            "name": variant.name,
            "system_prompt": variant.system_prompt,
            "include_gold": variant.include_gold,
            "layout": variant.layout,
        }
    )
    (output_dir / "system_prompt.txt").write_text(variant.system_prompt + "\n", encoding="utf-8")
    write_json_atomic(
        output_dir / "variant.json",
        {
            "name": variant.name,
            "family": variant.family,
            "description": variant.description,
            "include_gold": variant.include_gold,
            "layout": variant.layout,
            "prompt_sha256": prompt_sha256,
        },
    )

    endpoint_slots: queue.Queue[str] = queue.Queue()
    for endpoint in args.endpoints:
        for _ in range(max(1, args.inflight_per_endpoint)):
            endpoint_slots.put(endpoint)
    cache_root = args.cache_dir / args.variant
    progress_lock = threading.Lock()
    progress = {"done": 0, "cached": 0, "errors": 0}
    started_at = datetime.now().astimezone().isoformat()
    wall_started = time.perf_counter()

    def run_case(index: int, case: dict[str, Any]) -> dict[str, Any]:
        messages = build_messages(case, args.variant)
        if draft_sources:
            draft_lines = ["", "UNTRUSTED DRAFT JUDGMENTS FOR THIS CASE:"]
            for draft_index, source in enumerate(draft_sources, start=1):
                draft = source["rows"].get(case["case_id"])
                if draft is None:
                    raise ValueError(
                        f"Draft source {source['path']} has no case_id={case['case_id']}"
                    )
                draft_lines.extend(
                    [
                        f"Draft {draft_index} ({draft.get('variant') or 'unknown'}):",
                        f"  status: {draft.get('predicted_status') or 'format_error'}",
                        f"  answer: {str(draft.get('answer') or '')[:200]}",
                        f"  reason: {str(draft.get('reason') or '')[:600]}",
                    ]
                )
            messages[0]["content"] += "\n".join(draft_lines)
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"enable_thinking": args.enable_thinking},
        }
        if args.seed is not None:
            payload["seed"] = args.seed
        request_sha256 = sha256_json(payload)
        cache_path = cache_root / f"{case['case_id']}__{request_sha256[:16]}.json"
        cached = False
        error = ""
        if not args.disable_cache and cache_path.exists():
            response_record = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = True
        else:
            response_record: dict[str, Any] = {}
            last_error = ""
            for attempt in range(1, max(1, args.retries + 1) + 1):
                endpoint = endpoint_slots.get()
                request_started = time.perf_counter()
                try:
                    raw_api_response = post_chat(endpoint, payload, args.timeout)
                    content = str(
                        raw_api_response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    reasoning_content = str(
                        raw_api_response.get("choices", [{}])[0]
                        .get("message", {})
                        .get("reasoning_content", "")
                        or ""
                    )
                    response_record = {
                        "request_sha256": request_sha256,
                        "endpoint": endpoint,
                        "elapsed_s": time.perf_counter() - request_started,
                        "attempts": attempt,
                        "content": content,
                        "reasoning_content": reasoning_content,
                        "api_usage": raw_api_response.get("usage") or {},
                    }
                    if not args.disable_cache:
                        write_json_atomic(cache_path, response_record)
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
                        error = last_error
                        response_record = {
                            "request_sha256": request_sha256,
                            "endpoint": endpoint,
                            "elapsed_s": time.perf_counter() - request_started,
                            "attempts": attempt,
                            "content": "",
                            "api_usage": {},
                            "error": error,
                        }
                finally:
                    endpoint_slots.put(endpoint)

        parsed = parse_teacher_response(str(response_record.get("content") or ""))
        answer_is_supported = bool(
            parsed.get("parsed") and parsed.get("predicted_label") == "S"
        )
        teacher_gold_token_f1 = (
            answer_token_f1(parsed.get("answer"), case.get("gold_answers") or [])
            if answer_is_supported
            else 0.0
        )
        teacher_gold_exact_match = (
            answer_exact_match(parsed.get("answer"), case.get("gold_answers") or [])
            if answer_is_supported
            else 0.0
        )
        teacher_manual_answer_token_f1 = (
            answer_token_f1(parsed.get("answer"), [case.get("manual_answer")])
            if answer_is_supported and case.get("manual_answer")
            else 0.0
        )
        result = {
            "index": index,
            "case_id": case["case_id"],
            "uid": case["uid"],
            "step": case.get("step"),
            "step_layer": case.get("step_layer") or "",
            "split": case["split"],
            "question": case["question"],
            "gold_answers": case["gold_answers"],
            "manual_label": case["manual_label"],
            "manual_status": case["manual_status"],
            "manual_reason": case["manual_reason"],
            "manual_answer": case.get("manual_answer") or "",
            "teacher_called": bool(case.get("teacher_called")),
            "historical_teacher_label": case["historical_teacher_label"],
            "variant": args.variant,
            "family": variant.family,
            "include_gold": variant.include_gold,
            "prompt_sha256": prompt_sha256,
            "request_sha256": request_sha256,
            "messages": messages,
            "endpoint": response_record.get("endpoint") or "",
            "elapsed_s": float(response_record.get("elapsed_s") or 0.0),
            "attempts": int(response_record.get("attempts") or 0),
            "cached": cached,
            "error": error or str(response_record.get("error") or ""),
            "raw_content": str(response_record.get("content") or ""),
            "raw_reasoning_content": str(response_record.get("reasoning_content") or ""),
            "api_usage": response_record.get("api_usage") or {},
            "teacher_gold_token_f1": teacher_gold_token_f1,
            "teacher_gold_exact_match": teacher_gold_exact_match,
            "teacher_manual_answer_token_f1": teacher_manual_answer_token_f1,
            **parsed,
        }
        with progress_lock:
            progress["done"] += 1
            progress["cached"] += int(cached)
            progress["errors"] += int(bool(result["error"]))
            if progress["done"] % max(1, args.progress_interval) == 0 or progress["done"] == len(cases):
                print(
                    f"progress={progress['done']}/{len(cases)} cached={progress['cached']} "
                    f"errors={progress['errors']}",
                    flush=True,
                )
        return result

    max_slots = len(args.endpoints) * max(1, args.inflight_per_endpoint)
    max_workers = max(1, min(args.max_workers, max_slots, len(cases)))
    predictions: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="teacher-pe") as executor:
        futures = {executor.submit(run_case, index, case): index for index, case in enumerate(cases)}
        for future in as_completed(futures):
            predictions.append(future.result())
    predictions.sort(key=lambda row: row["index"])

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = score_by_split(predictions)
    finished_at = datetime.now().astimezone().isoformat()
    run_metadata = {
        "variant": args.variant,
        "family": variant.family,
        "description": variant.description,
        "include_gold": variant.include_gold,
        "layout": variant.layout,
        "split": args.split,
        "benchmark": str(args.benchmark),
        "selected_cases_sha256": sha256_json(cases),
        "case_count": len(cases),
        "model": args.model,
        "endpoints": args.endpoints,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
        "stop_after_status": False,
        "prompt_sha256": prompt_sha256,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_elapsed_s": time.perf_counter() - wall_started,
        "max_workers": max_workers,
        "inflight_per_endpoint": args.inflight_per_endpoint,
        "response_cache_enabled": not args.disable_cache,
        "cache_hits": progress["cached"],
        "request_errors": progress["errors"],
        "draft_predictions": [source["path"] for source in draft_sources],
    }
    write_json_atomic(output_dir / "run.json", run_metadata)
    write_json_atomic(output_dir / "metrics.json", metrics)
    write_errors(output_dir / "errors.tsv", predictions)
    write_answer_audit(output_dir / "answer_audit.tsv", predictions)
    (output_dir / "report.md").write_text(render_report(run_metadata, metrics), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "metrics": metrics["selected"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
