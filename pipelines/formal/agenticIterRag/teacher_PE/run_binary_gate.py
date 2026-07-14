#!/usr/bin/env python3
"""Run a one-token GLM I/non-I gate with constrained decoding and logprobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from prompt_variants import build_user_prompt
from run_ablation import DEFAULT_ENDPOINTS, HERE, load_cases, write_json_atomic


BINARY_GATE_PROMPTS = {
    "direct": (
        "Classify whether the supplied Search evidence can fully answer the Original question. Return I only when "
        "a required fact or relation is missing. Return N when at least one complete answer exists, including when "
        "multiple answers make it ambiguous. Use only the evidence."
    ),
    "exact_relation": (
        "Perform a binary evidence-entailment check for the Original question. Check its target entity, exact "
        "predicate, explicit scope, and every multi-hop bridge. Return I if any required element is unsupported. "
        "Return N if the passages establish one or more complete answers. Related entities or predicates and "
        "outside knowledge do not count; direct textual entailment and simple reading of stated values do count."
    ),
    "missing_fact": (
        "Decide whether there is a concrete missing fact that prevents answering the Original question from Search "
        "evidence. Return I only if you can identify such an absent required fact or bridge. Return N if no concrete "
        "gap remains and at least one answer is entailed, even if the question is ambiguous. Do not require exact "
        "wording, but do not invent aliases, locations, dates, identities, predicates, or world knowledge."
    ),
    "candidate_existential": (
        "Determine whether the Search evidence supports at least one complete candidate answer to the Original "
        "question. Return N when candidate count is one or more; return I when candidate count is zero. A complete "
        "candidate must satisfy the exact entity, predicate, scope, and all bridges. Multiple incompatible candidates "
        "are still N for this binary gate."
    ),
    "gold_support": (
        "The Reference gold answer is an untrusted hypothesis, not evidence. Return N only when Search evidence "
        "establishes a Reference gold answer as the answer to the Original question, allowing a clearly equivalent "
        "alias. Return I when that exact gold relation is unsupported. A mention of the gold without the requested "
        "predicate and bridges is insufficient."
    ),
}


def canonical_hash(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def binary_user_prompt(case: dict[str, Any], include_gold: bool) -> str:
    prompt = build_user_prompt(case, include_gold=include_gold)
    prompt = prompt.rsplit("\n   Now output the final result directly.", 1)[0]
    return prompt + "\n\n   Return exactly one uppercase choice: I or N."


def extract_probability(response: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
    choice = response.get("choices", [{}])[0]
    token_info = (choice.get("logprobs") or {}).get("content") or []
    if not token_info:
        raise ValueError("Response has no token logprobs")
    candidates = {
        item["token"]: float(item["logprob"])
        for item in token_info[0].get("top_logprobs") or []
        if item.get("token") in {"I", "N"}
    }
    if set(candidates) != {"I", "N"}:
        raise ValueError(f"I/N logprobs missing: {candidates}")
    maximum = max(candidates.values())
    exp_i = math.exp(candidates["I"] - maximum)
    exp_n = math.exp(candidates["N"] - maximum)
    return str(choice.get("message", {}).get("content") or ""), exp_i / (exp_i + exp_n), candidates


def binary_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for row in rows:
        expected = row["manual_label"] == "I"
        predicted = row["p_i"] >= threshold
        tp += int(expected and predicted)
        fp += int(not expected and predicted)
        fn += int(expected and not predicted)
        tn += int(not expected and not predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "case_count": len(rows),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / len(rows),
    }


def choose_threshold(rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    values = sorted({float(row["p_i"]) for row in rows})
    candidates = [0.0, 1.0]
    candidates.extend((left + right) / 2 for left, right in zip(values, values[1:]))
    scored = [(binary_metrics(rows, threshold), threshold) for threshold in candidates]
    metrics, threshold = max(
        scored,
        key=lambda item: (
            min(item[0]["precision"], item[0]["recall"]),
            item[0]["f1"],
            item[0]["accuracy"],
        ),
    )
    return threshold, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(BINARY_GATE_PROMPTS), required=True)
    parser.add_argument("--endpoints", nargs="+", default=DEFAULT_ENDPOINTS)
    parser.add_argument("--model", default="GLM-4.7-Flash")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--inflight-per-endpoint", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=HERE / "cache_binary_gate")
    parser.add_argument("--draft-predictions", nargs="*", type=Path, default=[])
    args = parser.parse_args()

    cases = load_cases("all")
    include_gold = args.variant == "gold_support"
    system_prompt = BINARY_GATE_PROMPTS[args.variant]
    draft_sources: list[dict[str, dict[str, Any]]] = []
    for path in args.draft_predictions:
        with path.open("r", encoding="utf-8") as handle:
            draft_sources.append({row["case_id"]: row for row in map(json.loads, handle)})
    endpoint_slots: queue.Queue[str] = queue.Queue()
    for endpoint in args.endpoints:
        for _ in range(args.inflight_per_endpoint):
            endpoint_slots.put(endpoint)
    cache_dir = args.cache_dir / args.variant
    lock = threading.Lock()
    progress = 0
    started = datetime.now().astimezone().isoformat()
    wall_start = time.perf_counter()

    def run_case(index: int, case: dict[str, Any]) -> dict[str, Any]:
        nonlocal progress
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": binary_user_prompt(case, include_gold)},
        ]
        if draft_sources:
            lines = ["", "Untrusted draft judgments for this same case; verify them against the evidence:"]
            for draft_index, source in enumerate(draft_sources, start=1):
                draft = source.get(case["case_id"])
                if draft is None:
                    raise ValueError(f"Draft {draft_index} has no case {case['case_id']}")
                lines.append(
                    f"Draft {draft_index}: status={draft.get('predicted_status') or draft.get('predicted_label')}; "
                    f"answer={str(draft.get('answer') or '')[:160]}; "
                    f"reason={str(draft.get('reason') or '')[:500]}"
                )
            messages[0]["content"] += "\n".join(lines)
        payload = {
            "model": args.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": 5,
            "structured_outputs": {"choice": ["I", "N"]},
            "chat_template_kwargs": {"enable_thinking": False},
        }
        request_hash = canonical_hash(payload)
        cache_path = cache_dir / f"{case['case_id']}__{request_hash[:16]}.json"
        cached = cache_path.exists()
        if cached:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            endpoint = endpoint_slots.get()
            request_start = time.perf_counter()
            try:
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    api_response = json.loads(response.read().decode("utf-8"))
                record = {
                    "endpoint": endpoint,
                    "elapsed_s": time.perf_counter() - request_start,
                    "api_response": api_response,
                }
                write_json_atomic(cache_path, record)
            finally:
                endpoint_slots.put(endpoint)
        emitted, p_i, logprobs = extract_probability(record["api_response"])
        with lock:
            progress += 1
            if progress % 50 == 0 or progress == len(cases):
                print(f"progress={progress}/{len(cases)}", flush=True)
        return {
            "index": index,
            "case_id": case["case_id"],
            "split": case["split"],
            "question": case["question"],
            "gold_answers": case["gold_answers"],
            "manual_label": case["manual_label"],
            "variant": args.variant,
            "messages": messages,
            "emitted": emitted,
            "p_i": p_i,
            "logprobs": logprobs,
            "endpoint": record.get("endpoint") or "",
            "elapsed_s": float(record.get("elapsed_s") or 0.0),
            "cached": cached,
            "usage": record["api_response"].get("usage") or {},
        }

    workers = min(args.max_workers, len(args.endpoints) * args.inflight_per_endpoint)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="binary-gate") as executor:
        futures = [executor.submit(run_case, index, case) for index, case in enumerate(cases)]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["index"])

    dev = [row for row in rows if row["split"] == "dev"]
    holdout = [row for row in rows if row["split"] == "holdout"]
    threshold, dev_metrics = choose_threshold(dev)
    holdout_metrics = binary_metrics(holdout, threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {
        "variant": args.variant,
        "include_gold": include_gold,
        "draft_predictions": [str(path) for path in args.draft_predictions],
        "prompt_sha256": canonical_hash(system_prompt),
        "threshold_selected_on": "dev",
        "threshold": threshold,
        "dev": dev_metrics,
        "holdout": holdout_metrics,
        "wall_elapsed_s": time.perf_counter() - wall_start,
        "avg_request_elapsed_s": sum(row["elapsed_s"] for row in rows) / len(rows),
        "started_at": started,
        "finished_at": datetime.now().astimezone().isoformat(),
    }
    write_json_atomic(args.output_dir / "metrics.json", result)
    (args.output_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
