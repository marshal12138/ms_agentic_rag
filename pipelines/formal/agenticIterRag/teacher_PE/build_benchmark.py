#!/usr/bin/env python3
"""Freeze the 237 manually judged SPAD teacher cases as a prompt benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
JUDGMENTS_PATH = HERE / "manual_judgments_237.tsv"
BENCHMARK_PATH = HERE / "benchmark_237.jsonl"
MANIFEST_PATH = HERE / "benchmark_237.manifest.json"

QUESTION_RE = re.compile(r"Question: (.*?)\n\nassistant", re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
DOC_HEADER_RE = re.compile(r'^\[(\d+)\]\s+(?:"(.*)"|(.*))$')


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def assign_split(question: str) -> str:
    """Assign whole question groups; the fixed split yields 178 dev / 59 holdout rows."""

    digest = hashlib.sha256(normalize_question(question).encode("utf-8")).hexdigest()
    return "holdout" if int(digest[:8], 16) % 10 < 3 else "dev"


def parse_question(raw_input: str) -> str:
    match = QUESTION_RE.search(raw_input)
    if not match:
        raise ValueError("Cannot extract Original question")
    return match.group(1).strip()


def parse_docs(raw_response: str) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    current_number: int | None = None
    current_title = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_number, current_title, current_lines
        if current_number is None:
            return
        docs.append(
            {
                "title": current_title,
                "contents": "\n".join(current_lines).strip(),
            }
        )
        current_number = None
        current_title = ""
        current_lines = []

    for line in raw_response.splitlines():
        match = DOC_HEADER_RE.match(line.strip())
        if match:
            flush()
            current_number = int(match.group(1))
            current_title = str(match.group(2) if match.group(2) is not None else match.group(3) or "").strip()
            expected_number = len(docs) + 1
            if current_number != expected_number:
                raise ValueError(f"Non-contiguous document number: got {current_number}, expected {expected_number}")
        elif current_number is not None:
            current_lines.append(line)
        elif line.strip():
            raise ValueError(f"Text before first document header: {line[:120]!r}")
    flush()
    if not docs:
        raise ValueError("No documents parsed from tool response")
    if len(docs) > 5:
        raise ValueError(f"Expected at most 5 visible documents, got {len(docs)}")
    return docs


def parse_evidence_steps(raw_output: str) -> list[dict[str, Any]]:
    raw_calls = TOOL_CALL_RE.findall(raw_output)
    raw_responses = TOOL_RESPONSE_RE.findall(raw_output)
    if len(raw_calls) != len(raw_responses):
        raise ValueError(f"tool call/response mismatch: {len(raw_calls)} != {len(raw_responses)}")
    if not raw_calls:
        raise ValueError("No search rounds found")

    evidence_steps: list[dict[str, Any]] = []
    for round_index, (raw_call, raw_response) in enumerate(zip(raw_calls, raw_responses), start=1):
        call = json.loads(raw_call)
        if call.get("name") != "search":
            raise ValueError(f"Unexpected tool in round {round_index}: {call.get('name')!r}")
        query = str((call.get("arguments") or {}).get("query") or "").strip()
        if not query:
            raise ValueError(f"Empty search query in round {round_index}")
        evidence_steps.append(
            {
                "round_index": round_index,
                "sub_query": query,
                "docs": parse_docs(raw_response),
            }
        )
    return evidence_steps


def load_source_record(source_file: str, source_line: int, cache: dict[Path, list[str]]) -> dict[str, Any]:
    path = REPO_ROOT / source_file
    if path not in cache:
        cache[path] = path.read_text(encoding="utf-8").splitlines()
    lines = cache[path]
    if source_line < 1 or source_line > len(lines):
        raise ValueError(f"Source line out of range: {path}:{source_line}")
    return json.loads(lines[source_line - 1])


def build_cases() -> list[dict[str, Any]]:
    source_cache: dict[Path, list[str]] = {}
    cases: list[dict[str, Any]] = []
    with JUDGMENTS_PATH.open("r", encoding="utf-8", newline="") as handle:
        for judgment in csv.DictReader(handle, delimiter="\t"):
            source_line = int(judgment["source_line"])
            source = load_source_record(judgment["source_file"], source_line, source_cache)
            if str(source.get("uid") or "") != judgment["uid"]:
                raise ValueError(f"UID mismatch for {judgment['audit_id']}")
            question = parse_question(str(source.get("input") or ""))
            if question != judgment["question"]:
                raise ValueError(f"Question mismatch for {judgment['audit_id']}")
            gold_answers = [str(item) for item in (source.get("gts") or {}).get("target") or []]
            judgment_gold = [str(item) for item in json.loads(judgment["gold_targets_reference_only"])]
            if gold_answers != judgment_gold:
                raise ValueError(f"Gold mismatch for {judgment['audit_id']}")
            evidence_steps = parse_evidence_steps(str(source.get("output") or ""))
            cases.append(
                {
                    "case_id": judgment["audit_id"],
                    "uid": judgment["uid"],
                    "source_file": judgment["source_file"],
                    "source_line": source_line,
                    "split": assign_split(question),
                    "question_group": normalize_question(question),
                    "question": question,
                    "gold_answers": gold_answers,
                    "manual_label": judgment["manual_label"],
                    "manual_status": judgment["manual_status"],
                    "manual_reason": judgment["manual_reason"],
                    "historical_teacher_label": judgment["teacher_label"],
                    "historical_teacher_status": judgment["teacher_bucket"],
                    "historical_teacher_answer": judgment["teacher_answer"],
                    "evidence_steps": evidence_steps,
                    "evidence_sha256": sha256_json(evidence_steps),
                }
            )
    return cases


def validate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cases) != 237:
        raise AssertionError(f"Expected 237 cases, got {len(cases)}")
    if len({case["case_id"] for case in cases}) != 237:
        raise AssertionError("case_id must be unique")
    group_splits: dict[str, set[str]] = {}
    for case in cases:
        group_splits.setdefault(case["question_group"], set()).add(case["split"])
    leaked = [group for group, splits in group_splits.items() if len(splits) != 1]
    if leaked:
        raise AssertionError(f"Question groups leaked across splits: {leaked[:5]}")

    label_counts = Counter(case["manual_label"] for case in cases)
    expected_labels = Counter({"S": 104, "I": 105, "A": 28})
    if label_counts != expected_labels:
        raise AssertionError(f"Unexpected manual labels: {label_counts}")
    split_counts = Counter(case["split"] for case in cases)
    if split_counts != Counter({"dev": 178, "holdout": 59}):
        raise AssertionError(f"Unexpected split counts: {split_counts}")
    split_labels = {
        split: dict(Counter(case["manual_label"] for case in cases if case["split"] == split))
        for split in ("dev", "holdout")
    }
    return {
        "case_count": len(cases),
        "question_group_count": len(group_splits),
        "label_counts": dict(label_counts),
        "split_counts": dict(split_counts),
        "split_label_counts": split_labels,
        "benchmark_sha256": sha256_json(cases),
    }


def main() -> None:
    cases = build_cases()
    manifest = validate_cases(cases)
    with BENCHMARK_PATH.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
