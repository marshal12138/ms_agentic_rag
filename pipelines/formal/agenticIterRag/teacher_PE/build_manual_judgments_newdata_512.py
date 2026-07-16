#!/usr/bin/env python3
"""Validate manual annotations for the new-data sample and build the final TSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BENCHMARK_PATH = HERE / "benchmark_newdata_512.jsonl"
BENCHMARK_MANIFEST_PATH = HERE / "benchmark_newdata_512.manifest.json"
ANNOTATIONS_PATH = HERE / "manual_annotations_newdata_512.jsonl"
OUTPUT_PATH = HERE / "manual_judgments_newdata_512.tsv"
MANIFEST_PATH = HERE / "manual_judgments_newdata_512.manifest.json"
LABEL_TO_STATUS = {
    "S": "supported_answer",
    "I": "insufficient_evidence",
    "A": "ambiguous_evidence",
}
OUTPUT_FIELDS = [
    "audit_id",
    "case_id",
    "step",
    "step_layer",
    "group_uid",
    "rollout_index",
    "data_source",
    "source_file",
    "source_line",
    "question",
    "gold_answers_reference_only",
    "actor_answer",
    "actor_em",
    "actor_f1",
    "search_count",
    "duplicate_query_count",
    "teacher_called",
    "group_all_em_zero",
    "historical_teacher_status",
    "historical_teacher_answer",
    "manual_label",
    "manual_status",
    "manual_answer",
    "manual_reason",
    "annotator",
    "annotated_at",
    "evidence_sha256",
    "source_row_sha256",
]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for source_line, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{source_line}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{source_line}")
            rows.append(value)
    return rows


def validate_annotations(
    benchmark: list[dict[str, Any]], annotations: list[dict[str, Any]], allow_incomplete: bool
) -> dict[str, dict[str, Any]]:
    benchmark_ids = {str(row["case_id"]) for row in benchmark}
    if len(benchmark_ids) != len(benchmark):
        raise ValueError("Benchmark case_id values are not unique")
    by_case: dict[str, dict[str, Any]] = {}
    for index, annotation in enumerate(annotations, start=1):
        case_id = str(annotation.get("case_id") or "")
        if case_id not in benchmark_ids:
            raise ValueError(f"Annotation {index} has unknown case_id={case_id!r}")
        if case_id in by_case:
            raise ValueError(f"Duplicate annotation for case_id={case_id}")
        label = str(annotation.get("manual_label") or "")
        if label not in LABEL_TO_STATUS:
            raise ValueError(f"Annotation {case_id} has invalid manual_label={label!r}")
        if not str(annotation.get("manual_reason") or "").strip():
            raise ValueError(f"Annotation {case_id} has an empty manual_reason")
        if label == "S" and not str(annotation.get("manual_answer") or "").strip():
            raise ValueError(f"Supported annotation {case_id} has an empty manual_answer")
        by_case[case_id] = annotation
    missing = benchmark_ids - set(by_case)
    if missing and not allow_incomplete:
        raise ValueError(f"Missing {len(missing)} annotations; examples: {sorted(missing)[:5]}")
    return by_case


def build_rows(
    benchmark: list[dict[str, Any]], annotations: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, case in enumerate(benchmark, start=1):
        annotation = annotations.get(str(case["case_id"]))
        if annotation is None:
            continue
        label = str(annotation["manual_label"])
        rows.append(
            {
                "audit_id": f"newdata512-{index:04d}",
                "case_id": case["case_id"],
                "step": case["step"],
                "step_layer": case["step_layer"],
                "group_uid": case["group_uid"],
                "rollout_index": case["rollout_index"],
                "data_source": case["data_source"],
                "source_file": case["source_file"],
                "source_line": case["source_line"],
                "question": case["question"],
                "gold_answers_reference_only": json.dumps(
                    case["gold_answers"], ensure_ascii=False
                ),
                "actor_answer": case["actor_answer"],
                "actor_em": case["actor_em"],
                "actor_f1": case["actor_f1"],
                "search_count": case["search_count"],
                "duplicate_query_count": case["duplicate_query_count"],
                "teacher_called": str(case["teacher_called"]).lower(),
                "group_all_em_zero": str(case["group_all_em_zero"]).lower(),
                "historical_teacher_status": case["historical_teacher_status"],
                "historical_teacher_answer": case["historical_teacher_answer"],
                "manual_label": label,
                "manual_status": LABEL_TO_STATUS[label],
                "manual_answer": str(annotation.get("manual_answer") or ""),
                "manual_reason": str(annotation["manual_reason"]),
                "annotator": str(annotation.get("annotator") or "Codex"),
                "annotated_at": str(annotation.get("annotated_at") or ""),
                "evidence_sha256": case["evidence_sha256"],
                "source_row_sha256": case["source_row_sha256"],
            }
        )
    return rows


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=BENCHMARK_PATH)
    parser.add_argument("--benchmark-manifest", type=Path, default=BENCHMARK_MANIFEST_PATH)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    benchmark = load_jsonl(args.benchmark)
    annotations = load_jsonl(args.annotations)
    by_case = validate_annotations(benchmark, annotations, args.allow_incomplete)
    rows = build_rows(benchmark, by_case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    benchmark_manifest = json.loads(args.benchmark_manifest.read_text(encoding="utf-8"))
    label_counts = Counter(row["manual_label"] for row in rows)
    layer_label_counts = {
        layer: dict(Counter(row["manual_label"] for row in rows if row["step_layer"] == layer))
        for layer in benchmark_manifest["step_layers"]
    }
    historical_agreement_count = sum(
        row["manual_status"] == row["historical_teacher_status"]
        for row in rows
        if row["historical_teacher_status"]
    )
    historical_comparable_count = sum(bool(row["historical_teacher_status"]) for row in rows)
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "benchmark_file": args.benchmark.name,
        "benchmark_sha256": benchmark_manifest["benchmark_sha256"],
        "annotations_file": args.annotations.name,
        "annotations_sha256": sha256_text(canonical_json(annotations)),
        "expected_case_count": len(benchmark),
        "annotated_case_count": len(rows),
        "complete": len(rows) == len(benchmark),
        "label_counts": dict(label_counts),
        "step_layer_label_counts": layer_label_counts,
        "historical_teacher_comparable_count": historical_comparable_count,
        "historical_teacher_agreement_count": historical_agreement_count,
        "historical_teacher_agreement_rate": (
            historical_agreement_count / historical_comparable_count
            if historical_comparable_count
            else 0.0
        ),
        "output_file": args.output.name,
    }
    write_json_atomic(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
