#!/usr/bin/env python3
"""Merge the frozen new-data annotations and assign a stratified PE split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_BENCHMARK = HERE / "benchmark_newdata_512.jsonl"
ANNOTATIONS = HERE / "manual_annotations_newdata_512.jsonl"
OUTPUT = HERE / "benchmark_newdata_512_ablation.jsonl"
MANIFEST = HERE / "benchmark_newdata_512_ablation.manifest.json"
SEED = 260715
LABEL_TO_STATUS = {
    "S": "supported_answer",
    "I": "insufficient_evidence",
    "A": "ambiguous_evidence",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def stable_key(case_id: str) -> str:
    return hashlib.sha256(f"{SEED}|split|{case_id}".encode("utf-8")).hexdigest()


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def largest_remainder_quotas(counts: Counter[str], total: int) -> dict[str, int]:
    exact = {label: total * count / sum(counts.values()) for label, count in counts.items()}
    quotas = {label: math.floor(value) for label, value in exact.items()}
    remaining = total - sum(quotas.values())
    ranking = sorted(counts, key=lambda label: (-(exact[label] - quotas[label]), label))
    for label in ranking[:remaining]:
        quotas[label] += 1
    return quotas


def assign_splits(rows: list[dict[str, Any]]) -> None:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[str(row["step_layer"])].append(row)
    for layer, layer_rows in sorted(by_layer.items()):
        if len(layer_rows) != 128:
            raise ValueError(f"Expected 128 cases in {layer}, got {len(layer_rows)}")
        label_counts = Counter(str(row["manual_label"]) for row in layer_rows)
        quotas = largest_remainder_quotas(label_counts, total=32)
        for label, quota in quotas.items():
            candidates = sorted(
                (row for row in layer_rows if row["manual_label"] == label),
                key=lambda row: stable_key(str(row["case_id"])),
            )
            for row in candidates[:quota]:
                row["split"] = "holdout"
        for row in layer_rows:
            row.setdefault("split", "dev")


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 512 or len({row["case_id"] for row in rows}) != 512:
        raise ValueError("Ablation benchmark must contain 512 unique case IDs")
    question_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        question_splits[normalize_question(str(row["question"]))].add(str(row["split"]))
    leaked = [question for question, splits in question_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"Question groups leaked across splits: {leaked[:5]}")
    split_counts = Counter(str(row["split"]) for row in rows)
    if split_counts != Counter({"dev": 384, "holdout": 128}):
        raise ValueError(f"Unexpected split counts: {split_counts}")
    layer_split_counts = {
        layer: dict(Counter(row["split"] for row in rows if row["step_layer"] == layer))
        for layer in sorted({str(row["step_layer"]) for row in rows})
    }
    if any(counts != {"dev": 96, "holdout": 32} for counts in layer_split_counts.values()):
        raise ValueError(f"Unexpected layer split counts: {layer_split_counts}")
    split_label_counts = {
        split: dict(Counter(row["manual_label"] for row in rows if row["split"] == split))
        for split in ("dev", "holdout")
    }
    return {
        "case_count": len(rows),
        "question_group_count": len(question_splits),
        "split_counts": dict(split_counts),
        "layer_split_counts": layer_split_counts,
        "split_label_counts": split_label_counts,
        "benchmark_sha256": sha256_json(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-benchmark", type=Path, default=SOURCE_BENCHMARK)
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args()

    source_rows = load_jsonl(args.source_benchmark)
    annotation_rows = load_jsonl(args.annotations)
    annotations = {str(row["case_id"]): row for row in annotation_rows}
    if len(annotations) != len(annotation_rows):
        raise ValueError("Annotation case IDs must be unique")

    rows: list[dict[str, Any]] = []
    for source in source_rows:
        case_id = str(source["case_id"])
        annotation = annotations.get(case_id)
        if annotation is None:
            raise ValueError(f"Missing annotation for {case_id}")
        label = str(annotation["manual_label"])
        if label not in LABEL_TO_STATUS:
            raise ValueError(f"Invalid manual label {label!r} for {case_id}")
        rows.append(
            {
                **source,
                "manual_label": label,
                "manual_status": LABEL_TO_STATUS[label],
                "manual_answer": str(annotation.get("manual_answer") or ""),
                "manual_reason": str(annotation["manual_reason"]),
                "annotator": str(annotation.get("annotator") or ""),
                "annotated_at": str(annotation.get("annotated_at") or ""),
            }
        )
    if set(annotations) != {str(row["case_id"]) for row in source_rows}:
        raise ValueError("Annotation and source benchmark case ID sets differ")

    assign_splits(rows)
    manifest = validate(rows)
    manifest.update(
        {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "split_seed": SEED,
            "split_policy": "32 holdout cases per step layer, label quotas by largest remainder",
            "source_benchmark": args.source_benchmark.name,
            "source_benchmark_sha256": sha256_json(source_rows),
            "annotations": args.annotations.name,
            "annotations_sha256": sha256_json(annotation_rows),
            "output": args.output.name,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
