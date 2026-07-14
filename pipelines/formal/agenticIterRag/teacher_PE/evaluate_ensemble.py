#!/usr/bin/env python3
"""Evaluate a frozen vote over persisted teacher predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from run_ablation import score_predictions, write_json_atomic


def load_predictions(path: Path, split: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if split == "all" or row["split"] == split:
                rows[row["case_id"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", type=Path, required=True)
    parser.add_argument("--i-threshold", type=int, required=True)
    parser.add_argument("--split", choices=("dev", "holdout", "all"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    components = [load_predictions(path, args.split) for path in args.predictions]
    if not 1 <= args.i_threshold <= len(components):
        parser.error("--i-threshold must be between 1 and the component count")
    case_ids = list(components[0])
    expected = set(case_ids)
    for path, component in zip(args.predictions, components):
        if set(component) != expected:
            raise ValueError(f"Component case IDs differ: {path}")

    outputs: list[dict[str, Any]] = []
    serial_latencies: list[float] = []
    parallel_latencies: list[float] = []
    total_tokens: list[int] = []
    for index, case_id in enumerate(case_ids):
        drafts = [component[case_id] for component in components]
        i_votes = sum(draft["predicted_label"] == "I" for draft in drafts)
        if i_votes >= args.i_threshold:
            final_label = "I"
        else:
            s_votes = sum(draft["predicted_label"] == "S" for draft in drafts)
            a_votes = sum(draft["predicted_label"] == "A" for draft in drafts)
            final_label = "A" if a_votes > s_votes else "S"
        elapsed = [float(draft.get("elapsed_s") or 0.0) for draft in drafts]
        tokens = sum(
            int((draft.get("api_usage") or {}).get("completion_tokens") or 0) for draft in drafts
        )
        serial_latencies.append(sum(elapsed))
        parallel_latencies.append(max(elapsed, default=0.0))
        total_tokens.append(tokens)
        base = drafts[0]
        outputs.append(
            {
                "index": index,
                "case_id": case_id,
                "split": base["split"],
                "question": base["question"],
                "manual_label": base["manual_label"],
                "predicted_label": final_label,
                "parsed": True,
                "reason": "",
                "answer": "",
                "api_usage": {"completion_tokens": tokens},
                "i_votes": i_votes,
                "component_labels": [draft["predicted_label"] for draft in drafts],
                "component_variants": [draft["variant"] for draft in drafts],
            }
        )

    metrics = score_predictions(outputs)
    cost = {
        "component_count": len(components),
        "avg_serial_request_latency_s": mean(serial_latencies),
        "avg_ideal_parallel_request_latency_s": mean(parallel_latencies),
        "avg_total_completion_tokens": mean(total_tokens),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in outputs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json_atomic(args.output_dir / "metrics.json", metrics)
    write_json_atomic(
        args.output_dir / "run.json",
        {
            "split": args.split,
            "i_threshold": args.i_threshold,
            "prediction_paths": [str(path) for path in args.predictions],
            "cost": cost,
        },
    )
    print(json.dumps({"metrics": metrics, "cost": cost}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
