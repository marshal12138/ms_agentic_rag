#!/usr/bin/env python3
"""Rescore frozen AIR evaluation traces with structured answer metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "AgenticIterRag"))

from agentic_iter_rag.metrics.answer_metrics import (  # noqa: E402
    answer_group_metrics,
    groups_from_ground_truth,
)


DEFAULT_DATA = ROOT / (
    "data/AgenticIterRag/structured_answer/260711a_search_r1_512_350/"
    "search_r1_structured.eval.parquet"
)
DEFAULT_OUTPUT = ROOT / (
    "reports/eval/agenticIterRag/260711-search_r1-structured-comparison"
)
RUNS = {
    "base_run1": "260710-qwen17_base_reval3_run1_search_eval_350",
    "base_run2": "260710-qwen17_base_reval3_run2_search_eval_350",
    "base_run3": "260710-qwen17_base_reval3_run3_search_eval_350",
    "legacy_search_r1_run1": (
        "260710-search_r1_original_qwen17_latest_gs8_reval3_run1_search_eval_350"
    ),
    "legacy_search_r1_run2": (
        "260710-search_r1_original_qwen17_latest_gs8_reval3_run2_search_eval_350"
    ),
    "legacy_search_r1_run3": (
        "260710-search_r1_original_qwen17_latest_gs8_reval3_run3_search_eval_350"
    ),
    "structured_search_r1_run1": "260711-search_r1_structured_qwen17_gs8_eval350",
    "structured_search_r1_run2": (
        "260711-search_r1_structured_qwen17_gs8_reval3_run2_search_eval_350"
    ),
    "structured_search_r1_run3": (
        "260711-search_r1_structured_qwen17_gs8_reval3_run3_search_eval_350"
    ),
}
QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*$", re.DOTALL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def mean(records: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(record[key]) for record in records) if records else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [record for record in records if record["structured_eligible"]]
    return {
        "n": len(records),
        "legacy_em": mean(records, "legacy_em"),
        "legacy_f1": mean(records, "legacy_f1"),
        "structured_n": len(eligible),
        "structured_em": mean(eligible, "structured_em"),
        "answer_group_f1": mean(eligible, "answer_group_f1"),
        "answer_group_recall": mean(eligible, "answer_group_recall"),
    }


def grouped_summary(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def aggregate_repeats(run_summaries: dict[str, dict[str, Any]], prefix: str) -> dict[str, Any]:
    selected = [summary for name, summary in run_summaries.items() if name.startswith(prefix)]
    metrics = ("legacy_em", "legacy_f1", "structured_em", "answer_group_f1", "answer_group_recall")
    result: dict[str, Any] = {"run_count": len(selected)}
    for metric in metrics:
        values = [float(summary["overall"][metric]) for summary in selected]
        result[f"{metric}_mean"] = statistics.fmean(values)
        result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Search-R1 Structured Answer Offline Rescore",
        "",
        f"- Dataset: `{summary['data_path']}`",
        f"- Dataset SHA256: `{summary['data_sha256']}`",
        "- Structured denominator excludes rows marked ineligible.",
        "",
        "## Per Run",
        "",
        "| Run | N | Legacy EM | Legacy F1 | Structured N | Structured EM | Group F1 | Group Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in summary["runs"].items():
        item = run["overall"]
        lines.append(
            f"| {name} | {item['n']} | {item['legacy_em']:.4f} | "
            f"{item['legacy_f1']:.4f} | {item['structured_n']} | "
            f"{item['structured_em']:.4f} | {item['answer_group_f1']:.4f} | "
            f"{item['answer_group_recall']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Repeat Aggregate",
            "",
            "| Model | Runs | Legacy EM | Structured EM | Group F1 | Group Recall |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, item in summary["repeat_aggregate"].items():
        lines.append(
            f"| {name} | {item['run_count']} | "
            f"{item['legacy_em_mean']:.4f} +/- {item['legacy_em_std']:.4f} | "
            f"{item['structured_em_mean']:.4f} +/- {item['structured_em_std']:.4f} | "
            f"{item['answer_group_f1_mean']:.4f} +/- {item['answer_group_f1_std']:.4f} | "
            f"{item['answer_group_recall_mean']:.4f} +/- {item['answer_group_recall_std']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(args.data).reset_index(drop=True)
    examples: list[dict[str, Any]] = []
    for index, row in data.iterrows():
        ground_truth = row.reward_model["ground_truth"]
        answers, groups, eligible, semantics = groups_from_ground_truth(ground_truth)
        examples.append(
            {
                "index": int(index),
                "data_source": str(row.data_source),
                "source_id": str(row.extra_info["source_id"]),
                "question": str(row.extra_info["question"]),
                "answers": answers,
                "groups": groups,
                "structured_eligible": eligible,
                "answer_semantics": semantics,
            }
        )

    all_records: list[dict[str, Any]] = []
    run_summaries: dict[str, dict[str, Any]] = {}
    for run_name, task_name in RUNS.items():
        trace_path = ROOT / "log/eval/agenticIterRag" / task_name / "trace/traces.jsonl"
        traces: dict[int, dict[str, Any]] = {}
        with trace_path.open(encoding="utf-8") as handle:
            for line in handle:
                trace = json.loads(line)
                index = int(trace["index"])
                if index in traces:
                    raise ValueError(f"duplicate index {index} in {trace_path}")
                traces[index] = trace
        expected = set(range(len(examples)))
        if set(traces) != expected:
            raise ValueError(f"index mismatch in {trace_path}")

        records: list[dict[str, Any]] = []
        for example in examples:
            trace = traces[example["index"]]
            if str(trace["data_source"]) != example["data_source"]:
                raise ValueError(
                    f"source mismatch at {example['index']}: "
                    f"{trace['data_source']} != {example['data_source']}"
                )
            prompt_content = str(trace["prompt"][-1]["content"])
            question_match = QUESTION_RE.search(prompt_content)
            trace_question = question_match.group(1).strip() if question_match else ""
            if trace_question != example["question"]:
                raise ValueError(
                    f"question mismatch at {example['index']}: "
                    f"{trace_question!r} != {example['question']!r}"
                )
            metrics = answer_group_metrics(
                trace.get("final_answer") or "",
                example["answers"],
                example["groups"],
                structured_eligible=example["structured_eligible"],
            ).to_dict()
            record = {
                "run": run_name,
                "task_name": task_name,
                "index": example["index"],
                "source_id": example["source_id"],
                "data_source": example["data_source"],
                "answer_semantics": example["answer_semantics"],
                "final_answer": str(trace.get("final_answer") or ""),
                **metrics,
            }
            records.append(record)
            all_records.append(record)
        run_summaries[run_name] = {
            "task_name": task_name,
            "trace_path": str(trace_path),
            "trace_sha256": sha256_file(trace_path),
            "overall": summarize(records),
            "by_data_source": grouped_summary(records, "data_source"),
            "by_answer_semantics": grouped_summary(records, "answer_semantics"),
        }

    records_path = args.output_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for record in all_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "version": "search-r1-structured-answer-v1",
        "data_path": str(args.data),
        "data_sha256": sha256_file(args.data),
        "runs": run_summaries,
        "repeat_aggregate": {
            "base": aggregate_repeats(run_summaries, "base_run"),
            "legacy_search_r1": aggregate_repeats(run_summaries, "legacy_search_r1_run"),
            "structured_search_r1": aggregate_repeats(run_summaries, "structured_search_r1_run"),
        },
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = args.output_dir / "report.md"
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    print(report_path)
    print(json.dumps(summary["repeat_aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
