#!/usr/bin/env python3
"""Summarize fresh three-repeat and cumulative stability results."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUTPUT = HERE / "REPLICA_STABILITY.md"

HISTORICAL_GROUPS = {
    "question-tail evidence-only": [
        "M2_question_tail_single_replica_all",
        "P1_question_tail_evidence_only_replica0_all",
        "P1_question_tail_evidence_only_replica1_all",
        "P1_question_tail_evidence_only_replica2_all",
        "P1_question_tail_evidence_only_replica3_all",
        "S0_question_tail_control_all",
    ],
    "question-tail with sub_query": [
        "N1_question_tail_v2_replica0_all",
        "N1_question_tail_v2_replica1_all",
        "N1_question_tail_v2_replica2_all",
        "N1_question_tail_v2_replica3_all",
    ],
    "title-free question-tail": [
        "Q2_text_question_tail_all",
        "R1_baseline_text_question_tail_v2_replica0_all",
        "R1_baseline_text_question_tail_v2_replica1_all",
        "R1_baseline_text_question_tail_v2_replica2_all",
        "R1_baseline_text_question_tail_v2_replica3_all",
    ],
    "short focused question-tail": [
        "Q4_focused_question_tail_all",
        "R2_focused_question_tail_evidence_only_v2_replica0_all",
        "R2_focused_question_tail_evidence_only_v2_replica1_all",
        "R2_focused_question_tail_evidence_only_v2_replica2_all",
        "R2_focused_question_tail_evidence_only_v2_replica3_all",
    ],
    "evidence-only without tail": [
        "L1_baseline_evidence_only_v2_all",
        "L6_baseline_evidence_only_v2_single_replica_all",
    ],
}

FRESH_TOP5_GROUPS = {
    "question-tail evidence-only": [
        "T1_top5_qtail_evidence_r1_all",
        "T1_top5_qtail_evidence_r2_fresh_all",
        "T1_top5_qtail_evidence_r3_all",
    ],
    "question-tail with sub_query": [
        "T2_top5_qtail_subquery_r1_all",
        "T2_top5_qtail_subquery_r2_fresh_all",
        "T2_top5_qtail_subquery_r3_all",
    ],
    "title-free question-tail": [
        "T3_top5_text_qtail_r1_all",
        "T3_top5_text_qtail_r2_fresh_all",
        "T3_top5_text_qtail_r3_all",
    ],
    "short focused question-tail": [
        "T4_top5_focused_qtail_r1_all",
        "T4_top5_focused_qtail_r2_fresh_all",
        "T4_top5_focused_qtail_r3_all",
    ],
    "evidence-only without tail": [
        "T5_top5_evidence_only_r1_all",
        "T5_top5_evidence_only_r2_all",
        "T5_top5_evidence_only_r3_all",
    ],
}


def holdout_metrics(result_name: str) -> dict[str, Any]:
    metrics = json.loads((RESULTS / result_name / "metrics.json").read_text(encoding="utf-8"))
    return metrics["holdout"]


def load_run(result_name: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    result_dir = RESULTS / result_name
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    if int(run.get("cache_hits", -1)) != 0:
        raise RuntimeError(f"Fresh repeat used response cache: {result_name}")
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = [
        json.loads(line)
        for line in (result_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    return run, metrics, predictions


def span(values: list[float]) -> str:
    return f"{mean(values):.4f} [{min(values):.4f}, {max(values):.4f}]"


def main() -> None:
    lines = [
        "# Teacher Prompt Replica Stability",
        "",
        "## Fresh Top 5 three-repeat comparison",
        "",
        "Each strategy was run three new times over all 237 cases on three different replicas. No response was "
        "reused: every included run has `cache_hits=0`. Accuracy columns are holdout mean [min, max]. Cost "
        "columns are means over the three full runs.",
        "",
        "| Rank | Strategy | I precision | I recall | I F1 | Parse rate | Wall s / 237 | Mean request s | Avg completion tokens |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    summaries: list[dict[str, Any]] = []
    for group, names in FRESH_TOP5_GROUPS.items():
        loaded = [load_run(name) for name in names]
        runs = [item[0] for item in loaded]
        rows = [item[1]["holdout"] for item in loaded]
        predictions = [prediction for item in loaded for prediction in item[2]]
        precision = [float(row["i_binary"]["precision"]) for row in rows]
        recall = [float(row["i_binary"]["recall"]) for row in rows]
        f1 = [float(row["i_binary"]["f1"]) for row in rows]
        parse = [float(row["parse_rate"]) for row in rows]
        elapsed = [float(row["elapsed_s"]) for row in predictions]
        completion_tokens = [
            float((row.get("api_usage") or {}).get("completion_tokens") or 0)
            for row in predictions
        ]
        summaries.append(
            {
                "group": group,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "parse": parse,
                "wall": mean(float(run["wall_elapsed_s"]) for run in runs),
                "elapsed": mean(elapsed),
                "tokens": mean(completion_tokens),
            }
        )
    summaries.sort(key=lambda row: mean(row["f1"]), reverse=True)
    for rank, row in enumerate(summaries, start=1):
        lines.append(
            f"| {rank} | {row['group']} | {span(row['precision'])} | {span(row['recall'])} | "
            f"{span(row['f1'])} | {span(row['parse'])} | {row['wall']:.2f} | "
            f"{row['elapsed']:.2f} | {row['tokens']:.1f} |"
        )
    leader = summaries[0]
    lines.extend(
        [
            "",
            f"The fresh three-repeat leader is `{leader['group']}` with mean holdout I "
            f"precision/recall/F1 `{mean(leader['precision']):.4f}/{mean(leader['recall']):.4f}/"
            f"{mean(leader['f1']):.4f}`.",
            "",
            "`short focused question-tail` is the fastest Top 5 candidate, but its lower I recall and non-perfect "
            "parse rate make it a speed-only option rather than the best accuracy/cost production choice. The "
            "balanced production candidate remains `question-tail evidence-only`.",
            "",
            "## Cumulative historical stability",
            "",
            "All entries below are one teacher call per sample. Values are holdout mean [min, max] across the "
            "independent runs completed before the fresh Top 5 comparison.",
            "",
            "| Strategy | Runs | I precision | I recall | I F1 | Parse rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, names in HISTORICAL_GROUPS.items():
        rows = [holdout_metrics(name) for name in names]
        precision = [float(row["i_binary"]["precision"]) for row in rows]
        recall = [float(row["i_binary"]["recall"]) for row in rows]
        f1 = [float(row["i_binary"]["f1"]) for row in rows]
        parse = [float(row["parse_rate"]) for row in rows]
        lines.append(
            f"| {group} | {len(rows)} | {span(precision)} | {span(recall)} | "
            f"{span(f1)} | {span(parse)} |"
        )
    lines.extend(
        [
            "",
            "The repeated-run spread is material even with temperature=0. Selection must use repeated-run means "
            "and cache-free requests rather than the best single replica result.",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
