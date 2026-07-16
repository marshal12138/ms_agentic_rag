#!/usr/bin/env python3
"""Summarize cache-free new-data PE candidate repeats."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_newdata"
OUTPUT = HERE / "NEW_DATA_STABILITY.md"
GROUPS = {
    "Training production prompt, fresh dev": [
        "260715_composite_stage_a_prod_dev",
        "260715_composite_stage_a_prod_rep2_dev",
        "260715_composite_stage_a_prod_rep3_dev",
    ],
    "R5 gold support, no sub-query, no tail": [
        "260715_round05_gold_no_subquery_no_tail_dev",
        "260715_stability_r5_rep2_dev",
        "260715_stability_r5_rep3_dev",
    ],
    "R9 gold binary support, no sub-query, no tail": [
        "260715_round09_gold_binary_support_dev",
        "260715_stability_r9_rep2_dev",
        "260715_stability_r9_rep3_dev",
    ],
    "R5 frozen holdout": [
        "260715_holdout_r5_rep1",
        "260715_holdout_r5_rep2",
        "260715_holdout_r5_rep3",
    ],
    "Training production prompt, fresh holdout": [
        "260715_composite_holdout_stage_a_rep1",
        "260715_composite_holdout_stage_a_rep2",
        "260715_composite_holdout_stage_a_rep3",
    ],
    "Dual-all v2 composite, dev": [
        "260715_composite_round05_dual_all_f108_literal_canon_dev",
        "260715_composite_stability_literal_canon_rep2_dev",
        "260715_composite_stability_literal_canon_rep3_dev",
    ],
    "Hard-gate v2 composite, dev": [
        "260715_composite_round06_hard_gate_literal_canon_dev",
        "260715_composite_stability_hard_gate_literal_canon_rep2_dev",
        "260715_composite_stability_hard_gate_literal_canon_rep3_dev",
    ],
    "Dual-all v2, consumed holdout diagnostic": [
        "260715_composite_holdout_literal_canon_rep1",
        "260715_composite_holdout_literal_canon_rep2",
        "260715_composite_holdout_literal_canon_rep3",
    ],
    "Hard-gate v2, reused holdout diagnostic": [
        "260715_composite_holdout_hard_gate_literal_canon_rep1",
        "260715_composite_holdout_hard_gate_literal_canon_rep2",
        "260715_composite_holdout_hard_gate_literal_canon_rep3",
    ],
}


def span(values: list[float]) -> str:
    return f"{mean(values):.4f} [{min(values):.4f}, {max(values):.4f}]"


def load_run(name: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    result_dir = RESULTS / name
    if not (result_dir / "run.json").exists() or not (result_dir / "metrics.json").exists():
        return None
    run = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((result_dir / "metrics.json").read_text(encoding="utf-8"))
    if int(run.get("cache_hits", -1)) != 0 or int(run.get("request_errors", -1)) != 0:
        raise RuntimeError(f"Invalid stability run cache/errors: {name}")
    return run, metrics


def metric_span(loaded: list[tuple[dict[str, Any], dict[str, Any]]], slice_name: str, path: tuple[str, ...]) -> str:
    values = []
    for _, metrics in loaded:
        value: Any = metrics[slice_name]
        for key in path:
            value = value[key]
        values.append(float(value))
    return span(values)


def budget_span(loaded: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    values = [
        float((run.get("budget") or {}).get("mean_elapsed_ratio_vs_stage_a", 1.0))
        for run, _ in loaded
    ]
    return span(values)


def main() -> None:
    lines = [
        "# New-Data Teacher PE Stability",
        "",
        f"Generated at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "Mean [min, max] over completed cache-free runs. The teacher-called objective is the primary operational selection metric. Reused holdout rows are diagnostics, not untouched final estimates.",
        "",
        "| Candidate | Runs | All I P | All I R | All I F1 | All gold F1 | All manual F1 | All objective | Called I P | Called I R | Called I F1 | Called gold F1 | Called manual F1 | Called objective | Parse | Elapsed x | Wall s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate, names in GROUPS.items():
        loaded = [item for name in names if (item := load_run(name)) is not None]
        if not loaded:
            continue
        walls = [float(run["wall_elapsed_s"]) for run, _ in loaded]
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate,
                    str(len(loaded)),
                    metric_span(loaded, "selected", ("i_binary", "precision")),
                    metric_span(loaded, "selected", ("i_binary", "recall")),
                    metric_span(loaded, "selected", ("i_binary", "f1")),
                    metric_span(loaded, "selected", ("answer_gold", "token_f1_coverage")),
                    metric_span(loaded, "selected", ("answer_gold", "manual_answer_token_f1_coverage")),
                    metric_span(loaded, "selected", ("equal_weight_objective",)),
                    metric_span(loaded, "teacher_called", ("i_binary", "precision")),
                    metric_span(loaded, "teacher_called", ("i_binary", "recall")),
                    metric_span(loaded, "teacher_called", ("i_binary", "f1")),
                    metric_span(loaded, "teacher_called", ("answer_gold", "token_f1_coverage")),
                    metric_span(loaded, "teacher_called", ("answer_gold", "manual_answer_token_f1_coverage")),
                    metric_span(loaded, "teacher_called", ("equal_weight_objective",)),
                    metric_span(loaded, "selected", ("parse_rate",)),
                    budget_span(loaded),
                    span(walls),
                ]
            )
            + " |"
        )
    lines.append("")
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
