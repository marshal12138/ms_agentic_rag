#!/usr/bin/env python3
"""Build the dual-objective index for persisted new-data PE runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "results_newdata"
DEFAULT_OUTPUT = HERE / "NEW_DATA_RESULTS_INDEX.md"


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def metric_cells(metrics: dict[str, Any]) -> list[str]:
    i_metrics = metrics["i_binary"]
    answer = metrics["answer_gold"]
    return [
        fmt(i_metrics["precision"]),
        fmt(i_metrics["recall"]),
        fmt(i_metrics["f1"]),
        fmt(answer["token_f1_coverage"]),
        fmt(answer["manual_answer_token_f1_coverage"]),
        fmt(metrics["equal_weight_objective"]),
    ]


def load_row(result_dir: Path, root: Path) -> list[str] | None:
    run_path = result_dir / "run.json"
    metrics_path = result_dir / "metrics.json"
    if not run_path.exists() or not metrics_path.exists():
        return None
    run = json.loads(run_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selected = metrics.get("selected")
    called = metrics.get("teacher_called")
    if not selected or not called:
        return None
    cache_hits = int(run.get("cache_hits", -1))
    request_errors = int(run.get("request_errors", -1))
    validity = "valid" if cache_hits == 0 and request_errors == 0 else "audit"
    budget = run.get("budget") or {}
    origin = "derived" if run.get("combination_postprocess_only") else "inference"
    return [
        f"`{result_dir.relative_to(root)}`",
        f"`{run['variant']}`",
        str(run["split"]),
        str(selected["case_count"]),
        *metric_cells(selected),
        fmt(called["i_binary"]["f1"]),
        fmt(called["answer_gold"]["token_f1_coverage"]),
        fmt(called["equal_weight_objective"]),
        fmt(selected["parse_rate"]),
        fmt(run.get("wall_elapsed_s"), 2),
        fmt(budget.get("mean_elapsed_ratio_vs_stage_a", 1.0)),
        fmt(budget.get("stage_b_call_rate", 0.0)),
        origin,
        str(cache_hits),
        str(request_errors),
        validity,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    if args.results.exists():
        for metrics_path in sorted(args.results.rglob("metrics.json")):
            row = load_row(metrics_path.parent, args.results)
            if row is not None:
                rows.append(row)
    lines = [
        "# New-Data Teacher PE Results Index",
        "",
        f"Generated at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "The equal objective is `0.5 * I_F1 + 0.5 * gold_token_F1_coverage_on_manual_S`. "
        "The teacher-called columns are the primary operational slice; all-dev columns retain the 512-sample design and controls.",
        "",
        "| Result | Variant | Split | N | I P | I R | I F1 | Gold F1 cov. | Manual-answer F1 cov. | Equal obj. | Called I F1 | Called gold F1 cov. | Called obj. | Parse | Wall s | Elapsed x | Stage-B rate | Origin | Cache | Errors | Audit |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
