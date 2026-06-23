#!/usr/bin/env python3
"""Generate schema-driven training metrics reports."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from report_io import (
    companion_env_path,
    companion_train_log_path,
    discover_role_rollout_dirs,
    fmt,
    numeric_series,
    read_env_file,
    read_metrics,
    rollout_cycles_by_step,
    rollout_data_dir_from_env,
    rollout_summary,
    summarize,
)
from report_schema import load_report_schema


def metric_value(row: dict[str, Any], key: str, rollout_cycles: dict[int, float]) -> Any:
    if key == "agent_rollout_num":
        return rollout_cycles.get(int(row["step"]))
    value = row["data"].get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return value if value is not None else None


def table_for_keys(series: dict[str, list[tuple[int, float]]], keys: list[str]) -> list[str]:
    lines = [
        "| metric | count | first | last | avg | min | p50 | p90 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in keys:
        values = [value for _, value in series.get(key, [])]
        stats = summarize(values)
        lines.append(
            f"| {key} | {int(stats['count'])} | {stats['first']:.6g} | {stats['last']:.6g} | "
            f"{stats['avg']:.6g} | {stats['min']:.6g} | {stats['p50']:.6g} | "
            f"{stats['p90']:.6g} | {stats['max']:.6g} |"
        )
    return lines


def build_report(
    metrics_path: Path,
    out: Path,
    step_limit: int | None,
    schema_path: Path | None,
    env_file: Path | None = None,
    train_log: Path | None = None,
    detailed: bool = False,
) -> str:
    schema = load_report_schema(schema_path)
    normalizer = schema["hooks"].get("normalize_metric_row")
    rows = read_metrics(metrics_path, step_limit, normalizer)
    env_path = companion_env_path(metrics_path, env_file)
    log_path = companion_train_log_path(metrics_path, train_log)
    env = read_env_file(env_path)
    rollout_root = rollout_data_dir_from_env(env)
    role_dirs = discover_role_rollout_dirs(
        rollout_root,
        list(schema.get("rollout_role_dirs", [])),
        schema["hooks"].get("discover_rollout_dirs"),
    )
    rollout_cycles = rollout_cycles_by_step(role_dirs, step_limit)
    rollout_info = rollout_summary(role_dirs)
    title_limit = f"steps <= {step_limit}" if step_limit is not None else "all completed steps"
    project_name = schema.get("project_name", "Training")
    report_name = "Detailed Metrics" if detailed else "Training Metrics"

    lines = [
        f"# {project_name} {report_name} Report ({title_limit})",
        "",
        f"- metrics_jsonl: `{metrics_path}`",
        f"- env_file: `{env_path}`",
        f"- train_log: `{log_path}`",
        f"- rollout_data_dir: `{rollout_root if rollout_root is not None else ''}`",
        f"- report_schema: `{schema.get('schema_path', '')}`",
        f"- output: `{out}`",
        f"- completed_train_steps: `{len(rows)}`",
        f"- max_step_in_report: `{rows[-1]['step'] if rows else 0}`",
    ]
    if rows:
        lines.extend([f"- first_step: `{rows[0]['step']}`", f"- last_step: `{rows[-1]['step']}`"])
    lines.extend(["", "## Rollout", ""])
    lines.append("| item | value |")
    lines.append("| --- | --- |")
    for key in [
        "rollout_role_dirs",
        "rollout_step_files",
        "rollout_rows_per_step_avg",
        "rollout_rows_per_step_min",
        "rollout_rows_per_step_max",
    ]:
        lines.append(f"| {key} | {fmt(rollout_info.get(key))} |")
    for key in ["N_ROLLOUTS", "TRAIN_BATCH_SIZE", "ACTOR_BATCH_SIZE", "ROLLOUT_DATA_DIR"]:
        lines.append(f"| env/{key} | {fmt(env.get(key, ''))} |")
    lines.append("")

    extra_hook = schema["hooks"].get("build_extra_markdown_sections")
    if extra_hook is not None:
        extra = extra_hook(
            {
                "metrics_path": metrics_path,
                "rows": rows,
                "env": env,
                "rollout_role_dirs": role_dirs,
                "step_limit": step_limit,
                "detailed": detailed,
            }
        )
        if isinstance(extra, str) and extra.strip():
            lines.extend([extra.strip(), ""])
        elif isinstance(extra, list):
            lines.extend([str(item) for item in extra])
            lines.append("")

    if not rows:
        lines.extend(["No completed training steps were found.", ""])
        return "\n".join(lines)

    if detailed:
        metric_keys = list(schema.get("detailed_metric_keys", []))
        if "agent_rollout_num" not in metric_keys:
            metric_keys.insert(0, "agent_rollout_num")
        lines.extend(["## Per-Step Metrics", ""])
        lines.append("| " + " | ".join(["step", *metric_keys]) + " |")
        lines.append("| " + " | ".join(["---:"] * (len(metric_keys) + 1)) + " |")
        for row in rows:
            values = [fmt(row["step"])]
            values.extend(fmt(metric_value(row, key, rollout_cycles)) for key in metric_keys)
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    else:
        series = numeric_series(rows)
        for group_name, keys in schema.get("metric_groups", {}).items():
            lines.extend([f"## {str(group_name).replace('_', ' ').title()}", ""])
            lines.extend(table_for_keys(series, list(keys)))
            lines.append("")
        lines.extend(["## Available Numeric Metrics", ""])
        for key in sorted(series):
            lines.append(f"- {key}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--report-schema", type=Path)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report(
            args.metrics_jsonl,
            args.out,
            args.step_limit,
            args.report_schema,
            args.env_file,
            args.train_log,
            args.detailed,
        ),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()

