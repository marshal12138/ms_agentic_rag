#!/usr/bin/env python3
"""Generate timing reports from training metrics and auxiliary logs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from report_io import (
    parse_gpu_ids,
    read_console_metrics,
    read_jsonl,
    read_metrics,
    read_nvidia_smi_csv,
    summarize,
    summarize_nvidia_group,
)
from report_schema import load_report_schema


def build_report(
    metrics_path: Path,
    search_path: Path,
    out: Path,
    step_limit: int | None,
    schema_path: Path | None,
    train_log: Path | None = None,
    nvidia_smi_csv: Path | None = None,
    main_gpu_ids: str | None = None,
    reranker_gpu_ids: str | None = None,
    ranker_gpu_ids: str | None = None,
) -> str:
    schema = load_report_schema(schema_path)
    normalizer = schema["hooks"].get("normalize_metric_row")
    metric_rows = read_metrics(metrics_path, step_limit, normalizer)
    metrics_source = str(metrics_path)
    if not metric_rows and train_log is not None:
        metric_rows = read_console_metrics(train_log, step_limit)
        metrics_source = f"{train_log} (console fallback)"
    search_rows = read_jsonl(search_path)

    action_values: dict[str, list[float]] = defaultdict(list)
    train_step_values: list[float] = []

    for row in metric_rows:
        data = row["data"]
        for key, value in data.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if key == "perf/time_per_step":
                train_step_values.append(float(value))
            if key.startswith("timing_s/"):
                action = key.removeprefix("timing_s/")
                action_values[action].append(float(value))
                if action == "gen":
                    action_values["rollout"].append(float(value))
                elif action.endswith("_update_actor"):
                    action_values[f"parameter_update/{action}"].append(float(value))
                    action_values["parameter_update/all_actor_updates"].append(float(value))
                elif action.endswith("_old_log_prob") or action.endswith("_ref_log_prob"):
                    action_values[f"log_prob/{action}"].append(float(value))
                elif "agent_loop/tool_calls/mean" in action:
                    action_values["rollout/tool_calls_mean"].append(float(value))
                elif "agent_loop/generate_sequences/mean" in action:
                    action_values["rollout/generate_sequences_mean"].append(float(value))

    for alias, keys in schema.get("timing_aliases", {}).items():
        for key in keys:
            source_key = key.removeprefix("timing_s/")
            if source_key in action_values:
                action_values[alias].extend(action_values[source_key])

    search_values = [
        float(row["elapsed_s"])
        for row in search_rows
        if row.get("action") == "search"
        and row.get("status") == "success"
        and isinstance(row.get("elapsed_s"), int | float)
    ]
    if search_values:
        action_values["search/http_retrieve"].extend(search_values)

    gpu_groups: dict[str, set[int]] = {}
    for group_name, env_name in schema.get("gpu_groups", {}).items():
        if env_name == "MAIN_GPU_IDS":
            gpu_groups[group_name] = parse_gpu_ids(main_gpu_ids)
        elif env_name == "RERANKER_GPU_IDS":
            gpu_groups[group_name] = parse_gpu_ids(reranker_gpu_ids)
        elif env_name == "RANKER_GPU_IDS":
            gpu_groups[group_name] = parse_gpu_ids(ranker_gpu_ids or reranker_gpu_ids)
    if not gpu_groups:
        gpu_groups = {
            "main_actor": parse_gpu_ids(main_gpu_ids),
            "reranker": parse_gpu_ids(reranker_gpu_ids or ranker_gpu_ids),
        }

    nvidia_rows = read_nvidia_smi_csv(nvidia_smi_csv)
    title_limit = f"steps <= {step_limit}" if step_limit is not None else "all completed steps"
    project_name = schema.get("project_name", "Training")
    lines = [
        f"# {project_name} Timing Report ({title_limit})",
        "",
        f"- metrics_source: `{metrics_source}`",
        f"- metrics_jsonl: `{metrics_path}`",
        f"- search_timing_jsonl: `{search_path}`",
        f"- nvidia_smi_csv: `{nvidia_smi_csv}`" if nvidia_smi_csv is not None else "- nvidia_smi_csv: `not provided`",
        f"- report_schema: `{schema.get('schema_path', '')}`",
        f"- output: `{out}`",
        f"- completed_train_steps: `{len(metric_rows)}`",
        f"- max_step_in_report: `{metric_rows[-1]['step'] if metric_rows else 0}`",
        f"- search_success_calls: `{len(search_values)}`",
        "",
        "## Train Step",
        "",
        "| metric | seconds |",
        "| --- | ---: |",
    ]

    step_summary = summarize(train_step_values)
    lines.append(f"| avg train step | {step_summary['avg']:.3f} |")
    lines.append(f"| p50 train step | {step_summary['p50']:.3f} |")
    lines.append(f"| p90 train step | {step_summary['p90']:.3f} |")
    lines.append(f"| max train step | {step_summary['max']:.3f} |")
    lines.extend(["", "## Action Timing", ""])
    lines.append("| action | count | avg_s | p50_s | p90_s | max_s |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for action in sorted(action_values):
        stats = summarize(action_values[action])
        lines.append(
            f"| {action} | {int(stats['count'])} | {stats['avg']:.3f} | "
            f"{stats['p50']:.3f} | {stats['p90']:.3f} | {stats['max']:.3f} |"
        )

    if nvidia_smi_csv is not None:
        lines.extend(["", "## GPU Utilization", ""])
        lines.append("| group | gpu_ids | sample_rows | timestamps | avg_gpu_util_% | avg_mem_util_% | avg_mem_used_gb_per_gpu | avg_group_mem_used_gb | avg_power_w_per_gpu | avg_group_power_w |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for group_name, ids in gpu_groups.items():
            stats = summarize_nvidia_group(nvidia_rows, ids)
            lines.append(
                f"| {group_name} | {','.join(str(i) for i in sorted(ids)) or 'n/a'} | "
                f"{int(stats['sample_rows'])} | {int(stats['timestamps'])} | "
                f"{stats['avg_gpu_util_pct']:.2f} | {stats['avg_mem_util_pct']:.2f} | "
                f"{stats['avg_mem_used_gb_per_gpu']:.2f} | {stats['avg_group_mem_used_gb']:.2f} | "
                f"{stats['avg_power_w_per_gpu']:.2f} | {stats['avg_group_power_w']:.2f} |"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--search-jsonl", type=Path, required=True)
    parser.add_argument("--train-log", type=Path)
    parser.add_argument("--nvidia-smi-csv", type=Path)
    parser.add_argument("--main-gpu-ids")
    parser.add_argument("--reranker-gpu-ids")
    parser.add_argument("--ranker-gpu-ids")
    parser.add_argument("--report-schema", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--step-limit", type=int)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        build_report(
            args.metrics_jsonl,
            args.search_jsonl,
            args.out,
            args.step_limit,
            args.report_schema,
            args.train_log,
            args.nvidia_smi_csv,
            args.main_gpu_ids,
            args.reranker_gpu_ids,
            args.ranker_gpu_ids,
        ),
        encoding="utf-8",
    )
    print(args.out)


if __name__ == "__main__":
    main()

