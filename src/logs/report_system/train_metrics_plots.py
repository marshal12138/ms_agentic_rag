#!/usr/bin/env python3
"""Plot schema-driven training metrics."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cosearch_report_matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from report_io import infer_run_prefix, read_metrics
from report_schema import DEFAULT_PLOT_COLORS, load_report_schema


def collect_series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        value = row["data"].get(key)
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        steps.append(int(row["step"]))
        values.append(float(value))
    return steps, values


def normalize_plot_groups(raw_groups: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    groups: dict[str, dict[str, list[str]]] = {}
    for group_name, spec in raw_groups.items():
        if isinstance(spec, dict):
            groups[str(group_name)] = {str(axis_name): list(keys) for axis_name, keys in spec.items()}
        elif isinstance(spec, list):
            groups[str(group_name)] = {"metrics": list(spec)}
    return groups


def plot_group(
    rows: list[dict[str, Any]],
    group: dict[str, list[str]],
    title: str,
    out_path: Path,
) -> None:
    axis_names = list(group)
    if len(axis_names) > 2:
        # Keep charts readable; extra axes become additional left-axis series.
        merged = []
        for name in axis_names[1:]:
            merged.extend(group[name])
        group = {axis_names[0]: group[axis_names[0]], "metrics": merged}
        axis_names = list(group)

    fig, ax_left = plt.subplots(figsize=(11, 5.8), dpi=150)
    axes = [ax_left]
    if len(axis_names) == 2:
        axes.append(ax_left.twinx())

    plotted = 0
    color_idx = 0
    for axis_idx, axis_name in enumerate(axis_names):
        ax = axes[axis_idx]
        for key in group[axis_name]:
            steps, values = collect_series(rows, key)
            if not steps:
                continue
            ax.plot(
                steps,
                values,
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=key,
                color=DEFAULT_PLOT_COLORS[color_idx % len(DEFAULT_PLOT_COLORS)],
            )
            color_idx += 1
            plotted += 1
        ax.set_ylabel(axis_name.replace("_", " "))

    ax_left.set_title(title)
    ax_left.set_xlabel("train step")
    ax_left.grid(True, linestyle="--", alpha=0.35)

    handles: list[Any] = []
    labels: list[str] = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
    if handles:
        ax_left.legend(handles, labels, loc="best", fontsize=8)
    if not plotted:
        ax_left.text(0.5, 0.5, "No metric data", ha="center", va="center", transform=ax_left.transAxes)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def build_default_out_prefix(metrics_path: Path) -> Path:
    return metrics_path.with_name(infer_run_prefix(metrics_path) + ".metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--out-prefix", type=Path)
    parser.add_argument("--step-limit", type=int)
    parser.add_argument("--report-schema", type=Path)
    args = parser.parse_args()

    schema = load_report_schema(args.report_schema)
    normalizer = schema["hooks"].get("normalize_metric_row")
    rows = read_metrics(args.metrics_jsonl, args.step_limit, normalizer)
    out_prefix = args.out_prefix or build_default_out_prefix(args.metrics_jsonl)
    plot_groups = normalize_plot_groups(schema.get("plot_groups", {}))
    extra_hook = schema["hooks"].get("build_extra_plot_specs")
    if extra_hook is not None:
        extra_groups = extra_hook({"rows": rows, "metrics_path": args.metrics_jsonl, "step_limit": args.step_limit})
        if isinstance(extra_groups, dict):
            plot_groups.update(normalize_plot_groups(extra_groups))

    outputs: list[Path] = []
    for name, group in plot_groups.items():
        out_path = out_prefix.with_name(out_prefix.name + f"_{name}.png")
        title = f"{schema.get('project_name', 'Training')} Metrics: {str(name).replace('_', ' ').title()}"
        plot_group(rows, group, title, out_path)
        outputs.append(out_path)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()

