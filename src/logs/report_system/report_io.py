#!/usr/bin/env python3
"""I/O helpers shared by training report generators."""

from __future__ import annotations

import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")
METRIC_RE = re.compile(r"(?P<key>[A-Za-z0-9_/@.-]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")


def infer_run_prefix(metrics_path: Path) -> str:
    suffix = ".metrics.jsonl"
    name = metrics_path.name
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return metrics_path.stem


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def read_env_file(path: Path | None) -> dict[str, str]:
    env: dict[str, str] = {}
    if path is None or not path.exists():
        return env
    with path.open("r", encoding="utf-8", errors="replace") as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip()
    return env


def companion_env_path(metrics_path: Path, env_file: Path | None) -> Path:
    if env_file is not None:
        return env_file
    prefix = infer_run_prefix(metrics_path)
    return metrics_path.with_name(f"{prefix}.env")


def companion_train_log_path(metrics_path: Path, train_log: Path | None) -> Path:
    if train_log is not None:
        return train_log
    prefix = infer_run_prefix(metrics_path)
    return metrics_path.with_name(f"{prefix}.train.log")


def extract_step_rows(
    metrics_rows: list[dict[str, Any]],
    step_limit: int | None,
    normalize_metric_row: Any | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metrics_rows:
        step = row.get("step")
        data = row.get("data")
        if not isinstance(step, int) or not isinstance(data, dict):
            continue
        if step_limit is not None and step > step_limit:
            continue
        normalized_data = dict(data)
        if normalize_metric_row is not None:
            maybe_data = normalize_metric_row({"step": step, "data": normalized_data})
            if isinstance(maybe_data, dict):
                normalized_data = dict(maybe_data.get("data", maybe_data))
        rows.append({"step": step, "data": normalized_data})
    rows.sort(key=lambda item: item["step"])
    return rows


def read_metrics(path: Path, step_limit: int | None, normalize_metric_row: Any | None = None) -> list[dict[str, Any]]:
    return extract_step_rows(read_jsonl(path), step_limit, normalize_metric_row)


def read_console_metrics(path: Path, step_limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace")).replace("\r", "\n")
    for line in text.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        step = int(match.group("step"))
        if step_limit is not None and step > step_limit:
            continue
        data: dict[str, float] = {}
        for metric in METRIC_RE.finditer(match.group("body")):
            data[metric.group("key")] = float(metric.group("value"))
        rows.append({"step": step, "data": data})
    rows.sort(key=lambda item: item["step"])
    return rows


def max_metric_step(path: Path) -> int:
    last_step = 0
    for row in read_jsonl(path):
        step = row.get("step")
        if isinstance(step, int):
            last_step = max(last_step, step)
    return last_step


def numeric_series(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        step = int(row["step"])
        for key, value in row["data"].items():
            if isinstance(value, bool):
                continue
            if isinstance(value, int | float):
                series[key].append((step, float(value)))
    return series


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "first": 0.0,
            "last": 0.0,
            "avg": 0.0,
            "min": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": float(len(values)),
        "first": values[0],
        "last": values[-1],
        "avg": statistics.fmean(values),
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return "N/A"
        if value.is_integer():
            return str(int(value))
        return f"{value:.6g}"
    return str(value)


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "N/A":
        return None
    match = re.search(r"-?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not match:
        return None
    return float(match.group(0))


def read_nvidia_smi_csv(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fp:
        reader = csv.reader(fp)
        for raw in reader:
            if not raw:
                continue
            values = [item.strip() for item in raw]
            if values[0] == "timestamp" or len(values) < 7:
                continue
            try:
                gpu_index = int(values[1])
            except ValueError:
                continue
            rows.append(
                {
                    "timestamp": values[0],
                    "index": gpu_index,
                    "gpu_util_pct": to_float(values[2]),
                    "mem_util_pct": to_float(values[3]),
                    "mem_used_mb": to_float(values[4]),
                    "mem_total_mb": to_float(values[5]),
                    "power_draw_w": to_float(values[6]),
                }
            )
    return rows


def parse_gpu_ids(value: str | None) -> set[int]:
    if not value:
        return set()
    ids: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            ids.add(int(item))
    return ids


def summarize_nvidia_group(rows: list[dict[str, Any]], gpu_ids: set[int]) -> dict[str, float]:
    selected = [row for row in rows if row.get("index") in gpu_ids]
    if not selected:
        return {
            "sample_rows": 0.0,
            "timestamps": 0.0,
            "avg_gpu_util_pct": 0.0,
            "avg_mem_util_pct": 0.0,
            "avg_mem_used_gb_per_gpu": 0.0,
            "avg_group_mem_used_gb": 0.0,
            "avg_power_w_per_gpu": 0.0,
            "avg_group_power_w": 0.0,
        }

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in selected if isinstance(row.get(name), int | float)]

    by_timestamp: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_timestamp[str(row.get("timestamp"))].append(row)

    group_mem_used_gb: list[float] = []
    group_power_w: list[float] = []
    for timestamp_rows in by_timestamp.values():
        mem_values = [
            float(row["mem_used_mb"]) / 1024.0
            for row in timestamp_rows
            if isinstance(row.get("mem_used_mb"), int | float)
        ]
        power_values = [
            float(row["power_draw_w"])
            for row in timestamp_rows
            if isinstance(row.get("power_draw_w"), int | float)
        ]
        if mem_values:
            group_mem_used_gb.append(sum(mem_values))
        if power_values:
            group_power_w.append(sum(power_values))

    return {
        "sample_rows": float(len(selected)),
        "timestamps": float(len(by_timestamp)),
        "avg_gpu_util_pct": statistics.fmean(values("gpu_util_pct")) if values("gpu_util_pct") else 0.0,
        "avg_mem_util_pct": statistics.fmean(values("mem_util_pct")) if values("mem_util_pct") else 0.0,
        "avg_mem_used_gb_per_gpu": statistics.fmean(values("mem_used_mb")) / 1024.0
        if values("mem_used_mb")
        else 0.0,
        "avg_group_mem_used_gb": statistics.fmean(group_mem_used_gb) if group_mem_used_gb else 0.0,
        "avg_power_w_per_gpu": statistics.fmean(values("power_draw_w")) if values("power_draw_w") else 0.0,
        "avg_group_power_w": statistics.fmean(group_power_w) if group_power_w else 0.0,
    }


def rollout_data_dir_from_env(env: dict[str, str]) -> Path | None:
    rollout_dir = env.get("ROLLOUT_DATA_DIR")
    if rollout_dir:
        return Path(rollout_dir)
    out_dir = env.get("OUT_DIR")
    if out_dir:
        return Path(out_dir) / "rollout_data"
    return None


def discover_role_rollout_dirs(
    rollout_data_dir: Path | None,
    role_names: list[str],
    discover_hook: Any | None = None,
) -> dict[str, Path]:
    if rollout_data_dir is None:
        return {}
    if discover_hook is not None:
        hook_result = discover_hook(rollout_data_dir)
        if isinstance(hook_result, dict):
            return {str(role): Path(path) for role, path in hook_result.items()}
    result: dict[str, Path] = {}
    for role in role_names:
        path = rollout_data_dir / role
        if path.exists():
            result[role] = path
    if not result and rollout_data_dir.exists():
        for path in sorted(rollout_data_dir.iterdir()):
            if path.is_dir() and any(path.glob("*.jsonl")):
                result[path.name] = path
    return result


def count_agent_cycles(output: str) -> int:
    if not output.strip():
        return 0
    return 1 + len(re.findall(r"(?:^|\n)assistant\n", output))


def rollout_cycles_by_step(role_dirs: dict[str, Path], step_limit: int | None) -> dict[int, float]:
    result: dict[int, float] = {}
    for role_dir in role_dirs.values():
        step_files = sorted(
            role_dir.glob("*.jsonl"),
            key=lambda path: int(path.stem) if path.stem.isdigit() else path.stem,
        )
        for path in step_files:
            fallback_step = int(path.stem) if path.stem.isdigit() else None
            if step_limit is not None and fallback_step is not None and fallback_step > step_limit:
                continue
            cycles_by_step: dict[int, list[int]] = defaultdict(list)
            with path.open("r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = row.get("step", fallback_step)
                    if not isinstance(step, int):
                        continue
                    if step_limit is not None and step > step_limit:
                        continue
                    cycles_by_step[step].append(count_agent_cycles(str(row.get("output", ""))))
            for step, cycles in cycles_by_step.items():
                if cycles:
                    result[step] = statistics.fmean(cycles)
    return result


def rollout_summary(role_dirs: dict[str, Path]) -> dict[str, Any]:
    row_counts: list[int] = []
    file_count = 0
    for role_dir in role_dirs.values():
        for path in role_dir.glob("*.jsonl"):
            file_count += 1
            count = 0
            with path.open("r", encoding="utf-8", errors="replace") as fp:
                for line in fp:
                    if line.strip():
                        count += 1
            row_counts.append(count)
    return {
        "rollout_role_dirs": ", ".join(f"{role}:{path}" for role, path in sorted(role_dirs.items())),
        "rollout_step_files": file_count,
        "rollout_rows_per_step_avg": statistics.fmean(row_counts) if row_counts else "",
        "rollout_rows_per_step_min": min(row_counts) if row_counts else "",
        "rollout_rows_per_step_max": max(row_counts) if row_counts else "",
    }

