#!/usr/bin/env python3
"""Project schema loading for training reports."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_PLOT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#17becf",
    "#8c564b",
    "#7f7f7f",
]


def _module_to_dict(module: ModuleType) -> dict[str, Any]:
    return {
        "project_name": getattr(module, "PROJECT_NAME", "Training"),
        "metric_groups": getattr(module, "METRIC_GROUPS", {}),
        "plot_groups": getattr(module, "PLOT_GROUPS", getattr(module, "METRIC_GROUPS", {})),
        "detailed_metric_keys": getattr(module, "DETAILED_METRIC_KEYS", []),
        "rollout_role_dirs": getattr(module, "ROLLOUT_ROLE_DIRS", ["main"]),
        "gpu_groups": getattr(module, "GPU_GROUPS", {}),
        "timing_aliases": getattr(module, "TIMING_ALIASES", {}),
        "hooks": {
            "normalize_metric_row": getattr(module, "normalize_metric_row", None),
            "build_extra_markdown_sections": getattr(module, "build_extra_markdown_sections", None),
            "build_extra_plot_specs": getattr(module, "build_extra_plot_specs", None),
            "discover_rollout_dirs": getattr(module, "discover_rollout_dirs", None),
        },
    }


def load_report_schema(schema_path: Path | None) -> dict[str, Any]:
    if schema_path is None:
        return _default_schema()
    if not schema_path.exists():
        raise FileNotFoundError(f"report schema not found: {schema_path}")

    spec = importlib.util.spec_from_file_location("project_report_schema", schema_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import report schema: {schema_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    schema = _module_to_dict(module)
    schema["schema_path"] = str(schema_path)
    return schema


def _default_schema() -> dict[str, Any]:
    return {
        "project_name": "Training",
        "metric_groups": {},
        "plot_groups": {},
        "detailed_metric_keys": [],
        "rollout_role_dirs": ["main", "main_agent"],
        "gpu_groups": {},
        "timing_aliases": {},
        "hooks": {
            "normalize_metric_row": None,
            "build_extra_markdown_sections": None,
            "build_extra_plot_specs": None,
            "discover_rollout_dirs": None,
        },
        "schema_path": "",
    }

