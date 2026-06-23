#!/usr/bin/env python3
"""Compatibility wrapper for the shared metrics plotter."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = ROOT / "scripts" / "cosearch_local" / "assets" / "report_schema.py"
TARGET = ROOT / "src" / "logs" / "report_system" / "train_metrics_plots.py"
sys.path.insert(0, str(TARGET.parent))

if "--report-schema" not in sys.argv:
    sys.argv.extend(["--report-schema", str(DEFAULT_SCHEMA)])

runpy.run_path(str(TARGET), run_name="__main__")
