#!/usr/bin/env python3
"""Compatibility wrapper for train_metrics_report.py."""

from __future__ import annotations

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("train_metrics_report.py")), run_name="__main__")

