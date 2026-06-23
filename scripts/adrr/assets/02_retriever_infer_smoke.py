#!/usr/bin/env python3
"""Compatibility wrapper for the renamed CoAgentic ranker inference smoke."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("02_ranker_infer_smoke.py")), run_name="__main__")
