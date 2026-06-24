#!/usr/bin/env python3
"""Compatibility wrapper for the renamed CoAgentic retrieval check."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("00_check_coagentic_tool_retrieval.py")), run_name="__main__")
