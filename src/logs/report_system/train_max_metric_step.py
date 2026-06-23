#!/usr/bin/env python3
"""Print the maximum completed step in a metrics JSONL file."""

from __future__ import annotations

import argparse
from pathlib import Path

from report_io import max_metric_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(max_metric_step(args.metrics_jsonl))


if __name__ == "__main__":
    main()

