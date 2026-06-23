#!/usr/bin/env python3
"""Extract CoSearch console metrics from official VERL logs."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(?P<step>\d+)\s+-\s+(?P<body>.*)")
METRIC_RE = re.compile(r"(?P<key>[A-Za-z0-9_/@.-]+):(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)")


def strip_noise(text: str) -> str:
    text = ANSI_RE.sub("", text)
    return text.replace("\r", "\n")


def parse_step_metrics(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        match = STEP_RE.search(line)
        if not match:
            continue
        metrics = {"step": int(match.group("step"))}
        for metric in METRIC_RE.finditer(match.group("body")):
            metrics[metric.group("key")] = float(metric.group("value"))
        rows.append(metrics)
    return rows


def parse_initial_validation_metrics(text: str) -> dict | None:
    marker = "Initial validation metrics: "
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = text.find("}", start)
    if end < 0:
        return None
    literal = text[start : end + 1]
    try:
        parsed = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    text = strip_noise(args.log.read_text(errors="replace"))
    payload = {
        "log": str(args.log),
        "initial_validation_metrics": parse_initial_validation_metrics(text),
        "step_metrics": parse_step_metrics(text),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None, sort_keys=True))


if __name__ == "__main__":
    main()
