#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


MODEL = Path("/data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash")
PATTERN = re.compile(r"vllm|vLLM|transformers|inference|serve|generate|convert|DeepseekV4|deepseek_v4", re.I)


def show_matches(path: Path) -> None:
    print(f"## {path}")
    if not path.exists():
        print("missing")
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for idx, line in enumerate(lines, start=1):
        if PATTERN.search(line):
            start = max(1, idx - 2)
            end = min(len(lines), idx + 3)
            print(f"-- match line {idx}")
            for j in range(start, end + 1):
                print(f"{j}: {lines[j - 1]}")


def main() -> None:
    show_matches(MODEL / "README.md")
    show_matches(MODEL / "inference/README.md")


if __name__ == "__main__":
    main()
