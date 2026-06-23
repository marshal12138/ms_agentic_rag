#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path("/data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash/inference")
KEYWORDS = ("hadamard", "fast_hadamard", "tilelang", "triton")


def main() -> None:
    for path in sorted(ROOT.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [line for line in text.splitlines() if any(k in line.lower() for k in KEYWORDS)]
        if hits:
            print(f"--- {path.name}")
            for line in hits[:100]:
                print(line)


if __name__ == "__main__":
    main()
