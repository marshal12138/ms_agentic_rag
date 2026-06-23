#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


MODEL = Path("/data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash")


def show_file(name: str, limit: int = 4000) -> None:
    path = MODEL / name
    print(f"--- {name} exists={path.exists()} size={path.stat().st_size if path.exists() else 'NA'}")
    if path.exists():
        print(path.read_text(encoding="utf-8", errors="replace")[:limit])


def main() -> None:
    for name in (
        "config.json",
        "configuration.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "generation_config.json",
        "README.md",
    ):
        show_file(name)

    inference = MODEL / "inference"
    print(f"--- inference exists={inference.exists()}")
    if inference.exists():
        for path in sorted(inference.rglob("*"))[:200]:
            print(path.relative_to(MODEL), "dir" if path.is_dir() else f"file size={path.stat().st_size}")

    for pattern in ("*.py", "inference/**/*.py", "assets/**/*", "encoding/**/*"):
        matches = sorted(MODEL.glob(pattern))
        print(f"--- pattern {pattern} count={len(matches)}")
        for path in matches[:100]:
            print(path.relative_to(MODEL), "dir" if path.is_dir() else f"file size={path.stat().st_size}")


if __name__ == "__main__":
    main()
