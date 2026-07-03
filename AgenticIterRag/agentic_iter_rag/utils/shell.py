"""Shell audit file helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def write_export_file(path: str | Path, env: Mapping[str, str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for key in sorted(env):
            f.write(f"export {key}={shell_quote(str(env[key]))}\n")

