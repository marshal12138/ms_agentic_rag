#!/usr/bin/env python3
"""Write an immutable identity manifest before an AIR evaluation run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    files: list[dict[str, Any]] = []
    candidates = [resolved] if resolved.is_file() else [resolved / name for name in IDENTITY_FILES]
    if resolved.is_dir():
        candidates.extend(sorted(resolved.glob("*.safetensors")))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        stat = candidate.stat()
        item: dict[str, Any] = {
            "name": candidate.name,
            "size": stat.st_size,
            "sha256": sha256_file(candidate),
        }
        files.append(item)
    if not files:
        raise ValueError(f"model path has no identifiable files: {resolved}")
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"resolved_path": str(resolved), "fingerprint": fingerprint, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--repeat-id", default="")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite eval manifest: {args.output}")
    data_path = args.data_path.resolve()
    manifest = {
        "version": "air-eval-run-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_name": args.task_name,
        "repeat_id": int(args.repeat_id) if args.repeat_id else None,
        "data": {
            "resolved_path": str(data_path),
            "sha256": sha256_file(data_path),
        },
        "model": model_identity(args.model_path),
        "max_samples": args.max_samples,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p},
        "output_reuse": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
