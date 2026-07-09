"""LLM reranker training entry for AgenticIterRag v1."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_iter_rag.reranker_training.trainer_entry import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgenticIterRag v1 LLM reranker.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs = run_from_config(args.config, args.manifest, dry_run=args.dry_run)
    print(f"wrote reranker training manifest to {args.manifest}: {outputs}")


if __name__ == "__main__":
    main()
