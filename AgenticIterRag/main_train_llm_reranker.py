"""LLM reranker training placeholder entry for AgenticIterRag v1."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_iter_rag.pipeline.manifest import write_stage_manifest
from agentic_iter_rag.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AgenticIterRag v1 LLM reranker.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    config = read_yaml(args.config)
    write_stage_manifest(
        args.manifest,
        stage="train_llm_reranker",
        config=config,
        outputs={
            "status": "not_started",
            "note": "training backend is intentionally scaffolded for v1 framework preparation",
        },
    )
    print(f"wrote reranker training manifest to {args.manifest}")


if __name__ == "__main__":
    main()

