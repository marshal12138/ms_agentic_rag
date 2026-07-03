"""AgenticIterRag v1 pipeline manifest entry."""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_iter_rag.pipeline.manifest import write_stage_manifest
from agentic_iter_rag.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Write AgenticIterRag v1 pipeline manifest.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    config = read_yaml(args.config)
    write_stage_manifest(args.manifest, stage="pipeline", config=config, outputs={"status": "compiled"})
    print(f"wrote pipeline manifest to {args.manifest}")


if __name__ == "__main__":
    main()

