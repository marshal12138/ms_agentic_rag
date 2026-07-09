"""AgenticIterRag v1 agent 训练入口。

这个入口只负责根据 pipeline.stage_configs.train_agent.impl 分发到 AIR 内部
训练实现。SPAD-RAG 的 sub-stage 细节不写在这里，避免顶层入口膨胀。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agentic_iter_rag.agent_training.train_agent_entry import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="AgenticIterRag v1 agent training.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs = run_from_config(args.config, args.manifest, dry_run=args.dry_run)
    print(f"train_agent outputs: {outputs}")


if __name__ == "__main__":
    main()
