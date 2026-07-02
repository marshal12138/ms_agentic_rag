#!/usr/bin/env python3
"""Compatibility wrapper for CoAgenticRetriever recall semantic preflight.

历史上 launcher 和部分手工检查命令直接调用这个脚本。现在真正的实现已经下沉到：

`scripts/coagenticRetriever_v2/assets/trainer_launcher/recall_preflight.py`

保留这个 wrapper 的目的只是维持旧入口可用，避免其它脚本立即跟随迁移。新代码应优先
调用 `trainer_launcher/recall_preflight.py semantic`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer_launcher.recall_preflight import RecallPreflightConfig, run_semantic_preflight
from trainer_launcher.recall_preflight import _print_semantic_success as print_semantic_success


def main() -> int:
    """兼容旧 CLI 参数，并委托给新的 recall_preflight semantic 实现。"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="CoAgenticRetriever")
    parser.add_argument("--url", default="http://127.0.0.1:8010/retrieve")
    parser.add_argument("--query", default="who got the first nobel prize in physics?")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--expect-contains", default="Röntgen")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    try:
        result = run_semantic_preflight(
            RecallPreflightConfig(
                url=args.url,
                query=args.query,
                top_n=args.top_n,
                top_m=args.top_m,
                expect_contains=args.expect_contains,
                timeout=args.timeout,
            )
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print_semantic_success(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
