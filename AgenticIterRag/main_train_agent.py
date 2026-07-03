"""AgenticIterRag v1 agent 训练入口占位。

当前 data produce 任务不执行 agent 训练。这里不委托任何外部框架入口，避免
AIR 训练入口在尚未实现前误调用其它链路。
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="AgenticIterRag v1 agent training placeholder.")
    parser.parse_args()
    raise NotImplementedError("AgenticIterRag v1 agent trainer has not been implemented yet.")


if __name__ == "__main__":
    main()
