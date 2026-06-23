#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path("/data04/envs/ms/deepseek_v4/lib/python3.11/site-packages")
NAMES = ("libcudart.so", "libcudart.so.13", "libnvrtc.so", "libnvrtc.so.13", "libcuda.so")


def main() -> None:
    for name in NAMES:
        print(f"--- {name}")
        for path in sorted(ROOT.rglob(name + "*")):
            print(path)


if __name__ == "__main__":
    main()
