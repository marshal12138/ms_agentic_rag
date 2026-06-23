#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=240)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    if args.start < 0:
        raise ValueError("--start must be non-negative")
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")

    subset = df.iloc[args.start : args.start + args.max_samples].copy()
    if subset.empty:
        raise ValueError(f"empty subset from {src} at start={args.start}")

    subset.to_parquet(dst, index=False)
    print(f"wrote {len(subset)} rows to {dst}")


if __name__ == "__main__":
    main()
