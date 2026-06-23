#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def write_subset(src: Path, dst: Path, count: int) -> None:
    table = pq.read_table(src)
    subset = table.slice(0, min(count, table.num_rows))
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset.replace_schema_metadata(table.schema.metadata), dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-src", type=Path, required=True)
    parser.add_argument("--eval-src", type=Path, required=True)
    parser.add_argument("--train-dst", type=Path, required=True)
    parser.add_argument("--eval-dst", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=4)
    parser.add_argument("--eval-count", type=int, default=1)
    args = parser.parse_args()

    write_subset(args.train_src, args.train_dst, args.train_count)
    write_subset(args.eval_src, args.eval_dst, args.eval_count)

    print(f"train_subset={args.train_dst}")
    print(f"eval_subset={args.eval_dst}")


if __name__ == "__main__":
    main()
