#!/usr/bin/env python3
"""Create CoAgenticRetriever ablation parquet files with reason tags renamed to think tags.

The script keeps the original parquet schema and only applies this literal prompt
text replacement:

    <reason>  -> <think>
    </reason> -> </think>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "coAgenticRetriever" / "albation_1"


def replace_prompt_tags(prompt: Any) -> list[dict[str, Any]]:
    new_prompt: list[dict[str, Any]] = []
    for message in prompt:
        item = dict(message)
        content = str(item.get("content", ""))
        item["content"] = content.replace("<reason>", "<think>").replace("</reason>", "</think>")
        new_prompt.append(item)
    return new_prompt


def count_tags(df: pd.DataFrame, open_tag: str, close_tag: str) -> int:
    count = 0
    for prompt in df["prompt"]:
        for message in prompt:
            content = str(message.get("content", ""))
            count += content.count(open_tag) + content.count(close_tag)
    return count


def convert_one(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        raise FileNotFoundError(f"source parquet not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"output exists, pass --overwrite to replace: {dst}")

    source_schema = pq.ParquetFile(src).schema_arrow
    df = pd.read_parquet(src).copy(deep=True)
    before_reason = count_tags(df, "<reason>", "</reason>")
    before_think = count_tags(df, "<think>", "</think>")

    df["prompt"] = df["prompt"].apply(replace_prompt_tags)
    after_reason = count_tags(df, "<reason>", "</reason>")
    after_think = count_tags(df, "<think>", "</think>")

    table = pa.Table.from_pandas(df, schema=source_schema, preserve_index=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, dst)

    print(
        f"wrote {dst} rows={table.num_rows} "
        f"reason_tags={before_reason}->{after_reason} "
        f"think_tags={before_think}->{after_think}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert <reason> prompt tags to <think> tags in CoAgenticRetriever parquet files.",
    )
    parser.add_argument(
        "--eval-src",
        type=Path,
        default=DEFAULT_DATA_DIR / "co_search_ablation.eval.parquet.bak",
    )
    parser.add_argument(
        "--train-src",
        type=Path,
        default=DEFAULT_DATA_DIR / "co_search_ablation.train.parquet.bak",
    )
    parser.add_argument(
        "--eval-out",
        type=Path,
        default=DEFAULT_DATA_DIR / "co_search_ablation_think.eval.parquet",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=DEFAULT_DATA_DIR / "co_search_ablation_think.train.parquet",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    convert_one(args.eval_src, args.eval_out, args.overwrite)
    convert_one(args.train_src, args.train_out, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
