#!/usr/bin/env python3
"""Download official Search-R1 retrieval assets from ModelScope mirrors."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/retrieval/wiki-18"))
    parser.add_argument("--index-repo", default="yamseyoung/wiki-18-e5-index")
    parser.add_argument("--corpus-repo", default="yamseyoung/wiki-18-corpus")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / ".modelscope_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.index_repo,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        local_dir=str(args.out_dir),
        allow_file_pattern=["part_aa", "part_ab"],
    )
    snapshot_download(
        repo_id=args.corpus_repo,
        repo_type="dataset",
        cache_dir=str(cache_dir),
        local_dir=str(args.out_dir),
        allow_file_pattern=["wiki-18.jsonl.gz"],
    )

    print(f"Downloaded ModelScope retrieval assets into {args.out_dir}")


if __name__ == "__main__":
    main()
