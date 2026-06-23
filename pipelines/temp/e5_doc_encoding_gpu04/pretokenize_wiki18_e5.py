#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def now_s() -> float:
    return time.time()


def json_log(log_path: Path, event: str, **payload) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_s(), "event": event, **payload}, ensure_ascii=False) + "\n")


class CpuSampler:
    def __init__(self, out_csv: Path, interval_s: float = 1.0):
        self.out_csv = out_csv
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["ts", "rss_mib", "load1", "load5", "load15"])
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                load1, load5, load15 = os.getloadavg()
                with self.out_csv.open("a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([now_s(), rss_mib, load1, load5, load15])
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def iter_jsonl_contents(corpus_path: Path, limit: int = 0) -> Iterable[str]:
    with corpus_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            item = json.loads(line)
            yield item.get("contents") or item.get("text") or ""


def count_lines(path: Path) -> int:
    out = subprocess.check_output(["wc", "-l", str(path)], text=True)
    return int(out.strip().split()[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize wiki-18 E5 passages into fixed-size memmaps.")
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-docs", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.out_dir / "pretokenize_events.jsonl"
    cpu_csv = args.out_dir / "cpu_samples_pretokenize.csv"
    meta_path = args.out_dir / "tokens_meta.json"
    input_ids_path = args.out_dir / "input_ids.uint16.memmap"
    attention_mask_path = args.out_dir / "attention_mask.uint8.memmap"

    num_docs = args.limit or args.num_docs or count_lines(args.corpus)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        raise ValueError(f"uint16 input_ids cannot hold vocab_size={tokenizer.vocab_size}")

    json_log(
        events_path,
        "start",
        corpus=str(args.corpus),
        model=str(args.model),
        out_dir=str(args.out_dir),
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_docs=num_docs,
        tokenizers_parallelism=os.environ.get("TOKENIZERS_PARALLELISM"),
        rayon_num_threads=os.environ.get("RAYON_NUM_THREADS"),
    )

    input_ids = np.memmap(input_ids_path, mode="w+", dtype=np.uint16, shape=(num_docs, args.max_length))
    attention_mask = np.memmap(attention_mask_path, mode="w+", dtype=np.uint8, shape=(num_docs, args.max_length))

    sampler = CpuSampler(cpu_csv)
    sampler.start()
    start = now_s()
    docs_seen = 0
    batch: list[str] = []
    try:
        pbar = tqdm(total=num_docs, desc="pretokenize wiki-18 docs", unit="doc")
        for text in iter_jsonl_contents(args.corpus, args.limit):
            batch.append(f"passage: {text}")
            if len(batch) < args.batch_size:
                continue
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="np",
            )
            end = docs_seen + len(batch)
            input_ids[docs_seen:end] = encoded["input_ids"].astype(np.uint16, copy=False)
            attention_mask[docs_seen:end] = encoded["attention_mask"].astype(np.uint8, copy=False)
            docs_seen = end
            pbar.update(len(batch))
            if docs_seen % (args.batch_size * 20) == 0:
                elapsed = now_s() - start
                json_log(
                    events_path,
                    "progress",
                    docs_seen=docs_seen,
                    elapsed_s=elapsed,
                    docs_per_s=docs_seen / elapsed if elapsed > 0 else None,
                    rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                )
            batch.clear()

        if batch:
            encoded = tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=args.max_length,
                return_tensors="np",
            )
            end = docs_seen + len(batch)
            input_ids[docs_seen:end] = encoded["input_ids"].astype(np.uint16, copy=False)
            attention_mask[docs_seen:end] = encoded["attention_mask"].astype(np.uint8, copy=False)
            docs_seen = end
            pbar.update(len(batch))
        pbar.close()

        input_ids.flush()
        attention_mask.flush()
        elapsed = now_s() - start
        meta = {
            "corpus": str(args.corpus),
            "model": str(args.model),
            "num_docs": docs_seen,
            "max_length": args.max_length,
            "input_ids_path": str(input_ids_path),
            "attention_mask_path": str(attention_mask_path),
            "input_ids_dtype": "uint16",
            "attention_mask_dtype": "uint8",
            "pretokenize_elapsed_s": elapsed,
            "pretokenize_docs_per_s": docs_seen / elapsed if elapsed > 0 else None,
            "input_ids_bytes": input_ids_path.stat().st_size,
            "attention_mask_bytes": attention_mask_path.stat().st_size,
            "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        json_log(events_path, "done", **meta)
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
