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

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel


def now_s() -> float:
    return time.time()


def json_log(log_path: Path, event: str, **payload) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_s(), "event": event, **payload}, ensure_ascii=False) + "\n")


class GpuSampler:
    def __init__(self, gpu_id: int, out_csv: Path, interval_s: float = 1.0):
        self.gpu_id = gpu_id
        self.out_csv = out_csv
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self.out_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "ts",
                    "gpu_id",
                    "memory_used_mib",
                    "memory_total_mib",
                    "utilization_gpu_pct",
                    "power_draw_w",
                    "temperature_c",
                ]
            )
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        query = "memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_id}",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if out:
                    fields = [x.strip() for x in out.splitlines()[0].split(",")]
                    with self.out_csv.open("a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([now_s(), self.gpu_id, *fields])
            except Exception:
                pass
            self._stop.wait(self.interval_s)


def pool_mean(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def summarize_gpu(csv_path: Path) -> dict:
    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "memory_used_mib": float(row["memory_used_mib"]),
                        "utilization_gpu_pct": float(row["utilization_gpu_pct"]),
                        "power_draw_w": float(row["power_draw_w"]),
                        "temperature_c": float(row["temperature_c"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    if not rows:
        return {}
    return {
        "sample_count": len(rows),
        "memory_used_peak_mib": max(x["memory_used_mib"] for x in rows),
        "memory_used_avg_mib": sum(x["memory_used_mib"] for x in rows) / len(rows),
        "gpu_util_avg_pct": sum(x["utilization_gpu_pct"] for x in rows) / len(rows),
        "gpu_util_peak_pct": max(x["utilization_gpu_pct"] for x in rows),
        "power_avg_w": sum(x["power_draw_w"] for x in rows) / len(rows),
        "temperature_peak_c": max(x["temperature_c"] for x in rows),
    }


@torch.no_grad()
def encode_batch(model, input_ids_np: np.ndarray, attention_mask_np: np.ndarray, device: str) -> np.ndarray:
    input_ids = torch.from_numpy(np.asarray(input_ids_np, dtype=np.int64)).to(device, non_blocking=True)
    attention_mask = torch.from_numpy(np.asarray(attention_mask_np, dtype=np.int64)).to(device, non_blocking=True)
    out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    emb = pool_mean(out.last_hidden_state, attention_mask)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.float().cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode pre-tokenized wiki-18 E5 passages into a FlatIP FAISS index.")
    parser.add_argument("--tokens-meta", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-id", type=int, default=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.out_dir / "encode_from_tokens_events.jsonl"
    gpu_csv = args.out_dir / "gpu_samples_encode_from_tokens.csv"
    summary_path = args.out_dir / "encode_from_tokens_summary.json"
    index_path = args.out_dir / "e5_Flat.index"

    meta = json.loads(args.tokens_meta.read_text(encoding="utf-8"))
    num_docs = int(meta["num_docs"])
    max_length = int(meta["max_length"])
    input_ids = np.memmap(meta["input_ids_path"], mode="r", dtype=np.uint16, shape=(num_docs, max_length))
    attention_mask = np.memmap(meta["attention_mask_path"], mode="r", dtype=np.uint8, shape=(num_docs, max_length))

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    json_log(
        events_path,
        "start",
        tokens_meta=str(args.tokens_meta),
        model=str(args.model),
        out_dir=str(args.out_dir),
        batch_size=args.batch_size,
        num_docs=num_docs,
        max_length=max_length,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )

    sampler = GpuSampler(args.gpu_id, gpu_csv)
    sampler.start()
    start = now_s()
    index = None
    docs_seen = 0
    try:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(args.device).eval()
        if args.device.startswith("cuda"):
            model = model.half()
            torch.cuda.reset_peak_memory_stats()

        pbar = tqdm(total=num_docs, desc="encode cached token docs", unit="doc")
        for batch_start in range(0, num_docs, args.batch_size):
            batch_end = min(batch_start + args.batch_size, num_docs)
            emb = encode_batch(model, input_ids[batch_start:batch_end], attention_mask[batch_start:batch_end], args.device)
            if index is None:
                index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            docs_seen = batch_end
            pbar.update(batch_end - batch_start)
            if docs_seen % (args.batch_size * 20) == 0:
                elapsed = now_s() - start
                json_log(
                    events_path,
                    "progress",
                    docs_seen=docs_seen,
                    elapsed_s=elapsed,
                    docs_per_s=docs_seen / elapsed if elapsed > 0 else None,
                    cuda_peak_allocated_mib=(
                        torch.cuda.max_memory_allocated() / 1024**2 if args.device.startswith("cuda") else None
                    ),
                    rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                )
        pbar.close()

        encode_elapsed = now_s() - start
        json_log(events_path, "encode_done", docs_seen=docs_seen, elapsed_s=encode_elapsed)
        write_start = now_s()
        faiss.write_index(index, str(index_path))
        write_elapsed = now_s() - write_start
        total_elapsed = now_s() - start

        cuda_summary = {}
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            cuda_summary = {
                "torch_cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "torch_cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            }
        summary = {
            "tokens_meta": str(args.tokens_meta),
            "model": str(args.model),
            "index_path": str(index_path),
            "docs_seen": docs_seen,
            "embedding_dim": index.d if index is not None else None,
            "index_ntotal": index.ntotal if index is not None else None,
            "batch_size": args.batch_size,
            "max_length": max_length,
            "device": args.device,
            "encode_elapsed_s": encode_elapsed,
            "write_index_elapsed_s": write_elapsed,
            "total_elapsed_s": total_elapsed,
            "index_bytes": index_path.stat().st_size,
            "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "gpu_samples": summarize_gpu(gpu_csv),
            **cuda_summary,
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        json_log(events_path, "summary_written", summary_path=str(summary_path))
    finally:
        sampler.stop()


if __name__ == "__main__":
    main()
