#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def now_s() -> float:
    return time.time()


def append_event(events_path: Path, event: str, **payload) -> None:
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now_s(), "event": event, **payload}, ensure_ascii=False) + "\n")


def sample_gpu(gpu_id: int) -> list[str] | None:
    query = "memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    return [x.strip() for x in out.splitlines()[0].split(",")]


def is_port_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/gpu_status", timeout=2) as resp:
            return resp.status < 500
    except (urllib.error.URLError, TimeoutError):
        return False


def get_gpu_status(port: int) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/gpu_status", timeout=2) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_gpu_header(gpu_csv: Path):
    f = gpu_csv.open("w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(
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
    return f, writer


def wait_ready(
    *,
    stage: str,
    port: int,
    timeout_s: float,
    gpu_id: int,
    gpu_csv: Path,
    events_path: Path,
    proc: subprocess.Popen,
) -> tuple[bool, float]:
    start = now_s()
    with write_gpu_header(gpu_csv)[0] as f:
        writer = csv.writer(f)
        while now_s() - start < timeout_s:
            sample = sample_gpu(gpu_id)
            if sample:
                writer.writerow([now_s(), gpu_id, *sample])
                f.flush()
            if proc.poll() is not None:
                append_event(
                    events_path,
                    f"{stage}_server_exited_before_ready",
                    returncode=proc.returncode,
                    elapsed_s=now_s() - start,
                )
                return False, now_s() - start
            try:
                payload = get_gpu_status(port)
                elapsed = now_s() - start
                append_event(events_path, f"{stage}_ready", elapsed_s=elapsed, gpu_status=payload)
                return True, elapsed
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(1)
    return False, now_s() - start


def summarize_gpu(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {}
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


def parse_doc_load_event(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    pattern = re.compile(r"\{.*doc_embeddings_loaded_to_gpu.*\}")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if pattern.search(line):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def build_server_cmd(args: argparse.Namespace, index_path: str) -> list[str]:
    return [
        os.environ.get("PY", "/data04/envs/ms/ms_cosearch_official/bin/python"),
        args.server_path,
        "--index_path",
        index_path,
        "--corpus_path",
        args.corpus_path,
        "--topk",
        "50",
        "--retriever_name",
        "e5",
        "--retriever_model",
        args.retriever_model,
        "--host",
        "0.0.0.0",
        "--port",
        str(args.port),
        "--device",
        "cuda",
        "--query_batch_size",
        str(args.query_batch_size),
        "--doc_dtype",
        args.doc_dtype,
    ]


def start_server(cmd: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    proc._codex_log_handle = log  # type: ignore[attr-defined]
    return proc


def close_proc_log(proc: subprocess.Popen) -> None:
    log = getattr(proc, "_codex_log_handle", None)
    if log is not None:
        log.close()


def stop_server(proc: subprocess.Popen, port: int, gpu_id: int, gpu_csv: Path, events_path: Path, timeout_s: float) -> dict:
    start = now_s()
    append_event(events_path, "old_shutdown_signal_sent", pid=proc.pid, signal="SIGTERM")
    with write_gpu_header(gpu_csv)[0] as f:
        writer = csv.writer(f)
        os.killpg(proc.pid, signal.SIGTERM)
        process_exited = False
        port_released = False
        killed = False
        while now_s() - start < timeout_s:
            sample = sample_gpu(gpu_id)
            if sample:
                writer.writerow([now_s(), gpu_id, *sample])
                f.flush()
            if not process_exited and proc.poll() is not None:
                process_exited = True
                append_event(events_path, "old_process_exited", returncode=proc.returncode, elapsed_s=now_s() - start)
            if process_exited and not is_port_ready(port):
                port_released = True
                break
            time.sleep(0.5)
        if not process_exited:
            killed = True
            append_event(events_path, "old_shutdown_kill_sent", pid=proc.pid, signal="SIGKILL")
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)
            process_exited = True
            port_released = not is_port_ready(port)
    elapsed = now_s() - start
    append_event(
        events_path,
        "old_shutdown_done",
        elapsed_s=elapsed,
        process_exited=process_exited,
        port_released=port_released,
        killed=killed,
    )
    close_proc_log(proc)
    return {
        "shutdown_elapsed_s": elapsed,
        "process_exited": process_exited,
        "port_released": port_released,
        "killed": killed,
        "gpu_samples": summarize_gpu(gpu_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure GPU retriever restart and full embedding load.")
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--old-index-path", required=True)
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--corpus-path", required=True)
    parser.add_argument("--retriever-model", required=True)
    parser.add_argument("--server-path", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--doc-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--timeout-s", type=float, default=3600)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    old_log_path = args.out_dir / "old_retriever_server.log"
    new_log_path = args.out_dir / "new_retriever_server.log"
    old_start_gpu_csv = args.out_dir / "gpu_samples_old_start.csv"
    old_shutdown_gpu_csv = args.out_dir / "gpu_samples_old_shutdown.csv"
    new_start_gpu_csv = args.out_dir / "gpu_samples_new_start.csv"
    events_path = args.out_dir / "load_events.jsonl"
    summary_path = args.out_dir / "load_summary.json"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    events_path.write_text("", encoding="utf-8")
    append_event(
        events_path,
        "start",
        old_index_path=args.old_index_path,
        new_index_path=args.index_path,
        cuda_visible_devices=env["CUDA_VISIBLE_DEVICES"],
        out_dir=str(args.out_dir),
    )

    if is_port_ready(args.port):
        raise RuntimeError(f"port {args.port} already has a ready retriever; choose a free PORT")

    old_cmd = build_server_cmd(args, args.old_index_path)
    new_cmd = build_server_cmd(args, args.index_path)
    old_proc = None
    new_proc = None
    try:
        append_event(events_path, "old_start", cmd=old_cmd, log_path=str(old_log_path))
        old_proc = start_server(old_cmd, env, old_log_path)
        old_ready, old_start_elapsed = wait_ready(
            stage="old",
            port=args.port,
            timeout_s=args.timeout_s,
            gpu_id=args.gpu_id,
            gpu_csv=old_start_gpu_csv,
            events_path=events_path,
            proc=old_proc,
        )
        old_doc_load = parse_doc_load_event(old_log_path)
        if not old_ready:
            raise RuntimeError(f"old retriever did not become ready; see {old_log_path}")

        shutdown_summary = stop_server(
            old_proc,
            args.port,
            args.gpu_id,
            old_shutdown_gpu_csv,
            events_path,
            timeout_s=300,
        )
        old_proc = None

        append_event(events_path, "new_start", cmd=new_cmd, log_path=str(new_log_path))
        new_proc = start_server(new_cmd, env, new_log_path)
        new_ready, new_start_elapsed = wait_ready(
            stage="new",
            port=args.port,
            timeout_s=args.timeout_s,
            gpu_id=args.gpu_id,
            gpu_csv=new_start_gpu_csv,
            events_path=events_path,
            proc=new_proc,
        )
        new_doc_load = parse_doc_load_event(new_log_path)

        summary = {
            "ready": new_ready,
            "old_service": {
                "startup_elapsed_s": old_start_elapsed,
                "index_path": args.old_index_path,
                "doc_embeddings_load_event": old_doc_load,
                "gpu_samples": summarize_gpu(old_start_gpu_csv),
                "log_path": str(old_log_path),
            },
            "old_service_shutdown": shutdown_summary,
            "new_service": {
                "startup_elapsed_s": new_start_elapsed,
                "index_path": args.index_path,
                "doc_embeddings_load_event": new_doc_load,
                "gpu_samples": summarize_gpu(new_start_gpu_csv),
                "log_path": str(new_log_path),
            },
            "restart_total_elapsed_s": shutdown_summary["shutdown_elapsed_s"] + new_start_elapsed,
            "process_pid": new_proc.pid,
            "corpus_path": args.corpus_path,
            "retriever_model": args.retriever_model,
            "gpu_id": args.gpu_id,
            "port": args.port,
            "doc_dtype": args.doc_dtype,
            "query_batch_size": args.query_batch_size,
            "summary_note": "restart_total_elapsed_s = old service shutdown elapsed + new service startup elapsed",
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        for proc in (old_proc, new_proc):
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
            if proc is not None:
                close_proc_log(proc)


if __name__ == "__main__":
    main()
