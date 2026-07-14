#!/usr/bin/env python3
"""Run Stage 2 Phase B teacher-labeling ablations on existing trajectories."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "AgenticIterRag"))

from agentic_iter_rag.agent_training.spad.refresh_rollout import (  # noqa: E402
    Stage2ResourceMonitor,
    _run_teacher_labeling_offline_batch_phase,
    _run_teacher_labeling_phase,
)
from agentic_iter_rag.agent_training.spad.service_manager import (  # noqa: E402
    SpadServiceManager,
    project_root,
)
from agentic_iter_rag.utils.io import write_json  # noqa: E402


DEFAULT_TRAJECTORY = (
    REPO_ROOT
    / "log/agenticIterRag/260710-005039-431413-pipeline-agentic_iter_rag_v1_spad_stage2_parallel_200_verl_dpo1epoch"
    / "outputs/stages/train_agent/spad_rag/answer_refresh_data/answer_refresh_actor_trajectories.jsonl"
)
DEFAULT_OUT_ROOT = REPO_ROOT / "docs/AgenticIterRag_v1/work_report/stage2_teacher_ablation_260710"


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline_http_6": {
        "scheduler": {
            "teacher_submit_batch_size": 32,
            "inflight_per_teacher": 4,
            "max_inflight_per_teacher": 6,
            "progress_log_interval": 20,
            "request_timeout_s": 240,
        },
        "teacher_common": {},
    },
    "http_32_inflight": {
        "scheduler": {
            "teacher_submit_batch_size": 128,
            "inflight_per_teacher": 16,
            "max_inflight_per_teacher": 32,
            "progress_log_interval": 20,
            "request_timeout_s": 600,
        },
        "teacher_common": {},
    },
    "http_32_batch_args": {
        "scheduler": {
            "teacher_submit_batch_size": 128,
            "inflight_per_teacher": 16,
            "max_inflight_per_teacher": 32,
            "progress_log_interval": 20,
            "request_timeout_s": 600,
        },
        "teacher_common": {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 65536,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
    },
    "http_32_batch_no_eager": {
        "scheduler": {
            "teacher_submit_batch_size": 128,
            "inflight_per_teacher": 16,
            "max_inflight_per_teacher": 32,
            "progress_log_interval": 20,
            "request_timeout_s": 600,
        },
        "teacher_common": {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 65536,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
        },
    },
    "offline_batch_no_eager": {
        "backend": "offline_vllm_batch",
        "scheduler": {
            "offline_shard_count": 4,
            "offline_batch_size": 64,
            "max_num_seqs": 128,
            "max_num_batched_tokens": 65536,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
            "progress_log_interval": 20,
            "offline_poll_interval_s": 5,
        },
        "teacher_common": {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 65536,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "enforce_eager": False,
        },
    },
}


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def cleanup_teacher_containers(resource_cfg: dict[str, Any], variant_name: str) -> None:
    names: list[str] = []
    for replica in resource_cfg.get("replicas", []):
        for key in ("container_name", "name"):
            value = replica.get(key)
            if value:
                names.append(str(value))
    names.extend(f"spad_ablate_{variant_name}_{port}" for port in (8067, 8068, 8069, 8070))
    names.extend(f"spad_glm47_vllm_{port}" for port in (8067, 8068, 8069, 8070))
    if names:
        subprocess.run(["docker", "rm", "-f", *sorted(set(names))], cwd=str(REPO_ROOT), check=False)


def variant_resource(base_resource: dict[str, Any], variant_name: str, variant_cfg: dict[str, Any]) -> dict[str, Any]:
    resource = copy.deepcopy(base_resource)
    common = dict(resource.get("common") or {})
    common.update(variant_cfg.get("teacher_common") or {})
    resource["common"] = common
    for replica in resource.get("replicas", []):
        port = int(replica["port"])
        replica["name"] = f"spad_ablate_{variant_name}_{port}"
        replica["container_name"] = f"spad_ablate_{variant_name}_{port}"
    return resource


def resource_summary(resource_jsonl: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    if resource_jsonl.exists():
        for line in resource_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                samples.append(json.loads(line))
    loaded = [
        sample
        for sample in samples
        if sample.get("phase") == "teacher_labeling"
        and sample.get("npu")
        and min(int(card.get("hbm_mb", 0)) for card in sample["npu"]) > 20000
    ]
    active = [
        sample
        for sample in loaded
        if int(sample.get("phase_b_seen", 0)) > 0 or int(sample.get("dpo_pairs", 0)) > 0
    ]
    target = active or loaded
    utils = [int(card.get("aicore_util", 0)) for sample in target for card in sample.get("npu", [])]
    hbms = [int(card.get("hbm_mb", 0)) for sample in loaded for card in sample.get("npu", [])]
    return {
        "monitor_samples": len(samples),
        "loaded_samples": len(loaded),
        "active_samples": len(active),
        "avg_aicore": (sum(utils) / len(utils)) if utils else 0.0,
        "max_aicore": max(utils) if utils else 0,
        "avg_hbm_mb": (sum(hbms) / len(hbms)) if hbms else 0.0,
        "max_hbm_mb": max(hbms) if hbms else 0,
    }


def write_markdown_summary(out_root: Path, summaries: list[dict[str, Any]]) -> Path:
    report = REPO_ROOT / "docs/AgenticIterRag_v1/work_report/260710_spad_stage2_teacher_ablation.md"
    lines = [
        "# SPAD Stage2 Teacher PhaseB Ablation",
        "",
        f"Generated at: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"Output root: `{out_root}`",
        "",
        "| Variant | Status | Elapsed(s) | Kept | Avg teacher(s) | Avg AICore | Max AICore | Avg HBM(MB) | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summaries:
        stats = row.get("stats", {}).get("teacher_labeling", {})
        resource = row.get("resource", {})
        notes = row.get("error") or ""
        lines.append(
            "| {name} | {status} | {elapsed:.1f} | {kept} | {avg_teacher:.2f} | {avg_aicore:.1f}% | {max_aicore}% | {avg_hbm:.0f} | {notes} |".format(
                name=row["variant"],
                status=row["status"],
                elapsed=float(row.get("elapsed_s") or 0.0),
                kept=int(stats.get("kept", 0) or 0),
                avg_teacher=float(stats.get("avg_teacher_elapsed_s", 0.0) or 0.0),
                avg_aicore=float(resource.get("avg_aicore", 0.0) or 0.0),
                max_aicore=int(resource.get("max_aicore", 0) or 0),
                avg_hbm=float(resource.get("avg_hbm_mb", 0.0) or 0.0),
                notes=str(notes).replace("|", "/"),
            )
        )
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_variant(
    *,
    variant_name: str,
    variant_cfg: dict[str, Any],
    trajectory_jsonl: Path,
    out_root: Path,
    spad_cfg: dict[str, Any],
    filter_cfg: dict[str, Any],
    base_teacher_resource: dict[str, Any],
    monitor_interval_s: float,
) -> dict[str, Any]:
    out_dir = out_root / variant_name
    out_dir.mkdir(parents=True, exist_ok=True)
    refresh_jsonl = out_dir / "refresh_rollouts.jsonl"
    dataset_jsonl = out_dir / "answer_distill_pairs.jsonl"
    stats_json = out_dir / "stage2_teacher_stats.json"
    resource_jsonl = out_dir / "stage2_teacher_resource_monitor.jsonl"
    monitor_report = out_dir / "stage2_teacher_resource_monitor.md"
    for path in (refresh_jsonl, dataset_jsonl, stats_json, resource_jsonl, monitor_report):
        path.unlink(missing_ok=True)

    teacher_resource = variant_resource(base_teacher_resource, variant_name, variant_cfg)
    cleanup_teacher_containers(teacher_resource, variant_name)
    progress: dict[str, int] = {"phase_a_total": 200, "phase_a_written": 200, "phase_b_seen": 0, "dpo_pairs": 0}
    progress_lock = threading.Lock()

    def set_progress(key: str, value: int) -> None:
        with progress_lock:
            progress[key] = value

    def inc_progress(key: str, amount: int = 1) -> None:
        with progress_lock:
            progress[key] = int(progress.get(key, 0)) + amount

    def get_progress() -> dict[str, int]:
        with progress_lock:
            return dict(progress)

    monitor = Stage2ResourceMonitor(
        jsonl_path=resource_jsonl,
        report_path=monitor_report,
        interval_s=monitor_interval_s,
        get_phase=lambda: "teacher_labeling",
        get_progress=get_progress,
    )
    manager = SpadServiceManager(runtime_dir=out_dir / "services", verl_root=project_root() / "verl")
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "variant": variant_name,
        "status": "failed",
        "output_dir": str(out_dir),
        "scheduler": variant_cfg["scheduler"],
        "teacher_common": variant_cfg.get("teacher_common") or {},
    }
    try:
        monitor.start()
        if str(variant_cfg.get("backend") or "http") == "offline_vllm_batch":
            stats = _run_teacher_labeling_offline_batch_phase(
                trajectory_jsonl=trajectory_jsonl,
                refresh_jsonl=refresh_jsonl,
                dataset_jsonl=dataset_jsonl,
                teacher_resource=teacher_resource,
                runtime_dir=out_dir / "services",
                spad_cfg=spad_cfg,
                filter_cfg=filter_cfg,
                teacher_request=dict(spad_cfg["teacher_answerer"].get("request") or {}),
                scheduler_cfg=dict(variant_cfg["scheduler"]),
                resume_existing=False,
                inc_progress=inc_progress,
                set_progress=set_progress,
            )
        else:
            teacher_outputs = manager.start_teacher_replicas(
                teacher_cfg=spad_cfg["teacher_answerer"],
                resource_cfg=teacher_resource,
            )
            stats = _run_teacher_labeling_phase(
                trajectory_jsonl=trajectory_jsonl,
                refresh_jsonl=refresh_jsonl,
                dataset_jsonl=dataset_jsonl,
                teacher_outputs=teacher_outputs,
                spad_cfg=spad_cfg,
                filter_cfg=filter_cfg,
                teacher_request=dict(spad_cfg["teacher_answerer"].get("request") or {}),
                scheduler_cfg=dict(variant_cfg["scheduler"]),
                resume_existing=False,
                inc_progress=inc_progress,
                set_progress=set_progress,
            )
        elapsed = time.perf_counter() - started
        summary.update(
            {
                "status": "completed",
                "elapsed_s": elapsed,
                "stats": {"teacher_labeling": stats},
                "resource": resource_summary(resource_jsonl),
                "refresh_jsonl": str(refresh_jsonl),
                "dataset_jsonl": str(dataset_jsonl),
                "resource_jsonl": str(resource_jsonl),
                "resource_report": str(monitor_report),
            }
        )
        write_json(stats_json, summary)
        return summary
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - started
        summary.update({"elapsed_s": elapsed, "error": f"{type(exc).__name__}: {exc}", "resource": resource_summary(resource_jsonl)})
        write_json(stats_json, summary)
        return summary
    finally:
        monitor.stop()
        manager.stop_all()
        cleanup_teacher_containers(teacher_resource, variant_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-jsonl", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--variants", default="baseline_http_6,http_32_inflight,http_32_batch_args")
    parser.add_argument("--monitor-interval-s", type=float, default=5.0)
    args = parser.parse_args()

    if not args.trajectory_jsonl.exists():
        raise FileNotFoundError(args.trajectory_jsonl)
    args.out_root.mkdir(parents=True, exist_ok=True)
    selected = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variants: {unknown}; known={sorted(VARIANTS)}")

    spad_cfg = load_yaml(REPO_ROOT / "AgenticIterRag/config/agent_training/spad_rag_base.yaml")
    resource_cfg = load_yaml(REPO_ROOT / "AgenticIterRag/config/resource/local_8gpu_0_7.yaml")
    base_teacher_resource = resource_cfg["stage_resources"]["train_agent"]["impls"]["spad_rag"]["sub_stages"][
        "answer_refresh_data"
    ]["services"]["teacher_answerer"]
    filter_cfg = {
        "require_teacher_format_valid": True,
        "require_evidence_sufficient": True,
        "min_teacher_f1": 0.0,
    }

    summaries: list[dict[str, Any]] = []
    for variant_name in selected:
        print(f"[ablation] start {variant_name}", flush=True)
        summary = run_variant(
            variant_name=variant_name,
            variant_cfg=VARIANTS[variant_name],
            trajectory_jsonl=args.trajectory_jsonl,
            out_root=args.out_root,
            spad_cfg=spad_cfg,
            filter_cfg=filter_cfg,
            base_teacher_resource=base_teacher_resource,
            monitor_interval_s=float(args.monitor_interval_s),
        )
        print(
            f"[ablation] done {variant_name}: status={summary['status']} elapsed={summary.get('elapsed_s', 0):.1f}s "
            f"avg_aicore={summary.get('resource', {}).get('avg_aicore', 0):.1f}%",
            flush=True,
        )
        summaries.append(summary)
        (args.out_root / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_markdown_summary(args.out_root, summaries)

    report = write_markdown_summary(args.out_root, summaries)
    print(f"[ablation] report={report}", flush=True)


if __name__ == "__main__":
    main()
