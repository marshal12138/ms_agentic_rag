#!/usr/bin/env python3
"""Run the AgenticIterRag v1 pipeline as one scheduled task."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ is None or __package__ == "":
    project_root_for_import = Path(__file__).resolve().parents[3] / "AgenticIterRag"
    sys.path.insert(0, str(project_root_for_import))

from agentic_iter_rag.infer_matrix.matrix import build_infer_matrix
from agentic_iter_rag.pipeline.manifest import write_stage_manifest
from agentic_iter_rag.trajectory.extract import extract_file
from agentic_iter_rag.utils.io import (
    copy_file,
    iter_jsonl,
    stable_config_hash,
    write_example,
    write_json,
    write_jsonl,
    read_yaml,
    read_json,
    write_yaml,
)


# ---------------------------------------------------------------------------
# 参数与基础工具：保持 shell 入口简单，所有业务配置都来自 final config。
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AgenticIterRag v1 pipeline.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(text: Any) -> str:
    raw = str(text or "")
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "unknown"


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in items]
    if isinstance(value, list):
        return [int(item) for item in value]
    return [int(value)]


def gpu_csv(value: Any) -> str:
    return ",".join(str(x) for x in as_int_list(value))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in iter_jsonl(path))


def first_jsonl_record(path: Path) -> dict[str, Any] | None:
    for record in iter_jsonl(path):
        return record
    return None


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"value": data}


def ensure_fresh_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists and overwrite=false: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_date_prefix() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%y%m%d")


def suffixed_version(version: str, suffix: str) -> str:
    match = re.match(r"^(\d{6})(_.*)$", version)
    if match:
        return f"{match.group(1)}{suffix}{match.group(2)}"
    return f"{version}_{suffix}"


def resolve_version_dir(base_dir: Path, version: str, *, overwrite: bool, manual_version: bool) -> tuple[str, Path]:
    """解析长期数据版本目录；自动版本冲突时在日期前缀追加 b/c/d。"""

    direct_dir = base_dir / version
    if overwrite or not direct_dir.exists():
        return version, direct_dir
    for suffix_idx in range(1, 26):
        retry_version = suffixed_version(version, chr(ord("a") + suffix_idx))
        retry_dir = base_dir / retry_version
        if not retry_dir.exists():
            return retry_version, retry_dir
    raise FileExistsError(f"cannot find free auto version under {base_dir} for base version={version}")


def selected_stages(pipeline: dict[str, Any]) -> list[str]:
    stages = as_list(pipeline.get("stages"))
    resume_from = pipeline.get("resume_from_stage")
    stop_after = pipeline.get("stop_after_stage")
    skip = set(as_list(pipeline.get("skip_stages")))
    if resume_from:
        if resume_from not in stages:
            raise ValueError(f"pipeline.resume_from_stage is not in stages: {resume_from}")
        stages = stages[stages.index(resume_from) :]
    if stop_after:
        if stop_after not in stages:
            raise ValueError(f"pipeline.stop_after_stage is not in selected stages: {stop_after}")
        stages = stages[: stages.index(stop_after) + 1]
    return [stage for stage in stages if stage not in skip]


def stage_manifest_path(artifact_root: Path, stage: str) -> Path:
    return artifact_root / "stages" / stage / "manifest.json"


def persist_final_config(config_path: Path, config: dict[str, Any]) -> None:
    """stage 产物路径写回 final config，保证后续 stage 仍然只从 YAML 读输入。"""

    write_yaml(config_path, config)
    final_json = os.environ.get("FINAL_CONFIG_JSON")
    if final_json:
        write_json(final_json, config)


def write_basic_stage(stage: str, config: dict[str, Any], manifest: Path, dry_run: bool, outputs: dict[str, Any]) -> None:
    payload = {"status": "compiled" if dry_run else "not_executed"}
    payload.update(outputs)
    write_stage_manifest(manifest, stage=stage, config=config, outputs=payload)


def write_dataset_readme(repo_root: Path, dataset_dir: Path) -> Path:
    """Generate a human-readable README for a produced dataset directory."""

    script = repo_root / "scripts" / "agenticIterRag_v1" / "assets" / "build_dataset_readme.py"
    subprocess.check_call([sys.executable, str(script), "--dataset-dir", str(dataset_dir)])
    return dataset_dir / "README.md"


# ---------------------------------------------------------------------------
# Stage-level resource planning.
# ---------------------------------------------------------------------------


LEGACY_RESOURCE_KEYS = {
    "agent",
    "recall",
    "judge",
    "original_llm_reranker",
    "trained_llm_reranker",
    "wait_for_gpus",
}


def require_new_resource_schema(config: dict[str, Any]) -> dict[str, Any]:
    resource = config.get("resource")
    if not isinstance(resource, dict):
        raise TypeError("resource config must be a mapping")
    legacy = sorted(key for key in LEGACY_RESOURCE_KEYS if key in resource)
    if legacy:
        joined = ", ".join(legacy)
        raise ValueError(f"legacy resource schema is not supported; move these keys under stage_resources: {joined}")
    if not isinstance(resource.get("hardware"), dict):
        raise ValueError("resource.hardware must be set")
    if not isinstance(resource.get("stage_resources"), dict):
        raise ValueError("resource.stage_resources must be set")
    return resource


def stage_resource_plan(config: dict[str, Any], stage: str) -> dict[str, Any]:
    resource = require_new_resource_schema(config)
    plans = resource["stage_resources"]
    if stage not in plans:
        raise ValueError(f"resource.stage_resources.{stage} must be set for selected pipeline stage")
    plan = plans[stage]
    if not isinstance(plan, dict):
        raise TypeError(f"resource.stage_resources.{stage} must be a mapping")
    return plan


def selected_stage_resource_plan(config: dict[str, Any], stages: list[str]) -> dict[str, Any]:
    return {stage: stage_resource_plan(config, stage) for stage in stages}


def normalize_stage_plan(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON/YAML friendly plan with GPU ids normalized to int lists."""

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, child in value.items():
                if key == "gpu_ids":
                    out[key] = as_int_list(child)
                else:
                    out[key] = normalize(child)
            return out
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    normalized = normalize(plan)
    return normalized if isinstance(normalized, dict) else {}


def validate_stage_resource_plan(config: dict[str, Any], stage: str, plan: dict[str, Any]) -> None:
    resource = require_new_resource_schema(config)
    hardware_gpus = set(as_int_list(resource["hardware"].get("gpu_ids")))
    if not hardware_gpus:
        raise ValueError("resource.hardware.gpu_ids must contain at least one GPU id")
    allow_gpu_overlap = bool(plan.get("allow_gpu_overlap"))

    owners: dict[int, str] = {}

    def visit(name: str, node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "gpu_ids" in node:
            gpu_ids = as_int_list(node["gpu_ids"])
            missing = sorted(set(gpu_ids) - hardware_gpus)
            if missing:
                raise ValueError(f"resource stage {stage}.{name} uses GPUs outside hardware.gpu_ids: {missing}")
            for gpu_id in gpu_ids:
                if gpu_id in owners and not allow_gpu_overlap:
                    raise ValueError(
                        f"resource stage {stage} has overlapping GPU {gpu_id}: {owners[gpu_id]} and {name}"
                    )
                owners[gpu_id] = name
            if "tensor_parallel_size" in node and int(node["tensor_parallel_size"]) != len(gpu_ids):
                raise ValueError(
                    f"resource stage {stage}.{name}.tensor_parallel_size must equal len(gpu_ids): "
                    f"{node['tensor_parallel_size']} != {len(gpu_ids)}"
                )
        for child_name, child in node.items():
            if isinstance(child, dict):
                visit(f"{name}.{child_name}" if name else str(child_name), child)

    visit("", plan)

    services = plan.get("services") if isinstance(plan.get("services"), dict) else {}
    recall = services.get("recall")
    if isinstance(recall, dict) and len(as_int_list(recall.get("gpu_ids"))) < 1:
        raise ValueError(f"resource stage {stage}.services.recall.gpu_ids must contain at least one GPU id")
    if isinstance(recall, dict) and recall.get("retrieval_service_url") and recall.get("port"):
        expected = f":{recall['port']}/"
        if expected not in str(recall["retrieval_service_url"]):
            raise ValueError(
                f"resource stage {stage}.services.recall.retrieval_service_url port does not match port={recall['port']}"
            )

    ports: dict[int, str] = {}

    def visit_ports(name: str, node: Any) -> None:
        if not isinstance(node, dict):
            return
        if "port" in node and node["port"] is not None:
            port = int(node["port"])
            if port in ports:
                raise ValueError(f"resource stage {stage} has duplicate port {port}: {ports[port]} and {name}")
            ports[port] = name
        for child_name, child in node.items():
            if isinstance(child, dict):
                visit_ports(f"{name}.{child_name}" if name else str(child_name), child)

    visit_ports("", plan)


def validate_selected_resource_plan(config: dict[str, Any], stages: list[str]) -> dict[str, Any]:
    plans = selected_stage_resource_plan(config, stages)
    for stage, plan in plans.items():
        validate_stage_resource_plan(config, stage, plan)
    return {stage: normalize_stage_plan(plan) for stage, plan in plans.items()}


def stage_wait_config(config: dict[str, Any], stage: str, plan: dict[str, Any]) -> dict[str, Any]:
    resource = require_new_resource_schema(config)
    defaults = resource.get("defaults", {})
    default_wait = defaults.get("wait_for_gpus", {}) if isinstance(defaults, dict) else {}
    stage_wait = plan.get("wait_for_gpus", {}) if isinstance(plan.get("wait_for_gpus"), dict) else {}
    wait_cfg = dict(default_wait)
    wait_cfg.update(stage_wait)
    wait_cfg.setdefault("enable", True)
    wait_cfg.setdefault("interval_seconds", 30)
    wait_cfg.setdefault("label", f"AgenticIterRag v1 {stage} GPU wait")
    return wait_cfg


# ---------------------------------------------------------------------------
# generate_traces：调用 AIR 自有推理 backend，沉淀 AIR canonical trajectory 数据集。
# ---------------------------------------------------------------------------


def trace_data_file(config: dict[str, Any]) -> str:
    files = as_list(config["data"].get("trace_generation_files"))
    if not files:
        raise ValueError("data.trace_generation_files must contain one file for generate_traces")
    if len(files) > 1:
        raise ValueError("generate_traces v1 currently supports exactly one trace_generation_files entry")
    return files[0]


def infer_max_samples(config: dict[str, Any]) -> int:
    value = config["data"].get("trace_max_samples")
    if value is None:
        value = config["infer_budget"].get("max_infer_num", -1)
    if value is None:
        return -1
    return int(value)


def air_backend_reranker_arg(config: dict[str, Any]) -> str:
    """AIR 配置用 none 表达无 reranker；no-ranker infer engine 参数使用 dense_e5 占位。"""

    value = str(config["infer_runtime"]["mode"].get("reranker") or "none")
    if value == "none":
        return "dense_e5"
    if value in {"dense_e5", "llm_as_judge"}:
        return value
    raise ValueError(f"unsupported infer_runtime.mode.reranker={value!r}")


def agent_checkpoint(config: dict[str, Any], stage_cfg: dict[str, Any]) -> str:
    value = stage_cfg.get("inputs", {}).get("agent_checkpoint") or config["infer_runtime"]["models"].get("trained_agent_model")
    if not value:
        raise ValueError("generate_traces requires infer_runtime.models.trained_agent_model or stage input agent_checkpoint")
    return str(value)


def auto_trajectory_version(config: dict[str, Any], stage_cfg: dict[str, Any]) -> str:
    stamp = local_date_prefix()
    data_slug = slugify(Path(trace_data_file(config)).stem)
    ckpt_slug = slugify(Path(agent_checkpoint(config, stage_cfg)).name)
    return f"{stamp}_AIR_v1_traj_{data_slug}_{ckpt_slug}"


def build_air_infer_command(
    *,
    repo_root: Path,
    stage_cfg: dict[str, Any],
) -> list[str]:
    """构造 AIR infer launcher 命令；业务运行参数全部通过子进程 env 注入。"""

    air_entry = repo_root / str(stage_cfg["entry"])
    return [
        "bash",
        str(air_entry),
    ]


def resolve_air_tool_config(repo_root: Path, config: dict[str, Any]) -> Path:
    explicit = config["infer_runtime"].get("tool", {}).get("tool_config")
    if explicit:
        return Path(str(explicit))
    project_root = repo_root / "AgenticIterRag"
    if str(config["infer_runtime"]["mode"].get("run_mode")) == "no-ranker":
        return project_root / "config" / "agentic_iter_rag_tool_config_no_ranker.yaml"
    return project_root / "config" / "agentic_iter_rag_tool_config.yaml"


def air_infer_runtime_env(
    *,
    repo_root: Path,
    config: dict[str, Any],
    stage_cfg: dict[str, Any],
    resource_plan: dict[str, Any],
    infer_trace_dir: Path,
    infer_report_path: Path,
) -> dict[str, str]:
    """把 AIR 推理配置翻译为 AIR infer launcher 使用的运行时环境变量。"""

    infer_budget = config["infer_budget"]
    infer_runtime = config["infer_runtime"]
    agent_runtime = infer_runtime.get("agent", {})
    if not isinstance(agent_runtime, dict):
        raise ValueError("infer_runtime.agent must be a mapping when set")
    services = resource_plan.get("services", {})
    if not isinstance(services, dict):
        raise ValueError("resource.stage_resources.generate_traces.services must be set")
    agent_vllm = services.get("agent_vllm")
    recall = services.get("recall")
    if not isinstance(agent_vllm, dict):
        raise ValueError("resource.stage_resources.generate_traces.services.agent_vllm must be set")
    if not isinstance(recall, dict):
        raise ValueError("resource.stage_resources.generate_traces.services.recall must be set")
    max_num_seqs = infer_budget.get("vllm", {}).get("max_num_seqs") or infer_budget.get("infer_batch_size")
    task_name = f"{os.environ.get('RUN_NAME', 'air')}-generate_traces"
    run_name = "air_generate_traces"
    project_root = repo_root / "AgenticIterRag"
    env = {
        "AIR_INFER_PRECOMPILED_ENV": "1",
        "AGENTIC_ITER_RAG_PROJECT_ROOT": str(project_root),
        "PROJECT_ROOT": str(project_root),
        "INFER_ENGINE": str(repo_root / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "infer_air_vllm.py"),
        "TOOL_CONFIG": str(resolve_air_tool_config(repo_root, config)),
        "INFER_TASK_NAME": "agentic_iter_rag_v1_generate_traces",
        "INFER_RUNTIME_CONFIG": str(config["main_run"]["config_groups"].get("infer_runtime")),
        "INFER_BUDGET_CONFIG": str(config["main_run"]["config_groups"].get("infer_budget")),
        "RUN_MODE": str(infer_runtime["mode"]["run_mode"]),
        "RERANKER": air_backend_reranker_arg(config),
        "AGENT_MODEL": agent_checkpoint(config, stage_cfg),
        "RECALL_MODEL_PATH": str(infer_runtime["models"]["recall_model_path"]),
        "DATA_PATH": trace_data_file(config),
        "MAX_INFER_NUM": str(infer_max_samples(config)),
        "INFER_BATCH_SIZE": str(infer_budget["infer_batch_size"]),
        "KEEP_TRACE": str(infer_runtime["artifacts"]["keep_trace"]),
        "FLUSH_EVERY_N": str(infer_runtime["artifacts"]["flush_every_n"]),
        "RECALL_FINAL_TOP_N": str(infer_runtime["retrieval"]["final_top_n"]),
        "SEARCH_TOOL_FINAL_TOP_M": str(infer_runtime["retrieval"]["visible_top_m"]),
        "TEMPERATURE": str(infer_budget["temperature"]),
        "TOP_P": str(infer_budget["top_p"]),
        "REQUEST_TIMEOUT": str(infer_budget["request_timeout"]),
        "AGENT_MAX_RETRIES": str(agent_runtime.get("max_retries", 3)),
        "AGENT_RETRY_DELAY": str(agent_runtime.get("retry_delay", 1.0)),
        "AGENT_RETRY_BACKOFF": str(agent_runtime.get("retry_backoff", 2.0)),
        "AGENT_HTTP_FORCE_CLOSE": str(agent_runtime.get("http_force_close", True)).lower(),
        "FAIL_ON_INFER_ERROR": str(agent_runtime.get("fail_on_error", True)).lower(),
        "MAX_MODEL_LEN": str(infer_budget["vllm"]["max_model_len"]),
        "AGENT_PORT": str(agent_vllm["port"]),
        "AGENT_SERVED_MODEL": str(agent_vllm["served_model_name"]),
        "AGENT_TP_SIZE": str(agent_vllm["tensor_parallel_size"]),
        "VLLM_STARTUP_TIMEOUT": str(infer_budget["vllm"]["startup_timeout"]),
        "GPU_MEMORY_UTILIZATION": str(infer_budget["vllm"]["gpu_memory_utilization"]),
        "MAX_NUM_SEQS": str(max_num_seqs),
        "PROXY_PORT": str(recall["port"]),
        "RETRIEVAL_SERVICE_URL": str(recall["retrieval_service_url"]),
        "RETRIEVAL_MAX_RETRIES": str(infer_runtime["retrieval"]["max_retries"]),
        "RETRIEVAL_RETRY_DELAY": str(infer_runtime["retrieval"]["retry_delay"]),
        "RETRIEVAL_RETRY_BACKOFF": str(infer_runtime["retrieval"]["retry_backoff"]),
        "TASK_NAME": task_name,
        "RUN_NAME": run_name,
        "EXP_NAME": str(os.environ.get("EXP_NAME", run_name)),
        "TRACE_DIR": str(infer_trace_dir),
        "OUT_DIR": str(infer_trace_dir),
        "RUNTIME_LOG_DIR": str(infer_trace_dir / "runtime_logs"),
        "LOG_DIR": str(infer_trace_dir / "runtime_logs"),
        "REPORT_PATH": str(infer_report_path),
        "AGENT_TIMING_JSONL": str(infer_trace_dir / "runtime_logs" / f"{run_name}.agent_timing.jsonl"),
        "AGENTIC_ITER_RAG_SEARCH_TIMING_JSONL": str(infer_trace_dir / "runtime_logs" / f"{run_name}.search_timing.jsonl"),
    }
    attention_backend = infer_budget.get("vllm", {}).get("attention_backend")
    if attention_backend is not None:
        env["VLLM_ATTENTION_BACKEND"] = str(attention_backend)
    stop_sequences = infer_budget.get("stop_sequences")
    if stop_sequences is not None:
        env["STOP_SEQUENCES"] = ",".join(as_list(stop_sequences))
    return env


def air_resource_env(config: dict[str, Any], stage: str, resource_plan: dict[str, Any]) -> dict[str, str]:
    """把 AIR 结构化资源字段翻译成 AIR infer launcher 使用的资源环境变量。"""

    services = resource_plan.get("services", {})
    if not isinstance(services, dict):
        raise ValueError(f"resource.stage_resources.{stage}.services must be set")
    agent_vllm = services.get("agent_vllm")
    recall = services.get("recall")
    dense_reranker = services.get("dense_reranker", {})
    llm_judge = services.get("llm_judge", {})
    if not isinstance(agent_vllm, dict):
        raise ValueError(f"resource.stage_resources.{stage}.services.agent_vllm must be set")
    if not isinstance(recall, dict):
        raise ValueError(f"resource.stage_resources.{stage}.services.recall must be set")
    if not isinstance(dense_reranker, dict):
        dense_reranker = {}
    if not isinstance(llm_judge, dict):
        llm_judge = {}
    wait_cfg = stage_wait_config(config, stage, resource_plan)
    return {
        "GROUP_NAME": "agenticIterRag",
        "AGENT_GPU_IDS": gpu_csv(agent_vllm["gpu_ids"]),
        "RANK_GPU_ID": gpu_csv(dense_reranker.get("gpu_ids")),
        "RECALL_GPU_ID": gpu_csv(recall["gpu_ids"]),
        "RECALL_BACKEND_BASE_PORT": str(recall.get("backend_base_port") or ""),
        "LLM_JUDGE_GPU_IDS": gpu_csv(llm_judge.get("gpu_ids")),
        "AUTO_START_RECALL_SERVICE": "1" if recall.get("auto_start") else "0",
        "AUTO_STOP_RECALL_SERVICE": "1" if recall.get("auto_stop") else "0",
        "RECALL_SERVICE_WAIT_SECONDS": str(recall.get("wait_seconds") or 240),
        "LLM_JUDGE_ENDPOINT": str(llm_judge.get("endpoint") or ""),
        "LLM_JUDGE_MODEL": str(llm_judge.get("model") or ""),
        "AUTO_START_LLM_JUDGE": "1" if llm_judge.get("auto_start") else "0",
        "AUTO_STOP_LLM_JUDGE": "1" if llm_judge.get("auto_stop") else "0",
        "LLM_JUDGE_PREFLIGHT": "1" if llm_judge.get("preflight") else "0",
        "LLM_JUDGE_WAIT_SECONDS": str(llm_judge.get("wait_seconds") or 240),
        "WAIT_FOR_GPU_RELEASE": "1" if wait_cfg.get("enable") else "0",
        "WAIT_FOR_GPU_INTERVAL_SECONDS": str(wait_cfg.get("interval_seconds") or 30),
        "WAIT_FOR_GPU_LABEL": str(wait_cfg.get("label") or f"AgenticIterRag v1 {stage} GPU wait"),
    }


def prepare_trajectory_dir(config: dict[str, Any], stage_cfg: dict[str, Any]) -> tuple[str, Path]:
    artifact_cfg = config["main_run"]["data_artifacts"]["trajectory"]
    base_dir = Path(str(config["main_run"]["data_artifacts"]["root"])) / "trajectory"
    requested = artifact_cfg.get("version")
    version_base = str(requested or auto_trajectory_version(config, stage_cfg))
    return resolve_version_dir(
        base_dir,
        version_base,
        overwrite=bool(artifact_cfg["overwrite"]),
        manual_version=bool(requested),
    )


def copy_or_empty(src: Path | None, dst: Path) -> None:
    if src and src.exists():
        copy_file(src, dst)
    else:
        write_jsonl(dst, [])


def run_generate_traces(config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    stage_cfg = config["pipeline"]["stage_configs"]["generate_traces"]
    resource_plan = stage_resource_plan(config, "generate_traces")
    validate_stage_resource_plan(config, "generate_traces", resource_plan)
    normalized_resource_plan = normalize_stage_plan(resource_plan)
    artifact_root = Path(os.environ["ARTIFACT_ROOT"])
    repo_root = Path(os.environ.get("REPO_ROOT", Path.cwd()))
    stage_work_dir = artifact_root / "stages" / "generate_traces"
    infer_trace_dir = stage_work_dir / "air_infer_trace"
    infer_report_path = stage_work_dir / "air_infer_report.md"
    version, trajectory_dir = prepare_trajectory_dir(config, stage_cfg)
    trajectory_jsonl = trajectory_dir / "trajectory.jsonl"
    raw_trace_jsonl = trajectory_dir / "raw_traces.jsonl"
    metrics_jsonl = trajectory_dir / "metrics.jsonl"
    summary_json = trajectory_dir / "summary.json"
    example_json = trajectory_dir / "example.json"
    trajectory_manifest = trajectory_dir / "manifest.json"
    final_config_copy = trajectory_dir / "final_config.yaml"

    infer_command = build_air_infer_command(
        repo_root=repo_root,
        stage_cfg=stage_cfg,
    )
    if dry_run:
        infer_env_preview = air_infer_runtime_env(
            repo_root=repo_root,
            config=config,
            stage_cfg=stage_cfg,
            resource_plan=resource_plan,
            infer_trace_dir=infer_trace_dir,
            infer_report_path=infer_report_path,
        )
        infer_env_preview.update(air_resource_env(config, "generate_traces", resource_plan))
        write_basic_stage(
            "generate_traces",
            stage_cfg,
            manifest,
            dry_run,
            {
                "resource_plan": normalized_resource_plan,
                "trajectory_version": version,
                "trajectory_dir": str(trajectory_dir),
                "canonical_trace_jsonl": str(trajectory_jsonl),
                "raw_trace_jsonl": str(raw_trace_jsonl),
                "trajectory_manifest": str(trajectory_manifest),
                "infer_command": infer_command,
                "infer_env": infer_env_preview,
            },
        )
        return

    ensure_fresh_dir(trajectory_dir, overwrite=bool(config["main_run"]["data_artifacts"]["trajectory"]["overwrite"]))
    existing_canonical = stage_cfg.get("inputs", {}).get("existing_canonical_trace_jsonl")
    existing_raw = stage_cfg.get("inputs", {}).get("existing_raw_trace_jsonl")
    raw_source: Path | None = None
    metrics_source: Path | None = None
    source_mode = "air_v1_infer"

    if existing_canonical:
        source_mode = "existing_canonical_trace_jsonl"
        copy_file(existing_canonical, trajectory_jsonl)
        copy_or_empty(Path(existing_raw) if existing_raw else None, raw_trace_jsonl)
    else:
        if existing_raw:
            source_mode = "existing_raw_trace_jsonl"
            raw_source = Path(str(existing_raw))
        else:
            stage_work_dir.mkdir(parents=True, exist_ok=True)
            infer_env = os.environ.copy()
            infer_env.update(air_resource_env(config, "generate_traces", resource_plan))
            infer_env.update(
                air_infer_runtime_env(
                    repo_root=repo_root,
                    config=config,
                    stage_cfg=stage_cfg,
                    resource_plan=resource_plan,
                    infer_trace_dir=infer_trace_dir,
                    infer_report_path=infer_report_path,
                )
            )
            subprocess.check_call(infer_command, cwd=repo_root, env=infer_env)
            raw_source = infer_trace_dir / "traces.jsonl"
            metrics_source = infer_trace_dir / "metrics.jsonl"
        if not raw_source.exists():
            raise FileNotFoundError(f"raw trace jsonl not found: {raw_source}")
        copy_file(raw_source, raw_trace_jsonl)
        extract_file(raw_trace_jsonl, trajectory_jsonl)
        copy_or_empty(metrics_source, metrics_jsonl)

    if existing_canonical:
        record_count = count_jsonl(trajectory_jsonl)
        if not metrics_jsonl.exists():
            write_jsonl(metrics_jsonl, [])
    else:
        record_count = count_jsonl(trajectory_jsonl)
    allow_empty = bool(config["main_run"]["data_artifacts"]["trajectory"]["allow_empty_trace"])
    if record_count == 0 and not allow_empty:
        raise ValueError("generate_traces produced zero canonical trajectory records")

    write_example(example_json, first_jsonl_record(trajectory_jsonl))
    source_summary = read_json_if_exists(infer_trace_dir / "summary.json")
    summary = {
        "dataset_type": "trajectory",
        "version": version,
        "created_at": utc_now(),
        "source_mode": source_mode,
        "record_count": record_count,
        "raw_trace_count": count_jsonl(raw_trace_jsonl),
        "metric_count": count_jsonl(metrics_jsonl),
        "source_summary": source_summary,
    }
    write_json(summary_json, summary)
    trajectory_payload = {
        "dataset_type": "trajectory",
        "version": version,
        "version_dir": str(trajectory_dir),
        "created_at": utc_now(),
        "source_mode": source_mode,
        "source_agent_checkpoint": agent_checkpoint(config, stage_cfg),
        "source_data_files": [trace_data_file(config)],
        "trace_max_samples": infer_max_samples(config),
        "run_mode": config["infer_runtime"]["mode"]["run_mode"],
        "reranker": config["infer_runtime"]["mode"]["reranker"],
        "trajectory_jsonl": str(trajectory_jsonl),
        "raw_trace_jsonl": str(raw_trace_jsonl),
        "metrics_jsonl": str(metrics_jsonl),
        "summary_json": str(summary_json),
        "example_json": str(example_json),
        "final_config_yaml": str(final_config_copy),
        "record_count": record_count,
        "raw_trace_count": count_jsonl(raw_trace_jsonl),
        "config_hash": stable_config_hash(stage_cfg),
    }
    write_json(trajectory_manifest, trajectory_payload)
    trajectory_readme = write_dataset_readme(repo_root, trajectory_dir)

    stage_outputs = {
        "status": "completed",
        "resource_plan": normalized_resource_plan,
        "trajectory_version": version,
        "trajectory_dir": str(trajectory_dir),
        "raw_trace_jsonl": str(raw_trace_jsonl),
        "canonical_trace_jsonl": str(trajectory_jsonl),
        "trajectory_manifest": str(trajectory_manifest),
        "trajectory_readme": str(trajectory_readme),
        "example_json": str(example_json),
        "record_count": record_count,
    }
    stage_cfg.setdefault("outputs", {}).update(
        {
            "raw_trace_jsonl": str(raw_trace_jsonl),
            "canonical_trace_jsonl": str(trajectory_jsonl),
            "trajectory_manifest": str(trajectory_manifest),
            "trajectory_version": version,
            "trajectory_readme": str(trajectory_readme),
            "manifest": str(manifest),
        }
    )
    build_inputs = config["pipeline"]["stage_configs"]["build_reranker_dataset"]["inputs"]
    build_inputs["canonical_trace_jsonl"] = str(trajectory_jsonl)
    build_inputs["trajectory_manifest"] = str(trajectory_manifest)
    persist_final_config(config_path, config)
    copy_file(config_path, final_config_copy)
    write_stage_manifest(manifest, stage="generate_traces", config=stage_cfg, outputs=stage_outputs)


# ---------------------------------------------------------------------------
# build_reranker_dataset：父 stage 调用数据构造器，并记录两个内部子阶段 manifest。
# ---------------------------------------------------------------------------


def write_build_dataset_dry_run(config: dict[str, Any], manifest: Path, artifact_root: Path) -> None:
    stage_cfg = config["pipeline"]["stage_configs"]["build_reranker_dataset"]
    resource_plan = normalize_stage_plan(stage_resource_plan(config, "build_reranker_dataset"))
    sub_manifests: dict[str, str] = {}
    for sub_stage in as_list(stage_cfg.get("sub_stage_order")):
        sub_cfg = stage_cfg["sub_stages"][sub_stage]
        sub_manifest = artifact_root / "stages" / "build_reranker_dataset" / sub_stage / "manifest.json"
        sub_manifests[sub_stage] = str(sub_manifest)
        write_stage_manifest(
            sub_manifest,
            stage=f"build_reranker_dataset.{sub_stage}",
            config=sub_cfg,
            outputs={"status": "compiled", "enabled": bool(sub_cfg.get("enabled")), "resource_plan": resource_plan},
        )
    write_basic_stage(
        "build_reranker_dataset",
        stage_cfg,
        manifest,
        True,
        {
            "resource_plan": resource_plan,
            "sub_stage_order": as_list(stage_cfg.get("sub_stage_order")),
            "sub_stage_manifests": sub_manifests,
        },
    )


def run_build_reranker_dataset(config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    artifact_root = Path(os.environ["ARTIFACT_ROOT"])
    resource_plan = normalize_stage_plan(stage_resource_plan(config, "build_reranker_dataset"))
    validate_stage_resource_plan(config, "build_reranker_dataset", stage_resource_plan(config, "build_reranker_dataset"))
    if dry_run:
        write_build_dataset_dry_run(config, manifest, artifact_root)
        return

    cmd = [
        sys.executable,
        "-m",
        "agentic_iter_rag.reranker_dataset.build_dataset",
        "--config",
        str(config_path),
        "--stage-manifest",
        str(manifest),
    ]
    subprocess.check_call(cmd)
    stage_manifest = read_json(manifest)
    outputs = stage_manifest.get("outputs") or {}
    outputs["resource_plan"] = resource_plan
    stage_manifest["outputs"] = outputs
    write_json(manifest, stage_manifest)
    stage_cfg = config["pipeline"]["stage_configs"]["build_reranker_dataset"]
    stage_cfg.setdefault("outputs", {}).update(
        {
            "input_dataset_manifest": outputs.get("input_dataset_manifest"),
            "train_dataset_manifest": outputs.get("train_dataset_manifest"),
            "reranker_train_set_manifest": outputs.get("reranker_train_set_manifest"),
            "manifest": str(manifest),
        }
    )
    if outputs.get("train_dataset_manifest"):
        config["reranker_training"]["dataset_manifest"] = outputs["train_dataset_manifest"]
    persist_final_config(config_path, config)


# ---------------------------------------------------------------------------
# stage 分发：正式 data produce 只执行 generate_traces 和 build_reranker_dataset。
# ---------------------------------------------------------------------------


def run_stage(stage: str, config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    stage_cfg = config["pipeline"]["stage_configs"].get(stage, {})
    resource_plan = normalize_stage_plan(stage_resource_plan(config, stage))
    if not bool(stage_cfg.get("enabled", True)):
        write_basic_stage(stage, stage_cfg, manifest, dry_run, {"status": "disabled", "resource_plan": resource_plan})
        return
    if stage == "train_agent":
        write_basic_stage(
            stage,
            stage_cfg,
            manifest,
            dry_run,
            {
                "resource_plan": resource_plan,
                "entry": "AgenticIterRag/main_train_agent.py",
                "note": "真实 agent 训练后续接入 AIR 自有 trainer；当前 pipeline 保留统一调度契约。",
            },
        )
    elif stage == "generate_traces":
        run_generate_traces(config, manifest, dry_run, config_path)
    elif stage == "build_reranker_dataset":
        run_build_reranker_dataset(config, manifest, dry_run, config_path)
    elif stage == "train_llm_reranker":
        write_basic_stage(
            stage,
            stage_cfg,
            manifest,
            dry_run,
            {
                "resource_plan": resource_plan,
                "base_model": config["reranker_training"].get("base_model"),
                "dataset_manifest": config["reranker_training"].get("dataset_manifest"),
            },
        )
    elif stage == "infer_matrix":
        matrix = build_infer_matrix({"baselines": config.get("infer_matrix", {}).get("baselines")})
        write_basic_stage(stage, stage_cfg, manifest, dry_run, {"resource_plan": resource_plan, "matrix": matrix})
    else:
        raise ValueError(f"unsupported pipeline stage: {stage}")


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    pipeline = config["pipeline"]
    artifact_root = Path(os.environ.get("ARTIFACT_ROOT", args.manifest.parent))
    stages = selected_stages(pipeline)
    stage_resource_plans = validate_selected_resource_plan(config, stages)
    plan = {
        "type": "agentic_iter_rag_execution_plan",
        "created_at": utc_now(),
        "dry_run": args.dry_run,
        "selected_stages": stages,
        "resume_from_stage": pipeline.get("resume_from_stage"),
        "stop_after_stage": pipeline.get("stop_after_stage"),
        "skip_stages": as_list(pipeline.get("skip_stages")),
        "force_rerun_stages": as_list(pipeline.get("force_rerun_stages")),
        "stage_manifests": {stage: str(stage_manifest_path(artifact_root, stage)) for stage in stages},
        "stage_resource_plan": stage_resource_plans,
    }
    write_yaml(args.execution_plan, plan)
    completed: list[str] = []
    skipped_existing: list[str] = []
    force_rerun = set(as_list(pipeline.get("force_rerun_stages")))
    for stage in stages:
        manifest = stage_manifest_path(artifact_root, stage)
        if manifest.exists() and stage not in force_rerun:
            skipped_existing.append(stage)
            continue
        run_stage(stage, config, manifest, args.dry_run, args.config)
        completed.append(stage)
    write_json(
        args.manifest,
        {
            "type": "agentic_iter_rag_pipeline_manifest",
            "created_at": utc_now(),
            "dry_run": args.dry_run,
            "final_config_yaml": str(args.config),
            "execution_plan": str(args.execution_plan),
            "selected_stages": stages,
            "completed_stages": completed,
            "skipped_existing_stages": skipped_existing,
            "stage_manifests": {stage: str(stage_manifest_path(artifact_root, stage)) for stage in completed},
        },
    )
    print(f"pipeline wrote manifest {args.manifest}")


if __name__ == "__main__":
    main()
