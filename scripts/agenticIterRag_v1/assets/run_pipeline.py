#!/usr/bin/env python3
"""Run the AgenticIterRag v1 pipeline as one scheduled task."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ is None or __package__ == "":
    project_root_for_import = Path(__file__).resolve().parents[3] / "AgenticIterRag"
    sys.path.insert(0, str(project_root_for_import))

from agentic_iter_rag.infer_matrix.matrix import build_infer_matrix
from agentic_iter_rag.agent_training.train_agent_entry import run_from_config as run_train_agent_from_config
from agentic_iter_rag.pipeline.manifest import write_stage_manifest
from agentic_iter_rag.reranker_training.branch_dataset import run_from_config as run_branch_dataset_from_config
from agentic_iter_rag.reranker_training.filter_branch_dataset import run_from_config as run_filter_branch_dataset_from_config
from agentic_iter_rag.reranker_training.service_bundle import run_from_config as run_service_bundle_from_config
from agentic_iter_rag.trajectory.enhanced import (
    CONTEXT_FORMAT_VERSION,
    ENHANCED_TRAJECTORY_SCHEMA_VERSION,
    TOOL_RESPONSE_FORMAT_VERSION,
    summarize_enhanced_records,
    validate_enhanced_record,
)
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


def stage_gap_seconds(pipeline: dict[str, Any]) -> float:
    """读取 stage 间资源冷却时间。

    这个字段只在配置显式设置时生效，用于全链路任务中等待 vLLM、retriever、Ray 和 NPU 显存彻底释放。
    """

    value = pipeline.get("stage_gap_seconds", 0)
    if value in (None, ""):
        return 0.0
    seconds = float(value)
    if seconds < 0:
        raise ValueError(f"pipeline.stage_gap_seconds must be >= 0, got {value}")
    return seconds


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


def recall_backend_type(recall: dict[str, Any]) -> str:
    """读取 recall backend 类型。

    新配置使用 backend_type 显式区分 cpu/npu/cuda；如果旧配置仍然只写 gpu_ids，
    这里按 npu 兜底，保证旧 dataproduce 配置迁移过程中错误更容易定位。
    """

    return str(recall.get("backend_type") or ("npu" if recall.get("gpu_ids") else "cpu")).lower()


def recall_active_accelerator_gpu_ids(recall: dict[str, Any]) -> list[int]:
    """只在 npu/cuda backend 下返回真实会被使用的 retriever 卡位。"""

    backend_type = recall_backend_type(recall)
    if backend_type not in {"npu", "cuda"}:
        return []
    accelerator = recall.get("accelerator_backend") if isinstance(recall.get("accelerator_backend"), dict) else {}
    return as_int_list(accelerator.get("gpu_ids", recall.get("gpu_ids")))


def recall_proxy_config(recall: dict[str, Any]) -> dict[str, Any]:
    proxy = recall.get("proxy")
    return proxy if isinstance(proxy, dict) else {}


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


def process_tree_pgids(root_pid: int) -> set[int]:
    """收集 root_pid 子进程树里所有进程组。

    某些 stage 内部会再启动新的进程组，比如 VERL、vLLM、retriever 和 reporter。
    仅 kill root 进程组会遗漏这些服务，所以这里基于 ps 快照追踪完整后代树。
    """

    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,ppid=,pgid="],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return {root_pid}
    children_by_parent: dict[int, list[tuple[int, int]]] = {}
    pgids: set[int] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, pgid = (int(part) for part in parts)
        except ValueError:
            continue
        if pid == root_pid:
            pgids.add(pgid)
        children_by_parent.setdefault(ppid, []).append((pid, pgid))
    stack = [root_pid]
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        for child_pid, child_pgid in children_by_parent.get(pid, []):
            pgids.add(child_pgid)
            stack.append(child_pid)
    pgids.add(root_pid)
    return pgids


def signal_process_groups(pgids: set[int], sig: int) -> None:
    """向多个进程组发送信号；忽略已经退出的进程组。"""

    for pgid in sorted(pgids, reverse=True):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            continue
        except PermissionError:
            continue


def check_call_process_group(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    """以独立进程组运行 stage 子进程，保证中断时能把子进程树一起清掉。"""

    process = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True)
    try:
        return_code = process.wait()
    except BaseException:
        # pipeline 收到 Ctrl-C / TERM / 异常时，先温和停止整个 stage 子进程树；
        # 超时后再强杀，避免 vLLM、retriever、Ray worker 残留占用端口和 NPU 显存。
        pgids = process_tree_pgids(process.pid)
        try:
            signal_process_groups(pgids, signal.SIGTERM)
            process.wait(timeout=60)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            signal_process_groups(process_tree_pgids(process.pid) | pgids, signal.SIGKILL)
            process.wait(timeout=30)
        raise
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def write_dataset_readme(repo_root: Path, dataset_dir: Path) -> Path:
    """Generate a human-readable README for a produced dataset directory."""

    script = repo_root / "scripts" / "agenticIterRag_v1" / "assets" / "build_dataset_readme.py"
    check_call_process_group([sys.executable, str(script), "--dataset-dir", str(dataset_dir)])
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
    train_agent_impl = (
        config.get("pipeline", {})
        .get("stage_configs", {})
        .get("train_agent", {})
        .get("impl")
    )
    if stage == "train_agent" and train_agent_impl != "spad_rag" and "impls" in plan:
        plan = {key: value for key, value in plan.items() if key != "impls"}
    if stage == "train_agent" and train_agent_impl == "spad_rag":
        impls = plan.get("impls") if isinstance(plan.get("impls"), dict) else {}
        spad = impls.get("spad_rag") if isinstance(impls.get("spad_rag"), dict) else None
        if spad is None:
            raise ValueError("resource.stage_resources.train_agent.impls.spad_rag must be set")
        sub_stages = spad.get("sub_stages") if isinstance(spad.get("sub_stages"), dict) else {}
        for sub_stage_name, sub_stage_plan in sub_stages.items():
            if not isinstance(sub_stage_plan, dict):
                raise TypeError(f"resource train_agent.impls.spad_rag.sub_stages.{sub_stage_name} must be a mapping")
            if sub_stage_name == "answer_distillation" and isinstance(sub_stage_plan.get("phases"), dict):
                for phase_name, phase_plan in sub_stage_plan["phases"].items():
                    if not isinstance(phase_plan, dict):
                        raise TypeError(
                            "resource train_agent.impls.spad_rag.sub_stages.answer_distillation."
                            f"phases.{phase_name} must be a mapping"
                        )
                    validate_stage_resource_plan(
                        config,
                        f"train_agent.impls.spad_rag.sub_stages.answer_distillation.phases.{phase_name}",
                        phase_plan,
                    )
                remainder = {k: v for k, v in sub_stage_plan.items() if k != "phases"}
                if remainder:
                    validate_stage_resource_plan(
                        config,
                        f"train_agent.impls.spad_rag.sub_stages.{sub_stage_name}",
                        remainder,
                    )
            else:
                validate_stage_resource_plan(
                    config,
                    f"train_agent.impls.spad_rag.sub_stages.{sub_stage_name}",
                    sub_stage_plan,
                )
        return
    allow_gpu_overlap = bool(plan.get("allow_gpu_overlap"))

    owners: dict[int, str] = {}

    def validate_recall_node(name: str, node: dict[str, Any]) -> None:
        backend_type = recall_backend_type(node)
        if backend_type not in {"cpu", "npu", "cuda"}:
            raise ValueError(f"resource stage {stage}.{name}.backend_type must be cpu/npu/cuda, got {backend_type!r}")
        if node.get("retrieval_service_url") and node.get("port"):
            expected = f":{node['port']}/"
            if expected not in str(node["retrieval_service_url"]):
                raise ValueError(
                    f"resource stage {stage}.{name}.retrieval_service_url port does not match port={node['port']}"
                )
        if backend_type == "cpu":
            instance_count = int(node.get("instance_count") or 8)
            if instance_count < 1:
                raise ValueError(f"resource stage {stage}.{name}.instance_count must be >= 1")
            cpu_backend = node.get("cpu_backend") if isinstance(node.get("cpu_backend"), dict) else {}
            doc_dtype = str(cpu_backend.get("doc_dtype") or "float32")
            if doc_dtype != "float32":
                raise ValueError(f"resource stage {stage}.{name}.cpu_backend.doc_dtype must be float32")
        else:
            gpu_ids = recall_active_accelerator_gpu_ids(node)
            if not gpu_ids:
                raise ValueError(
                    f"resource stage {stage}.{name}.accelerator_backend.gpu_ids must contain at least one GPU id"
                )

    def visit(name: str, node: Any) -> None:
        if not isinstance(node, dict):
            return
        is_recall_node = name.endswith("recall")
        if is_recall_node:
            validate_recall_node(name, node)
        is_multi_instance_container = str(node.get("backend_type") or "").lower() == "multi_instance_proxy"
        if "gpu_ids" in node and not is_multi_instance_container:
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
            if "tensor_parallel_size" in node:
                tensor_parallel_size = int(node["tensor_parallel_size"])
                if tensor_parallel_size <= 0:
                    raise ValueError(
                        f"resource stage {stage}.{name}.tensor_parallel_size must be positive: {tensor_parallel_size}"
                    )
                # tensor_parallel_size 可以小于 gpu_ids 数量，此时表示同一组 GPU 上启动多个 rollout replica。
                # 例如 stage1_format_actor 使用 8 张 NPU、TP=1，会形成 8 个单卡 vLLM replica，提高 64*16 rollout 吞吐。
                if len(gpu_ids) % tensor_parallel_size != 0:
                    raise ValueError(
                        f"resource stage {stage}.{name}.gpu_ids count must be divisible by tensor_parallel_size: "
                        f"{len(gpu_ids)} % {tensor_parallel_size} != 0"
                    )
        backend_type = recall_backend_type(node) if is_recall_node else None
        for child_name, child in node.items():
            # CPU retriever 不读取 accelerator_backend，accelerator retriever 也不读取 cpu_backend。
            # 校验阶段同步跳过 inactive 分支，避免配置里保留的切换模板被误算成当前资源占用。
            if is_recall_node and backend_type == "cpu" and child_name == "accelerator_backend":
                continue
            if is_recall_node and backend_type in {"npu", "cuda"} and child_name == "cpu_backend":
                continue
            if isinstance(child, dict):
                visit(f"{name}.{child_name}" if name else str(child_name), child)

    visit("", plan)

    ports: dict[int, str] = {}

    def visit_ports(name: str, node: Any) -> None:
        if not isinstance(node, dict):
            return
        # multi_instance_proxy 节点自身的 port 可能来自旧单实例配置继承；
        # 真实监听端口在 proxy.port 和 instances[*].port，避免把容器节点误判成重复端口。
        is_multi_instance_container = str(node.get("backend_type") or "").lower() == "multi_instance_proxy"
        if "port" in node and node["port"] is not None and not is_multi_instance_container:
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
        "RECALL_BACKEND_TYPE": recall_backend_type(recall),
        "RECALL_INSTANCE_COUNT": str(recall.get("instance_count") or ""),
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
    backend_type = recall_backend_type(recall)
    cpu_backend = recall.get("cpu_backend") if isinstance(recall.get("cpu_backend"), dict) else {}
    accelerator_backend = recall.get("accelerator_backend") if isinstance(recall.get("accelerator_backend"), dict) else {}
    proxy = recall_proxy_config(recall)
    return {
        "GROUP_NAME": "agenticIterRag",
        "AGENT_GPU_IDS": gpu_csv(agent_vllm["gpu_ids"]),
        "RANK_GPU_ID": gpu_csv(dense_reranker.get("gpu_ids")),
        "RECALL_BACKEND_TYPE": backend_type,
        "RECALL_INSTANCE_COUNT": str(recall.get("instance_count") or ""),
        "RECALL_GPU_ID": gpu_csv(accelerator_backend.get("gpu_ids", recall.get("gpu_ids"))),
        "RECALL_BACKEND_BASE_PORT": str(recall.get("backend_base_port") or ""),
        "RECALL_PROXY_STRATEGY": str(proxy.get("strategy") or recall.get("proxy_strategy") or "least_inflight"),
        "RECALL_PROXY_TIMEOUT": str(proxy.get("timeout") or recall.get("proxy_timeout") or ""),
        "RECALL_PROXY_FAILURE_COOLDOWN_SECONDS": str(
            proxy.get("failure_cooldown_seconds") or recall.get("proxy_failure_cooldown_seconds") or ""
        ),
        "RECALL_PROXY_LATENCY_EWMA_ALPHA": str(
            proxy.get("latency_ewma_alpha") or recall.get("proxy_latency_ewma_alpha") or ""
        ),
        "RECALL_PROXY_MAX_RETRIES_PER_REQUEST": str(
            proxy.get("max_retries_per_request") or recall.get("proxy_max_retries_per_request") or ""
        ),
        "RECALL_ASSET_PRECHECK": "1" if recall.get("asset_precheck") else "0",
        "RECALL_QUERY_PREFLIGHT": "1" if recall.get("query_preflight") else "0",
        "RETRIEVAL_PREFLIGHT_QUERY": str(recall.get("preflight_query") or "who got the first nobel prize in physics?"),
        "RETRIEVAL_PREFLIGHT_EXPECT": str(recall.get("preflight_expect") or ""),
        "RECALL_CPU_THREADS_PER_INSTANCE": str(cpu_backend.get("cpu_threads_per_instance") or ""),
        "RECALL_CPU_QUERY_BATCH_SIZE": str(cpu_backend.get("query_batch_size") or ""),
        "RECALL_CPU_DOC_DTYPE": str(cpu_backend.get("doc_dtype") or ""),
        "RECALL_ACCELERATOR_QUERY_BATCH_SIZE": str(accelerator_backend.get("query_batch_size") or ""),
        "RECALL_ACCELERATOR_DOC_DTYPE": str(accelerator_backend.get("doc_dtype") or ""),
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


def extract_enhanced_trajectory_file(
    *,
    raw_trace_jsonl: Path,
    output_jsonl: Path,
    example_json: Path,
    summary_json: Path,
    raw_trace_ref_path: Path,
    no_ranker: bool,
    top_n: int,
    top_m: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    missing_lines: list[int] = []
    for line_index, raw in enumerate(iter_jsonl(raw_trace_jsonl)):
        enhanced = raw.get("enhanced_trajectory")
        if not isinstance(enhanced, dict):
            missing_lines.append(line_index)
            continue
        record = dict(enhanced)
        record["raw_trace_ref"] = {"path": str(raw_trace_ref_path), "line_index": line_index}
        validate_enhanced_record(record, no_ranker=no_ranker, top_m=top_m)
        records.append(record)
    if missing_lines:
        preview = ", ".join(str(item) for item in missing_lines[:10])
        raise ValueError(
            "raw trace jsonl does not contain strict enhanced_trajectory records; "
            f"missing lines: {preview}. Old raw_traces.jsonl cannot be backfilled strictly."
        )
    write_jsonl(output_jsonl, records)
    write_example(example_json, records[0] if records else None)
    summary = summarize_enhanced_records(records, top_n=top_n, top_m=top_m)
    write_json(summary_json, summary)
    return summary


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
    enhanced_trajectory_jsonl = trajectory_dir / "enhanced_trajectory.jsonl"
    enhanced_example_json = trajectory_dir / "enhanced_example.json"
    enhanced_summary_json = trajectory_dir / "enhanced_summary.json"
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
                "enhanced_trajectory_jsonl": str(enhanced_trajectory_jsonl),
                "enhanced_example_json": str(enhanced_example_json),
                "enhanced_summary_json": str(enhanced_summary_json),
                "trajectory_manifest": str(trajectory_manifest),
                "infer_command": infer_command,
                "infer_env": infer_env_preview,
            },
        )
        # dry-run 不会真实写 trajectory manifest，但后续 stage 仍需要看到计划中的 manifest 路径，
        # 这样整条 generate_traces -> train_llm_reranker 链路可以在编译阶段被完整校验。
        stage_cfg.setdefault("outputs", {}).update(
            {
                "raw_trace_jsonl": str(raw_trace_jsonl),
                "canonical_trace_jsonl": str(trajectory_jsonl),
                "enhanced_trajectory_jsonl": str(enhanced_trajectory_jsonl),
                "enhanced_summary_json": str(enhanced_summary_json),
                "trajectory_manifest": str(trajectory_manifest),
                "trajectory_version": version,
                "manifest": str(manifest),
            }
        )
        build_inputs = config["pipeline"]["stage_configs"]["build_reranker_dataset"]["inputs"]
        build_inputs["canonical_trace_jsonl"] = str(trajectory_jsonl)
        build_inputs["trajectory_manifest"] = str(trajectory_manifest)
        branch_inputs = config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"]["inputs"]
        branch_inputs["enhanced_trajectory_manifest"] = str(trajectory_manifest)
        config["reranker_training"].setdefault("input", {})["enhanced_trajectory_manifest"] = str(trajectory_manifest)
        persist_final_config(config_path, config)
        return

    ensure_fresh_dir(trajectory_dir, overwrite=bool(config["main_run"]["data_artifacts"]["trajectory"]["overwrite"]))
    existing_canonical = stage_cfg.get("inputs", {}).get("existing_canonical_trace_jsonl")
    existing_raw = stage_cfg.get("inputs", {}).get("existing_raw_trace_jsonl")
    raw_source: Path | None = None
    metrics_source: Path | None = None
    source_mode = "air_v1_infer"
    enhanced_summary: dict[str, Any] | None = None

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
            check_call_process_group(infer_command, cwd=repo_root, env=infer_env)
            raw_source = infer_trace_dir / "traces.jsonl"
            metrics_source = infer_trace_dir / "metrics.jsonl"
        if not raw_source.exists():
            raise FileNotFoundError(f"raw trace jsonl not found: {raw_source}")
        copy_file(raw_source, raw_trace_jsonl)
        extract_file(raw_trace_jsonl, trajectory_jsonl)
        copy_or_empty(metrics_source, metrics_jsonl)
        enhanced_summary = extract_enhanced_trajectory_file(
            raw_trace_jsonl=raw_trace_jsonl,
            output_jsonl=enhanced_trajectory_jsonl,
            example_json=enhanced_example_json,
            summary_json=enhanced_summary_json,
            raw_trace_ref_path=raw_trace_jsonl,
            no_ranker=str(config["infer_runtime"]["mode"].get("run_mode")) == "no-ranker",
            top_n=int(config["infer_runtime"]["retrieval"]["final_top_n"]),
            top_m=int(config["infer_runtime"]["retrieval"]["visible_top_m"]),
        )

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
        "enhanced_record_count": enhanced_summary.get("record_count", 0) if enhanced_summary else 0,
        "enhanced_search_step_count": enhanced_summary.get("search_step_count", 0) if enhanced_summary else 0,
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
        "enhanced_trajectory_jsonl": str(enhanced_trajectory_jsonl) if enhanced_summary else None,
        "enhanced_example_json": str(enhanced_example_json) if enhanced_summary else None,
        "enhanced_summary_json": str(enhanced_summary_json) if enhanced_summary else None,
        "enhanced_schema_version": ENHANCED_TRAJECTORY_SCHEMA_VERSION if enhanced_summary else None,
        "enhanced_record_count": enhanced_summary.get("record_count", 0) if enhanced_summary else 0,
        "enhanced_record_count_actual": enhanced_summary.get("record_count", 0) if enhanced_summary else 0,
        "enhanced_search_step_count": enhanced_summary.get("search_step_count", 0) if enhanced_summary else 0,
        "context_format_version": CONTEXT_FORMAT_VERSION if enhanced_summary else None,
        "tool_response_format_version": TOOL_RESPONSE_FORMAT_VERSION if enhanced_summary else None,
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
        "enhanced_trajectory_jsonl": str(enhanced_trajectory_jsonl) if enhanced_summary else None,
        "enhanced_example_json": str(enhanced_example_json) if enhanced_summary else None,
        "enhanced_summary_json": str(enhanced_summary_json) if enhanced_summary else None,
        "enhanced_record_count": enhanced_summary.get("record_count", 0) if enhanced_summary else 0,
        "enhanced_record_count_actual": enhanced_summary.get("record_count", 0) if enhanced_summary else 0,
        "enhanced_search_step_count": enhanced_summary.get("search_step_count", 0) if enhanced_summary else 0,
        "trajectory_manifest": str(trajectory_manifest),
        "trajectory_readme": str(trajectory_readme),
        "example_json": str(example_json),
        "record_count": record_count,
    }
    stage_cfg.setdefault("outputs", {}).update(
        {
            "raw_trace_jsonl": str(raw_trace_jsonl),
            "canonical_trace_jsonl": str(trajectory_jsonl),
            "enhanced_trajectory_jsonl": str(enhanced_trajectory_jsonl) if enhanced_summary else None,
            "enhanced_summary_json": str(enhanced_summary_json) if enhanced_summary else None,
            "trajectory_manifest": str(trajectory_manifest),
            "trajectory_version": version,
            "trajectory_readme": str(trajectory_readme),
            "manifest": str(manifest),
        }
    )
    build_inputs = config["pipeline"]["stage_configs"]["build_reranker_dataset"]["inputs"]
    build_inputs["canonical_trace_jsonl"] = str(trajectory_jsonl)
    build_inputs["trajectory_manifest"] = str(trajectory_manifest)
    branch_inputs = config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"]["inputs"]
    branch_inputs["enhanced_trajectory_manifest"] = str(trajectory_manifest)
    config["reranker_training"].setdefault("input", {})["enhanced_trajectory_manifest"] = str(trajectory_manifest)
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
        stage_cfg = config["pipeline"]["stage_configs"]["build_reranker_dataset"]
        train_set_manifest = (
            artifact_root
            / "stages"
            / "build_reranker_dataset"
            / "build_train_dataset"
            / "planned_train_dataset_manifest.json"
        )
        stage_cfg.setdefault("outputs", {}).update(
            {
                "train_dataset_manifest": str(train_set_manifest),
                "reranker_train_set_manifest": str(train_set_manifest),
                "manifest": str(manifest),
            }
        )
        branch_inputs = config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"]["inputs"]
        branch_inputs["reranker_train_set_manifest"] = str(train_set_manifest)
        config["reranker_training"].setdefault("input", {})["reranker_train_set_manifest"] = str(train_set_manifest)
        persist_final_config(config_path, config)
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
    check_call_process_group(cmd)
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
    if outputs.get("reranker_train_set_manifest"):
        branch_inputs = config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"]["inputs"]
        branch_inputs["reranker_train_set_manifest"] = outputs["reranker_train_set_manifest"]
        config["reranker_training"].setdefault("input", {})["reranker_train_set_manifest"] = outputs[
            "reranker_train_set_manifest"
        ]
    persist_final_config(config_path, config)


# ---------------------------------------------------------------------------
# build_reranker_branch_dataset / filter_reranker_branch_dataset / train_llm_reranker / build_service_bundle：
# AIR LLM reranker branch GRPO 新链路。
# ---------------------------------------------------------------------------


def run_build_reranker_branch_dataset(config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    resource_plan = normalize_stage_plan(stage_resource_plan(config, "build_reranker_branch_dataset"))
    validate_stage_resource_plan(
        config,
        "build_reranker_branch_dataset",
        stage_resource_plan(config, "build_reranker_branch_dataset"),
    )
    outputs = run_branch_dataset_from_config(config_path, manifest, dry_run=dry_run)
    outputs["resource_plan"] = resource_plan
    stage_manifest = read_json(manifest)
    stage_manifest["outputs"] = outputs
    write_json(manifest, stage_manifest)

    branch_manifest = outputs.get("branch_dataset_manifest")
    if not branch_manifest:
        raise ValueError("build_reranker_branch_dataset did not produce branch_dataset_manifest")
    stage_cfg = config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"]
    stage_cfg.setdefault("outputs", {}).update(
        {
            "branch_dataset_manifest": branch_manifest,
            "branch_dataset_jsonl": outputs.get("branch_dataset_jsonl"),
            "branch_dataset_version": outputs.get("branch_dataset_version"),
            "manifest": str(manifest),
        }
    )
    filter_stage = config["pipeline"]["stage_configs"].get("filter_reranker_branch_dataset")
    if isinstance(filter_stage, dict):
        filter_stage.setdefault("inputs", {})["source_branch_dataset_manifest"] = branch_manifest
    train_inputs = config["pipeline"]["stage_configs"]["train_llm_reranker"]["inputs"]
    train_inputs["branch_dataset_manifest"] = branch_manifest
    rt_input = config["reranker_training"].setdefault("input", {})
    rt_input["branch_dataset_manifest"] = branch_manifest
    persist_final_config(config_path, config)
    if dry_run:
        return


def run_filter_reranker_branch_dataset(config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    resource_plan = normalize_stage_plan(stage_resource_plan(config, "filter_reranker_branch_dataset"))
    validate_stage_resource_plan(
        config,
        "filter_reranker_branch_dataset",
        stage_resource_plan(config, "filter_reranker_branch_dataset"),
    )
    outputs = run_filter_branch_dataset_from_config(config_path, manifest, dry_run=dry_run)
    outputs["resource_plan"] = resource_plan
    stage_manifest = read_json(manifest)
    stage_manifest["outputs"] = outputs
    write_json(manifest, stage_manifest)

    filtered_manifest = outputs.get("filtered_branch_dataset_manifest") or outputs.get("subset_manifest")
    if not filtered_manifest:
        raise ValueError("filter_reranker_branch_dataset did not produce filtered_branch_dataset_manifest")
    stage_cfg = config["pipeline"]["stage_configs"]["filter_reranker_branch_dataset"]
    stage_cfg.setdefault("outputs", {}).update(
        {
            "filtered_branch_dataset_manifest": filtered_manifest,
            "filtered_branch_dataset_jsonl": outputs.get("filtered_branch_dataset_jsonl") or outputs.get("subset_jsonl"),
            "filtered_branch_dataset_version": outputs.get("filtered_branch_dataset_version") or outputs.get("subset_version"),
            "manifest": str(manifest),
        }
    )
    train_inputs = config["pipeline"]["stage_configs"]["train_llm_reranker"]["inputs"]
    train_inputs["branch_dataset_manifest"] = filtered_manifest
    rt_input = config["reranker_training"].setdefault("input", {})
    rt_input["branch_dataset_manifest"] = filtered_manifest
    rt_input["filtered_branch_dataset_manifest"] = filtered_manifest
    persist_final_config(config_path, config)
    if dry_run:
        return


def run_train_llm_reranker(config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    resource_plan = normalize_stage_plan(stage_resource_plan(config, "train_llm_reranker"))
    validate_stage_resource_plan(config, "train_llm_reranker", stage_resource_plan(config, "train_llm_reranker"))
    cmd = [
        sys.executable,
        str(Path(os.environ.get("REPO_ROOT", Path.cwd())) / "AgenticIterRag" / "main_train_llm_reranker.py"),
        "--config",
        str(config_path),
        "--manifest",
        str(manifest),
    ]
    if dry_run:
        cmd.append("--dry-run")
    check_call_process_group(cmd, cwd=Path(os.environ.get("REPO_ROOT", Path.cwd())))
    stage_manifest = read_json(manifest)
    outputs = stage_manifest.get("outputs") or {}
    outputs["resource_plan"] = resource_plan
    stage_manifest["outputs"] = outputs
    write_json(manifest, stage_manifest)

    reranker_model = outputs.get("reranker_model")
    if not reranker_model:
        raise ValueError("train_llm_reranker did not produce reranker_model")
    stage_cfg = config["pipeline"]["stage_configs"]["train_llm_reranker"]
    stage_cfg.setdefault("outputs", {}).update(
        {
            "reranker_model": reranker_model,
            "reranker_checkpoint": outputs.get("reranker_checkpoint"),
            "manifest": str(manifest),
        }
    )
    config["reranker_training"].setdefault("runtime", {})["manifest_path"] = str(manifest)
    config["reranker_training"]["runtime"]["trained_model_path"] = reranker_model
    config["infer_runtime"].setdefault("models", {})["trained_llm_reranker_model_path"] = reranker_model
    config["infer_runtime"]["models"]["trained_llm_reranker_model"] = Path(str(reranker_model)).name
    bundle_inputs = config["pipeline"]["stage_configs"]["build_service_bundle"]["inputs"]
    bundle_inputs["reranker_model"] = reranker_model
    persist_final_config(config_path, config)
    if dry_run:
        return


def run_build_service_bundle(config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    resource_plan = normalize_stage_plan(stage_resource_plan(config, "build_service_bundle"))
    validate_stage_resource_plan(config, "build_service_bundle", stage_resource_plan(config, "build_service_bundle"))
    outputs = run_service_bundle_from_config(config_path, manifest, dry_run=dry_run)
    outputs["resource_plan"] = resource_plan
    stage_manifest = read_json(manifest)
    stage_manifest["outputs"] = outputs
    write_json(manifest, stage_manifest)
    if dry_run:
        return

    stage_cfg = config["pipeline"]["stage_configs"]["build_service_bundle"]
    stage_cfg.setdefault("outputs", {}).update(
        {
            "service_bundle_dir": outputs.get("service_bundle_dir"),
            "manifest": outputs.get("manifest") or str(manifest),
        }
    )
    config["reranker_training"].setdefault("runtime", {})["service_bundle_dir"] = outputs.get("service_bundle_dir")
    persist_final_config(config_path, config)


def run_stage(stage: str, config: dict[str, Any], manifest: Path, dry_run: bool, config_path: Path) -> None:
    stage_cfg = config["pipeline"]["stage_configs"].get(stage, {})
    resource_plan = normalize_stage_plan(stage_resource_plan(config, stage))
    if not bool(stage_cfg.get("enabled", True)):
        write_basic_stage(stage, stage_cfg, manifest, dry_run, {"status": "disabled", "resource_plan": resource_plan})
        return
    if stage == "train_agent":
        outputs = run_train_agent_from_config(config_path, manifest, dry_run=dry_run)
        outputs["resource_plan"] = resource_plan
        stage_manifest = read_json(manifest)
        stage_manifest["outputs"] = outputs
        write_json(manifest, stage_manifest)
        if dry_run:
            return
        stage_cfg.setdefault("outputs", {}).update(
            {
                "agent_checkpoint": outputs.get("agent_checkpoint"),
                "agent_training_manifest": outputs.get("agent_training_manifest"),
                "manifest": str(manifest),
            }
        )
        if outputs.get("agent_checkpoint"):
            config.setdefault("infer_runtime", {}).setdefault("models", {})["trained_agent_model"] = outputs["agent_checkpoint"]
        persist_final_config(config_path, config)
    elif stage == "generate_traces":
        run_generate_traces(config, manifest, dry_run, config_path)
    elif stage == "build_reranker_dataset":
        run_build_reranker_dataset(config, manifest, dry_run, config_path)
    elif stage == "build_reranker_branch_dataset":
        run_build_reranker_branch_dataset(config, manifest, dry_run, config_path)
    elif stage == "filter_reranker_branch_dataset":
        run_filter_reranker_branch_dataset(config, manifest, dry_run, config_path)
    elif stage == "train_llm_reranker":
        run_train_llm_reranker(config, manifest, dry_run, config_path)
    elif stage == "build_service_bundle":
        run_build_service_bundle(config, manifest, dry_run, config_path)
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
    gap_seconds = stage_gap_seconds(pipeline)
    for index, stage in enumerate(stages):
        manifest = stage_manifest_path(artifact_root, stage)
        if manifest.exists() and stage not in force_rerun:
            skipped_existing.append(stage)
            continue
        run_stage(stage, config, manifest, args.dry_run, args.config)
        completed.append(stage)
        if gap_seconds > 0 and not args.dry_run and index < len(stages) - 1:
            print(f"waiting {gap_seconds:g}s before next stage to let runtime resources settle")
            time.sleep(gap_seconds)
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
