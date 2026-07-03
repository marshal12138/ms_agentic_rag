#!/usr/bin/env python3
"""Compile AgenticIterRag v1 pipeline config into runtime audit files."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    project_root_for_import = Path(__file__).resolve().parents[3] / "AgenticIterRag"
    sys.path.insert(0, str(project_root_for_import))

from agentic_iter_rag.utils.io import deep_merge, read_yaml, write_json, write_yaml
from agentic_iter_rag.utils.shell import write_export_file
from agentic_iter_rag.utils.validation import require_no_shell_only_business_env


GROUP_DIRS = {
    "data": "data",
    "pipeline": "pipeline",
    "resource": "resource",
    "infer_runtime": "infer_runtime",
    "infer_budget": "infer_budget",
    "reranker_training": "reranker_training",
    "model": "model",
    "rollout": "rollout",
}

GROUP_ARG_DESTS = {
    "data_config": "data",
    "pipeline_config": "pipeline",
    "resource_config": "resource",
    "infer_runtime_config": "infer_runtime",
    "infer_budget_config": "infer_budget",
    "reranker_training_config": "reranker_training",
    "model_config": "model",
    "rollout_config": "rollout",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile AgenticIterRag v1 config.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--script-dir", required=True, type=Path)
    parser.add_argument("--main-run-config", "--main_run_config", default="agentic_iter_rag_main")
    parser.add_argument("--DATA_CONFIG", "--data-config", dest="data_config")
    parser.add_argument("--PIPELINE_CONFIG", "--pipeline-config", dest="pipeline_config")
    parser.add_argument("--RESOURCE_CONFIG", "--resource-config", dest="resource_config")
    parser.add_argument("--INFER_RUNTIME_CONFIG", "--infer-runtime-config", dest="infer_runtime_config")
    parser.add_argument("--INFER_BUDGET_CONFIG", "--infer-budget-config", dest="infer_budget_config")
    parser.add_argument("--RERANKER_TRAINING_CONFIG", "--reranker-training-config", dest="reranker_training_config")
    parser.add_argument("--MODEL_CONFIG", "--model-config", dest="model_config")
    parser.add_argument("--ROLLOUT_CONFIG", "--rollout-config", dest="rollout_config")
    parser.add_argument("--OVERLAY_YAML", "--overlay-yaml", dest="overlay_yaml", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args, cli_overrides = parser.parse_known_args()
    args.cli_overrides = cli_overrides
    return args


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    return slug or "run"


def parse_cli_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        if not item.startswith("--"):
            raise ValueError(f"CLI override must use --key=value syntax: {item}")
        if "=" not in item:
            raise ValueError(f"CLI override must use --key=value syntax: {item}")
        key, raw_value = item[2:].split("=", 1)
        if not key or any(part == "" for part in key.split(".")):
            raise ValueError(f"CLI override key must be a non-empty dotted path: {item}")
        value: Any
        if raw_value in {"true", "True"}:
            value = True
        elif raw_value in {"false", "False"}:
            value = False
        elif raw_value in {"null", "None"}:
            value = None
        else:
            try:
                value = json.loads(raw_value)
            except json.JSONDecodeError:
                value = raw_value
        cur = out
        parts = key.split(".")
        for part in parts[:-1]:
            child = cur.get(part)
            if not isinstance(child, dict):
                child = {}
                cur[part] = child
            cur = child
        cur[parts[-1]] = value
    return out


def load_group(project_root: Path, group: str, name: str) -> dict[str, Any]:
    rel_dir = GROUP_DIRS[group]
    path = project_root / "config" / rel_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing config group {group}={name}: {path}")
    return read_yaml(path)


def load_main(project_root: Path, name: str) -> dict[str, Any]:
    path = project_root / "config" / "main_run" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"missing main_run config: {path}")
    return read_yaml(path)


def resolve_overlay(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    if not path.exists():
        raise FileNotFoundError(f"overlay does not exist: {path}")
    return path


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    main_cfg = load_main(args.project_root, args.main_run_config)
    cfg: dict[str, Any] = {"main_run": main_cfg}
    groups = dict(main_cfg.get("config_groups", {}))
    if not isinstance(groups, dict):
        raise TypeError("main_run.config_groups must be a mapping")
    for arg_dest, group in GROUP_ARG_DESTS.items():
        value = getattr(args, arg_dest, None)
        if value:
            groups[group] = str(value)
    main_cfg["config_groups"] = groups
    for group in GROUP_DIRS:
        name = groups.get(group)
        if not name:
            raise ValueError(f"main_run.config_groups.{group} must be set")
        cfg[group] = load_group(args.project_root, group, str(name))
    for overlay in args.overlay_yaml:
        cfg = deep_merge(cfg, read_yaml(resolve_overlay(args.repo_root, overlay)))
    cfg = deep_merge(cfg, parse_cli_overrides(args.cli_overrides))
    return cfg


def deep_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def require_path(cfg: dict[str, Any], dotted: str) -> Any:
    sentinel = object()
    value = deep_get(cfg, dotted, sentinel)
    if value is sentinel:
        raise ValueError(f"missing required AgenticIterRag v1 config field: {dotted}")
    return value


def validate_current_config(cfg: dict[str, Any]) -> None:
    """拒绝旧字段，校验 v1 data produce 需要的新配置结构。"""

    unsupported_paths = [
        "reranker_training.label_policy",
        "reranker_training.candidate_top_n",
        "reranker_training.positive_top_k",
        "reranker_training.prompt",
        "resource.agent",
        "resource.recall",
        "resource.judge",
        "resource.original_llm_reranker",
        "resource.trained_llm_reranker",
        "resource.wait_for_gpus",
    ]
    for old_path in unsupported_paths:
        sentinel = object()
        if deep_get(cfg, old_path, sentinel) is not sentinel:
            raise ValueError(f"unsupported config field in AgenticIterRag v1: {old_path}")

    required_paths = [
        "main_run.data_artifacts.root",
        "main_run.data_artifacts.trajectory.version",
        "main_run.data_artifacts.trajectory.overwrite",
        "main_run.data_artifacts.trajectory.allow_empty_trace",
        "main_run.data_artifacts.llm_reranker_train_set.version",
        "main_run.data_artifacts.llm_reranker_train_set.overwrite",
        "main_run.data_artifacts.llm_reranker_train_set.derive_version_from_trajectory",
        "infer_runtime.models.trained_agent_model",
        "pipeline.stage_configs.generate_traces.entry_type",
        "pipeline.stage_configs.generate_traces.entry",
        "pipeline.stage_configs.generate_traces.inputs.existing_raw_trace_jsonl",
        "pipeline.stage_configs.generate_traces.inputs.existing_canonical_trace_jsonl",
        "pipeline.stage_configs.build_reranker_dataset.sub_stage_order",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.enabled",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.version",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.overwrite",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.derive_version_from_trajectory",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.schema_version",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.builder_policy",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.candidate_source",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.candidate_top_n",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.dedupe_policy",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.label_policy",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.positive_policy",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_input_dataset.target_ranking_policy",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.enabled",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.version",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.overwrite",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.format",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.prompt_template_version",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.formatter",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.ground_truth_policy",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.reward_policy",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.output_schema",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.reranker_top_m",
        "pipeline.stage_configs.build_reranker_dataset.sub_stages.build_train_dataset.max_doc_chars",
        "infer_runtime.artifacts.flush_every_n",
        "infer_runtime.agent.max_retries",
        "infer_runtime.agent.retry_delay",
        "infer_runtime.agent.retry_backoff",
        "infer_runtime.agent.http_force_close",
        "infer_runtime.agent.fail_on_error",
        "resource.hardware.gpu_ids",
        "resource.stage_resources.generate_traces.services.agent_vllm.gpu_ids",
        "resource.stage_resources.generate_traces.services.agent_vllm.tensor_parallel_size",
        "resource.stage_resources.generate_traces.services.agent_vllm.port",
        "resource.stage_resources.generate_traces.services.agent_vllm.served_model_name",
        "resource.stage_resources.generate_traces.services.recall.gpu_ids",
        "resource.stage_resources.generate_traces.services.recall.port",
        "resource.stage_resources.generate_traces.services.recall.retrieval_service_url",
        "resource.stage_resources.build_reranker_dataset",
        "resource.stage_resources.train_agent",
        "resource.stage_resources.train_llm_reranker",
        "resource.stage_resources.infer_matrix",
    ]
    for path in required_paths:
        require_path(cfg, path)


def build_runtime(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, Path]]:
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S-%f")
    group = str(cfg["main_run"]["project"].get("group_name") or "agenticIterRag")
    exp = os.environ.get("EXP_NAME") or str(cfg["main_run"]["project"].get("experiment_name") or "agentic_iter_rag_v1")
    run_name = slugify(f"{timestamp}-pipeline-{exp}")
    run_root = Path(str(cfg["main_run"]["runtime"]["run_root"])) / group / run_name
    report_root = Path(str(cfg["main_run"]["runtime"]["report_root"])) / group
    artifact_root = Path(str(cfg["main_run"]["runtime"]["artifact_root"])) / group / run_name
    log_dir = run_root / "runtime_logs"
    files = {
        "run_root": run_root,
        "report_root": report_root,
        "artifact_root": artifact_root,
        "log_dir": log_dir,
        "final_yaml": log_dir / "pipeline.final_config.yaml",
        "final_json": log_dir / "pipeline.final_config.json",
        "env_file": log_dir / "pipeline.env",
        "runtime_env": log_dir / "pipeline.runtime_env.sh",
        "args_file": log_dir / "pipeline.args.txt",
        "manifest": artifact_root / "pipeline.manifest.json",
        "execution_plan": artifact_root / "execution_plan.yaml",
    }
    for path in files.values():
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path.mkdir(parents=True, exist_ok=True)
    env = {
        "AGENTIC_ITER_RAG_CONFIG_MODE": "1",
        "AGENTIC_ITER_RAG_PROJECT_ROOT": str(args.project_root),
        "REPO_ROOT": str(args.repo_root),
        "GROUP_NAME": group,
        "EXP_NAME": exp,
        "RUN_NAME": run_name,
        "RUN_ROOT": str(run_root),
        "LOG_DIR": str(log_dir),
        "ARTIFACT_ROOT": str(artifact_root),
        "REPORT_ROOT": str(report_root),
        "FINAL_CONFIG_YAML": str(files["final_yaml"]),
        "FINAL_CONFIG_JSON": str(files["final_json"]),
        "ENV_PATH": str(files["env_file"]),
        "MANIFEST_PATH": str(files["manifest"]),
        "EXECUTION_PLAN_PATH": str(files["execution_plan"]),
        "DRY_RUN": "1" if args.dry_run else "0",
    }
    return env, files


def main() -> None:
    args = parse_args()
    if os.environ.get("AGENTIC_ITER_RAG_ALLOW_SHELL_CONFIG") != "1":
        require_no_shell_only_business_env(os.environ)
    cfg = build_config(args)
    validate_current_config(cfg)
    env, files = build_runtime(args, cfg)
    cfg.setdefault("runtime_compiled", {})
    cfg["runtime_compiled"].update({k: str(v) for k, v in env.items()})
    write_yaml(files["final_yaml"], cfg)
    write_json(files["final_json"], cfg)
    write_json(
        files["manifest"],
        {
            "type": "agentic_iter_rag_compiled_pipeline",
            "final_config_yaml": str(files["final_yaml"]),
            "final_config_json": str(files["final_json"]),
            "env_path": str(files["env_file"]),
            "execution_plan": str(files["execution_plan"]),
            "dry_run": args.dry_run,
        },
    )
    with files["args_file"].open("w", encoding="utf-8") as f:
        for item in sys.argv[1:]:
            f.write(item + "\n")
    write_export_file(files["env_file"], env)
    write_export_file(files["runtime_env"], env)
    print(files["runtime_env"])


if __name__ == "__main__":
    main()
