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
    "agent_training": "agent_training",
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
    "agent_training_config": "agent_training",
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
    parser.add_argument("--AGENT_TRAINING_CONFIG", "--agent-training-config", dest="agent_training_config")
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


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def selected_stages(pipeline: dict[str, Any]) -> list[str]:
    """按 pipeline resume/stop/skip 推导本次会执行的 stage。

    compiler 的字段校验必须尊重 selected stages。比如 dataproduce 不会执行 branch reranker 训练，
    就不能强制要求 llm_reranker_grpo_branch 独有字段。
    """

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


def validate_current_config(cfg: dict[str, Any]) -> None:
    """拒绝旧字段，校验 v1 data produce 需要的新配置结构。"""

    unsupported_paths = [
        "reranker_training.label_policy",
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

    active_stages = set(selected_stages(cfg["pipeline"]))
    uses_branch_reranker_training = bool(
        active_stages
        & {
            "build_reranker_branch_dataset",
            "filter_reranker_branch_dataset",
            "train_llm_reranker",
            "build_service_bundle",
        }
    )

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
        "main_run.config_groups.reranker_training",
        "reranker_training.base_model",
        "reranker_training.trainer.method",
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
        "resource.stage_resources.generate_traces.services.recall.backend_type",
        "resource.stage_resources.generate_traces.services.recall.port",
        "resource.stage_resources.generate_traces.services.recall.retrieval_service_url",
        "resource.stage_resources.build_reranker_dataset",
        "resource.stage_resources.train_agent",
        "resource.stage_resources.infer_matrix",
    ]
    if uses_branch_reranker_training:
        required_paths.extend(
            [
                "pipeline.stage_configs.build_reranker_branch_dataset.inputs.enhanced_trajectory_manifest",
                "pipeline.stage_configs.build_reranker_branch_dataset.outputs.branch_dataset_manifest",
                "pipeline.stage_configs.filter_reranker_branch_dataset.inputs.source_branch_dataset_manifest",
                "pipeline.stage_configs.filter_reranker_branch_dataset.outputs.filtered_branch_dataset_manifest",
                "pipeline.stage_configs.train_llm_reranker.inputs.branch_dataset_manifest",
                "pipeline.stage_configs.build_service_bundle.outputs.service_bundle_dir",
                "main_run.data_artifacts.llm_reranker_branch_train_set.version",
                "main_run.data_artifacts.llm_reranker_branch_train_set.overwrite",
                "main_run.data_artifacts.llm_reranker_branch_train_set.derive_version_from_trajectory",
                "reranker_training.input.enhanced_trajectory_manifest",
                "reranker_training.input.branch_dataset_manifest",
                "reranker_training.input.reranker_train_set_manifest",
                "reranker_training.branch_dataset.enabled",
                "reranker_training.branch_dataset.version",
                "reranker_training.branch_dataset.overwrite",
                "reranker_training.branch_dataset.step_policy",
                "reranker_training.branch_dataset.random_seed",
                "reranker_training.branch_dataset.allow_no_search",
                "reranker_training.branch_dataset.candidate_top_n",
                "reranker_training.branch_dataset.visible_top_m",
                "reranker_training.branch_dataset.prompt_template_version",
                "reranker_training.branch_dataset.formatter",
                "reranker_training.branch_dataset.max_doc_chars",
                "reranker_training.branch_filter.enabled",
                "reranker_training.branch_filter.version",
                "reranker_training.branch_filter.overwrite",
                "reranker_training.branch_filter.max_samples",
                "reranker_training.branch_filter.sample_mode",
                "reranker_training.branch_filter.random_seed",
                "reranker_training.branch_filter.strategy.kind",
                "reranker_training.branch_filter.strategy.name",
                "reranker_training.branch_filter.strategy.builtin_name",
                "reranker_training.branch_filter.strategy.callable",
                "reranker_training.branch_filter.strategy.script_path",
                "reranker_training.branch_filter.strategy.kwargs",
                "reranker_training.continuation.agent_model",
                "reranker_training.continuation.use_frozen_agent",
                "reranker_training.continuation.search_tool_mode",
                "reranker_training.reward.strategy",
                "reranker_training.reward.format_penalty",
                "reranker_training.reward.answer_reward_function.path",
                "reranker_training.reward.answer_reward_function.name",
                "reranker_training.trainer.backend",
                "resource.stage_resources.build_reranker_branch_dataset",
                "resource.stage_resources.filter_reranker_branch_dataset",
                "resource.stage_resources.train_llm_reranker",
                "resource.stage_resources.build_service_bundle",
            ]
        )
    for path in required_paths:
        require_path(cfg, path)

    if not uses_branch_reranker_training:
        if "train_agent" in active_stages and deep_get(cfg, "pipeline.stage_configs.train_agent.impl") == "spad_rag":
            validate_spad_train_agent_config(cfg)
        return

    # AIR branch reranker 训练是主 pipeline 的一个配置组，不能绕过 main_run 单独运行。
    if deep_get(cfg, "main_run.config_groups.reranker_training") != "llm_reranker_grpo_branch":
        raise ValueError("main_run.config_groups.reranker_training must be llm_reranker_grpo_branch")

    # step policy 使用明确枚举，避免旧 type0/type1 这类难以读懂的值继续进入数据 manifest。
    step_policy = str(deep_get(cfg, "reranker_training.branch_dataset.step_policy"))
    allowed_step_policies = {"first_point", "end_point", "random_point", "all_steps"}
    if step_policy not in allowed_step_policies:
        raise ValueError(
            "reranker_training.branch_dataset.step_policy must be one of "
            f"{sorted(allowed_step_policies)}; got {step_policy!r}"
        )
    if step_policy == "all_steps":
        raise ValueError("reranker_training.branch_dataset.step_policy=all_steps is not supported in AIR v1")

    candidate_top_n = int(deep_get(cfg, "reranker_training.branch_dataset.candidate_top_n"))
    visible_top_m = int(deep_get(cfg, "reranker_training.branch_dataset.visible_top_m"))
    if visible_top_m > candidate_top_n:
        raise ValueError("reranker_training.branch_dataset.visible_top_m must be <= candidate_top_n")
    filter_kind = str(deep_get(cfg, "reranker_training.branch_filter.strategy.kind"))
    allowed_filter_kinds = {"builtin", "python_callable", "script"}
    if filter_kind not in allowed_filter_kinds:
        raise ValueError(
            "reranker_training.branch_filter.strategy.kind must be one of "
            f"{sorted(allowed_filter_kinds)}; got {filter_kind!r}"
        )
    sample_mode = str(deep_get(cfg, "reranker_training.branch_filter.sample_mode"))
    allowed_sample_modes = {"none", "first", "random"}
    if sample_mode not in allowed_sample_modes:
        raise ValueError(
            "reranker_training.branch_filter.sample_mode must be one of "
            f"{sorted(allowed_sample_modes)}; got {sample_mode!r}"
        )
    if str(deep_get(cfg, "reranker_training.continuation.search_tool_mode")) != "retriever_only":
        raise ValueError("reranker_training.continuation.search_tool_mode must be retriever_only")
    if not bool(deep_get(cfg, "reranker_training.continuation.use_frozen_agent")):
        raise ValueError("reranker_training.continuation.use_frozen_agent must be true")
    if str(deep_get(cfg, "reranker_training.trainer.method")) != "grpo":
        raise ValueError("reranker_training.trainer.method must be grpo")
    if "train_agent" in active_stages and deep_get(cfg, "pipeline.stage_configs.train_agent.impl") == "spad_rag":
        validate_spad_train_agent_config(cfg)


def validate_spad_train_agent_config(cfg: dict[str, Any]) -> None:
    """校验 SPAD-RAG 作为 train_agent impl 时必须存在的配置骨架。"""

    required_paths = [
        "main_run.config_groups.agent_training",
        "pipeline.stage_configs.train_agent.impl",
        "pipeline.stage_configs.train_agent.impl_config_ref",
        "pipeline.stage_configs.train_agent.inputs.train_files",
        "pipeline.stage_configs.train_agent.inputs.val_files",
        "pipeline.stage_configs.train_agent.inputs.init_actor_model",
        "pipeline.stage_configs.train_agent.outputs.agent_checkpoint",
        "pipeline.stage_configs.train_agent.outputs.agent_training_manifest",
        "agent_training.impl",
        "agent_training.sub_stage_order",
        "agent_training.sub_stages.search_policy_rl",
        "agent_training.sub_stages.answer_refresh_data",
        "agent_training.sub_stages.answer_distillation",
        "agent_training.teacher_answerer.default_service_profile",
        "agent_training.teacher_answerer.service_profiles",
        "resource.stage_resources.train_agent.impls.spad_rag",
    ]
    for path in required_paths:
        require_path(cfg, path)

    if deep_get(cfg, "agent_training.impl") != "spad_rag":
        raise ValueError("agent_training.impl must be spad_rag when train_agent.impl=spad_rag")
    if deep_get(cfg, "pipeline.stage_configs.train_agent.impl_config_ref") != "agent_training":
        raise ValueError("train_agent.impl_config_ref must be agent_training for SPAD-RAG")

    sub_stage_order = as_list(deep_get(cfg, "agent_training.sub_stage_order"))
    expected = {"search_policy_rl", "answer_refresh_data", "answer_distillation"}
    missing = sorted(expected - set(sub_stage_order))
    if missing:
        raise ValueError(f"agent_training.sub_stage_order is missing SPAD sub-stages: {missing}")


def build_runtime(args: argparse.Namespace, cfg: dict[str, Any]) -> tuple[dict[str, str], dict[str, Path]]:
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S-%f")
    group = str(cfg["main_run"]["project"].get("group_name") or "agenticIterRag")
    exp = os.environ.get("EXP_NAME") or str(cfg["main_run"]["project"].get("experiment_name") or "agentic_iter_rag_v1")
    run_name = slugify(f"{timestamp}-pipeline-{exp}")
    runtime_cfg = cfg["main_run"]["runtime"]
    outputs_leaf = Path(str(runtime_cfg.get("outputs_dir") or "outputs"))
    if outputs_leaf.is_absolute() or ".." in outputs_leaf.parts:
        raise ValueError("main_run.runtime.outputs_dir must be a run-local relative directory name")
    run_root = Path(str(runtime_cfg["run_root"])) / run_name
    report_root = Path(str(cfg["main_run"]["runtime"]["report_root"])) / group
    checkpoint_base = Path(str(runtime_cfg.get("checkpoint_root") or (args.repo_root / "checkpoints" / "AIR")))
    checkpoint_root = checkpoint_base / run_name
    artifact_root = run_root / outputs_leaf
    log_dir = run_root / "runtime_logs"
    pipeline_log_dir = log_dir / "pipeline"
    files = {
        "run_root": run_root,
        "report_root": report_root,
        "outputs_dir": artifact_root,
        "artifact_root": artifact_root,
        "checkpoint_root": checkpoint_root,
        "log_dir": log_dir,
        "pipeline_log_dir": pipeline_log_dir,
        "final_yaml": pipeline_log_dir / "pipeline.final_config.yaml",
        "final_json": pipeline_log_dir / "pipeline.final_config.json",
        "env_file": pipeline_log_dir / "pipeline.env",
        "runtime_env": pipeline_log_dir / "pipeline.runtime_env.sh",
        "args_file": pipeline_log_dir / "pipeline.args.txt",
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
        "RUNTIME_LOG_ROOT": str(log_dir),
        "PIPELINE_LOG_DIR": str(pipeline_log_dir),
        "ARTIFACT_ROOT": str(artifact_root),
        "CHECKPOINT_ROOT": str(checkpoint_root),
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
            "outputs_dir": str(files["outputs_dir"]),
            "checkpoint_root": str(files["checkpoint_root"]),
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
