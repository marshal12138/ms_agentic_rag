#!/usr/bin/env python3
"""Compile CoAgenticRetriever evaluation configuration into runtime files."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trainer_launcher.paths import (
    normalize_config_group_value,
    normalize_main_run_config,
    resolve_repo_path,
    slugify_name,
)
from trainer_launcher.resource import RESOURCE_KEYS, load_canonical_resource_env
from trainer_launcher.shell_quote import write_export_file
from trainer_launcher.tool_config import read_static_tool_config
from trainer_launcher.yaml_utils import dump_mapping, load_mapping


@dataclass
class EvalSelection:
    main_run_config: str = ""
    eval_runtime_config: str = ""
    eval_budget_config: str = ""
    resource_config: str = ""
    data_config: str = ""
    model_config: str = ""
    rollout_config: str = ""
    ranker_base_config: str = ""
    overlay_yamls: list[Path] = field(default_factory=list)
    eval_cli_overrides: list[str] = field(default_factory=list)
    passthrough_args: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompilerContext:
    repo_root: Path
    script_dir: Path
    assets_dir: Path
    project_root: Path
    external_model_root: Path
    external_retrieval_root: Path
    device_prefix: str
    visible_devices_var: str
    accelerator: str

    def device_spec(self, index: str | int | None = None) -> str:
        return f"{self.device_prefix}:{index}" if index not in (None, "") else self.device_prefix


EVAL_ENV_TO_CONFIG_PATH = {
    "EVAL_TASK_NAME": "identity.eval_task_name",
    "RUN_MODE": "mode.run_mode",
    "RERANKER": "mode.reranker",
    "AGENT_MODEL": "models.agent_model",
    "RANKER_MODEL": "models.ranker_model",
    "RANKER_BASE_MODEL": "models.ranker_base_model",
    "RANKER_ENCODER_PATH": "models.ranker_encoder_path",
    "RECALL_MODEL_PATH": "models.recall_model_path",
    "DATA_PATH": "data.data_path",
    "MAX_EVAL_NUM": "data.max_eval_num",
    "EVAL_BATCH_SIZE": "data.eval_batch_size",
    "MAX_EVAL_STEPS": "data.max_eval_steps",
    "MAX_RANKER_STEPS": "data.max_ranker_steps",
    "KEEP_TRACE": "data.keep_trace",
    "TEMPERATURE": "generation.temperature",
    "TOP_P": "generation.top_p",
    "REQUEST_TIMEOUT": "generation.request_timeout",
    "STOP_SEQUENCES": "generation.stop_sequences",
    "AGENT_PORT": "vllm.agent_port",
    "AGENT_SERVED_MODEL": "vllm.agent_served_model",
    "VLLM_STARTUP_TIMEOUT": "vllm.startup_timeout",
    "GPU_MEMORY_UTILIZATION": "vllm.gpu_memory_utilization",
    "MAX_NUM_SEQS": "vllm.max_num_seqs",
    "VLLM_ATTENTION_BACKEND": "vllm.attention_backend",
    "PROXY_PORT": "retrieval.proxy_port",
    "RETRIEVAL_SERVICE_URL": "retrieval.retrieval_service_url",
    "RETRIEVAL_MAX_RETRIES": "retrieval.max_retries",
    "RETRIEVAL_RETRY_DELAY": "retrieval.retry_delay",
    "RETRIEVAL_RETRY_BACKOFF": "retrieval.retry_backoff",
    "RETRIEVAL_PREFLIGHT_QUERY": "retrieval.preflight_query",
    "RETRIEVAL_PREFLIGHT_EXPECT": "retrieval.preflight_expect",
    "RANKER_DEVICE": "ranker.device",
    "RANKER_CUDA_VISIBLE_DEVICES": "ranker.cuda_visible_devices",
    "RANKER_CONFIG_DEVICE": "ranker.config_device",
    "RANKER_MAX_QUERY_LENGTH": "ranker.max_query_length",
    "RANKER_MAX_DOC_LENGTH": "ranker.max_doc_length",
    "LLM_JUDGE_ENDPOINT": "llm_judge.endpoint",
    "LLM_JUDGE_MODEL": "llm_judge.model",
    "LLM_JUDGE_PROMPT_PATH": "llm_judge.prompt_path",
    "LLM_JUDGE_MAX_CHUNK_CHARS": "llm_judge.max_chunk_chars",
    "LLM_JUDGE_MAX_TOKENS": "llm_judge.max_tokens",
    "LLM_JUDGE_TEMPERATURE": "llm_judge.temperature",
    "LLM_JUDGE_REQUEST_TIMEOUT": "llm_judge.request_timeout",
    "LLM_JUDGE_MAX_RETRIES": "llm_judge.max_retries",
    "LLM_JUDGE_RETRY_DELAY": "llm_judge.retry_delay",
    "LLM_JUDGE_RETRY_BACKOFF": "llm_judge.retry_backoff",
    "TOOL_CONFIG": "tool.tool_config",
    "INJECT_TOOL_SCHEMA": "tool.inject_tool_schema",
    "TRUST_REMOTE_CODE": "tool.trust_remote_code",
    "RUN_NAME": "artifacts.run_name",
    "EXP_NAME": "artifacts.exp_name",
    "TASK_NAME": "artifacts.task_name",
    "TRACE_DIR": "artifacts.trace_dir",
    "RUNTIME_LOG_DIR": "artifacts.runtime_log_dir",
    "OUT_DIR": "artifacts.out_dir",
    "LOG_DIR": "artifacts.log_dir",
    "REPORT_PATH": "artifacts.report_path",
    "METRICS_JSONL": "artifacts.metrics_jsonl",
    "SEARCH_TIMING_JSONL": "artifacts.search_timing_jsonl",
    "LLM_IO_JSONL": "artifacts.llm_io_jsonl",
    "INFER_LOG": "artifacts.infer_log",
    "RECALL_SERVICE_LOG": "artifacts.recall_service_log",
    "RANKER_OUTPUT_JSONL": "artifacts.ranker_output_jsonl",
    "ROLLOUT_DATA_DIR": "artifacts.rollout_data_dir",
    "VALIDATION_DATA_DIR": "artifacts.validation_data_dir",
    "ENV_PATH": "artifacts.env_path",
}

CLI_KEY_ALIASES = {
    "eval_task_name": "identity.eval_task_name",
    "run_mode": "mode.run_mode",
    "reranker": "mode.reranker",
    "agent_model": "models.agent_model",
    "ranker_model": "models.ranker_model",
    "ranker_base_model": "models.ranker_base_model",
    "data_path": "data.data_path",
    "eval_batch_size": "data.eval_batch_size",
    "max_eval_num": "data.max_eval_num",
    "inject_tool_schema": "tool.inject_tool_schema",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile CoAgenticRetriever eval launcher config.")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--script-dir", required=True, type=Path)
    parser.add_argument("--assets-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--external-model-root", required=True, type=Path)
    parser.add_argument("--external-retrieval-root", required=True, type=Path)
    parser.add_argument("--device-prefix", required=True)
    parser.add_argument("--visible-devices-var", required=True)
    parser.add_argument("--accelerator", required=True)
    parser.add_argument("launcher_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.launcher_args and args.launcher_args[0] == "--":
        args.launcher_args = args.launcher_args[1:]
    return args


def _require_value(option: str, value: str | None) -> str:
    if value is None or value == "":
        raise ValueError(f"{option} requires a non-empty value")
    return value


def _consume_value(argv: list[str], index: int, option: str, inline_value: str | None) -> tuple[str, int]:
    if inline_value is not None:
        return _require_value(option, inline_value), index + 1
    if index + 1 >= len(argv):
        raise ValueError(f"{option} requires a value")
    return _require_value(option, argv[index + 1]), index + 2


def parse_launcher_args(argv: list[str], repo_root: Path) -> EvalSelection:
    selection = EvalSelection()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--":
            selection.passthrough_args.extend(argv[i + 1 :])
            break
        option = arg
        inline_value: str | None = None
        if arg.startswith("--") and "=" in arg:
            option, inline_value = arg.split("=", 1)
        if option == "--main_run_config":
            selection.main_run_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--EVAL_RUNTIME_CONFIG":
            selection.eval_runtime_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--EVAL_BUDGET_CONFIG":
            selection.eval_budget_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--RESOURCE_CONFIG":
            selection.resource_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--DATA_CONFIG":
            selection.data_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--MODEL_CONFIG":
            selection.model_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--ROLLOUT_CONFIG":
            selection.rollout_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--RANKER_BASE_CONFIG":
            selection.ranker_base_config, i = _consume_value(argv, i, option, inline_value)
        elif option == "--OVERLAY_YAML":
            value, i = _consume_value(argv, i, option, inline_value)
            selection.overlay_yamls.append(resolve_repo_path(repo_root, value))
        elif "=" in arg and not arg.startswith("--"):
            selection.eval_cli_overrides.append(arg)
            i += 1
        else:
            selection.passthrough_args.append(arg)
            i += 1
    return selection


def normalize_group_file(
    *,
    repo_root: Path,
    project_root: Path,
    option: str,
    group: str,
    value: str,
) -> tuple[str, Path]:
    name = normalize_config_group_value(repo_root, project_root, option=option, group=group, value=value)
    return name, project_root / "config" / group / f"{name}.yaml"


def deep_get(data: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def deep_set(data: dict[str, Any], path: str, value: Any) -> None:
    cur = data
    parts = path.split(".")
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


def budget_get(budget: Mapping[str, Any], flat_key: str, nested_key: str, default: Any = None) -> Any:
    value = budget.get(flat_key)
    if value is not None:
        return value
    return deep_get(budget, nested_key, default)


def coerce_by_existing(value: str, existing: Any) -> Any:
    if isinstance(existing, bool):
        return value.lower() in {"1", "true", "yes", "on"}
    if isinstance(existing, int) and not isinstance(existing, bool):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    if isinstance(existing, list):
        return [part for part in value.split(",") if part]
    if existing is None:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def merge_strict(base: dict[str, Any], overlay: Mapping[str, Any], *, source: Path, path: str = "") -> None:
    for key, value in overlay.items():
        if path == "" and key in RESOURCE_KEYS:
            continue
        if path == "" and key == "resources":
            continue
        if key not in base:
            dotted = f"{path}.{key}" if path else key
            raise ValueError(f"{source} defines unknown eval config key: {dotted}")
        current = base[key]
        if isinstance(current, dict) and isinstance(value, Mapping):
            merge_strict(current, value, source=source, path=f"{path}.{key}" if path else key)
        else:
            base[key] = value


def normalize_run_mode(value: Any) -> str:
    text = str(value or "full").strip()
    if text in {"full", "co-training", "co_training"}:
        return "full"
    if text in {"no-ranker", "no_ranker"}:
        return "no-ranker"
    if text in {"ranker-only", "ranker_only"}:
        return "ranker-only"
    raise ValueError(f"unsupported eval run_mode={text}; use full, no-ranker, or ranker-only")


def normalize_reranker(value: Any) -> str:
    text = str(value or "dense_e5").strip()
    if text in {"dense", "e5", "dense-e5", "dense_e5"}:
        return "dense_e5"
    if text in {"llm-as-judge", "llm_judge", "judge", "llm_as_judge"}:
        return "llm_as_judge"
    raise ValueError(f"unsupported eval reranker={text}; use dense_e5 or llm_as_judge")


def normalize_device(device: str | None, *, ctx: CompilerContext) -> str:
    if not device:
        return ctx.device_prefix
    text = str(device)
    if ctx.device_prefix == "npu":
        if text == "cuda":
            return "npu"
        if text.startswith("cuda:"):
            return "npu:" + text.split(":", 1)[1]
    if ctx.device_prefix == "cuda":
        if text == "npu":
            return "cuda"
        if text.startswith("npu:"):
            return "cuda:" + text.split(":", 1)[1]
    return text


def resolve_config_path(repo_root: Path, value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(resolve_repo_path(repo_root, str(value)))


def resolve_prompt_path(repo_root: Path, project_root: Path, value: Any) -> str:
    if value in (None, ""):
        return ""
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        return str(path)
    if text.startswith("CoAgenticRetriever/"):
        return str(repo_root / text)
    return str(project_root / text)


def parse_main_run(ctx: CompilerContext, selection: EvalSelection) -> dict[str, Any]:
    if not selection.main_run_config:
        return {}
    ref = normalize_main_run_config(ctx.repo_root, ctx.project_root, selection.main_run_config)
    data = load_mapping(ref.path, label="main_run_config")
    data["_main_run_name"] = ref.name
    data["_main_run_file"] = str(ref.path)
    return data


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def build_eval_args(env: Mapping[str, str]) -> list[str]:
    args = [
        "run",
        "--run-mode", env["RUN_MODE"],
        "--reranker", env["RERANKER"],
        "--data-path", env["DATA_PATH"],
        "--max-eval-num", env["MAX_EVAL_NUM"],
        "--max-ranker-steps", env["MAX_RANKER_STEPS"],
        "--batch-size", env["EVAL_BATCH_SIZE"],
        "--keep-trace", env["KEEP_TRACE"],
        "--trace-dir", env["TRACE_DIR"],
        "--report-path", env["REPORT_PATH"],
        "--eval-task-name", env["EVAL_TASK_NAME"],
        "--retrieval-url", env["RETRIEVAL_SERVICE_URL"],
        "--agent-served-model", env["AGENT_SERVED_MODEL"],
        "--top-n", env["RECALL_FINAL_TOP_N"],
        "--top-m", env["SEARCH_TOOL_FINAL_TOP_M"],
        "--ranker-top-k", env["RANKER_FINAL_TOP_K"],
        "--max-assistant-turns", env["MAX_ASSISTANT_TURNS"],
        "--max-user-turns", env["MAX_USER_TURNS"],
        "--max-tool-response-length", env["MAX_TOOL_RESPONSE_LENGTH"],
        "--max-prompt-length", env["MAX_PROMPT_LENGTH"],
        "--max-response-length", env["MAX_RESPONSE_LENGTH"],
        "--max-model-len", env["MAX_MODEL_LEN"],
        "--temperature", env["TEMPERATURE"],
        "--top-p", env["TOP_P"],
        "--request-timeout", env["REQUEST_TIMEOUT"],
        "--max-retries", env["RETRIEVAL_MAX_RETRIES"],
        "--retry-delay", env["RETRIEVAL_RETRY_DELAY"],
        "--retry-backoff", env["RETRIEVAL_RETRY_BACKOFF"],
        "--metrics-jsonl", env["METRICS_JSONL"],
        "--search-timing-jsonl", env["SEARCH_TIMING_JSONL"],
        "--ranker-output-jsonl", env["RANKER_OUTPUT_JSONL"],
        "--validation-data-dir", env["VALIDATION_DATA_DIR"],
        "--rollout-data-dir", env["ROLLOUT_DATA_DIR"],
        "--ranker-device", env["RANKER_DEVICE"],
        "--ranker-max-query-length", env["RANKER_MAX_QUERY_LENGTH"],
        "--ranker-max-doc-length", env["RANKER_MAX_DOC_LENGTH"],
        "--tool-config-path", env["TOOL_CONFIG"],
        "--llm-io-max-records", env["LLM_IO_MAX_RECORDS"],
    ]
    if env.get("LLM_IO_JSONL"):
        args.extend(["--llm-io-jsonl", env["LLM_IO_JSONL"]])
    if env["RUN_MODE"] != "ranker-only":
        args.extend(["--agent-model", env["AGENT_MODEL"]])
        args.extend(["--agent-base-url", f"http://127.0.0.1:{env['AGENT_PORT']}"])
    if env["RUN_MODE"] != "no-ranker" and env["RERANKER"] == "dense_e5":
        args.extend(["--ranker-model", env["RANKER_MODEL"], "--ranker-base-model", env["RANKER_BASE_MODEL"]])
        if env.get("RANKER_ENCODER_PATH"):
            args.extend(["--ranker-encoder", env["RANKER_ENCODER_PATH"]])
    if env["RUN_MODE"] != "no-ranker" and env["RERANKER"] == "llm_as_judge":
        args.extend(
            [
                "--llm-judge-endpoint", env["LLM_JUDGE_ENDPOINT"],
                "--llm-judge-model", env["LLM_JUDGE_MODEL"],
                "--llm-judge-prompt-path", env["LLM_JUDGE_PROMPT_PATH"],
                "--llm-judge-max-chunk-chars", env["LLM_JUDGE_MAX_CHUNK_CHARS"],
                "--llm-judge-max-tokens", env["LLM_JUDGE_MAX_TOKENS"],
                "--llm-judge-temperature", env["LLM_JUDGE_TEMPERATURE"],
                "--llm-judge-request-timeout", env["LLM_JUDGE_REQUEST_TIMEOUT"],
                "--llm-judge-max-retries", env["LLM_JUDGE_MAX_RETRIES"],
                "--llm-judge-retry-delay", env["LLM_JUDGE_RETRY_DELAY"],
                "--llm-judge-retry-backoff", env["LLM_JUDGE_RETRY_BACKOFF"],
            ]
        )
    for stop in [part for part in env.get("STOP_SEQUENCES", "").split(",") if part]:
        args.extend(["--stop-sequence", stop])
    args.append("--trust-remote-code" if env["TRUST_REMOTE_CODE"].lower() == "true" else "--no-trust-remote-code")
    args.append("--enable-thinking" if env["ENABLE_THINKING"].lower() == "true" else "--no-enable-thinking")
    args.append("--inject-tool-schema" if env["INJECT_TOOL_SCHEMA"].lower() == "true" else "--no-inject-tool-schema")
    return args


def compile_config(ctx: CompilerContext, launcher_args: list[str], environ: Mapping[str, str]) -> Path:
    selection = parse_launcher_args(launcher_args, ctx.repo_root)
    main_run = parse_main_run(ctx, selection)
    eval_groups = main_run.get("eval_config_groups") or {}
    trainer_groups = main_run.get("trainer_config_groups") or {}
    if not isinstance(eval_groups, dict):
        raise TypeError("main_run eval_config_groups must be a mapping")
    if not isinstance(trainer_groups, dict):
        raise TypeError("main_run trainer_config_groups must be a mapping")

    eval_runtime_name, eval_runtime_file = normalize_group_file(
        repo_root=ctx.repo_root,
        project_root=ctx.project_root,
        option="--EVAL_RUNTIME_CONFIG",
        group="eval_runtime",
        value=selection.eval_runtime_config or str(eval_groups.get("eval_runtime") or "coagentic_retriever_vllm"),
    )
    eval_budget_name, eval_budget_file = normalize_group_file(
        repo_root=ctx.repo_root,
        project_root=ctx.project_root,
        option="--EVAL_BUDGET_CONFIG",
        group="eval_budget",
        value=selection.eval_budget_config or str(eval_groups.get("eval_budget") or "coagentic_retriever_aligned_budget"),
    )
    resource_config = selection.resource_config or str(eval_groups.get("resource") or main_run.get("resource_config") or "local_eval_4gpu_0_3")
    data_config = selection.data_config or str(trainer_groups.get("data") or "co_search_ablation")
    model_config = selection.model_config or str(trainer_groups.get("model") or "qwen3_4b")
    rollout_config = selection.rollout_config or str(trainer_groups.get("rollout") or "cosearch_async_qwen3_4b")
    ranker_base_config = selection.ranker_base_config or str(trainer_groups.get("ranker_base") or "ranker_contrastive")

    data_name, data_file = normalize_group_file(
        repo_root=ctx.repo_root, project_root=ctx.project_root, option="--DATA_CONFIG", group="data", value=data_config
    )
    model_name, model_file = normalize_group_file(
        repo_root=ctx.repo_root, project_root=ctx.project_root, option="--MODEL_CONFIG", group="model", value=model_config
    )
    rollout_name, rollout_file = normalize_group_file(
        repo_root=ctx.repo_root, project_root=ctx.project_root, option="--ROLLOUT_CONFIG", group="rollout", value=rollout_config
    )
    ranker_base_name, ranker_base_file = normalize_group_file(
        repo_root=ctx.repo_root,
        project_root=ctx.project_root,
        option="--RANKER_BASE_CONFIG",
        group="experimental/ranker_base",
        value=ranker_base_config,
    )

    config = load_mapping(eval_runtime_file, label="eval runtime config")
    budget = load_mapping(eval_budget_file, label="eval budget config")
    base_config = copy.deepcopy(config)
    for overlay in selection.overlay_yamls:
        data = load_mapping(overlay, label="eval overlay YAML")
        merge_strict(config, data, source=overlay)

    for raw in selection.eval_cli_overrides:
        key, value = raw.split("=", 1)
        path = CLI_KEY_ALIASES.get(key, key)
        existing = deep_get(config, path, None)
        if existing is None and deep_get(base_config, path, None) is None:
            raise ValueError(f"unknown eval CLI override key: {key}")
        deep_set(config, path, coerce_by_existing(value, existing))

    for env_name, path in EVAL_ENV_TO_CONFIG_PATH.items():
        if env_name in environ and environ[env_name] != "":
            existing = deep_get(config, path, None)
            deep_set(config, path, coerce_by_existing(environ[env_name], existing))

    _, resource_base_file, resource_file, resource_env = load_canonical_resource_env(
        repo_root=ctx.repo_root,
        project_root=ctx.project_root,
        resource_config=resource_config,
        overlay_yamls=selection.overlay_yamls,
        environ=environ,
    )

    data_cfg = load_mapping(data_file, label="data config")
    model_cfg = load_mapping(model_file, label="model config")
    rollout_cfg = load_mapping(rollout_file, label="rollout config")
    ranker_base_cfg = load_mapping(ranker_base_file, label="ranker base config")
    recall_cfg = ranker_base_cfg.get("recall_retriever") or {}
    ranker_cfg = ranker_base_cfg.get("ranker") or {}
    if not isinstance(recall_cfg, dict) or not isinstance(ranker_cfg, dict):
        raise TypeError("ranker base config must contain recall_retriever and ranker mappings")

    run_mode = normalize_run_mode(deep_get(config, "mode.run_mode"))
    reranker = normalize_reranker(deep_get(config, "mode.reranker"))
    deep_set(config, "mode.run_mode", run_mode)
    deep_set(config, "mode.reranker", reranker)

    eval_task_name = str(deep_get(config, "identity.eval_task_name") or "default")
    if environ.get("EVAL_TASK_NAME"):
        eval_task_name = environ["EVAL_TASK_NAME"]
    eval_task_slug = slugify_name(eval_task_name)

    group_name = resource_env["GROUP_NAME"]
    group_slug = slugify_name(group_name)
    run_stamp = environ.get("RUN_STAMP") or datetime.now().strftime("%y%m%d-%H%M")
    eval_log_root = deep_get(config, "artifacts.eval_log_root") or str(ctx.repo_root / "log" / "eval_res" / group_slug)
    eval_report_root = deep_get(config, "artifacts.eval_report_root") or str(ctx.repo_root / "reports" / "eval" / group_slug)
    task_name = deep_get(config, "artifacts.task_name") or environ.get("TASK_NAME") or f"{run_stamp}-{eval_task_slug}"
    run_name = deep_get(config, "artifacts.run_name") or environ.get("RUN_NAME") or eval_task_slug
    exp_name = deep_get(config, "artifacts.exp_name") or environ.get("EXP_NAME") or run_name
    trace_dir = Path(str(deep_get(config, "artifacts.trace_dir") or Path(eval_log_root) / task_name))
    runtime_log_dir = Path(str(deep_get(config, "artifacts.runtime_log_dir") or trace_dir / "runtime_logs"))
    out_dir = Path(str(deep_get(config, "artifacts.out_dir") or trace_dir))
    log_dir = Path(str(deep_get(config, "artifacts.log_dir") or runtime_log_dir))
    report_path = Path(str(deep_get(config, "artifacts.report_path") or Path(eval_report_root) / f"{task_name}.report.md"))
    log_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_log_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    source_tool_config = resolve_config_path(ctx.repo_root, deep_get(config, "tool.tool_config"))
    no_ranker_tool_config = resolve_config_path(ctx.repo_root, deep_get(config, "tool.no_ranker_tool_config"))
    if run_mode == "no-ranker" and "TOOL_CONFIG" not in environ:
        source_tool_config = no_ranker_tool_config
    tool_data = load_mapping(Path(source_tool_config), label="tool config")
    runtime_tool_config = log_dir / f"{run_name}.tool_config.yaml"
    dump_mapping(runtime_tool_config, tool_data)
    static_tool = read_static_tool_config(Path(source_tool_config))

    parsed_url = urlparse(static_tool.retrieval_service_url)
    proxy_port = str(deep_get(config, "retrieval.proxy_port") or parsed_url.port or "8030")
    retrieval_service_url = (
        str(deep_get(config, "retrieval.retrieval_service_url") or "")
        or static_tool.retrieval_service_url
        or f"http://127.0.0.1:{proxy_port}/retrieve"
    )
    if "PROXY_PORT" in environ and "RETRIEVAL_SERVICE_URL" not in environ:
        proxy_port = environ["PROXY_PORT"]
        retrieval_service_url = f"http://127.0.0.1:{proxy_port}/retrieve"

    val_files = data_cfg.get("val_files") or []
    default_data_path = val_files[0] if isinstance(val_files, list) and val_files else ""
    data_path = str(deep_get(config, "data.data_path") or default_data_path)
    agent_model = str(deep_get(config, "models.agent_model") or "")
    ranker_model = str(deep_get(config, "models.ranker_model") or "")
    ranker_base_model = str(
        deep_get(config, "models.ranker_base_model")
        or ranker_cfg.get("model_path")
        or ""
    )
    ranker_encoder_path = str(deep_get(config, "models.ranker_encoder_path") or "")
    recall_model_path = str(deep_get(config, "models.recall_model_path") or recall_cfg.get("model_path") or ctx.external_model_root / "retriever" / "e5-base-v2")

    recall_top_k = str(recall_cfg.get("recall_final_top_n") or "50")
    top_m = static_tool.search_tool_final_top_m or "5"
    ranker_top_k = str(ranker_cfg.get("top_k") or recall_top_k)
    ranker_device = normalize_device(str(deep_get(config, "ranker.device") or ctx.device_spec(0)), ctx=ctx)
    ranker_max_query = str(deep_get(config, "ranker.max_query_length") or ranker_cfg.get("max_query_length") or "256")
    ranker_max_doc = str(deep_get(config, "ranker.max_doc_length") or ranker_cfg.get("max_doc_length") or "512")

    wait_for_gpus = resource_env.get("WAIT_FOR_GPUS") or ""
    if not wait_for_gpus:
        if run_mode == "no-ranker":
            wait_for_gpus = f"{resource_env['AGENT_GPU_IDS']},{resource_env['RECALL_GPU_ID']}"
        elif reranker == "llm_as_judge":
            wait_for_gpus = f"{resource_env['AGENT_GPU_IDS']},{resource_env['RECALL_GPU_ID']},{resource_env['LLM_JUDGE_GPU_IDS']}"
        else:
            wait_for_gpus = f"{resource_env['AGENT_GPU_IDS']},{resource_env['RANK_GPU_ID']},{resource_env['RECALL_GPU_ID']}"

    max_num_seqs = (
        deep_get(config, "vllm.max_num_seqs")
        or budget_get(budget, "max_num_seqs", "actor_rollout_ref.rollout.max_num_seqs")
        or deep_get(config, "data.eval_batch_size")
    )
    max_ranker_steps = deep_get(config, "data.max_ranker_steps") or deep_get(config, "data.max_eval_steps") or "1"
    stop_sequences = deep_get(config, "generation.stop_sequences") or []
    if isinstance(stop_sequences, str):
        stop_sequences_text = stop_sequences
    else:
        stop_sequences_text = ",".join(str(item) for item in stop_sequences)

    llm_judge_prompt = resolve_prompt_path(ctx.repo_root, ctx.project_root, deep_get(config, "llm_judge.prompt_path"))
    evaluator = resolve_config_path(ctx.repo_root, deep_get(config, "entrypoint.evaluator"))
    env: dict[str, str] = {
        "EVAL_CONFIG_COMPILER_USED": "1",
        "MAIN_RUN_CONFIG": selection.main_run_config,
        "MAIN_RUN_CONFIG_FILE": str(main_run.get("_main_run_file", "")),
        "EVAL_RUNTIME_CONFIG": eval_runtime_name,
        "EVAL_RUNTIME_CONFIG_FILE": str(eval_runtime_file),
        "EVAL_BUDGET_CONFIG": eval_budget_name,
        "EVAL_BUDGET_CONFIG_FILE": str(eval_budget_file),
        "DATA_CONFIG": data_name,
        "MODEL_CONFIG": model_name,
        "ROLLOUT_CONFIG": rollout_name,
        "RANKER_BASE_CONFIG": ranker_base_name,
        "RESOURCE_CONFIG": resource_config,
        "RESOURCE_BASE_CONFIG_FILE": str(resource_base_file),
        "RESOURCE_CONFIG_FILE": str(resource_file),
        "PROJECT_ROOT": str(ctx.project_root),
        "COAGENTIC_PROJECT_ROOT": str(ctx.project_root),
        "COSEARCH_ACCELERATOR": ctx.accelerator,
        "VISIBLE_DEVICES_VAR": ctx.visible_devices_var,
        "DEVICE_PREFIX": ctx.device_prefix,
        "GROUP_NAME": group_name,
        "GROUP_SLUG": group_slug,
        "EVAL_TASK_NAME": eval_task_name,
        "EVAL_TASK_SLUG": eval_task_slug,
        "TASK_NAME": str(task_name),
        "RUN_NAME": str(run_name),
        "EXP_NAME": str(exp_name),
        "RUN_STAMP": run_stamp,
        "EVALUATOR": evaluator,
        "RUN_MODE": run_mode,
        "RERANKER": reranker,
        "AGENT_MODEL": agent_model,
        "RECALL_MODEL_PATH": recall_model_path,
        "RANKER_MODEL": ranker_model,
        "RANKER_BASE_MODEL": ranker_base_model,
        "RANKER_ENCODER_PATH": ranker_encoder_path,
        "CORPUS_JSONL": str(environ.get("CORPUS_JSONL") or ctx.external_retrieval_root / "wiki-18" / "wiki-18.jsonl"),
        "DATA_PATH": data_path,
        "MAX_EVAL_NUM": str(deep_get(config, "data.max_eval_num")),
        "EVAL_BATCH_SIZE": str(deep_get(config, "data.eval_batch_size")),
        "MAX_EVAL_STEPS": str(deep_get(config, "data.max_eval_steps")),
        "MAX_RANKER_STEPS": str(max_ranker_steps),
        "KEEP_TRACE": str(deep_get(config, "data.keep_trace")),
        "RECALL_FINAL_TOP_N": recall_top_k,
        "SEARCH_TOOL_FINAL_TOP_M": top_m,
        "RANKER_FINAL_TOP_K": ranker_top_k,
        "PROXY_PORT": proxy_port,
        "RETRIEVAL_SERVICE_URL": retrieval_service_url,
        "RETRIEVAL_MAX_RETRIES": str(static_tool.max_concurrent_per_worker and deep_get(config, "retrieval.max_retries") or deep_get(config, "retrieval.max_retries")),
        "RETRIEVAL_RETRY_DELAY": str(deep_get(config, "retrieval.retry_delay")),
        "RETRIEVAL_RETRY_BACKOFF": str(deep_get(config, "retrieval.retry_backoff")),
        "RETRIEVAL_PREFLIGHT_QUERY": str(deep_get(config, "retrieval.preflight_query")),
        "RETRIEVAL_PREFLIGHT_EXPECT": str(deep_get(config, "retrieval.preflight_expect") or ""),
        "RETRIEVER_DEVICE": normalize_device(str(deep_get(config, "retrieval.retriever_device") or ctx.device_prefix), ctx=ctx),
        "AGENT_GPU_IDS": resource_env["AGENT_GPU_IDS"],
        "RANK_GPU_ID": resource_env["RANK_GPU_ID"],
        "RECALL_GPU_ID": resource_env["RECALL_GPU_ID"],
        "LLM_JUDGE_GPU_IDS": resource_env["LLM_JUDGE_GPU_IDS"],
        "WAIT_FOR_GPUS": wait_for_gpus,
        "WAIT_FOR_GPU_RELEASE": resource_env["WAIT_FOR_GPU_RELEASE"],
        "WAIT_FOR_GPU_INTERVAL_SECONDS": resource_env["WAIT_FOR_GPU_INTERVAL_SECONDS"],
        "WAIT_FOR_GPU_LABEL": resource_env["WAIT_FOR_GPU_LABEL"],
        "AUTO_START_RECALL_SERVICE": resource_env["AUTO_START_RECALL_SERVICE"],
        "AUTO_STOP_RECALL_SERVICE": resource_env["AUTO_STOP_RECALL_SERVICE"],
        "RECALL_SERVICE_WAIT_SECONDS": resource_env["RECALL_SERVICE_WAIT_SECONDS"],
        "AUTO_START_LLM_JUDGE": resource_env["AUTO_START_LLM_JUDGE"],
        "AUTO_STOP_LLM_JUDGE": resource_env["AUTO_STOP_LLM_JUDGE"],
        "LLM_JUDGE_PREFLIGHT": resource_env["LLM_JUDGE_PREFLIGHT"],
        "LLM_JUDGE_WAIT_SECONDS": resource_env["LLM_JUDGE_WAIT_SECONDS"],
        "AGENT_TP_SIZE": str(len([part for part in resource_env["AGENT_GPU_IDS"].split(",") if part])),
        "AGENT_PORT": str(deep_get(config, "vllm.agent_port")),
        "AGENT_HOST": str(deep_get(config, "vllm.agent_host")),
        "AGENT_SERVED_MODEL": str(deep_get(config, "vllm.agent_served_model")),
        "VLLM_STARTUP_TIMEOUT": str(deep_get(config, "vllm.startup_timeout")),
        "GPU_MEMORY_UTILIZATION": str(deep_get(config, "vllm.gpu_memory_utilization")),
        "MAX_NUM_SEQS": str(max_num_seqs),
        "VLLM_DTYPE": str(deep_get(config, "vllm.dtype")),
        "VLLM_ENFORCE_EAGER": str(deep_get(config, "vllm.enforce_eager")).lower(),
        "VLLM_TRUST_REMOTE_CODE_FOR_SERVER": str(deep_get(config, "vllm.trust_remote_code_for_server")).lower(),
        "VLLM_ATTENTION_BACKEND": str(deep_get(config, "vllm.attention_backend")),
        "VLLM_DISABLE_FLASHINFER": "1" if deep_get(config, "vllm.disable_flashinfer") else "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "1" if deep_get(config, "vllm.use_flashinfer_sampler") else "0",
        "TOKENIZERS_PARALLELISM": str(deep_get(config, "vllm.tokenizers_parallelism")).lower(),
        "VLLM_REUSE_EXISTING_SERVER": str(deep_get(config, "vllm.reuse_existing_server")).lower(),
        "MAX_MODEL_LEN": str(budget_get(budget, "max_model_len", "actor_rollout_ref.rollout.max_model_len")),
        "MAX_ASSISTANT_TURNS": str(budget_get(budget, "max_assistant_turns", "actor_rollout_ref.rollout.multi_turn.max_assistant_turns")),
        "MAX_USER_TURNS": str(budget_get(budget, "max_user_turns", "actor_rollout_ref.rollout.multi_turn.max_user_turns")),
        "MAX_PROMPT_LENGTH": str(budget_get(budget, "max_prompt_length", "data.max_prompt_length")),
        "MAX_RESPONSE_LENGTH": str(budget_get(budget, "max_response_length", "data.max_response_length")),
        "MAX_TOOL_RESPONSE_LENGTH": str(budget_get(budget, "max_tool_response_length", "actor_rollout_ref.rollout.multi_turn.max_tool_response_length")),
        "ENABLE_THINKING": str(budget_get(budget, "enable_thinking", "data.apply_chat_template_kwargs.enable_thinking")).lower(),
        "TEMPERATURE": str(deep_get(config, "generation.temperature")),
        "TOP_P": str(deep_get(config, "generation.top_p")),
        "REQUEST_TIMEOUT": str(deep_get(config, "generation.request_timeout")),
        "STOP_SEQUENCES": stop_sequences_text,
        "RANKER_DEVICE": ranker_device,
        "RANKER_CUDA_VISIBLE_DEVICES": str(deep_get(config, "ranker.cuda_visible_devices") or resource_env["RANK_GPU_ID"]),
        "RANKER_CONFIG_DEVICE": normalize_device(str(deep_get(config, "ranker.config_device") or ranker_device), ctx=ctx),
        "RANKER_MAX_QUERY_LENGTH": ranker_max_query,
        "RANKER_MAX_DOC_LENGTH": ranker_max_doc,
        "LLM_JUDGE_ENDPOINT": str(deep_get(config, "llm_judge.endpoint")),
        "LLM_JUDGE_MODEL": str(deep_get(config, "llm_judge.model")),
        "LLM_JUDGE_PROMPT_PATH": llm_judge_prompt,
        "LLM_JUDGE_MAX_CHUNK_CHARS": str(deep_get(config, "llm_judge.max_chunk_chars")),
        "LLM_JUDGE_MAX_TOKENS": str(deep_get(config, "llm_judge.max_tokens")),
        "LLM_JUDGE_TEMPERATURE": str(deep_get(config, "llm_judge.temperature")),
        "LLM_JUDGE_REQUEST_TIMEOUT": str(deep_get(config, "llm_judge.request_timeout")),
        "LLM_JUDGE_MAX_RETRIES": str(deep_get(config, "llm_judge.max_retries")),
        "LLM_JUDGE_RETRY_DELAY": str(deep_get(config, "llm_judge.retry_delay")),
        "LLM_JUDGE_RETRY_BACKOFF": str(deep_get(config, "llm_judge.retry_backoff")),
        "TOOL_CONFIG": str(runtime_tool_config),
        "SOURCE_TOOL_CONFIG": source_tool_config,
        "INJECT_TOOL_SCHEMA": str(deep_get(config, "tool.inject_tool_schema")).lower(),
        "TRUST_REMOTE_CODE": str(deep_get(config, "tool.trust_remote_code")).lower(),
        "EVAL_LOG_ROOT": str(eval_log_root),
        "EVAL_REPORT_ROOT": str(eval_report_root),
        "TRACE_DIR": str(trace_dir),
        "RUNTIME_LOG_DIR": str(runtime_log_dir),
        "OUT_DIR": str(out_dir),
        "LOG_DIR": str(log_dir),
        "REPORT_PATH": str(report_path),
        "METRICS_JSONL": str(deep_get(config, "artifacts.metrics_jsonl") or log_dir / f"{run_name}.metrics.jsonl"),
        "SEARCH_TIMING_JSONL": str(deep_get(config, "artifacts.search_timing_jsonl") or log_dir / f"{run_name}.search_timing.jsonl"),
        "LLM_IO_JSONL": str(deep_get(config, "artifacts.llm_io_jsonl") or log_dir / f"{run_name}.llm_io.jsonl"),
        "INFER_LOG": str(deep_get(config, "artifacts.infer_log") or log_dir / f"{run_name}.infer.log"),
        "RECALL_SERVICE_LOG": str(deep_get(config, "artifacts.recall_service_log") or log_dir / f"{run_name}.recall_retriever_server.log"),
        "RANKER_OUTPUT_JSONL": str(deep_get(config, "artifacts.ranker_output_jsonl") or out_dir / "ranker_infer_smoke.jsonl"),
        "ROLLOUT_DATA_DIR": str(deep_get(config, "artifacts.rollout_data_dir") or out_dir / "rollout_data"),
        "VALIDATION_DATA_DIR": str(deep_get(config, "artifacts.validation_data_dir") or out_dir / "validation_data"),
        "ENV_PATH": str(deep_get(config, "artifacts.env_path") or log_dir / f"{run_name}.env"),
        "LLM_IO_MAX_RECORDS": str(deep_get(config, "logging.llm_io_max_records") or "20"),
    }

    for path_key in ["ROLLOUT_DATA_DIR", "VALIDATION_DATA_DIR"]:
        Path(env[path_key]).mkdir(parents=True, exist_ok=True)

    if env["RUN_MODE"] != "ranker-only" and not env["AGENT_MODEL"]:
        raise ValueError("models.agent_model is required for eval unless run_mode=ranker-only")
    if env["RUN_MODE"] != "no-ranker" and env["RERANKER"] == "dense_e5" and not env["RANKER_MODEL"]:
        raise ValueError("models.ranker_model is required for full/ranker-only dense_e5 eval")

    final_config = {
        "main_run": {
            "name": selection.main_run_config,
            "file": str(main_run.get("_main_run_file", "")),
        },
        "groups": {
            "eval_runtime": eval_runtime_name,
            "eval_budget": eval_budget_name,
            "resource": resource_config,
            "data": data_name,
            "model": model_name,
            "rollout": rollout_name,
            "ranker_base": ranker_base_name,
        },
        "eval_runtime": config,
        "eval_budget": budget,
        "resource_env": resource_env,
        "hydra_facts": {
            "data": {
                "val_files": data_cfg.get("val_files"),
                "max_prompt_length": data_cfg.get("max_prompt_length"),
                "max_response_length": data_cfg.get("max_response_length"),
            },
            "model": {
                "path": model_cfg.get("path"),
            },
            "rollout": {
                "max_model_len": deep_get(rollout_cfg, "max_model_len"),
            },
            "recall_retriever": recall_cfg,
            "ranker": ranker_cfg,
        },
        "env": env,
    }

    final_yaml = log_dir / f"{run_name}.final_eval_config.yaml"
    final_json = log_dir / f"{run_name}.final_eval_config.json"
    runtime_env_sh = log_dir / f"{run_name}.eval_runtime_env.sh"
    eval_args_file = log_dir / f"{run_name}.eval_args.txt"
    overlay_file = log_dir / f"{run_name}.eval_overlay_yamls.txt"
    passthrough_file = log_dir / f"{run_name}.eval_passthrough_args.txt"
    env["FINAL_EVAL_CONFIG_YAML"] = str(final_yaml)
    env["FINAL_EVAL_CONFIG_JSON"] = str(final_json)
    env["EVAL_RUNTIME_ENV_SH"] = str(runtime_env_sh)
    env["EVAL_ARGS_FILE"] = str(eval_args_file)
    env["EVAL_OVERLAY_YAMLS_FILE"] = str(overlay_file)
    env["EVAL_PASSTHROUGH_ARGS_FILE"] = str(passthrough_file)

    eval_args = build_eval_args(env)
    dump_mapping(final_yaml, final_config)
    final_json.write_text(json.dumps(final_config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    write_lines(eval_args_file, eval_args)
    write_lines(overlay_file, [str(path) for path in selection.overlay_yamls])
    write_lines(passthrough_file, selection.passthrough_args)
    Path(env["ENV_PATH"]).write_text("\n".join(f"{key}={value}" for key, value in env.items()) + "\n", encoding="utf-8")
    write_export_file(runtime_env_sh, env)
    return runtime_env_sh


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = CompilerContext(
        repo_root=args.repo_root.resolve(),
        script_dir=args.script_dir.resolve(),
        assets_dir=args.assets_dir.resolve(),
        project_root=args.project_root.resolve(),
        external_model_root=args.external_model_root.resolve(),
        external_retrieval_root=args.external_retrieval_root.resolve(),
        device_prefix=args.device_prefix,
        visible_devices_var=args.visible_devices_var,
        accelerator=args.accelerator,
    )
    runtime_env = compile_config(ctx, args.launcher_args, os.environ)
    print(runtime_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
