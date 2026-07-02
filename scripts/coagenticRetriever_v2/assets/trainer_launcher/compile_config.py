#!/usr/bin/env python3
"""Compile CoAgenticRetriever train-launcher configuration into runtime files.

这个脚本是 `scripts/coagenticRetriever_v2/01_train_launcher.sh` 调用的 Python
配置编译入口。它现在只保留“入口 + 主流程编排”，具体工作已经拆到同目录模块：

- `cli.py`：解析 launcher CLI。
- `main_run_config.py`：读取并合并 launcher 级 main-run config。
- `resource.py`：合并 resource/base、resource/<selected>、overlay 和外部 env。
- `runtime_env.py`：生成 run identity、日志路径、GPU/service 等运行态 env。
- `runtime_overrides.py`：生成 Hydra runtime override YAML。
- `final_config.py`：compose 并导出最终完整 Hydra 配置。
- `validators.py`：执行不启动服务的静态校验。
- `audit_files.py`：写 `.env`、`hydra_args.txt`、`launcher_runtime_env.sh` 等文件。

它的边界仍然很明确：

- 不启动 recall retriever。
- 不启动 LLM judge。
- 不等待 GPU。
- 不执行训练。
- 不清理后台进程。

这些有副作用的步骤继续留在 Bash launcher 中。Python 只负责把“配置应该是什么”
编译成确定文件，Bash 只负责“按照这些文件去做事”。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Mapping

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainer_launcher.audit_files import (
    build_run_files,
    build_runtime_env_for_bash,
    populate_reward_audit,
    populate_ranker_training_sample_builder_audit,
    write_audit_env,
    write_canonical_files,
    write_final_config_files,
    write_legacy_files,
)
from trainer_launcher.cli import append_unique_cli_override, parse_launcher_args
from trainer_launcher.context import CompiledConfig, CompilerContext
from trainer_launcher.final_config import compose_final_config
from trainer_launcher.hydra_args import build_hydra_args, normalize_canonical_selection
from trainer_launcher.main_run_config import merge_main_run_selection
from trainer_launcher.resource import load_canonical_resource_env
from trainer_launcher.run_mode import build_run_mode_hydra_overrides, resolve_run_mode
from trainer_launcher.runtime_env import (
    apply_common_defaults,
    apply_tool_config,
    normalize_run_mode,
    resolve_run_identity,
    safe_mkdir,
    setup_log_defaults,
)
from trainer_launcher.runtime_overrides import build_runtime_override_yaml
from trainer_launcher.shell_quote import write_export_file
from trainer_launcher.tool_config import write_runtime_tool_config_from_hydra_ranker
from trainer_launcher.yaml_utils import dump_mapping
from trainer_launcher.validators import (
    check_required_paths,
    infer_needs_llm_judge_service,
    reject_canonical_deprecated_env_overrides,
    validate_async_ranker_training_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 Python compiler 自己的参数。

    Bash launcher 会把两类参数传给本脚本：

    1. compiler 运行所需的路径和设备上下文，例如 repo root、project root、device prefix。
    2. `--` 之后的原始 launcher 参数，例如 `--main_run_config`、`--OVERLAY_YAML`。

    这里只消费第 1 类参数；第 2 类参数保持原样交给 `cli.py`，避免 argparse 提前
    误解 Hydra override。
    """

    parser = argparse.ArgumentParser(
        description="Compile CoAgenticRetriever launcher config into runtime env and Hydra arg files."
    )
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


def _normalize_canonical_cli_overrides(raw_overrides: list[str]) -> list[str]:
    """校验并去重 canonical 模式下 task 末尾透传的 Hydra override。

    这些 override 优先级最高，会出现在 `hydra_args.txt` 的最后。这里不合并 YAML，
    只保证语法是 Hydra dotlist 能接受的 `key=value` 或 `~key` 形式。
    """

    normalized: list[str] = []
    for raw_override in raw_overrides:
        append_unique_cli_override(normalized, raw_override)
    return normalized


def _apply_canonical_selection_to_env(
    *,
    ctx: CompilerContext,
    env: dict[str, str],
    selection,
    environ: Mapping[str, str],
):
    """把 canonical 模式的 Hydra/resource 选择规整后写入运行态 env。

    这一步会同时完成两件事：

    - 规范化 Hydra group 名称，确保写入 `hydra_args.txt` 的都是 group 内短名。
    - 按 resource 优先级计算最终环境变量，并写回 env，供后续 GPU wait、服务启动
      和 runtime override 复用。
    """

    hydra_selection = normalize_canonical_selection(
        repo_root=ctx.repo_root,
        project_root=ctx.project_root,
        selection=selection,
    )
    resource_config, resource_base, resource_file, resource_env = load_canonical_resource_env(
        repo_root=ctx.repo_root,
        project_root=ctx.project_root,
        resource_config=selection.resource_config,
        overlay_yamls=selection.overlay_yamls,
        environ=environ,
    )
    env.update(resource_env)
    env["RESOURCE_CONFIG"] = resource_config
    env["RESOURCE_BASE_CONFIG_FILE"] = str(resource_base)
    env["RESOURCE_CONFIG_FILE"] = str(resource_file)
    env["TRAINER_MAIN_HYDRA_CONFIG"] = hydra_selection.trainer_main_hydra_config
    env["DATA_CONFIG"] = hydra_selection.data_config
    env["MODEL_CONFIG"] = hydra_selection.model_config
    env["ROLLOUT_CONFIG"] = hydra_selection.rollout_config
    env["RANKER_BASE_CONFIG"] = hydra_selection.ranker_base_config
    env["ASYNC_RANKER_TRAINING_BASE_CONFIG"] = hydra_selection.async_ranker_training_base_config
    return hydra_selection


def _write_generated_file_env(compiled: CompiledConfig) -> None:
    """把 compiler 生成的文件路径注册到 env。

    `.env` 审计文件和 Bash source 文件都需要知道这些路径。集中在这里写入可以
    避免各个 writer 自己拼路径，保持“编译结果里只有一套文件名”。
    """

    assert compiled.files is not None
    env = compiled.env
    env["MAIN_RUN_CONFIG"] = compiled.selection.main_run_config
    env["MAIN_RUN_CONFIG_FILE"] = str(compiled.manifest.ref.path) if compiled.manifest.ref else ""
    env["CANONICAL_HYDRA_ARGS_FILE"] = str(compiled.files.hydra_args_file) if compiled.canonical else ""
    env["CANONICAL_TRAINER_MAIN_HYDRA_CONFIG_FILE"] = (
        str(compiled.files.trainer_main_hydra_config_file) if compiled.canonical else ""
    )
    env["CANONICAL_HYDRA_GROUPS_FILE"] = str(compiled.files.hydra_groups_file) if compiled.canonical else ""
    env["CANONICAL_CLI_OVERRIDES_FILE"] = str(compiled.files.hydra_cli_overrides_file) if compiled.canonical else ""
    env["CANONICAL_OVERLAY_YAMLS_FILE"] = str(compiled.files.overlay_yamls_file) if compiled.canonical else ""
    env["CANONICAL_RUN_MODE_OVERRIDE_YAML"] = str(compiled.files.run_mode_override_yaml) if compiled.canonical else ""
    env["CANONICAL_RUNTIME_OVERRIDE_YAML"] = str(compiled.files.runtime_override_yaml) if compiled.canonical else ""
    env["CANONICAL_RUNTIME_TOOL_CONFIG_YAML"] = (
        str(compiled.files.runtime_tool_config_yaml) if compiled.canonical else ""
    )
    env["CANONICAL_FINAL_CONFIG_YAML"] = str(compiled.files.final_config_yaml) if compiled.canonical else ""
    env["CANONICAL_FINAL_CONFIG_JSON"] = str(compiled.files.final_config_json) if compiled.canonical else ""
    env["LEGACY_CLI_ARGS_FILE"] = str(compiled.files.legacy_cli_args_file)


def compile_config(ctx: CompilerContext, launcher_args: list[str], environ: Mapping[str, str]) -> CompiledConfig:
    """把 launcher CLI 和外部环境变量编译成确定的运行文件。

    主流程按固定顺序执行，顺序本身就是配置优先级的一部分：

    1. 解析 task 传入的 launcher 参数。
    2. 合并 main_run_config 默认值。
    3. canonical 模式下合并 resource，并拒绝历史 shell env 覆盖训练超参。
    4. 物化运行态 env，例如 run 目录、GPU 列表、服务 URL、日志路径。
    5. 写 runtime override YAML，再生成最终 Hydra args。
    6. 做静态校验，最后写 `.env` 和 Bash runtime env 文件。

    函数只做配置编译，不启动服务、不等待 GPU、不执行训练。
    """

    original_selection = parse_launcher_args(launcher_args, repo_root=ctx.repo_root)
    selection, manifest = merge_main_run_selection(
        repo_root=ctx.repo_root,
        project_root=ctx.project_root,
        selection=original_selection,
    )
    canonical = selection.has_canonical_signal()
    if canonical:
        # canonical 模式要求训练语义来自 YAML/overlay/显式 Hydra CLI，不能再被历史
        # shell 默认值悄悄覆盖。
        reject_canonical_deprecated_env_overrides(environ)
        selection.trainer_cli_overrides = _normalize_canonical_cli_overrides(selection.trainer_cli_overrides)
        run_mode_resolution = resolve_run_mode(
            main_run_mode=selection.run_mode,
            overlay_yamls=selection.overlay_yamls,
            trainer_cli_overrides=selection.trainer_cli_overrides,
        )
        selection.run_mode = run_mode_resolution.run_mode
        selection.trainer_cli_overrides = run_mode_resolution.trainer_cli_overrides
    else:
        run_mode_resolution = None

    env = dict(environ)
    env["CANONICAL_CONFIG_MODE"] = "1" if canonical else "0"
    hydra_selection = None
    if canonical:
        # resource 也是 canonical 模式的一部分：它先于普通 overlay 生效，低于显式
        # 外部环境变量，最终结果同时用于 shell runtime 和 Hydra resources.*。
        hydra_selection = _apply_canonical_selection_to_env(
            ctx=ctx,
            env=env,
            selection=selection,
            environ=environ,
        )

    if selection.llm_judge_service_config:
        env["LLM_JUDGE_SERVICE_CONFIG"] = str(selection.llm_judge_service_config)
    if manifest.tool_config:
        env["TOOL_CONFIG"] = str(manifest.tool_config)

    resolve_run_identity(ctx, env, require_exp_name=True)
    setup_log_defaults(ctx, env)
    if run_mode_resolution is not None:
        env["RUN_MODE"] = run_mode_resolution.run_mode
        env["EFFECTIVE_RUN_MODE"] = run_mode_resolution.effective_run_mode
        env["RUN_MODE_SOURCE"] = run_mode_resolution.source
    normalize_run_mode(env)
    apply_common_defaults(ctx, env, canonical=canonical)
    if env["EFFECTIVE_RUN_MODE"] == "no-ranker":
        # no-ranker 是 recall-only 训练形态。除了关闭 trainer/ranker 侧 Hydra 配置，
        # rollout tool 也必须默认切到 no-ranker tool config，否则工具层仍可能要求
        # 共享 ranker actor。显式外部 TOOL_CONFIG 仍然拥有最高优先级。
        if "TOOL_CONFIG" not in environ:
            env["TOOL_CONFIG"] = str(ctx.project_root / "config" / "coagentic_retriever_tool_config_no_ranker.yaml")
        env["CHECKPOINT_TRAINABLE_ROLES"] = "actor"
    apply_tool_config(ctx, env)

    # 只有在 run identity 和日志目录确定后，才能安全确定所有生成文件的位置。
    files = build_run_files(env)
    safe_mkdir(Path(env["LOG_DIR"]))
    compiled = CompiledConfig(
        env=env,
        canonical=canonical,
        selection=selection,
        manifest=manifest,
        hydra_selection=hydra_selection,
        files=files,
    )
    _write_generated_file_env(compiled)

    if canonical:
        # 先用原始 tool config 生成一版 runtime override，compose 出最终 Hydra 配置。
        # 如果 full 模式启用了 shared inference ranker，随后会用 Hydra 中的 actor 标识
        # 生成本次 run 专用 tool config，再重写 runtime override 和 hydra args。
        dump_mapping(files.run_mode_override_yaml, build_run_mode_hydra_overrides(env["RUN_MODE"]))
        build_runtime_override_yaml(ctx, env, files.runtime_override_yaml)
        hydra_args, _, _ = build_hydra_args(
            hydra_selection=hydra_selection,  # type: ignore[arg-type]
            overlay_yamls=selection.overlay_yamls,
            run_mode_override_yaml=files.run_mode_override_yaml,
            runtime_override_yaml=files.runtime_override_yaml,
            trainer_cli_overrides=selection.trainer_cli_overrides,
        )
        compiled.hydra_args = hydra_args
        write_canonical_files(compiled)
        final_config = compose_final_config(ctx.project_root, files.hydra_args_file)

        forbidden_tool_overlay_keys = ["default_top_n", "default_top_m", "searchTool_final_top_m"]
        present_tool_overlay_keys = [key for key in forbidden_tool_overlay_keys if key in final_config]
        if present_tool_overlay_keys:
            raise ValueError(
                "Tool top fields are not Hydra training overrides: "
                + ", ".join(present_tool_overlay_keys)
                + ". Set searchTool_final_top_m in the static tool config, or migrate top-M into Hydra before using overlays."
            )

        shared_ranker = final_config.get("ranker_training", {}).get("shared_inference_ranker", {})
        if not isinstance(shared_ranker, dict):
            raise TypeError("final Hydra ranker_training.shared_inference_ranker must be a mapping")
        hydra_recall_retriever = final_config.get("recall_retriever", {})
        if not isinstance(hydra_recall_retriever, dict):
            raise TypeError("final Hydra recall_retriever must be a mapping")
        hydra_ranker = final_config.get("ranker", {})
        if not isinstance(hydra_ranker, dict):
            raise TypeError("final Hydra ranker must be a mapping")
        if "top_k" in hydra_recall_retriever:
            raise ValueError(
                "Deprecated training override recall_retriever.top_k is present in final Hydra config. "
                "Use recall_retriever.recall_final_top_n for the recall candidate pool size."
            )
        if "final_top_k" in hydra_ranker:
            raise ValueError(
                "Deprecated training override ranker.final_top_k is present in final Hydra config. "
                "Use ranker.top_k in the Hydra ranker base; runtime tool config will receive ranker.final_top_k."
            )
        recall_final_top_n = hydra_recall_retriever.get("recall_final_top_n")
        if recall_final_top_n is None or recall_final_top_n == "":
            raise ValueError("final Hydra recall_retriever.recall_final_top_n must be configured")
        ranker_final_top_k = hydra_ranker.get("top_k")
        if ranker_final_top_k is None or ranker_final_top_k == "":
            raise ValueError("final Hydra ranker.top_k must be configured")
        env["RECALL_TOP_K"] = str(recall_final_top_n)
        env["TOP_N"] = env["RECALL_TOP_K"]
        env["HYDRA_RECALL_FINAL_TOP_N"] = env["RECALL_TOP_K"]
        env["HYDRA_RANKER_FINAL_TOP_K"] = str(ranker_final_top_k)
        env["RUNTIME_TOOL_RECALL_FINAL_TOP_N"] = env["HYDRA_RECALL_FINAL_TOP_N"]
        env["RUNTIME_TOOL_RANKER_FINAL_TOP_K"] = env["HYDRA_RANKER_FINAL_TOP_K"]
        env["RUNTIME_TOOL_SEARCH_TOOL_FINAL_TOP_M"] = env["SEARCH_TOOL_FINAL_TOP_M"]
        env["SOURCE_TOOL_CONFIG"] = env["TOOL_CONFIG"]
        env["TOOL_CONFIG"] = str(files.runtime_tool_config_yaml)
        env["HYDRA_RANKER_CONFIG_SOURCE"] = env.get("RANKER_BASE_CONFIG", "")
        env["HYDRA_RANKER_MODEL_PATH"] = str(hydra_ranker.get("model_path") or "")
        env["HYDRA_RANKER_ENCODER_PATH"] = str(hydra_ranker.get("encoder_path") or "")
        env["HYDRA_RANKER_DEVICE"] = str(hydra_ranker.get("device") or "")
        env["HYDRA_RANKER_MAX_QUERY_LENGTH"] = str(hydra_ranker.get("max_query_length") or "")
        env["HYDRA_RANKER_MAX_DOC_LENGTH"] = str(hydra_ranker.get("max_doc_length") or "")
        env["RUNTIME_TOOL_RANKER_MAX_QUERY_LENGTH"] = env["HYDRA_RANKER_MAX_QUERY_LENGTH"]
        env["RUNTIME_TOOL_RANKER_MAX_DOC_LENGTH"] = env["HYDRA_RANKER_MAX_DOC_LENGTH"]
        write_runtime_tool_config_from_hydra_ranker(
            source_path=Path(env["SOURCE_TOOL_CONFIG"]),
            output_path=files.runtime_tool_config_yaml,
            actor_name=shared_ranker.get("actor_name"),
            actor_namespace=shared_ranker.get("actor_namespace"),
            hydra_recall_retriever_config=hydra_recall_retriever,
            hydra_ranker_config=hydra_ranker,
        )
        build_runtime_override_yaml(ctx, env, files.runtime_override_yaml)
        hydra_args, _, _ = build_hydra_args(
            hydra_selection=hydra_selection,  # type: ignore[arg-type]
            overlay_yamls=selection.overlay_yamls,
            run_mode_override_yaml=files.run_mode_override_yaml,
            runtime_override_yaml=files.runtime_override_yaml,
            trainer_cli_overrides=selection.trainer_cli_overrides,
        )
        compiled.hydra_args = hydra_args
        write_canonical_files(compiled)
        final_config = compose_final_config(ctx.project_root, files.hydra_args_file)

        populate_reward_audit(env, final_config)
        populate_ranker_training_sample_builder_audit(env, final_config)
        write_final_config_files(compiled, final_config)
        env["NEEDS_LLM_JUDGE_SERVICE"] = infer_needs_llm_judge_service(final_config)
        validate_async_ranker_training_config(ctx, env, selection.overlay_yamls, final_config)
        check_required_paths(ctx, compiled, final_config=final_config)
    else:
        # legacy 模式只保留旧透传行为，方便未迁移的调用继续走老 asset runner。
        write_legacy_files(compiled)
        check_required_paths(ctx, compiled)

    # 审计文件最后写入，确保它记录的是已经通过校验的最终配置。
    write_audit_env(compiled, ctx)
    runtime_env = build_runtime_env_for_bash(compiled, ctx)
    write_export_file(files.runtime_env_sh, runtime_env)
    return compiled


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    成功时只向 stdout 打印 `launcher_runtime_env.sh` 路径，Bash launcher 会 source
    该文件继续执行；失败时返回 2，让 Bash 立即停止。
    """

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
    try:
        compiled = compile_config(ctx, args.launcher_args, os.environ)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    assert compiled.files is not None
    print(compiled.files.runtime_env_sh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
