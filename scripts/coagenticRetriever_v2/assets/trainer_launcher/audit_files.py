"""Generated audit/runtime file writers for the launcher compiler.

这个模块负责把内存中的编译结果写到磁盘：

- `.env` 人类审计文件。
- `hydra_args.txt` 和相关 canonical 审计文件。
- legacy CLI passthrough 文件。
- Bash launcher source 的 `launcher_runtime_env.sh` 内容。

它不做配置合并，不做校验，也不启动任何服务。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .context import CompiledConfig, CompilerContext, RunFiles


def _select_config_value(config_data: Mapping[str, Any], dotted_key: str) -> Any:
    """从 resolved final config 中读取简单 dot path。"""

    value: Any = config_data
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _audit_bool(value: Any) -> str:
    """把审计布尔值写成稳定的小写字符串。"""

    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def populate_reward_audit(env: dict[str, str], config_data: Mapping[str, Any]) -> None:
    """Record reward-only audit fields derived from final Hydra config."""

    format_penalty = _select_config_value(config_data, "custom_reward_function.reward_kwargs.format_penalty")
    env["REWARD_FORMAT_PENALTY"] = "" if format_penalty is None else str(format_penalty)
    env["REWARD_FORMAT_PENALTY_SOURCE"] = (
        "custom_reward_function.reward_kwargs.format_penalty" if format_penalty is not None else ""
    )


def populate_ranker_training_sample_builder_audit(
    env: dict[str, str],
    config_data: Mapping[str, Any],
) -> None:
    """记录最终启用的 ranker sample builder 路径。

    ranker_training 下同时存在 pseudo-rank sample_builder 和 async sample_builder。
    审计文件只摘出最终实际启用的那条路径，避免读 `.env` 时把两套配置混在一起。
    """

    signal_source = str(_select_config_value(config_data, "ranker_training.signal_source") or "")
    ranker_trainable = _select_config_value(config_data, "trainer.ranker_trainable")
    ranker_update_mode = str(_select_config_value(config_data, "trainer.ranker_update_mode") or "")
    async_enabled = _select_config_value(config_data, "ranker_training.async_ranker_training.enable")

    env["RANKER_TRAINING_SIGNAL_SOURCE"] = signal_source
    env["RANKER_TRAINING_RANKER_TRAINABLE"] = _audit_bool(ranker_trainable)
    env["RANKER_TRAINING_UPDATE_MODE"] = ranker_update_mode
    env["RANKER_TRAINING_ASYNC_ENABLE"] = _audit_bool(async_enabled)
    env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH"] = ""
    env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_TYPE"] = ""
    env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_JSON"] = ""
    env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_DISABLED_REASON"] = ""

    if ranker_trainable is False or ranker_update_mode == "disabled":
        env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH"] = "disabled"
        env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_DISABLED_REASON"] = "ranker_training_disabled"
        return

    if signal_source == "async_ranker_training":
        if async_enabled is not True:
            env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH"] = "disabled"
            env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_DISABLED_REASON"] = "async_ranker_training_disabled"
            return
        path = "ranker_training.async_ranker_training.sample_builder"
    elif signal_source == "pseudo_rank":
        path = "ranker_training.sample_builder"
    else:
        env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH"] = "unknown"
        env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_DISABLED_REASON"] = f"unknown_signal_source:{signal_source}"
        return

    builder = _select_config_value(config_data, path)
    if not isinstance(builder, Mapping):
        env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH"] = path
        env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_DISABLED_REASON"] = "sample_builder_missing"
        return
    builder_dict = dict(builder)
    env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH"] = path
    env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_TYPE"] = str(builder_dict.get("type") or "")
    env["RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_JSON"] = json.dumps(
        builder_dict,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def build_run_files(env: Mapping[str, str]) -> RunFiles:
    """根据最终 run identity 生成本次运行的全部输出文件路径。

    只有在 `RUN_NAME` 和 `LOG_DIR` 已经确定后才能调用。这里不创建目录，也不写文件，
    只返回路径对象。
    """

    log_dir = Path(env["LOG_DIR"])
    run_name = env["RUN_NAME"]
    return RunFiles(
        runtime_env_sh=log_dir / f"{run_name}.launcher_runtime_env.sh",
        env_file=log_dir / f"{run_name}.env",
        runtime_override_yaml=log_dir / f"{run_name}.runtime_env_overrides.yaml",
        run_mode_override_yaml=log_dir / f"{run_name}.run_mode_overrides.yaml",
        runtime_tool_config_yaml=log_dir / f"{run_name}.tool_config.yaml",
        hydra_args_file=log_dir / f"{run_name}.hydra_args.txt",
        final_config_yaml=log_dir / f"{run_name}.final_config.yaml",
        final_config_json=log_dir / f"{run_name}.final_config.json",
        trainer_main_hydra_config_file=log_dir / f"{run_name}.trainer_main_hydra_config.txt",
        hydra_groups_file=log_dir / f"{run_name}.hydra_groups.txt",
        hydra_cli_overrides_file=log_dir / f"{run_name}.hydra_cli_overrides.txt",
        overlay_yamls_file=log_dir / f"{run_name}.overlay_yamls.txt",
        legacy_cli_args_file=log_dir / f"{run_name}.legacy_cli_args.txt",
    )


def write_audit_env(compiled: CompiledConfig, ctx: CompilerContext) -> None:
    """写入给人阅读的 `.env` 审计文件。

    `.env` 不是 Bash source 文件，主要用途是复盘一次运行到底采用了哪些配置：

    - main_run/resource/Hydra group 选择。
    - 最终 GPU/service/log/checkpoint 环境变量。
    - canonical 模式生成的 Hydra 参数文件路径。
    - legacy 模式仍然使用的旧训练参数。
    """

    assert compiled.files is not None
    env = compiled.env
    lines = [
        f"RUN_NAME={env.get('RUN_NAME', '')}",
        f"EXP_NAME={env.get('EXP_NAME', '')}",
        f"GROUP_NAME={env.get('GROUP_NAME', '')}",
        f"GROUP_SLUG={env.get('GROUP_SLUG', '')}",
        f"RUN_STAMP={env.get('RUN_STAMP', '')}",
        f"RUN_MODE={env.get('RUN_MODE', '')}",
        f"EFFECTIVE_RUN_MODE={env.get('EFFECTIVE_RUN_MODE', '')}",
        f"RUN_MODE_SOURCE={env.get('RUN_MODE_SOURCE', '')}",
        f"PROJECT_ROOT={ctx.project_root}",
        f"CONFIG_NAME={env.get('CONFIG_NAME', '')}",
        f"COSEARCH_ACCELERATOR={ctx.accelerator}",
        f"VISIBLE_DEVICES_VAR={ctx.visible_devices_var}",
        f"DEVICE_PREFIX={ctx.device_prefix}",
        f"CANONICAL_CONFIG_MODE={'1' if compiled.canonical else '0'}",
        f"MAIN_RUN_CONFIG={compiled.selection.main_run_config}",
        f"MAIN_RUN_CONFIG_FILE={compiled.manifest.ref.path if compiled.manifest.ref else ''}",
        f"TRAINER_MAIN_HYDRA_CONFIG={env.get('TRAINER_MAIN_HYDRA_CONFIG', '')}",
        f"DATA_CONFIG={env.get('DATA_CONFIG', '')}",
        f"MODEL_CONFIG={env.get('MODEL_CONFIG', '')}",
        f"ROLLOUT_CONFIG={env.get('ROLLOUT_CONFIG', '')}",
        f"RANKER_BASE_CONFIG={env.get('RANKER_BASE_CONFIG', '')}",
        f"ASYNC_RANKER_TRAINING_BASE_CONFIG={env.get('ASYNC_RANKER_TRAINING_BASE_CONFIG', '')}",
        f"RESOURCE_CONFIG={env.get('RESOURCE_CONFIG', '')}",
        f"RESOURCE_BASE_CONFIG_FILE={env.get('RESOURCE_BASE_CONFIG_FILE', '')}",
        f"RESOURCE_CONFIG_FILE={env.get('RESOURCE_CONFIG_FILE', '')}",
        f"CANONICAL_HYDRA_ARGS_FILE={env.get('CANONICAL_HYDRA_ARGS_FILE', '')}",
        f"CANONICAL_TRAINER_MAIN_HYDRA_CONFIG_FILE={env.get('CANONICAL_TRAINER_MAIN_HYDRA_CONFIG_FILE', '')}",
        f"CANONICAL_HYDRA_GROUPS_FILE={env.get('CANONICAL_HYDRA_GROUPS_FILE', '')}",
        f"CANONICAL_CLI_OVERRIDES_FILE={env.get('CANONICAL_CLI_OVERRIDES_FILE', '')}",
        f"CANONICAL_OVERLAY_YAMLS_FILE={env.get('CANONICAL_OVERLAY_YAMLS_FILE', '')}",
        f"CANONICAL_RUN_MODE_OVERRIDE_YAML={env.get('CANONICAL_RUN_MODE_OVERRIDE_YAML', '')}",
        f"CANONICAL_RUNTIME_OVERRIDE_YAML={env.get('CANONICAL_RUNTIME_OVERRIDE_YAML', '')}",
        f"CANONICAL_RUNTIME_TOOL_CONFIG_YAML={env.get('CANONICAL_RUNTIME_TOOL_CONFIG_YAML', '')}",
        f"CANONICAL_FINAL_CONFIG_YAML={env.get('CANONICAL_FINAL_CONFIG_YAML', '')}",
        f"CANONICAL_FINAL_CONFIG_JSON={env.get('CANONICAL_FINAL_CONFIG_JSON', '')}",
        f"LEGACY_CLI_ARGS_FILE={env.get('LEGACY_CLI_ARGS_FILE', '')}",
    ]
    # 这些字段是 canonical 和 legacy 都需要审计的 launcher runtime 变量。
    audit_keys = [
        "GPU_IDS",
        "AGENT_GPU_IDS",
        "AGENT_N_GPUS_PER_NODE",
        "WAIT_FOR_GPUS",
        "WAIT_FOR_GPU_RELEASE",
        "WAIT_FOR_GPU_INTERVAL_SECONDS",
        "WAIT_FOR_GPU_LABEL",
        "NCCL_TIMEOUT",
        "ACTOR_USE_TORCH_COMPILE",
        "MAIN_GPU_IDS",
        "RANKER_GPU_IDS",
        "REPORT_STEPS",
        "NVIDIA_SMI_INTERVAL",
        "REPORT_INTERVAL_SECONDS",
        "COAGENTIC_ROLLOUT_PROGRESS_INTERVAL",
        "COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL",
        "CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS",
        "CHECKPOINT_DELETE_OLD_GLOBAL_STEPS",
        "CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS",
        "CHECKPOINT_TRAINABLE_ROLES",
        "CHECKPOINT_REMOVE_ROOT_DIRS",
        "CHECKPOINT_REMOVE_ROOT_GLOBS",
        "ALLOW_RUN_REUSE",
        "ALLOW_DIR_REUSE",
        "RECALL_MODEL_PATH",
        "RANKER_BASE_MODEL_PATH",
        "CORPUS_JSONL",
        "OUT_DIR",
        "CHECKPOINT_ROOT",
        "ROLLOUT_DATA_DIR",
        "VALIDATION_DATA_DIR",
        "LOG_DIR",
        "TRAIN_LOG",
        "METRICS_JSONL",
        "SEARCH_TIMING_JSONL",
        "NVIDIA_SMI_CSV",
        "CHECKPOINT_CONVERSION_LOG",
        "REPORT_PREFIX",
        "REPORT_SCHEMA_PATH",
        "RETRIEVAL_SERVICE_URL",
        "AUTO_START_RECALL_SERVICE",
        "AUTO_STOP_RECALL_SERVICE",
        "NEEDS_LLM_JUDGE_SERVICE",
        "ASYNC_RANKER_TRAINING_YAML",
        "AUTO_START_LLM_JUDGE",
        "AUTO_STOP_LLM_JUDGE",
        "LLM_JUDGE_SERVICE_CONFIG",
        "LLM_JUDGE_GPU_IDS",
        "LLM_JUDGE_ENDPOINT",
        "LLM_JUDGE_HYDRA_STAGE_COUNT",
        "LLM_JUDGE_HYDRA_STAGE_ENDPOINTS",
        "LLM_JUDGE_HYDRA_STAGE_MODELS",
        "LLM_JUDGE_SERVICE_ENDPOINT",
        "LLM_JUDGE_SERVICE_MODEL",
        "LLM_JUDGE_PREFLIGHT",
        "LLM_JUDGE_WAIT_SECONDS",
        "ASYNC_RANKER_TRAINING_LOG_DIR",
        "RECALL_GPU_ID",
        "RANK_GPU_ID",
        "RANKER_VISIBLE_DEVICE_INDEX",
        "TOOL_CONFIG",
        "RECALL_TOP_K",
        "TOP_N",
        "TOP_M",
        "SEARCH_TOOL_FINAL_TOP_M",
        "RECALL_RETRIEVER_CONFIG_DEVICE",
        "REWARD_FORMAT_PENALTY",
        "REWARD_FORMAT_PENALTY_SOURCE",
        "SOURCE_TOOL_CONFIG",
        "HYDRA_RECALL_FINAL_TOP_N",
        "HYDRA_RANKER_FINAL_TOP_K",
        "RUNTIME_TOOL_RECALL_FINAL_TOP_N",
        "RUNTIME_TOOL_RANKER_FINAL_TOP_K",
        "RUNTIME_TOOL_SEARCH_TOOL_FINAL_TOP_M",
        "HYDRA_RANKER_CONFIG_SOURCE",
        "HYDRA_RANKER_MODEL_PATH",
        "HYDRA_RANKER_ENCODER_PATH",
        "HYDRA_RANKER_DEVICE",
        "HYDRA_RANKER_MAX_QUERY_LENGTH",
        "HYDRA_RANKER_MAX_DOC_LENGTH",
        "RUNTIME_TOOL_RANKER_MAX_QUERY_LENGTH",
        "RUNTIME_TOOL_RANKER_MAX_DOC_LENGTH",
        "RANKER_TRAINING_SIGNAL_SOURCE",
        "RANKER_TRAINING_RANKER_TRAINABLE",
        "RANKER_TRAINING_UPDATE_MODE",
        "RANKER_TRAINING_ASYNC_ENABLE",
        "RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_PATH",
        "RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_TYPE",
        "RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_JSON",
        "RANKER_TRAINING_ACTIVE_SAMPLE_BUILDER_DISABLED_REASON",
    ]
    lines.extend(f"{key}={env.get(key, '')}" for key in audit_keys)
    if not compiled.canonical:
        # legacy 模式仍然允许旧 Bash 风格训练参数，因此额外记录这些字段。canonical 模式
        # 下这些训练语义参数应由 Hydra YAML/overlay 管理，不再写入这里。
        legacy_keys = [
            "TRAIN_BATCH_SIZE",
            "VAL_BATCH_SIZE",
            "TRAIN_MAX_SAMPLES",
            "VAL_MAX_SAMPLES",
            "ACTOR_BATCH_SIZE",
            "TOTAL_STEPS",
            "N_ROLLOUTS",
            "LORA_RANK",
            "LORA_ALPHA",
            "MODEL_PATH",
            "TRAIN_DATA",
            "VAL_DATA",
            "DUMP_ROLLOUT_EVERY_STEP_NUM",
            "DUMP_ROLLOUT_NUM_EVERYTIME",
            "MAX_ROLLOUT_DUMP_NUM",
            "ROLLOUT_TRACE_MODE",
            "RANKER_CONTRASTIVE_BATCH_SIZE",
            "RANKER_GRADIENT_ACCUMULATION_STEPS",
            "RANKER_NUM_GROUPS_PER_STEP",
            "RANKER_STEPS_PER_GLOBAL_STEP",
            "RANKER_INFERENCE_SYNC_INTERVAL",
            "RANKER_INFERENCE_ACTOR_NAME",
            "RANKER_NEG_PER_POS",
            "RANKER_POSITIVE_TOP_K",
            "RANKER_TEMPERATURE",
            "RANK_TOP_K",
            "RANKER_CONFIG_DEVICE",
            "HYDRA_OVERRIDE_YAMLS",
            "RANKER_STRATEGY_YAML",
            "TRAIN_BUDGET_YAML",
            "INJECT_TOOL_SCHEMA",
            "COAGENTIC_EXTRA_ARGS",
        ]
        lines.extend(f"{key}={env.get(key, '')}" for key in legacy_keys)
    compiled.files.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_canonical_files(compiled: CompiledConfig) -> None:
    """写 canonical 模式的 Hydra 审计文件。

    这些文件都服务于可审计性：

    - trainer main config 单独一行。
    - config group 选择单独一份。
    - task 尾部 CLI override 单独一份。
    - overlay YAML 列表单独一份。
    - `hydra_args.txt` 是最终训练入口真正读取的完整参数序列。
    """

    assert compiled.files is not None
    assert compiled.hydra_selection is not None
    files = compiled.files
    files.trainer_main_hydra_config_file.write_text(
        compiled.hydra_selection.trainer_main_hydra_config + "\n", encoding="utf-8"
    )
    files.hydra_groups_file.write_text("\n".join(compiled.hydra_selection.group_args) + "\n", encoding="utf-8")
    files.hydra_cli_overrides_file.write_text(
        "\n".join(compiled.selection.trainer_cli_overrides) + ("\n" if compiled.selection.trainer_cli_overrides else ""),
        encoding="utf-8",
    )
    files.overlay_yamls_file.write_text(
        "\n".join(str(path) for path in compiled.selection.overlay_yamls)
        + ("\n" if compiled.selection.overlay_yamls else ""),
        encoding="utf-8",
    )
    files.hydra_args_file.write_text("\n".join(compiled.hydra_args) + "\n", encoding="utf-8")
    files.legacy_cli_args_file.write_text("", encoding="utf-8")


def write_final_config_files(compiled: CompiledConfig, config_data: Mapping[str, object]) -> None:
    """写入最终完整 Hydra 配置的 YAML 和 JSON 快照。

    这里的 `config_data` 必须已经是同一个 resolved Python container。YAML 和 JSON 都从
    这份对象写出，确保两个文件内容一一对应；区别只是序列化格式不同。
    """

    assert compiled.files is not None
    from .yaml_utils import dump_mapping

    data = dict(config_data)
    dump_mapping(compiled.files.final_config_yaml, data)
    compiled.files.final_config_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def write_legacy_files(compiled: CompiledConfig) -> None:
    """写 legacy 模式的 CLI passthrough 文件。

    legacy 模式不生成 canonical Hydra args，只把未识别参数交回旧 asset runner。
    """

    assert compiled.files is not None
    compiled.files.legacy_cli_args_file.write_text(
        "\n".join(compiled.selection.trainer_cli_overrides)
        + ("\n" if compiled.selection.trainer_cli_overrides else ""),
        encoding="utf-8",
    )


def build_runtime_env_for_bash(compiled: CompiledConfig, ctx: CompilerContext) -> dict[str, str]:
    """挑选需要写入 `launcher_runtime_env.sh` 并由 Bash source 的变量。

    和 `.env` 不同，这里生成的是机器要执行的环境变量文件。原则是：

    - Bash 后续启动 GPU wait、recall service、judge service、训练入口所需的变量必须写入。
    - 纯审计字段不一定写入。
    - 所有值都已经在 Python 中物化，Bash 只消费，不再重新推导配置优先级。
    """

    source_env = compiled.env
    # 这个白名单是 Bash launcher 和下游 shell helper 的运行契约。新增变量时应先确认它
    # 真的是 Bash 后续步骤需要 source 的运行态变量，而不是仅供审计的信息。
    export_names = [
        "RUN_NAME",
        "EXP_NAME",
        "GROUP_NAME",
        "GROUP_SLUG",
        "RUN_STAMP",
        "TRAIN_LOG_ROOT",
        "LOG_DIR",
        "CONFIG_NAME",
        "RUN_MODE",
        "EFFECTIVE_RUN_MODE",
        "RUN_MODE_SOURCE",
        "MAIN_RUN_CONFIG",
        "MAIN_RUN_CONFIG_FILE",
        "TRAINER_MAIN_HYDRA_CONFIG",
        "DATA_CONFIG",
        "MODEL_CONFIG",
        "ROLLOUT_CONFIG",
        "RANKER_BASE_CONFIG",
        "ASYNC_RANKER_TRAINING_BASE_CONFIG",
        "RESOURCE_CONFIG",
        "RESOURCE_BASE_CONFIG_FILE",
        "RESOURCE_CONFIG_FILE",
        "CANONICAL_HYDRA_ARGS_FILE",
        "CANONICAL_TRAINER_MAIN_HYDRA_CONFIG_FILE",
        "CANONICAL_HYDRA_GROUPS_FILE",
        "CANONICAL_CLI_OVERRIDES_FILE",
        "CANONICAL_OVERLAY_YAMLS_FILE",
        "CANONICAL_RUN_MODE_OVERRIDE_YAML",
        "CANONICAL_RUNTIME_OVERRIDE_YAML",
        "CANONICAL_FINAL_CONFIG_YAML",
        "CANONICAL_FINAL_CONFIG_JSON",
        "LEGACY_CLI_ARGS_FILE",
        "GPU_IDS",
        "AGENT_GPU_IDS",
        "AGENT_N_GPUS_PER_NODE",
        "RANK_GPU_ID",
        "RECALL_GPU_ID",
        "LLM_JUDGE_GPU_IDS",
        "WAIT_FOR_GPUS",
        "WAIT_FOR_GPU_RELEASE",
        "WAIT_FOR_GPU_INTERVAL_SECONDS",
        "WAIT_FOR_GPU_LABEL",
        "NCCL_TIMEOUT",
        "ACTOR_USE_TORCH_COMPILE",
        "MAIN_GPU_IDS",
        "RANKER_GPU_IDS",
        "RERANKER_GPU_IDS",
        "REPORT_STEPS",
        "NVIDIA_SMI_INTERVAL",
        "REPORT_INTERVAL_SECONDS",
        "COAGENTIC_ROLLOUT_PROGRESS_INTERVAL",
        "COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL",
        "CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS",
        "CHECKPOINT_DELETE_OLD_GLOBAL_STEPS",
        "CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS",
        "CHECKPOINT_TRAINABLE_ROLES",
        "CHECKPOINT_REMOVE_ROOT_DIRS",
        "CHECKPOINT_REMOVE_ROOT_GLOBS",
        "TRAIN_BATCH_SIZE",
        "ACTOR_BATCH_SIZE",
        "TOTAL_STEPS",
        "N_ROLLOUTS",
        "VAL_BATCH_SIZE",
        "TRAIN_MAX_SAMPLES",
        "VAL_MAX_SAMPLES",
        "LORA_RANK",
        "LORA_ALPHA",
        "MODEL_PATH",
        "TRAIN_DATA",
        "VAL_DATA",
        "RECALL_MODEL_PATH",
        "RANKER_BASE_MODEL_PATH",
        "RANKER_ENCODER_PATH",
        "CORPUS_JSONL",
        "CHECKPOINT_ROOT",
        "OUT_DIR",
        "ROLLOUT_DATA_DIR",
        "VALIDATION_DATA_DIR",
        "DUMP_ROLLOUT_EVERY_STEP_NUM",
        "DUMP_ROLLOUT_NUM_EVERYTIME",
        "MAX_ROLLOUT_DUMP_NUM",
        "ROLLOUT_TRACE_MODE",
        "PROXY_PORT",
        "RETRIEVAL_SERVICE_URL",
        "RETRIEVER_DEVICE",
        "AUTO_START_RECALL_SERVICE",
        "AUTO_STOP_RECALL_SERVICE",
        "RECALL_SERVICE_WAIT_SECONDS",
        "NEEDS_LLM_JUDGE_SERVICE",
        "ASYNC_RANKER_TRAINING_YAML",
        "AUTO_START_LLM_JUDGE",
        "AUTO_STOP_LLM_JUDGE",
        "LLM_JUDGE_SERVICE_CONFIG",
        "LLM_JUDGE_ENDPOINT",
        "LLM_JUDGE_PREFLIGHT",
        "LLM_JUDGE_WAIT_SECONDS",
        "ASYNC_RANKER_TRAINING_LOG_DIR",
        "RANKER_CONTRASTIVE_BATCH_SIZE",
        "RANKER_GRADIENT_ACCUMULATION_STEPS",
        "RANKER_NUM_GROUPS_PER_STEP",
        "RANKER_STEPS_PER_GLOBAL_STEP",
        "RANKER_INFERENCE_SYNC_INTERVAL",
        "RANKER_INFERENCE_ACTOR_NAME",
        "RANKER_NEG_PER_POS",
        "RANKER_POSITIVE_TOP_K",
        "RANKER_TEMPERATURE",
        "RANK_TOP_K",
        "RANKER_CONFIG_DEVICE",
        "HYDRA_RECALL_FINAL_TOP_N",
        "HYDRA_RANKER_FINAL_TOP_K",
        "RUNTIME_TOOL_RECALL_FINAL_TOP_N",
        "RUNTIME_TOOL_RANKER_FINAL_TOP_K",
        "SEARCH_TOOL_FINAL_TOP_M",
        "RUNTIME_TOOL_SEARCH_TOOL_FINAL_TOP_M",
        "RECALL_TOP_K",
        "TOP_N",
        "TOP_M",
        "RECALL_RETRIEVER_CONFIG_DEVICE",
        "RANKER_VISIBLE_DEVICE_INDEX",
        "TOOL_CONFIG",
        "COAGENTIC_TOOL_CLASS_NAME",
        "TOOL_MAX_CONCURRENT_PER_WORKER",
        "COAGENTIC_RANKER_ENABLED",
        "RECALL_SERVICE_LOG",
        "TRAIN_LOG",
        "METRICS_JSONL",
        "SEARCH_TIMING_JSONL",
        "NVIDIA_SMI_CSV",
        "CHECKPOINT_CONVERSION_LOG",
        "REPORT_PREFIX",
        "REPORT_SCHEMA_PATH",
        "VERL_FILE_LOGGER_PATH",
        "RETRIEVAL_PREFLIGHT_QUERY",
        "RETRIEVAL_PREFLIGHT_EXPECT",
        "HYDRA_OVERRIDE_YAMLS",
        "RANKER_STRATEGY_YAML",
        "TRAIN_BUDGET_YAML",
        "INJECT_TOOL_SCHEMA",
        "COAGENTIC_EXTRA_ARGS",
    ]
    env = {name: source_env[name] for name in export_names if name in source_env}
    # 以下字段不是用户配置，而是 Bash 运行所需的派生上下文。集中追加在这里，避免主
    # 流程把“配置输入”和“执行派生值”混在一起。
    env["PYTHON_CONFIG_COMPILER_USED"] = "1"
    env["CANONICAL_CONFIG_MODE"] = "1" if compiled.canonical else "0"
    env["PROJECT_ROOT"] = str(ctx.project_root)
    env["COSEARCH_ACCELERATOR"] = ctx.accelerator
    env["VISIBLE_DEVICES_VAR"] = ctx.visible_devices_var
    env["DEVICE_PREFIX"] = ctx.device_prefix
    env["COAGENTIC_PROJECT_ROOT"] = str(ctx.project_root)
    env["CHECKPOINT_VERL_ROOT"] = env.get("CHECKPOINT_VERL_ROOT") or str(ctx.project_root / "verl")
    env["N_GPUS_PER_NODE"] = env.get("N_GPUS_PER_NODE") or env["AGENT_N_GPUS_PER_NODE"]
    env["COAGENTIC_MAIN"] = str(ctx.project_root / "main_coagentic_retriever.py")
    env["SAVE_TOP_N_DOCUMENTS"] = env.get("SAVE_TOP_N_DOCUMENTS") or "true"
    env["COAGENTIC_RETRIEVER_SEARCH_TIMING_JSONL"] = (
        env.get("COAGENTIC_RETRIEVER_SEARCH_TIMING_JSONL") or env["SEARCH_TIMING_JSONL"]
    )
    env["COAGENTIC_RETRIEVER_LLM_IO_JSONL"] = (
        env.get("COAGENTIC_RETRIEVER_LLM_IO_JSONL")
        or str(Path(env["LOG_DIR"]) / f"{env['RUN_NAME']}.llm_io.jsonl")
    )
    env["COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS"] = env.get("COAGENTIC_RETRIEVER_LLM_IO_MAX_RECORDS") or "20"

    if compiled.files is not None:
        env["LAUNCHER_RUNTIME_ENV_SH"] = str(compiled.files.runtime_env_sh)
        env["CANONICAL_HYDRA_ARGS_FILE"] = str(compiled.files.hydra_args_file) if compiled.canonical else ""
        env["CANONICAL_TRAINER_MAIN_HYDRA_CONFIG_FILE"] = (
            str(compiled.files.trainer_main_hydra_config_file) if compiled.canonical else ""
        )
        env["CANONICAL_HYDRA_GROUPS_FILE"] = str(compiled.files.hydra_groups_file) if compiled.canonical else ""
        env["CANONICAL_CLI_OVERRIDES_FILE"] = str(compiled.files.hydra_cli_overrides_file) if compiled.canonical else ""
        env["CANONICAL_OVERLAY_YAMLS_FILE"] = str(compiled.files.overlay_yamls_file) if compiled.canonical else ""
        env["CANONICAL_RUN_MODE_OVERRIDE_YAML"] = (
            str(compiled.files.run_mode_override_yaml) if compiled.canonical else ""
        )
        env["CANONICAL_RUNTIME_OVERRIDE_YAML"] = str(compiled.files.runtime_override_yaml) if compiled.canonical else ""
        env["CANONICAL_RUNTIME_TOOL_CONFIG_YAML"] = (
            str(compiled.files.runtime_tool_config_yaml) if compiled.canonical else ""
        )
        env["CANONICAL_FINAL_CONFIG_YAML"] = str(compiled.files.final_config_yaml) if compiled.canonical else ""
        env["CANONICAL_FINAL_CONFIG_JSON"] = str(compiled.files.final_config_json) if compiled.canonical else ""
        env["LEGACY_CLI_ARGS_FILE"] = str(compiled.files.legacy_cli_args_file)
    return {key: str(value) for key, value in env.items()}
