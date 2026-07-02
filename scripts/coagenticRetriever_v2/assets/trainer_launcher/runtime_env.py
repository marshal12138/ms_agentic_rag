"""Runtime environment materialization for the launcher config compiler.

这个模块只负责把“已经选择好的配置”转成 Bash 后续需要的运行态环境变量：

- run identity：`GROUP_NAME`、`RUN_NAME`、`LOG_DIR`。
- 日志/report/checkpoint 路径默认值。
- GPU/service/report/checkpoint 等 launcher 运行态默认值。
- 从静态 tool config 同步 retrieval URL、agent 可见 top-M 等工具侧参数。

它不写 Hydra 参数文件，也不启动任何服务。这样 `compile_config.py` 可以只描述
高层编译流程，而不用承载一长串 shell env 默认值。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping

from .context import CompilerContext
from .paths import slugify_name
from .tool_config import read_static_tool_config


def truthy(value: str | None) -> bool:
    """解析 Bash 常见 truthy 字符串，用于保留旧 launcher 行为。"""

    return (value or "") in {"1", "true", "TRUE", "yes", "YES", "on", "ON"}


def set_default(env: dict[str, str], name: str, value: object | None) -> None:
    """只在变量为空时写默认值。

    这个函数体现 launcher 的一个基本约定：显式外部环境变量优先级最高，Python
    compiler 只能填补缺省值，不能覆盖用户已经 export 的值。
    """

    if not env.get(name):
        env[name] = "" if value is None else str(value)


def count_csv(csv: str) -> str:
    """统计逗号分隔 GPU 列表中的非空项数量。"""

    parts = [part for part in csv.split(",") if part != ""]
    return str(len(parts))


def safe_mkdir(path: Path) -> None:
    """创建目录；调用点负责决定这个目录是否允许存在。"""

    path.mkdir(parents=True, exist_ok=True)


def path_has_payload(path: Path) -> bool:
    """Return True when a file/dir already exists and is non-empty."""

    if not path.exists():
        return False
    if path.is_dir():
        return any(path.iterdir())
    return True


def assert_safe_run_target(path: Path, label: str, env: Mapping[str, str]) -> None:
    """Match old Bash safety behavior: do not reuse non-empty run dirs by default."""

    if truthy(env.get("ALLOW_RUN_REUSE")) or truthy(env.get("ALLOW_DIR_REUSE")):
        return
    if env.get("RESUME_MODE") and env.get("RESUME_MODE") != "disable":
        return
    if path_has_payload(path):
        raise RuntimeError(
            f"{label} already exists and is non-empty: {path}\n"
            "Refusing to reuse it by default because this may overwrite checkpoints or logs.\n"
            "Set ALLOW_RUN_REUSE=1 (or ALLOW_DIR_REUSE=1), or use a new EXP_NAME/RUN_NAME."
        )


def resolve_run_identity(ctx: CompilerContext, env: dict[str, str], *, require_exp_name: bool = True) -> None:
    """Compute GROUP_SLUG/RUN_NAME/LOG_DIR using the same shape as the Bash helper."""

    group_name = env.get("GROUP_NAME") or "coAgenticRetriever"
    group_slug = env.get("GROUP_SLUG") or slugify_name(group_name)
    env["GROUP_NAME"] = group_name
    env["GROUP_SLUG"] = group_slug
    env.setdefault("TRAIN_LOG_ROOT", str(ctx.repo_root / "log" / "train_logs" / group_slug))

    if env.get("RUN_NAME"):
        env["RUN_NAME"] = slugify_name(env["RUN_NAME"])
        env.setdefault("LOG_DIR", str(Path(env["TRAIN_LOG_ROOT"]) / env["RUN_NAME"]))
        env.setdefault("CONFIG_NAME", env["RUN_NAME"])
        return

    exp_name = env.get("EXP_NAME", "")
    if require_exp_name and not exp_name:
        raise ValueError(
            "EXP_NAME is required to build a unique RUN_NAME. "
            "Example: EXP_NAME=my_rule_v1 bash <script>"
        )
    exp_name = exp_name or "default"
    env["EXP_NAME"] = exp_name
    env.setdefault("RUN_STAMP", datetime.now().strftime("%y%m%d-%H%M%S"))
    run_name = f"{env['RUN_STAMP']}-{slugify_name(exp_name)}"
    env["RUN_NAME"] = run_name
    env.setdefault("LOG_DIR", str(Path(env["TRAIN_LOG_ROOT"]) / run_name))
    env.setdefault("CONFIG_NAME", run_name)


def setup_log_defaults(ctx: CompilerContext, env: dict[str, str]) -> None:
    """Materialize the same log/report paths old Bash launcher wrote."""

    log_dir = Path(env["LOG_DIR"])
    run_name = env["RUN_NAME"]
    env.setdefault("TRAIN_LOG", str(log_dir / f"{run_name}.train.log"))
    env.setdefault("METRICS_JSONL", str(log_dir / f"{run_name}.metrics.jsonl"))
    env.setdefault("SEARCH_TIMING_JSONL", str(log_dir / f"{run_name}.search_timing.jsonl"))
    env.setdefault("NVIDIA_SMI_CSV", str(log_dir / f"{run_name}.nvidia_smi.csv"))
    env.setdefault("CHECKPOINT_CONVERSION_LOG", str(log_dir / f"{run_name}.checkpoint_conversion.log"))
    env.setdefault("REPORT_PREFIX", str(log_dir / f"{run_name}.timing_report"))
    env.setdefault("REPORT_SCHEMA_PATH", str(ctx.assets_dir / "report_schema.py"))
    env.setdefault("VERL_FILE_LOGGER_PATH", env["METRICS_JSONL"])


def normalize_run_mode(env: dict[str, str]) -> None:
    """Normalize canonical launcher run mode spelling."""

    run_mode = env.get("RUN_MODE") or "full"
    if run_mode in {"full", "co-training", "co_training"}:
        env["RUN_MODE"] = "full"
        env["EFFECTIVE_RUN_MODE"] = "full"
        return
    if run_mode in {"no-ranker", "no_ranker"}:
        env["RUN_MODE"] = "no-ranker"
        env["EFFECTIVE_RUN_MODE"] = "no-ranker"
        return
    raise ValueError(f"unsupported RUN_MODE={run_mode}; use full or no-ranker")


def apply_common_defaults(ctx: CompilerContext, env: dict[str, str], *, canonical: bool) -> None:
    """填充 launcher 运行态默认值。

    这里分两类处理：

    - canonical 模式：训练语义参数必须来自 YAML/overlay/Hydra CLI，所以这里只填充
      shell runtime 必需的派生变量和非训练语义默认值。
    - legacy 模式：为了保留旧调用可用性，仍然补齐旧 Bash launcher 中的训练参数默认值。

    函数不会覆盖已经存在的外部 env。
    """

    set_default(env, "ALLOW_RUN_REUSE", "0")
    set_default(env, "ALLOW_DIR_REUSE", "0")

    if not canonical:
        # legacy 模式保留旧脚本顶部的资源默认值；canonical 模式改由 resource YAML 管理。
        set_default(env, "GROUP_NAME", "coAgenticRetriever")
        set_default(env, "AGENT_GPU_IDS", "0,1,2,3")
        set_default(env, "RECALL_GPU_ID", "5")
        set_default(env, "RANK_GPU_ID", "4")
        set_default(env, "LLM_JUDGE_GPU_IDS", "6,7")
        set_default(env, "AUTO_START_RECALL_SERVICE", "1")
        set_default(env, "AUTO_STOP_RECALL_SERVICE", "1")
        set_default(env, "RECALL_SERVICE_WAIT_SECONDS", "240")
        set_default(env, "AUTO_START_LLM_JUDGE", "0")
        set_default(env, "AUTO_STOP_LLM_JUDGE", "0")
        set_default(env, "LLM_JUDGE_PREFLIGHT", "1")
        set_default(env, "LLM_JUDGE_WAIT_SECONDS", "600")

    set_default(env, "AGENT_N_GPUS_PER_NODE", count_csv(env["AGENT_GPU_IDS"]))
    # 训练主进程可见 GPU。
    #
    # full 模式需要 actor GPU + ranker GPU；recall/judge 服务单独设置设备。
    # no-ranker 模式不会创建 ranker worker，也不会启动共享 ranker 推理 actor，因此默认
    # 只暴露 agent GPU。显式外部 GPU_IDS 仍然最高优先级，会被 set_default 保留下来。
    if env.get("EFFECTIVE_RUN_MODE") == "no-ranker":
        set_default(env, "GPU_IDS", env["AGENT_GPU_IDS"])
    else:
        set_default(env, "GPU_IDS", f"{env['AGENT_GPU_IDS']},{env['RANK_GPU_ID']}")
    set_default(env, "RANKER_VISIBLE_DEVICE_INDEX", env["AGENT_N_GPUS_PER_NODE"])
    set_default(env, "MAIN_GPU_IDS", env["AGENT_GPU_IDS"])
    set_default(env, "RANKER_GPU_IDS", env["RANK_GPU_ID"])
    set_default(env, "RERANKER_GPU_IDS", env["RANKER_GPU_IDS"])
    set_default(env, "REPORT_STEPS", "10")
    set_default(env, "NVIDIA_SMI_INTERVAL", "10")
    set_default(env, "REPORT_INTERVAL_SECONDS", "60")
    set_default(env, "COAGENTIC_ROLLOUT_PROGRESS_INTERVAL", "60")
    set_default(env, "COAGENTIC_ROLLOUT_ITEM_PROGRESS_INTERVAL", "32")
    set_default(env, "CHECKPOINT_KEEP_LATEST_GLOBAL_STEPS", "1")
    set_default(env, "CHECKPOINT_DELETE_OLD_GLOBAL_STEPS", "1")
    set_default(env, "CHECKPOINT_DELETE_EMPTY_GLOBAL_STEPS", "1")
    set_default(env, "CHECKPOINT_TRAINABLE_ROLES", "actor ranker")
    set_default(env, "CHECKPOINT_REMOVE_ROOT_DIRS", "ranker retriever rollout_data validation_data")
    set_default(env, "CHECKPOINT_REMOVE_ROOT_GLOBS", "ranker_contrastive_smoke_metrics.jsonl")

    if not canonical:
        # 这些训练超参在 canonical 模式已经迁入 Hydra YAML 链路。这里只保留 legacy 兼容。
        set_default(env, "TRAIN_BATCH_SIZE", "64")
        set_default(env, "ACTOR_BATCH_SIZE", "64")
        set_default(env, "TOTAL_STEPS", "auto")
        set_default(env, "N_ROLLOUTS", "8")
        set_default(env, "VAL_BATCH_SIZE", "8")
        set_default(env, "TRAIN_MAX_SAMPLES", "5100")
        set_default(env, "VAL_MAX_SAMPLES", "8")
        set_default(env, "LORA_RANK", "0")
        set_default(env, "LORA_ALPHA", "16")

    set_default(env, "ACTOR_MICRO_BATCH_SIZE_PER_GPU", "4" if canonical else "2")
    set_default(env, "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU", "8" if canonical else "4")
    set_default(env, "NNODES", "1")
    if not env.get("NCCL_TIMEOUT"):
        env["NCCL_TIMEOUT"] = env.get("HCCL_TIMEOUT", "")
    if ctx.accelerator in {"npu", "ascend"}:
        set_default(env, "NCCL_TIMEOUT", "1800")
        set_default(env, "ACTOR_USE_TORCH_COMPILE", "False")
    set_default(env, "NCCL_TIMEOUT", "600")
    set_default(env, "ACTOR_USE_TORCH_COMPILE", "true")
    set_default(env, "ACTOR_LR", "1e-6")
    set_default(env, "KL_LOSS_COEF", "0.001")
    set_default(env, "SAVE_FREQ", "10")
    set_default(env, "TEST_FREQ", "-1")
    set_default(env, "RESUME_MODE", "disable")
    set_default(env, "MAX_ACTOR_CKPT_TO_KEEP", "1")

    if not canonical:
        set_default(env, "MODEL_PATH", str(ctx.external_model_root / "llm" / "Qwen3-4B"))
        set_default(
            env,
            "TRAIN_DATA",
            str(ctx.repo_root / "data" / "coAgenticRetriever" / "albation_1" / "co_search_ablation.train.parquet"),
        )
        set_default(
            env,
            "VAL_DATA",
            str(ctx.repo_root / "data" / "coAgenticRetriever" / "albation_1" / "co_search_ablation.eval.parquet"),
        )
    set_default(
        env,
        "RECALL_MODEL_PATH",
        env.get("RETRIEVER_MODEL_PATH") or ctx.external_model_root / "retriever" / "e5-base-v2",
    )
    set_default(env, "RANKER_BASE_MODEL_PATH", "")
    set_default(env, "RANKER_ENCODER_PATH", "")
    set_default(env, "CORPUS_JSONL", ctx.external_retrieval_root / "wiki-18" / "wiki-18.jsonl")
    set_default(env, "CHECKPOINT_ROOT", ctx.repo_root / "checkpoints" / "qwen3_4b_probe")

    checkpoint_group_root = Path(env["CHECKPOINT_ROOT"]) / env["GROUP_SLUG"]
    safe_mkdir(checkpoint_group_root)
    set_default(env, "OUT_DIR", checkpoint_group_root / env["RUN_NAME"])
    # 防止误复用已有 run 目录。显式 resume 或 ALLOW_RUN_REUSE 才允许继续写入。
    assert_safe_run_target(Path(env["LOG_DIR"]), "log dir", env)
    assert_safe_run_target(Path(env["OUT_DIR"]), "checkpoint dir", env)
    set_default(env, "ROLLOUT_DATA_DIR", Path(env["LOG_DIR"]) / "rollout_data")
    set_default(env, "VALIDATION_DATA_DIR", Path(env["LOG_DIR"]) / "validation_data")
    set_default(env, "DUMP_ROLLOUT_EVERY_STEP_NUM", "10")
    set_default(env, "DUMP_ROLLOUT_NUM_EVERYTIME", "1")
    set_default(env, "MAX_ROLLOUT_DUMP_NUM", "-1")
    set_default(env, "ROLLOUT_TRACE_MODE", "full")
    if env["ROLLOUT_TRACE_MODE"] not in {"full", "partial"}:
        raise ValueError(f"unsupported ROLLOUT_TRACE_MODE={env['ROLLOUT_TRACE_MODE']}; use full or partial")

    set_default(env, "PROXY_PORT", "8030")
    set_default(env, "RETRIEVAL_SERVICE_URL", f"http://127.0.0.1:{env['PROXY_PORT']}/retrieve")
    set_default(env, "RETRIEVER_DEVICE", ctx.device_prefix)
    set_default(env, "RETRIEVAL_PREFLIGHT_QUERY", "who got the first nobel prize in physics?")
    set_default(env, "RETRIEVAL_PREFLIGHT_EXPECT", "")
    set_default(env, "ASYNC_RANKER_TRAINING_YAML", "")
    set_default(
        env,
        "LLM_JUDGE_SERVICE_CONFIG",
        ctx.project_root / "async_ranker_training" / "configs" / "llm_judge_vllm_deepseek_flash_gpu06_07.yaml",
    )
    set_default(env, "LLM_JUDGE_ENDPOINT", "http://127.0.0.1:8067/v1/chat/completions")
    set_default(env, "ASYNC_RANKER_TRAINING_LOG_DIR", Path(env["LOG_DIR"]) / "async_ranker_training")

    for name in [
        "RANKER_CONTRASTIVE_BATCH_SIZE",
        "RANKER_GRADIENT_ACCUMULATION_STEPS",
        "RANKER_NUM_GROUPS_PER_STEP",
        "RANKER_STEPS_PER_GLOBAL_STEP",
        "RANKER_INFERENCE_SYNC_INTERVAL",
        "RANKER_INFERENCE_ACTOR_NAME",
        "RANKER_NEG_PER_POS",
        "RANKER_POSITIVE_TOP_K",
        "RANKER_TEMPERATURE",
        "RANKER_CONFIG_DEVICE",
    ]:
        set_default(env, name, "")
    set_default(env, "RECALL_TOP_K", "50")
    set_default(env, "TOP_N", env["RECALL_TOP_K"])
    set_default(env, "TOP_M", "5")
    set_default(env, "RECALL_RETRIEVER_CONFIG_DEVICE", ctx.device_spec(env["RECALL_GPU_ID"]))
    set_default(env, "TOOL_CONFIG", ctx.project_root / "config" / "coagentic_retriever_tool_config.yaml")


def apply_tool_config(ctx: CompilerContext, env: dict[str, str]) -> None:
    """用静态 tool YAML 同步检索服务相关参数。

    retrieval URL、agent 可见 top-M 等信息存在于 tool config 中。
    launcher 读取它们后用于三件事：服务预检、`.env` 审计、Hydra runtime override。
    """

    tool = read_static_tool_config(Path(env["TOOL_CONFIG"]))
    if tool.retrieval_service_url:
        env["RETRIEVAL_SERVICE_URL"] = tool.retrieval_service_url
    if tool.retrieval_port:
        env["PROXY_PORT"] = tool.retrieval_port
    if not tool.search_tool_final_top_m:
        raise ValueError(f"tool config must explicitly set config.searchTool_final_top_m: {env['TOOL_CONFIG']}")
    env["TOP_M"] = tool.search_tool_final_top_m
    env["SEARCH_TOOL_FINAL_TOP_M"] = tool.search_tool_final_top_m
    if tool.class_name:
        env["COAGENTIC_TOOL_CLASS_NAME"] = tool.class_name
    if tool.max_concurrent_per_worker:
        env["TOOL_MAX_CONCURRENT_PER_WORKER"] = tool.max_concurrent_per_worker
    if tool.ranker_enabled:
        env["COAGENTIC_RANKER_ENABLED"] = tool.ranker_enabled

    env["RECALL_RETRIEVER_CONFIG_DEVICE"] = ctx.device_spec(env["RECALL_GPU_ID"])
    if not env.get("GPU_IDS"):
        if env.get("EFFECTIVE_RUN_MODE") == "no-ranker":
            env["GPU_IDS"] = env["AGENT_GPU_IDS"]
        else:
            env["GPU_IDS"] = f"{env['AGENT_GPU_IDS']},{env['RANK_GPU_ID']}"
    # 如果 resource 没显式给 WAIT_FOR_GPUS，就用最终 GPU 分配拼出等待列表。这样 GPU
    # wait、服务启动和 Hydra resources.* 看到的是同一套资源值。
    if env.get("EFFECTIVE_RUN_MODE") == "no-ranker":
        wait_for_gpus = f"{env['AGENT_GPU_IDS']},{env['RECALL_GPU_ID']}"
    else:
        wait_for_gpus = f"{env['AGENT_GPU_IDS']},{env['RANK_GPU_ID']},{env['RECALL_GPU_ID']},{env['LLM_JUDGE_GPU_IDS']}"
    set_default(env, "WAIT_FOR_GPUS", wait_for_gpus)
    set_default(env, "RECALL_SERVICE_LOG", Path(env["LOG_DIR"]) / f"{env['RUN_NAME']}.recall_retriever_server.log")
