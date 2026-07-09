"""AIR LLM reranker branch GRPO training entry.

这个模块负责把 pipeline final config 翻译成训练 stage 行为。当前实现包含：

1. smoke 后端：不启动大模型训练，只消费 branch dataset、解析合法 reranker 输出、计算格式 reward、
   产出可审计 checkpoint manifest 和 service bundle 所需的 reranker_model 路径。
2. verl 后端：启动 frozen agent + retriever-only recall 服务，然后调用 VERL 执行真实 1-step 或完整 GRPO。
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import shlex
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_iter_rag.pipeline.manifest import write_stage_manifest
from agentic_iter_rag.reranker_training.reward import compute_format_only_reward
from agentic_iter_rag.reranker_training.parser import render_identity_rerank_response
from agentic_iter_rag.reranker_training.service_manager import TrainingServiceManager, as_int_list, csv_ids, tail_text
from agentic_iter_rag.reranker_training.training_report import build_report_paths, generate_once, resolve_compatible_python
from agentic_iter_rag.utils.io import (
    copy_file,
    deep_merge,
    iter_jsonl,
    read_json,
    read_yaml,
    stable_config_hash,
    write_json,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def project_root() -> Path:
    return repo_root() / "AgenticIterRag"


def default_verl_root() -> Path:
    """选择真实训练使用的 VERL 副本。

    当前 AIR 自带的 AgenticIterRag/verl 缺少 NPU 训练必需的 verl.models 目录；
    同仓库 CoSearch/verl 同时具备 verl.models 和 config/data/legacy_data.yaml，所以默认优先使用它。
    """

    return repo_root() / "CoSearch" / "verl"


def shell_quote(value: Any) -> str:
    return shlex.quote(str(value))


def as_list(value: Any) -> list[Any]:
    """把 YAML 中的单值或列表统一成列表，方便透传 stop sequences 等配置。"""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def resolve_branch_manifest(config: dict[str, Any], *, must_exist: bool = True) -> Path:
    """解析 branch dataset manifest。

    训练入口只接受 branch dataset，不直接消费旧的静态 reranker train set；这样可以保证训练样本里
    带有 continuation 必需的 messages_before_tool_response、baseline reward 和完整 top50 doc。
    """

    stage_cfg = config["pipeline"]["stage_configs"]["train_llm_reranker"]
    rt_cfg = config["reranker_training"]
    value = (
        stage_cfg.get("inputs", {}).get("branch_dataset_manifest")
        or rt_cfg.get("input", {}).get("branch_dataset_manifest")
    )
    if not value:
        raise ValueError("train_llm_reranker requires branch_dataset_manifest")
    path = Path(str(value))
    if must_exist and not path.exists():
        raise FileNotFoundError(f"branch_dataset_manifest does not exist: {path}")
    return path


def load_branch_dataset(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "air_reranker_branch_dataset_v1":
        raise ValueError("branch dataset schema_version must be air_reranker_branch_dataset_v1")
    dataset_jsonl = manifest.get("dataset_jsonl")
    if not dataset_jsonl:
        raise ValueError("branch dataset manifest is missing dataset_jsonl")
    data_path = Path(str(dataset_jsonl))
    if not data_path.exists():
        raise FileNotFoundError(f"branch dataset jsonl does not exist: {data_path}")
    rows = list(iter_jsonl(data_path))
    if not rows:
        raise ValueError("branch dataset is empty")
    return manifest, rows


def resolve_training_schedule(trainer_cfg: dict[str, Any], branch_manifest: dict[str, Any]) -> dict[str, Any]:
    """Resolve AIR epoch settings into VERL-compatible integer epoch/step values.

    VERL uses ``range(total_epochs)``, so fractional epochs must be represented as
    an integer ``total_training_steps`` cap before the command reaches VERL.
    """

    requested_value = trainer_cfg.get("total_epochs")
    if isinstance(requested_value, bool):
        raise TypeError("reranker_training.trainer.total_epochs must be numeric, not bool")
    try:
        requested_epochs = float(requested_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"reranker_training.trainer.total_epochs must be numeric: {requested_value!r}") from exc
    if not math.isfinite(requested_epochs):
        raise ValueError("reranker_training.trainer.total_epochs must be finite")

    explicit_steps = trainer_cfg.get("total_training_steps")
    train_batch_size = int(trainer_cfg["train_batch_size"])
    if train_batch_size <= 0:
        raise ValueError("reranker_training.trainer.train_batch_size must be > 0")

    if explicit_steps is not None:
        resolved_epochs = max(1, math.ceil(requested_epochs)) if requested_epochs > 0 else int(requested_epochs)
        return {
            "requested_total_epochs": requested_value,
            "resolved_total_epochs": resolved_epochs,
            "resolved_total_training_steps": int(explicit_steps),
            "fractional_epoch_resolved": False,
            "explicit_total_training_steps": True,
            "effective_sample_count": branch_manifest.get("sample_count"),
            "train_batch_size": train_batch_size,
            "steps_per_epoch": None,
            "drop_last": True,
        }

    if requested_epochs <= 0:
        raise ValueError("reranker_training.trainer.total_epochs must be > 0 when total_training_steps is null")

    resolved_integer_epochs = int(requested_epochs)
    if requested_epochs.is_integer():
        return {
            "requested_total_epochs": requested_value,
            "resolved_total_epochs": resolved_integer_epochs,
            "resolved_total_training_steps": None,
            "fractional_epoch_resolved": False,
            "explicit_total_training_steps": False,
            "effective_sample_count": branch_manifest.get("sample_count"),
            "train_batch_size": train_batch_size,
            "steps_per_epoch": None,
            "drop_last": True,
        }

    sample_count_value = branch_manifest.get("sample_count")
    if sample_count_value is None:
        raise ValueError("fractional total_epochs requires branch dataset manifest sample_count")
    sample_count = int(sample_count_value)
    train_max_samples = int(trainer_cfg.get("train_max_samples", -1))
    if train_max_samples >= 0:
        sample_count = min(sample_count, train_max_samples)
    steps_per_epoch = sample_count // train_batch_size
    if steps_per_epoch <= 0:
        raise ValueError(
            "fractional total_epochs requires at least one full training batch; "
            f"effective_sample_count={sample_count}, train_batch_size={train_batch_size}"
        )

    resolved_steps = max(1, math.ceil(steps_per_epoch * requested_epochs))
    return {
        "requested_total_epochs": requested_value,
        "resolved_total_epochs": max(1, math.ceil(requested_epochs)),
        "resolved_total_training_steps": resolved_steps,
        "fractional_epoch_resolved": True,
        "explicit_total_training_steps": False,
        "effective_sample_count": sample_count,
        "train_batch_size": train_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "drop_last": True,
    }


def planned_branch_manifest_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Build enough branch manifest metadata for dry-run planning.

    During a full pipeline dry-run, build_reranker_branch_dataset only plans the
    output manifest path; the file may not exist yet. Fractional epochs still
    need a sample count to resolve planned step limits.
    """

    stage_inputs = config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"].get("inputs", {})
    rt_input = config["reranker_training"].get("input", {})
    manifest_value = stage_inputs.get("enhanced_trajectory_manifest") or rt_input.get("enhanced_trajectory_manifest")
    if not manifest_value:
        return {"sample_count": None, "dataset_jsonl": None, "dataset_parquet": None}

    source_manifest_path = Path(str(manifest_value))
    if not source_manifest_path.exists():
        return {"sample_count": None, "dataset_jsonl": None, "dataset_parquet": None}

    source_manifest = read_json(source_manifest_path)
    sample_count = (
        source_manifest.get("enhanced_record_count_actual")
        or source_manifest.get("enhanced_record_count")
        or source_manifest.get("raw_trace_count")
    )
    return {
        "sample_count": sample_count,
        "dataset_jsonl": None,
        "dataset_parquet": None,
        "source_enhanced_trajectory_manifest": str(source_manifest_path),
        "sample_count_source": "enhanced_trajectory_manifest",
    }


def trainer_output_dir(config: dict[str, Any], backend: str) -> Path:
    trainer_cfg = config["reranker_training"]["trainer"]
    explicit = trainer_cfg.get("output_dir")
    if explicit:
        return Path(str(explicit))
    artifact_root = Path(str(config["runtime_compiled"]["ARTIFACT_ROOT"]))
    return artifact_root / "stages" / "train_llm_reranker" / f"reranker_model_{backend}"


def phase_output_dir(config: dict[str, Any], backend: str, phase_name: str) -> Path:
    """解析单个 phase 的输出目录。

    训练内部现在分 stage1/stage2 两个 phase。即使只有 stage1 默认启用，也必须把 checkpoint
    和日志按 phase 隔离，避免以后开启 stage2 后覆盖 stage1 产物。
    """

    base = trainer_output_dir(config, backend)
    return base / phase_name


def runtime_service_dir(config: dict[str, Any], phase_name: str | None = None) -> Path:
    artifact_root = Path(str(config["runtime_compiled"]["ARTIFACT_ROOT"]))
    base = artifact_root / "stages" / "train_llm_reranker" / "runtime_services"
    return base / phase_name if phase_name else base


def training_report_dir(config: dict[str, Any], phase_name: str | None = None) -> Path:
    artifact_root = Path(str(config["runtime_compiled"]["ARTIFACT_ROOT"]))
    base = artifact_root / "stages" / "train_llm_reranker" / "training_reports"
    return base / phase_name if phase_name else base


def training_log_root(config: dict[str, Any]) -> Path:
    """返回本次 run 的动态训练日志根目录。

    compiler 每次运行都会生成新的 runtime_compiled.LOG_DIR；LLM reranker 的 rollout dump
    必须挂在这里，而不是写死到配置文件或 checkpoint 目录，否则多次运行会互相覆盖，排查样本输出也很困难。
    """

    runtime = config.get("runtime_compiled") or {}
    log_dir = runtime.get("LOG_DIR")
    if log_dir:
        return Path(str(log_dir)) / "train_llm_reranker"
    artifact_root = Path(str(runtime["ARTIFACT_ROOT"]))
    return artifact_root / "stages" / "train_llm_reranker" / "runtime_logs"


def resolve_phase_dump_dir(config: dict[str, Any], phase_name: str, key: str, default_leaf: str) -> Path:
    """解析 stage1/stage2 共用的 VERL 样本 dump 目录。

    配置项表示 dump 根目录，而不是 VERL 最终写入目录。无论配置为空、相对路径还是绝对路径，
    最终都会追加 <phase>/<default_leaf>。这样两个训练 stage 的日志配置方式一致，且不会把 stage1/stage2
    的模型输入输出混到同一个 JSONL 目录里。
    """

    trainer_cfg = config["reranker_training"]["trainer"]
    root = training_log_root(config)
    configured = trainer_cfg.get(key)
    if configured:
        path = Path(str(configured))
        base = path if path.is_absolute() else root / path
    else:
        base = root
    return base / phase_name / default_leaf


def ordered_training_phases(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """按固定顺序读取 LLM reranker 训练 phase。

    这版不兼容旧的单 reward 训练配置；没有 training_phases 就直接报错，避免静默回到旧链路。
    """

    phases = config["reranker_training"].get("training_phases")
    if not isinstance(phases, dict):
        raise ValueError("reranker_training.training_phases is required for AIR LLM reranker training")
    ordered: list[tuple[str, dict[str, Any]]] = []
    for name in ("stage1_format", "stage2_agentic"):
        phase_cfg = phases.get(name)
        if isinstance(phase_cfg, dict):
            ordered.append((name, phase_cfg))
    extra = [name for name in phases if name not in {"stage1_format", "stage2_agentic"}]
    if extra:
        raise ValueError(f"unsupported reranker training phases: {extra}")
    return ordered


def phase_enabled(phase_cfg: dict[str, Any]) -> bool:
    return bool(phase_cfg.get("enabled", False))


def phase_actor_key(phase_name: str, phase_cfg: dict[str, Any], services: dict[str, Any]) -> str:
    configured = phase_cfg.get("actor_service")
    if configured and str(configured) in services:
        return str(configured)
    # 新资源结构已经按 phase 分层，所以每个 phase 内部都可以统一叫 reranker_actor。
    # 如果旧 phase 配置还写着 stage1_format_actor/stage2_agentic_actor，则回退到当前 phase 内的 reranker_actor。
    if configured and "reranker_actor" in services:
        return "reranker_actor"
    if configured:
        return str(configured)
    default_key = f"{phase_name}_actor"
    if default_key in services:
        return default_key
    return "reranker_actor"


def phase_services_for_config(config: dict[str, Any], phase_name: str | None = None) -> dict[str, Any]:
    """读取当前训练 phase 的资源服务配置。

    新结构使用 ``resource.stage_resources.train_llm_reranker.phase_services.<phase>.services``，
    这样 stage1/stage2 的 actor、frozen agent、retriever 归属在 YAML 中一眼可见。
    旧的平铺 ``services`` 仅作为过渡 fallback，避免历史 dry-run 配置或临时 overlay 直接失效。
    """

    rt_cfg = config.get("reranker_training", {})
    active_phase = str(phase_name or rt_cfg.get("_active_phase_name") or "legacy")
    train_resource = config["resource"]["stage_resources"]["train_llm_reranker"]
    phase_services = train_resource.get("phase_services")
    if isinstance(phase_services, dict) and active_phase in phase_services:
        phase_node = phase_services[active_phase]
        if not isinstance(phase_node, dict):
            raise TypeError(f"resource.stage_resources.train_llm_reranker.phase_services.{active_phase} must be a mapping")
        services = phase_node.get("services")
        if not isinstance(services, dict):
            raise ValueError(
                f"resource.stage_resources.train_llm_reranker.phase_services.{active_phase}.services must be set"
            )
        return services
    services = train_resource.get("services")
    if isinstance(services, dict):
        return services
    raise ValueError(
        "resource.stage_resources.train_llm_reranker.phase_services.<phase>.services must be set "
        "for AIR LLM reranker training"
    )


def effective_phase_config(
    config: dict[str, Any],
    phase_name: str,
    phase_cfg: dict[str, Any],
    *,
    init_model: str | Path | None = None,
) -> dict[str, Any]:
    """生成当前 phase 的 effective config。

    phase 配置覆盖公共 trainer 配置；reward 相关字段保留在 phase 自己的配置里。这样 VERL 命令生成只读
    effective_config，不需要 shell 侧维护任何业务参数。
    """

    out = deepcopy(config)
    rt_cfg = out["reranker_training"]
    trainer_cfg = rt_cfg.get("trainer", {})
    trainer_overrides = {
        key: value
        for key, value in phase_cfg.items()
        if key
        in {
            "total_epochs",
            "total_training_steps",
            "train_max_samples",
            "learning_rate",
            "train_batch_size",
            "ppo_mini_batch_size",
            "micro_batch_size_per_gpu",
            "log_prob_micro_batch_size_per_gpu",
            "ppo_max_token_len_per_gpu",
            "log_prob_max_token_len_per_gpu",
            "use_remove_padding",
            "use_dynamic_bsz",
            "calculate_rollout_log_probs",
            "use_rollout_log_probs",
            "rollout_correction_bypass_mode",
            "rollout_correction_use_policy_gradient",
            "actor_activation_offload",
            "actor_param_offload",
            "actor_optimizer_offload",
            "actor_forward_prefetch",
            "ref_param_offload",
            "ref_forward_prefetch",
            "actor_progress_interval",
            "n_samples_per_prompt",
            "val_n_samples_per_prompt",
            "val_rollout_temperature",
            "val_rollout_top_p",
            "val_do_sample",
            "use_kl_loss",
            "kl_loss_coef",
            "kl_loss_type",
            "entropy_coeff",
            "save_freq",
            "max_response_length",
            "max_prompt_length",
            "rollout_max_model_len",
            "truncation",
            "rollout_mode",
            "rollout_temperature",
            "rollout_top_p",
            "rollout_gpu_memory_utilization",
            "max_num_batched_tokens",
            "max_num_seqs",
            "enable_chunked_prefill",
            "enable_prefix_caching",
            "enforce_eager",
            "sampling_stop",
            "include_stop_str_in_output",
            "agent_loop_num_workers",
            "rollout_data_parallel_size",
            "train_max_samples",
            "val_max_samples",
            "val_batch_size",
            "val_before_train",
            "val_only",
            "rollout_detail_logging_enabled",
            "rollout_detail_log_path",
            "rollout_detail_expected_count",
            "rollout_detail_max_index",
            "timeout_seconds",
            "output_dir",
        }
    }
    rt_cfg["trainer"] = deep_merge(trainer_cfg, trainer_overrides)
    rt_cfg["_active_phase_name"] = phase_name
    rt_cfg["_active_phase_config"] = dict(phase_cfg)
    if init_model is not None:
        rt_cfg["_active_init_model"] = str(init_model)
    return out


def ensure_clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"training output directory already exists and overwrite=false: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_agent_model(config: dict[str, Any]) -> Path:
    """解析 continuation frozen agent 的可加载 HF 模型目录。"""

    rt_cfg = config["reranker_training"]
    value = (
        rt_cfg.get("continuation", {}).get("agent_model")
        or config.get("infer_runtime", {}).get("models", {}).get("trained_agent_model")
    )
    if not value:
        raise ValueError("reranker_training.continuation.agent_model or infer_runtime.models.trained_agent_model is required")
    infer_engine = repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "infer_air_vllm.py"
    cmd = [
        sys.executable,
        str(infer_engine),
        "resolve-model",
        "--path",
        str(value),
        "--role",
        "agent",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root()}:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(cmd, cwd=repo_root(), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"failed to resolve agent model from {value}: {result.stderr.strip()}")
    resolved = Path(result.stdout.strip())
    if not resolved.exists():
        raise FileNotFoundError(f"resolved agent model does not exist: {resolved}")
    return resolved


def ensure_parquet_dataset(branch_manifest: dict[str, Any]) -> Path:
    dataset_parquet = branch_manifest.get("dataset_parquet")
    if not dataset_parquet:
        raise ValueError("branch dataset manifest must contain dataset_parquet for VERL training")
    path = Path(str(dataset_parquet))
    if not path.exists():
        raise FileNotFoundError(f"branch dataset parquet does not exist: {path}")
    return path


def env_export_overrides(env_vars: dict[str, str]) -> dict[str, str]:
    """把 reward 环境变量同步注入 Ray runtime_env。

    VERL 的 custom reward 在 Ray worker 里执行；只设置 driver 环境变量是不够的。
    """

    return {f"+ray_kwargs.ray_init.runtime_env.env_vars.{key}": value for key, value in env_vars.items()}


def list_override(key: str, values: list[str]) -> str:
    quoted = ",".join("'" + value.replace("'", "\\'") + "'" for value in values)
    return f"{key}=[{quoted}]"


def hydra_string_literal(value: Any) -> str:
    """生成 Hydra 明确按字符串解析的字面量。

    Ray runtime_env.env_vars 要求 Dict[str, str]。如果这里直接传 50/false，
    Hydra 会把它们解析成 int/bool，Ray 初始化时会直接报类型错误。
    """

    text = str(value).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{text}'"


def string_override(key: str, value: Any) -> str:
    return f"{key}={hydra_string_literal(value)}"


def scalar_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        raw = "true" if value else "false"
    elif isinstance(value, (int, float)):
        raw = str(value)
    elif value is None:
        raw = "null"
    else:
        raw = shell_quote(value)
    return f"{key}={raw}"


def build_base_verl_env_vars(config: dict[str, Any]) -> dict[str, str]:
    trainer_cfg = config["reranker_training"]["trainer"]
    network_interface = str(trainer_cfg.get("network_interface") or "lo")
    return {
        # 单机 NPU 训练默认强制使用 lo，避免 torch/Gloo 自动选中没有 IPv4 的物理网卡。
        "GLOO_SOCKET_IFNAME": network_interface,
        "NCCL_SOCKET_IFNAME": network_interface,
        "HCCL_SOCKET_IFNAME": network_interface,
        "TP_SOCKET_IFNAME": network_interface,
        # Ascend 通信算子使用 AIV 展开，避免长 step 里 HCCL 走低效 FFTS 路径。
        "HCCL_OP_EXPANSION_MODE": "AIV",
        # actor update 阶段耗时较长时打印 micro batch 进度，避免训练过程变成黑盒。
        "COSEARCH_ACTOR_PROGRESS_INTERVAL": str(trainer_cfg.get("actor_progress_interval", 8)),
    }


def build_verl_env_vars(config: dict[str, Any], agent_model: Path) -> dict[str, str]:
    rt_cfg = config["reranker_training"]
    resource_cfg = phase_services_for_config(config)
    frozen_agent = resource_cfg["frozen_agent_vllm"]
    recall = resource_cfg["recall"]
    env_vars = build_base_verl_env_vars(config)
    phase_cfg = rt_cfg.get("_active_phase_config") or {}
    sub_strategy = phase_cfg.get("sub_strategy") or rt_cfg.get("reward", {}).get("strategy", "answer_reward")
    format_invalid_score = phase_cfg.get("format_invalid_score", rt_cfg.get("reward", {}).get("format_penalty", -0.5))
    evidence_hit_weight = phase_cfg.get("evidence_hit_weight", rt_cfg.get("reward", {}).get("evidence_hit_weight", 0.0))
    env_vars.update(
        {
            "AIR_CONTINUATION_AGENT_MODEL": str(agent_model),
            "AIR_CONTINUATION_TOKENIZER_PATH": str(agent_model),
            # frozen agent 既可能是旧单实例，也可能是 stage2 多实例 proxy；
            # continuation reward 始终只访问统一 base_url，不能直接感知 backend 实例。
            "AIR_CONTINUATION_AGENT_BASE_URL": frozen_agent_base_url(frozen_agent),
            "AIR_CONTINUATION_AGENT_SERVED_MODEL": str(frozen_agent.get("served_model_name") or "agentic-iter-rag-frozen-agent"),
            "AIR_CONTINUATION_RETRIEVAL_URL": str(recall["retrieval_service_url"]),
            "AIR_CONTINUATION_BATCH_WORKERS": str(rt_cfg["trainer"].get("agent_loop_num_workers", 64)),
            "AIR_CONTINUATION_CANDIDATE_TOP_N": str(rt_cfg["branch_dataset"]["candidate_top_n"]),
            "AIR_CONTINUATION_VISIBLE_TOP_M": str(rt_cfg["branch_dataset"]["visible_top_m"]),
            "AIR_CONTINUATION_FORMAT_PENALTY": str(format_invalid_score),
            "AIR_CONTINUATION_REWARD_STRATEGY": str(sub_strategy),
            "AIR_CONTINUATION_EVIDENCE_HIT_WEIGHT": str(evidence_hit_weight),
            "AIR_CONTINUATION_MAX_ASSISTANT_TURNS": str(rt_cfg["continuation"]["max_assistant_turns"]),
            "AIR_CONTINUATION_MAX_USER_TURNS": str(rt_cfg["continuation"]["max_user_turns"]),
            "AIR_CONTINUATION_MAX_PROMPT_LENGTH": str(rt_cfg["continuation"]["max_prompt_length"]),
            "AIR_CONTINUATION_MAX_RESPONSE_LENGTH": str(rt_cfg["continuation"]["max_response_length"]),
            "AIR_CONTINUATION_MAX_TOOL_RESPONSE_LENGTH": str(rt_cfg["continuation"]["max_tool_response_length"]),
            "AIR_CONTINUATION_TEMPERATURE": str(rt_cfg["continuation"]["temperature"]),
            "AIR_CONTINUATION_TOP_P": str(rt_cfg["continuation"]["top_p"]),
            "AIR_CONTINUATION_REQUEST_TIMEOUT": str(rt_cfg["continuation"].get("request_timeout", 180)),
            "AIR_CONTINUATION_ENABLE_THINKING": str(rt_cfg["continuation"].get("enable_thinking", False)).lower(),
        }
    )
    return env_vars


def frozen_agent_base_url(frozen_agent_cfg: dict[str, Any]) -> str:
    """解析 continuation reward 应该访问的 frozen agent 统一入口。"""

    if str(frozen_agent_cfg.get("backend_type") or "").lower() == "multi_instance_proxy":
        proxy_cfg = frozen_agent_cfg.get("proxy") if isinstance(frozen_agent_cfg.get("proxy"), dict) else {}
        host = str(proxy_cfg.get("host") or "127.0.0.1")
        port = int(proxy_cfg.get("port") or frozen_agent_cfg.get("port") or 8140)
        return f"http://{host}:{port}"
    return f"http://127.0.0.1:{int(frozen_agent_cfg['port'])}"


def build_verl_command_plan(
    config: dict[str, Any],
    branch_manifest: dict[str, Any],
    output_dir: Path,
    *,
    agent_model: Path | None = None,
) -> dict[str, Any]:
    """生成真实 VERL 训练命令计划。"""

    rt_cfg = config["reranker_training"]
    phase_name = str(rt_cfg.get("_active_phase_name") or "legacy")
    phase_cfg = rt_cfg.get("_active_phase_config") or {}
    trainer_cfg = rt_cfg["trainer"]
    resource_cfg = phase_services_for_config(config, phase_name)
    actor_key = phase_actor_key(phase_name, phase_cfg, resource_cfg)
    actor = resource_cfg[actor_key]
    frozen_agent = resource_cfg.get("frozen_agent_vllm", {})
    recall = resource_cfg.get("recall", {})
    dataset_parquet = (
        ensure_parquet_dataset(branch_manifest)
        if branch_manifest.get("dataset_parquet")
        else Path("<branch_dataset_parquet>")
    )
    actor_gpu_ids = as_int_list(actor["gpu_ids"])
    verl_root = Path(str(trainer_cfg.get("verl_root") or default_verl_root()))
    resolved_agent_model = agent_model or Path("<resolved-at-runtime>")
    reward_name = str(phase_cfg.get("reward_name") or "agentic_rag_rollout_reward")
    needs_continuation = reward_name == "agentic_rag_rollout_reward"
    expected_count = int(phase_cfg.get("expected_count", rt_cfg["branch_dataset"]["visible_top_m"]))
    max_index = int(phase_cfg.get("max_index", rt_cfg["branch_dataset"]["candidate_top_n"]))
    if expected_count != int(rt_cfg["branch_dataset"]["visible_top_m"]):
        raise ValueError("phase expected_count must match reranker_training.branch_dataset.visible_top_m")
    if max_index != int(rt_cfg["branch_dataset"]["candidate_top_n"]):
        raise ValueError("phase max_index must match reranker_training.branch_dataset.candidate_top_n")
    env_vars = build_verl_env_vars(config, resolved_agent_model) if needs_continuation and agent_model else build_base_verl_env_vars(config)
    max_prompt_length = int(trainer_cfg["max_prompt_length"])
    max_response_length = int(trainer_cfg["max_response_length"])
    max_model_len = int(trainer_cfg.get("rollout_max_model_len") or (max_prompt_length + max_response_length))
    rollout_n = int(trainer_cfg["n_samples_per_prompt"])
    train_batch_size = int(trainer_cfg["train_batch_size"])
    ppo_mini_batch_size = int(trainer_cfg.get("ppo_mini_batch_size") or train_batch_size)
    use_kl_loss = bool(trainer_cfg.get("use_kl_loss", False))
    rollout_data_parallel_size = int(trainer_cfg.get("rollout_data_parallel_size", 1))
    rollout_world_size = int(actor["tensor_parallel_size"]) * rollout_data_parallel_size
    if len(actor_gpu_ids) % rollout_world_size != 0:
        raise ValueError(
            "actor gpu count must be divisible by tensor_parallel_size * rollout_data_parallel_size; "
            f"got gpu_count={len(actor_gpu_ids)}, tensor_parallel_size={actor['tensor_parallel_size']}, "
            f"rollout_data_parallel_size={rollout_data_parallel_size}"
        )
    training_schedule = resolve_training_schedule(trainer_cfg, branch_manifest)
    total_training_steps_override = training_schedule["resolved_total_training_steps"]
    rollout_data_dir = resolve_phase_dump_dir(config, phase_name, "rollout_data_dir", "rollout_data")
    validation_data_dir = resolve_phase_dump_dir(config, phase_name, "validation_data_dir", "validation_data")
    val_rollout_n = int(trainer_cfg.get("val_n_samples_per_prompt", rollout_n))
    if bool(trainer_cfg.get("rollout_detail_logging_enabled", False)):
        detail_log_path = trainer_cfg.get("rollout_detail_log_path")
        if detail_log_path:
            detail_log_dir = Path(str(detail_log_path))
            if not detail_log_dir.is_absolute():
                detail_log_dir = validation_data_dir / detail_log_dir
        else:
            detail_log_dir = validation_data_dir / "rollout_details"
        env_vars.update(
            {
                "AIR_RERANKER_ROLLOUT_DETAIL_LOG": str(detail_log_dir),
                "AIR_RERANKER_EXPECTED_COUNT": str(
                    trainer_cfg.get("rollout_detail_expected_count", expected_count)
                ),
                "AIR_RERANKER_MAX_INDEX": str(trainer_cfg.get("rollout_detail_max_index", max_index)),
            }
        )

    hydra_overrides: list[str] = [
        scalar_override("algorithm.adv_estimator", "grpo"),
        scalar_override("algorithm.use_kl_in_reward", False),
        scalar_override("critic.enable", False),
        scalar_override("reward_model.enable", False),
        # 当前 CoSearch/verl 的 RayPPOTrainer 会直接访问这个字段；
        # AIR reranker 不启用 reward loop。stage2 会在下面切到 batch reward manager 做并发 continuation。
        scalar_override("+reward_model.use_reward_loop", False),
        scalar_override(
            "reward_model.reward_manager",
            str(trainer_cfg.get("reward_manager") or ("batch" if needs_continuation else "naive")),
        ),
        scalar_override("data.train_files", str(dataset_parquet)),
        scalar_override("data.val_files", str(dataset_parquet)),
        scalar_override("data.train_max_samples", int(trainer_cfg.get("train_max_samples", -1))),
        scalar_override("data.val_max_samples", int(trainer_cfg.get("val_max_samples", 1))),
        scalar_override("data.prompt_key", "prompt"),
        scalar_override("data.reward_fn_key", "data_source"),
        scalar_override("data.max_prompt_length", max_prompt_length),
        scalar_override("data.max_response_length", max_response_length),
        scalar_override("data.train_batch_size", train_batch_size),
        scalar_override("data.val_batch_size", int(trainer_cfg.get("val_batch_size", 1))),
        scalar_override("data.truncation", str(trainer_cfg.get("truncation", "left"))),
        scalar_override("data.filter_overlong_prompts", bool(trainer_cfg.get("filter_overlong_prompts", False))),
        scalar_override("data.filter_overlong_prompts_workers", int(trainer_cfg.get("filter_overlong_prompts_workers", 1))),
        scalar_override("data.dataloader_num_workers", int(trainer_cfg.get("dataloader_num_workers", 0))),
        scalar_override("data.return_raw_chat", True),
        scalar_override("data.trust_remote_code", True),
        scalar_override("+data.apply_chat_template_kwargs.enable_thinking", False),
        scalar_override("actor_rollout_ref.model.path", str(rt_cfg.get("_active_init_model") or rt_cfg["base_model"])),
        scalar_override("actor_rollout_ref.model.trust_remote_code", True),
        scalar_override("actor_rollout_ref.model.use_remove_padding", bool(trainer_cfg.get("use_remove_padding", False))),
        scalar_override("actor_rollout_ref.model.enable_gradient_checkpointing", True),
        # stage1 的大 batch * 多 rollout 会让 actor update 峰值显存很高；
        # activation offload 只改变显存/速度 tradeoff，不改变 batch、rollout 数或生成长度。
        scalar_override("actor_rollout_ref.model.enable_activation_offload", bool(trainer_cfg.get("actor_activation_offload", False))),
        scalar_override("actor_rollout_ref.actor.optim.lr", float(trainer_cfg["learning_rate"])),
        scalar_override("actor_rollout_ref.actor.ppo_mini_batch_size", ppo_mini_batch_size),
        scalar_override("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu", int(trainer_cfg["micro_batch_size_per_gpu"])),
        scalar_override("actor_rollout_ref.actor.use_dynamic_bsz", bool(trainer_cfg.get("use_dynamic_bsz", False))),
        scalar_override("actor_rollout_ref.actor.ppo_max_token_len_per_gpu", int(trainer_cfg.get("ppo_max_token_len_per_gpu", 16384))),
        # rollout-logprob bypass 时必须显式告诉 actor update 使用 batch 里的 old_log_probs；
        # 否则单 mini-batch + ppo_epochs=1 会被 on-policy 分支改成当前 log_prob.detach()。
        scalar_override("+actor_rollout_ref.actor.use_rollout_log_probs", bool(trainer_cfg.get("use_rollout_log_probs", False))),
        # stage1 只用格式/长度 reward 时不需要 reference policy；关闭 KL 可跳过 ref worker 初始化。
        scalar_override("actor_rollout_ref.actor.use_kl_loss", use_kl_loss),
        scalar_override("actor_rollout_ref.actor.kl_loss_coef", float(trainer_cfg.get("kl_loss_coef", 0.001))),
        scalar_override("actor_rollout_ref.actor.kl_loss_type", str(trainer_cfg.get("kl_loss_type", "low_var_kl"))),
        scalar_override("actor_rollout_ref.actor.entropy_coeff", float(trainer_cfg.get("entropy_coeff", 0.0))),
        scalar_override("actor_rollout_ref.actor.use_torch_compile", False),
        scalar_override("actor_rollout_ref.actor.fsdp_config.param_offload", bool(trainer_cfg.get("actor_param_offload", False))),
        scalar_override("actor_rollout_ref.actor.fsdp_config.optimizer_offload", bool(trainer_cfg.get("actor_optimizer_offload", False))),
        # forward_prefetch 会提前拉取下一层参数，速度可能更好，但 stage2 长 prompt + batch=64 时会抬高峰值显存。
        # 因此这里做成配置项，默认保持原行为，OOM 消融时可在 YAML 中关闭。
        scalar_override(
            "actor_rollout_ref.actor.fsdp_config.forward_prefetch",
            bool(trainer_cfg.get("actor_forward_prefetch", True)),
        ),
        scalar_override("actor_rollout_ref.ref.use_torch_compile", False),
        scalar_override("actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu", int(trainer_cfg.get("log_prob_micro_batch_size_per_gpu", 1))),
        scalar_override("actor_rollout_ref.ref.fsdp_config.param_offload", bool(trainer_cfg.get("ref_param_offload", True))),
        # ref 只参与 KL/logprob，不应该为了 prefetch 抢占 actor backward 的显存余量。
        scalar_override(
            "actor_rollout_ref.ref.fsdp_config.forward_prefetch",
            bool(trainer_cfg.get("ref_forward_prefetch", True)),
        ),
        scalar_override("actor_rollout_ref.rollout.name", "vllm"),
        # stage1 是单轮 reranker 生成，不需要 async AgentLoopManager；sync 直接走 vLLM rollout，减少 Python 调度长尾。
        scalar_override("actor_rollout_ref.rollout.mode", str(trainer_cfg.get("rollout_mode", "async"))),
        scalar_override("actor_rollout_ref.rollout.tensor_model_parallel_size", int(actor["tensor_parallel_size"])),
        # 这些 rollout 吞吐参数不改变训练语义，只控制 train_batch_size*n 条采样如何进入 vLLM 队列。
        scalar_override("actor_rollout_ref.rollout.data_parallel_size", rollout_data_parallel_size),
        scalar_override("actor_rollout_ref.rollout.n", rollout_n),
        scalar_override("actor_rollout_ref.rollout.temperature", float(trainer_cfg.get("rollout_temperature", 1.0))),
        scalar_override("actor_rollout_ref.rollout.top_p", float(trainer_cfg.get("rollout_top_p", 1.0))),
        # validation 在 ray_trainer 里先按 val_kwargs.n 展开，再把每条交给 rollout engine。
        # 因此这里必须显式透传 AIR phase 的 val_n_samples_per_prompt，避免 eval 仍使用 VERL 默认 n=1。
        scalar_override("actor_rollout_ref.rollout.val_kwargs.n", val_rollout_n),
        scalar_override(
            "actor_rollout_ref.rollout.val_kwargs.temperature",
            float(trainer_cfg.get("val_rollout_temperature", trainer_cfg.get("rollout_temperature", 0.0))),
        ),
        scalar_override(
            "actor_rollout_ref.rollout.val_kwargs.top_p",
            float(trainer_cfg.get("val_rollout_top_p", trainer_cfg.get("rollout_top_p", 1.0))),
        ),
        scalar_override("actor_rollout_ref.rollout.val_kwargs.do_sample", bool(trainer_cfg.get("val_do_sample", False))),
        scalar_override("actor_rollout_ref.rollout.gpu_memory_utilization", float(trainer_cfg.get("rollout_gpu_memory_utilization", 0.45))),
        scalar_override("actor_rollout_ref.rollout.max_model_len", max_model_len),
        scalar_override("actor_rollout_ref.rollout.max_num_batched_tokens", int(trainer_cfg.get("max_num_batched_tokens", max_model_len))),
        scalar_override("actor_rollout_ref.rollout.max_num_seqs", int(trainer_cfg.get("max_num_seqs", rollout_n))),
        scalar_override("actor_rollout_ref.rollout.enable_chunked_prefill", bool(trainer_cfg.get("enable_chunked_prefill", False))),
        scalar_override("actor_rollout_ref.rollout.enable_prefix_caching", bool(trainer_cfg.get("enable_prefix_caching", False))),
        scalar_override("actor_rollout_ref.rollout.enforce_eager", bool(trainer_cfg.get("enforce_eager", True))),
        # 让 async rollout 直接返回 logprob，后面通过 rollout-correction bypass 复用，跳过 old_log_prob 二次 forward。
        scalar_override("actor_rollout_ref.rollout.calculate_log_probs", bool(trainer_cfg.get("calculate_rollout_log_probs", False))),
        scalar_override("actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu", int(trainer_cfg.get("log_prob_micro_batch_size_per_gpu", 1))),
        scalar_override("actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu", int(trainer_cfg.get("log_prob_max_token_len_per_gpu", trainer_cfg.get("ppo_max_token_len_per_gpu", 16384)))),
        scalar_override("actor_rollout_ref.rollout.agent.num_workers", int(trainer_cfg.get("agent_loop_num_workers", 8))),
        # bypass_mode=true 时，trainer 会把 rollout_log_probs 直接写成 old_log_probs；
        # 这是当前不降低 batch/rollout/response 底线下最直接的训练阶段提速手段。
        scalar_override("algorithm.rollout_correction.bypass_mode", bool(trainer_cfg.get("rollout_correction_bypass_mode", False))),
        scalar_override(
            "algorithm.rollout_correction.use_policy_gradient",
            bool(trainer_cfg.get("rollout_correction_use_policy_gradient", False)),
        ),
        scalar_override("trainer.critic_warmup", 0),
        list_override("trainer.logger", [str(item) for item in trainer_cfg.get("logger", ["console"])]),
        scalar_override("trainer.project_name", str(trainer_cfg.get("project_name", "agentic_iter_rag_reranker"))),
        scalar_override("trainer.experiment_name", str(config["main_run"]["project"]["experiment_name"])),
        scalar_override("trainer.n_gpus_per_node", len(actor_gpu_ids)),
        scalar_override("trainer.nnodes", int(trainer_cfg.get("nnodes", 1))),
        scalar_override("trainer.default_local_dir", str(output_dir)),
        scalar_override("trainer.device", str(trainer_cfg.get("device", "npu"))),
        scalar_override("trainer.resume_mode", str(trainer_cfg.get("resume_mode", "disable"))),
        scalar_override("trainer.val_before_train", bool(trainer_cfg.get("val_before_train", False))),
        # eval-only 调试时只跑 validation generation/reward，不进入 PPO update。
        scalar_override("trainer.val_only", bool(trainer_cfg.get("val_only", False))),
        scalar_override("trainer.save_freq", int(trainer_cfg["save_freq"])),
        scalar_override("trainer.test_freq", int(trainer_cfg.get("test_freq", -1))),
        scalar_override("trainer.total_epochs", int(training_schedule["resolved_total_epochs"])),
        # null 表示让 VERL 按 dataloader 长度和 total_epochs 自动计算总 step；
        # 小样本 1-step 调试则显式传 1。
        scalar_override("trainer.total_training_steps", total_training_steps_override),
        scalar_override("trainer.max_actor_ckpt_to_keep", int(trainer_cfg.get("max_actor_ckpt_to_keep", 1))),
        # 两个 reranker 训练 phase 共用同一套日志配置逻辑，但按 phase 拆目录，方便直接定位模型输入输出。
        scalar_override("trainer.rollout_data_dir", str(rollout_data_dir)),
        scalar_override("trainer.validation_data_dir", str(validation_data_dir)),
        scalar_override("+trainer.num_examine", int(trainer_cfg.get("num_examine", 0))),
        scalar_override("+trainer.val_num_examine", int(trainer_cfg.get("val_num_examine", 0))),
        scalar_override("+ray_kwargs.ray_init.num_cpus", int(trainer_cfg.get("ray_num_cpus", 24))),
        scalar_override("+ray_kwargs.ray_init.include_dashboard", False),
        scalar_override("+ray_kwargs.ray_init.ignore_reinit_error", True),
    ]
    sampling_stop = trainer_cfg.get("sampling_stop")
    if sampling_stop:
        hydra_overrides.append(list_override("actor_rollout_ref.rollout.stop", [str(item) for item in as_list(sampling_stop)]))
        hydra_overrides.append(
            scalar_override(
                "actor_rollout_ref.rollout.include_stop_str_in_output",
                bool(trainer_cfg.get("include_stop_str_in_output", True)),
            )
        )
    if reward_name == "reranker_format_reward":
        hydra_overrides.extend(
            [
                scalar_override(
                    "custom_reward_function.path",
                    str(project_root() / "agentic_iter_rag" / "reranker_training" / "rewards" / "reranker_format_reward.py"),
                ),
                scalar_override("custom_reward_function.name", "compute_reranker_format_reward"),
                # CoSearch 对齐协议：模型只输出 visible_top_m 个 index，但 index 合法范围来自 candidate_top_n。
                scalar_override("+custom_reward_function.reward_kwargs.expected_count", expected_count),
                scalar_override("+custom_reward_function.reward_kwargs.max_index", max_index),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.format_invalid_score",
                    float(phase_cfg.get("format_invalid_score", -0.5)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.short_valid_score",
                    float(phase_cfg.get("short_valid_score", 1.0)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.long_valid_score",
                    float(phase_cfg.get("long_valid_score", 0.8)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.length_threshold_tokens",
                    int(phase_cfg.get("length_threshold_tokens", 512)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.partial_credit_enabled",
                    bool(phase_cfg.get("partial_credit_enabled", True)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.empty_rerank_score",
                    float(phase_cfg.get("empty_rerank_score", -0.2)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.wrong_index_count_base_score",
                    float(phase_cfg.get("wrong_index_count_base_score", 0.0)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.wrong_index_count_span_score",
                    float(phase_cfg.get("wrong_index_count_span_score", 0.4)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.wrong_index_count_max_score",
                    float(phase_cfg.get("wrong_index_count_max_score", 0.4)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.duplicate_or_out_of_range_score",
                    float(phase_cfg.get("duplicate_or_out_of_range_score", 0.2)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.invalid_rerank_text_score",
                    float(phase_cfg.get("invalid_rerank_text_score", -0.1)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.missing_reason_with_valid_rerank_score",
                    float(phase_cfg.get("missing_reason_with_valid_rerank_score", 0.1)),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.tokenizer_path",
                    str(rt_cfg.get("_active_init_model") or rt_cfg["base_model"]),
                ),
            ]
        )
    elif reward_name == "agentic_rag_rollout_reward":
        hydra_overrides.extend(
            [
                scalar_override(
                    "custom_reward_function.path",
                    str(project_root() / "agentic_iter_rag" / "reranker_training" / "rewards" / "agentic_rag_rollout_reward.py"),
                ),
                # stage2 使用 batch reward manager，让 64*n 条 continuation 在 AIR reward 层并发执行。
                scalar_override("custom_reward_function.name", "compute_agentic_rag_rollout_reward_batch"),
                # stage2 使用同一套 parser 参数；reranker 输出 top5 后直接变成 agent observation。
                scalar_override("+custom_reward_function.reward_kwargs.expected_count", expected_count),
                scalar_override("+custom_reward_function.reward_kwargs.max_index", max_index),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.format_penalty",
                    float(phase_cfg.get("format_invalid_score", rt_cfg.get("reward", {}).get("format_penalty", -0.5))),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.reward_strategy",
                    str(phase_cfg.get("sub_strategy", rt_cfg.get("reward", {}).get("strategy", "answer_reward"))),
                ),
                scalar_override(
                    "+custom_reward_function.reward_kwargs.evidence_hit_weight",
                    float(phase_cfg.get("evidence_hit_weight", rt_cfg.get("reward", {}).get("evidence_hit_weight", 0.0))),
                ),
            ]
        )
    else:
        raise ValueError(f"unsupported reranker training reward_name={reward_name!r}")
    if trainer_cfg.get("ray_object_store_memory") is not None:
        hydra_overrides.append(scalar_override("+ray_kwargs.ray_init.object_store_memory", int(trainer_cfg["ray_object_store_memory"])))
    if env_vars:
        for key, value in env_export_overrides(env_vars).items():
            hydra_overrides.append(string_override(key, value))

    # recall 资源既支持旧式 gpu_ids，也支持现在的 backend 分层配置；
    # manifest 里要把真实加速卡展示出来，避免 dry-run 审计时误以为 retriever 没有占卡。
    recall_gpus = recall.get("gpu_ids")
    if recall_gpus is None:
        recall_gpus = recall.get("accelerator_backend", {}).get("gpu_ids")
    frozen_agent_instances = frozen_agent.get("instances") if isinstance(frozen_agent.get("instances"), list) else []
    if str(frozen_agent.get("backend_type") or "").lower() == "multi_instance_proxy" and frozen_agent_instances:
        frozen_agent_gpus = [
            gpu_id
            for instance in frozen_agent_instances
            if isinstance(instance, dict)
            for gpu_id in as_int_list(instance.get("gpu_ids"))
        ]
    else:
        frozen_agent_gpus = frozen_agent.get("gpu_ids")

    return {
        "status": "planned",
        "entry": "python -m verl.trainer.main_ppo",
        "phase_name": phase_name,
        "reward_name": reward_name,
        "verl_root": str(verl_root),
        "dataset_files": [str(dataset_parquet)],
        "model_path": str(rt_cfg.get("_active_init_model") or rt_cfg["base_model"]),
        "output_dir": str(output_dir),
        "rollout_data_dir": str(rollout_data_dir),
        "validation_data_dir": str(validation_data_dir),
        "training_schedule": training_schedule,
        "hydra_overrides": hydra_overrides,
        "env_vars": env_vars,
        "resource_plan": {
            "actor_service": actor_key,
            "reranker_actor_gpus": actor["gpu_ids"],
            "frozen_agent_gpus": frozen_agent_gpus if needs_continuation else [],
            "frozen_agent_instances": frozen_agent_instances if needs_continuation else [],
            "frozen_agent_proxy_url": frozen_agent_base_url(frozen_agent) if needs_continuation and frozen_agent else None,
            "recall_gpus": recall_gpus if needs_continuation else [],
            "recall_url": recall.get("retrieval_service_url") if needs_continuation else None,
            "continuation_services_enabled": needs_continuation,
        },
    }


def run_shell_script(script_path: Path, log_path: Path, timeout_s: int | None = None) -> dict[str, Any]:
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # VERL 日志既写入磁盘给 reporter/复盘使用，也实时打印到 terminal，方便长任务现场观察。
    # 用 shell tee 而不是 Python 手动读管道，是为了保留 wait(timeout=...) 对无输出卡死场景的约束。
    tee_command = f"set -o pipefail; bash {shell_quote(script_path)} 2>&1 | tee {shell_quote(log_path)}"
    process = subprocess.Popen(
        ["bash", "-lc", tee_command],
        cwd=str(repo_root()),
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait(timeout=30)
    return {
        "return_code": return_code,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }


def start_periodic_reporter(
    *,
    config: dict[str, Any],
    log_path: Path,
    report_dir: Path,
    runtime_dir: Path,
    phase_name: str | None = None,
) -> tuple[subprocess.Popen[Any] | None, Path | None]:
    """启动 AIR LLM reranker 周期报告进程。

    reporter 只读 VERL 日志并刷新 latest 图；如果它启动失败，训练本身仍继续。
    """

    reporting_cfg = config["reranker_training"].get("reporting", {})
    if not bool(reporting_cfg.get("enabled", True)):
        return None, None

    stop_file = runtime_dir / "air_reranker_reporter.stop"
    if stop_file.exists():
        stop_file.unlink()
    report_dir.mkdir(parents=True, exist_ok=True)
    final_config = Path(str(config["runtime_compiled"]["FINAL_CONFIG_YAML"]))
    reporter_log = runtime_dir / "air_reranker_reporter.log"
    cmd = [
        resolve_compatible_python(repo_root()),
        "-m",
        "agentic_iter_rag.reranker_training.training_report",
        "--mode",
        "periodic",
        "--verl-log",
        str(log_path),
        "--out-dir",
        str(report_dir),
        "--repo-root",
        str(repo_root()),
        "--schema-path",
        str(repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "report_schema.py"),
        "--config-yaml",
        str(final_config),
        "--interval-seconds",
        str(int(reporting_cfg.get("interval_seconds", 60))),
        "--step-interval",
        str(int(reporting_cfg.get("step_interval", 1))),
        "--stop-file",
        str(stop_file),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root()}:{repo_root()}:{env.get('PYTHONPATH', '')}"
    reporter_log.parent.mkdir(parents=True, exist_ok=True)
    log_fp = reporter_log.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(repo_root()),
            env=env,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fp.close()
    return process, stop_file


def stop_periodic_reporter(
    process: subprocess.Popen[Any] | None,
    stop_file: Path | None,
    *,
    timeout_s: int = 20,
) -> None:
    """通知周期 reporter 退出；超时后再终止进程组。

    使用 stop-file 是为了让 reporter 完成当前一轮写文件，避免半写的 png/manifest。
    """

    if process is None:
        return
    if stop_file is not None:
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text("stop\n", encoding="utf-8")
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def final_training_report_outputs(
    *,
    config: dict[str, Any],
    log_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    """训练结束后强制生成最终报告，并返回可写入 manifest 的路径字段。"""

    reporting_cfg = config["reranker_training"].get("reporting", {})
    if not bool(reporting_cfg.get("enabled", True)):
        return {
            "periodic_reporter_enabled": False,
            "periodic_reporter_interval_seconds": int(reporting_cfg.get("interval_seconds", 60)),
        }

    manifest: dict[str, Any] = {}
    if bool(reporting_cfg.get("generate_final_on_exit", True)):
        manifest = generate_once(
            verl_log=log_path,
            out_dir=report_dir,
            repo_root=repo_root(),
            schema_path=repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "report_schema.py",
            config=config,
            mode="final",
        )
    paths = build_report_paths(report_dir)
    curve_paths = manifest.get("curve_paths") if isinstance(manifest, dict) else None
    if not isinstance(curve_paths, dict):
        curve_paths = {key: str(value) for key, value in paths.curve_paths.items()}
    return {
        "training_metrics_jsonl": str(paths.metrics_jsonl),
        "training_report": str(paths.training_report),
        "detailed_training_report": str(paths.detailed_training_report),
        "training_curve_paths": curve_paths,
        "training_report_manifest": str(paths.report_manifest),
        "periodic_reporter_enabled": True,
        "periodic_reporter_interval_seconds": int(reporting_cfg.get("interval_seconds", 60)),
    }


def write_verl_launch_script(
    *,
    config: dict[str, Any],
    plan: dict[str, Any],
    output_dir: Path,
    runtime_dir: Path,
) -> Path:
    trainer_cfg = config["reranker_training"]["trainer"]
    rt_cfg = config["reranker_training"]
    phase_name = str(rt_cfg.get("_active_phase_name") or "legacy")
    phase_cfg = rt_cfg.get("_active_phase_config") or {}
    services = phase_services_for_config(config, phase_name)
    actor_cfg = services[phase_actor_key(phase_name, phase_cfg, services)]
    actor_gpu_csv = csv_ids(actor_cfg["gpu_ids"])
    network_interface = str(trainer_cfg.get("network_interface") or "lo")
    verl_root = Path(str(plan["verl_root"]))
    compat_python = repo_root() / "src" / "env_manage" / "compatible_python.sh"
    air_accel = repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "infer_backend" / "00_air_accelerator.sh"
    overrides = [str(item) for item in plan["hydra_overrides"]]

    script = runtime_dir / "run_verl_reranker_grpo.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shell_quote(repo_root())}",
        f"source {shell_quote(compat_python)}",
        f"source {shell_quote(air_accel)}",
        f"export ASCEND_RT_VISIBLE_DEVICES={shell_quote(actor_gpu_csv)}",
        f"export CUDA_VISIBLE_DEVICES={shell_quote(actor_gpu_csv)}",
        "export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1",
        "export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1",
        # 单机训练强制使用 lo，避免自动选择无 IPv4 地址的物理网卡导致 Gloo 初始化失败。
        f"export GLOO_SOCKET_IFNAME={shell_quote(network_interface)}",
        f"export NCCL_SOCKET_IFNAME={shell_quote(network_interface)}",
        f"export HCCL_SOCKET_IFNAME={shell_quote(network_interface)}",
        f"export TP_SOCKET_IFNAME={shell_quote(network_interface)}",
        "export TOKENIZERS_PARALLELISM=false",
        "export VLLM_DISABLE_FLASHINFER=1",
        "export VLLM_USE_FLASHINFER_SAMPLER=0",
        "export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}",
        "export VLLM_ALLREDUCE_USE_SYMM_MEM=0",
        "export WANDB_MODE=disabled",
        f"export PYTHONPATH={shell_quote(verl_root)}:{shell_quote(project_root())}:${{PYTHONPATH:-}}",
        f"mkdir -p {shell_quote(output_dir)}",
    ]
    for key, value in plan.get("env_vars", {}).items():
        lines.append(f"export {key}={shell_quote(value)}")
    lines.extend(
        [
            "cmd=(\"$PY\" -m verl.trainer.main_ppo)",
            "cmd+=(",
        ]
    )
    for override in overrides:
        lines.append(f"  {shell_quote(override)}")
    lines.extend(
        [
            ")",
            "printf '%s\\n' \"${cmd[@]}\" > " + shell_quote(runtime_dir / "verl_command.argv"),
            "exec \"${cmd[@]}\"",
        ]
    )
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o750)
    return script


def find_latest_checkpoint(output_dir: Path) -> Path:
    """返回 VERL 训练后的可追溯 checkpoint 目录。"""

    candidates = [path for path in output_dir.rglob("global_step_*") if path.is_dir()]
    if not candidates:
        return output_dir

    def step_num(path: Path) -> int:
        suffix = path.name.rsplit("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else -1

    return sorted(candidates, key=lambda item: (step_num(item), str(item)))[-1]


def run_verl_training(
    *,
    config: dict[str, Any],
    branch_manifest_path: Path,
    branch_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """执行真实 VERL GRPO 训练。

    这个函数把训练语义串完整：reranker 输出排序，custom reward 解析排序，构造新的 top5
    observation，交给 frozen agent continuation，后续 search 只走 retriever，最后用 answer reward 计分。
    """

    rt_cfg = config["reranker_training"]
    trainer_cfg = rt_cfg["trainer"]
    phase_name = str(rt_cfg.get("_active_phase_name") or "legacy")
    phase_cfg = rt_cfg.get("_active_phase_config") or {}
    reward_name = str(phase_cfg.get("reward_name") or "agentic_rag_rollout_reward")
    needs_continuation = reward_name == "agentic_rag_rollout_reward"
    ensure_parquet_dataset(branch_manifest)
    ensure_clean_dir(output_dir, overwrite=bool(trainer_cfg.get("overwrite", True)))
    runtime_dir = runtime_service_dir(config, phase_name)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    report_dir = training_report_dir(config, phase_name)

    agent_model = resolve_agent_model(config) if needs_continuation else None
    plan = build_verl_command_plan(config, branch_manifest, output_dir, agent_model=agent_model)
    rollout_data_dir = Path(str(plan["rollout_data_dir"]))
    validation_data_dir = Path(str(plan["validation_data_dir"]))
    rollout_data_dir.mkdir(parents=True, exist_ok=True)
    validation_data_dir.mkdir(parents=True, exist_ok=True)
    write_json(runtime_dir / "verl_command_plan.json", plan)

    services_cfg = phase_services_for_config(config)
    manager = TrainingServiceManager(
        repo_root=repo_root(),
        project_root=project_root(),
        verl_root=Path(str(plan["verl_root"])),
        runtime_dir=runtime_dir,
        config=config,
    )
    service_outputs: dict[str, Any] = {}
    reporter_process: subprocess.Popen[Any] | None = None
    reporter_stop_file: Path | None = None
    try:
        if needs_continuation:
            recall_cfg = services_cfg["recall"]
            frozen_agent_cfg = services_cfg["frozen_agent_vllm"]
            if bool(recall_cfg.get("auto_start", True)):
                service_outputs["recall"] = manager.start_recall(recall_cfg)
            else:
                service_outputs["recall"] = {"status": "external", "retrieval_url": recall_cfg["retrieval_service_url"]}
            if bool(frozen_agent_cfg.get("auto_start", True)):
                if agent_model is None:
                    raise RuntimeError("agent_model must be resolved before starting frozen agent")
                service_outputs["frozen_agent"] = manager.start_frozen_agent(frozen_agent_cfg, agent_model=agent_model)
            else:
                service_outputs["frozen_agent"] = {
                    "status": "external",
                    "base_url": f"http://127.0.0.1:{frozen_agent_cfg['port']}",
                    "served_model": frozen_agent_cfg["served_model_name"],
                }
        else:
            # stage1 只训练格式和长度，不启动 frozen agent/retriever，避免无意义占用端口和 NPU。
            service_outputs["continuation_services"] = "skipped_for_format_phase"

        launch_script = write_verl_launch_script(
            config=config,
            plan=plan,
            output_dir=output_dir,
            runtime_dir=runtime_dir,
        )
        log_path = runtime_dir / "verl_train.log"
        reporter_process, reporter_stop_file = start_periodic_reporter(
            config=config,
            log_path=log_path,
            report_dir=report_dir,
            runtime_dir=runtime_dir,
            phase_name=phase_name,
        )
        try:
            run_info = run_shell_script(
                launch_script,
                log_path,
                timeout_s=trainer_cfg.get("timeout_seconds"),
            )
        finally:
            stop_periodic_reporter(reporter_process, reporter_stop_file)
        if int(run_info["return_code"]) != 0:
            raise RuntimeError(
                f"VERL training exited with code {run_info['return_code']}; log={log_path}\n{tail_text(log_path)}"
            )

        report_outputs = final_training_report_outputs(
            config=config,
            log_path=log_path,
            report_dir=report_dir,
        )
        checkpoint = find_latest_checkpoint(output_dir)
        checkpoint_manifest = output_dir / "air_reranker_training_manifest.json"
        write_json(
            checkpoint_manifest,
            {
                "type": "air_llm_reranker_verl_checkpoint",
                "created_at": utc_now(),
                "phase_name": phase_name,
                "reward_name": reward_name,
                "init_model": str(rt_cfg.get("_active_init_model") or rt_cfg["base_model"]),
                "base_model": str(rt_cfg["base_model"]),
                "branch_dataset_manifest": str(branch_manifest_path),
                "sample_count": len(rows),
                "n_samples_per_prompt": int(trainer_cfg["n_samples_per_prompt"]),
                "checkpoint": str(checkpoint),
                "runtime_dir": str(runtime_dir),
                "verl_log": str(log_path),
                "rollout_data_dir": str(rollout_data_dir),
                "validation_data_dir": str(validation_data_dir),
                "training_schedule": plan.get("training_schedule"),
                **report_outputs,
            },
        )
        return {
            "status": "completed",
            "backend": "verl",
            "phase_name": phase_name,
            "reward_name": reward_name,
            "reranker_model": str(checkpoint),
            "reranker_checkpoint": str(checkpoint),
            "checkpoint_manifest": str(checkpoint_manifest),
            "branch_dataset_manifest": str(branch_manifest_path),
            "branch_dataset_version": branch_manifest.get("version"),
            "sample_count": len(rows),
            "n_samples_per_prompt": int(trainer_cfg["n_samples_per_prompt"]),
            "verl_command_plan": plan,
            "service_outputs": service_outputs,
            "runtime_dir": str(runtime_dir),
            "verl_log": str(log_path),
            "rollout_data_dir": str(rollout_data_dir),
            "validation_data_dir": str(validation_data_dir),
            "training_schedule": plan.get("training_schedule"),
            **report_outputs,
            "config_hash": stable_config_hash(config["reranker_training"]),
        }
    finally:
        stop_periodic_reporter(reporter_process, reporter_stop_file)
        if bool(trainer_cfg.get("auto_stop_services", True)):
            manager.stop_all()


def run_smoke_training(
    *,
    config: dict[str, Any],
    branch_manifest_path: Path,
    branch_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    """执行本地 smoke 训练。

    smoke 的目的不是更新模型参数，而是验证：branch dataset 可读、prompt/extra_info 完整、parser/reward
    规则可执行、manifest 和 service bundle 能串起来。
    """

    rt_cfg = config["reranker_training"]
    phase_name = str(rt_cfg.get("_active_phase_name") or "legacy")
    phase_cfg = rt_cfg.get("_active_phase_config") or {}
    reward_name = str(phase_cfg.get("reward_name") or "reranker_format_reward")
    branch_cfg = rt_cfg["branch_dataset"]
    trainer_cfg = rt_cfg["trainer"]
    overwrite = bool(trainer_cfg.get("overwrite", True))
    ensure_clean_dir(output_dir, overwrite=overwrite)

    expected_count = int(branch_cfg["visible_top_m"])
    max_index = int(branch_cfg["candidate_top_n"])
    format_penalty = float(phase_cfg.get("format_invalid_score", -0.5))
    short_valid_score = float(phase_cfg.get("short_valid_score", 1.0))
    long_valid_score = float(phase_cfg.get("long_valid_score", 0.5))
    length_threshold_tokens = int(phase_cfg.get("length_threshold_tokens", 512))
    response_text = render_identity_rerank_response(expected_count, max_index=max_index)

    reward_rows: list[dict[str, Any]] = []
    format_valid_count = 0
    for sample_index, row in enumerate(rows):
        reward = compute_format_only_reward(
            response_text,
            row["extra_info"],
            expected_count=expected_count,
            max_index=max_index,
            format_penalty=format_penalty,
            short_valid_score=short_valid_score,
            long_valid_score=long_valid_score,
            length_threshold_tokens=length_threshold_tokens,
            tokenizer_path=str(rt_cfg.get("_active_init_model") or rt_cfg["base_model"]),
        )
        if reward["format_valid"]:
            format_valid_count += 1
        reward_rows.append(
            {
                "sample_index": sample_index,
                "sample_id": row.get("sample_id"),
                "trajectory_id": row.get("extra_info", {}).get("trajectory_id"),
                "step_index": row.get("extra_info", {}).get("step_index"),
                "score": reward["score"],
                "format_valid": reward["format_valid"],
                "continuation_status": reward["continuation_status"],
                "visible_doc_ids": reward.get("visible_doc_ids", []),
                "format_error_code": reward.get("format_error_code"),
                "response_length_tokens": reward.get("response_length_tokens"),
                "length_penalty_applied": reward.get("length_penalty_applied"),
            }
        )

    rewards_jsonl = output_dir / "smoke_rewards.jsonl"
    checkpoint_manifest = output_dir / "checkpoint_manifest.json"
    readme = output_dir / "README.md"
    model_marker = output_dir / "SMOKE_MODEL_DO_NOT_DEPLOY.txt"
    config_copy = output_dir / "final_config.yaml"
    write_jsonl(rewards_jsonl, reward_rows)
    write_json(
        checkpoint_manifest,
        {
            "type": "air_llm_reranker_smoke_checkpoint",
            "created_at": utc_now(),
            "phase_name": phase_name,
            "reward_name": reward_name,
            "init_model": str(rt_cfg.get("_active_init_model") or rt_cfg["base_model"]),
            "base_model": str(rt_cfg["base_model"]),
            "branch_dataset_manifest": str(branch_manifest_path),
            "sample_count": len(rows),
            "format_valid_count": format_valid_count,
            "n_samples_per_prompt": int(trainer_cfg["n_samples_per_prompt"]),
            "note": "This directory is a smoke artifact. It is not a trained deployable model.",
        },
    )
    write_json(
        output_dir / "training_metrics.json",
        {
            "backend": "smoke",
            "phase_name": phase_name,
            "reward_name": reward_name,
            "sample_count": len(rows),
            "format_valid_rate": format_valid_count / max(len(rows), 1),
            "mean_format_reward": sum(float(item["score"]) for item in reward_rows) / max(len(reward_rows), 1),
            "continuation_executed_count": 0,
            "rollout_n": int(trainer_cfg["n_samples_per_prompt"]),
        },
    )
    model_marker.write_text(
        "AIR LLM reranker smoke artifact only. Real GRPO training requires the VERL continuation reward worker.\n",
        encoding="utf-8",
    )
    readme.write_text(
        "\n".join(
            [
                "# AIR LLM Reranker Smoke Artifact",
                "",
                "这是 train_llm_reranker stage 的 smoke 产物，用来验证新增训练链路可以跑通。",
                "它没有更新模型参数，不能作为真实 reranker 模型部署。",
                "",
                f"- branch_dataset_manifest: {branch_manifest_path}",
                f"- sample_count: {len(rows)}",
                f"- rollout_n: {trainer_cfg['n_samples_per_prompt']}",
                f"- rewards_jsonl: {rewards_jsonl}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    final_yaml = Path(str(config["runtime_compiled"]["FINAL_CONFIG_YAML"]))
    if final_yaml.exists():
        copy_file(final_yaml, config_copy)

    return {
        "status": "completed",
        "backend": "smoke",
        "phase_name": phase_name,
        "reward_name": reward_name,
        "reranker_model": str(output_dir),
        "reranker_checkpoint": str(output_dir),
        "checkpoint_manifest": str(checkpoint_manifest),
        "smoke_rewards_jsonl": str(rewards_jsonl),
        "sample_count": len(rows),
        "format_valid_count": format_valid_count,
        "continuation_executed_count": 0,
        "branch_dataset_manifest": str(branch_manifest_path),
        "branch_dataset_version": branch_manifest.get("version"),
        "n_samples_per_prompt": int(trainer_cfg["n_samples_per_prompt"]),
        "config_hash": stable_config_hash(rt_cfg),
        "warning": "smoke backend does not perform real GRPO parameter updates",
    }


def resolve_phase_init_model(
    phase_name: str,
    phase_cfg: dict[str, Any],
    rt_cfg: dict[str, Any],
    completed_outputs: dict[str, dict[str, Any]],
) -> str:
    init_model = str(phase_cfg.get("init_model") or "base_model")
    if init_model == "base_model":
        return str(rt_cfg["base_model"])
    if init_model == "stage1_checkpoint":
        stage1 = completed_outputs.get("stage1_format")
        if not stage1 or not stage1.get("reranker_checkpoint"):
            raise RuntimeError("stage2_agentic requires completed stage1_format checkpoint")
        return str(stage1["reranker_checkpoint"])
    path = Path(init_model)
    if not path.exists():
        raise FileNotFoundError(f"phase {phase_name} init_model does not exist: {path}")
    return str(path)


def build_phase_plan_outputs(
    *,
    config: dict[str, Any],
    branch_manifest: dict[str, Any],
    backend: str,
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    phase_plans: list[dict[str, Any]] = []
    last_model: str | None = None
    last_checkpoint: str | None = None
    completed_for_plan: dict[str, dict[str, Any]] = {}
    for phase_name, phase_cfg in ordered_training_phases(config):
        enabled = phase_enabled(phase_cfg)
        init_model = None
        if enabled:
            try:
                init_model = resolve_phase_init_model(phase_name, phase_cfg, config["reranker_training"], completed_for_plan)
            except RuntimeError:
                init_model = "<stage1_checkpoint_after_training>"
        elif str(phase_cfg.get("init_model") or "") == "stage1_checkpoint":
            init_model = "<stage1_checkpoint_after_training>"
        phase_config = effective_phase_config(config, phase_name, phase_cfg, init_model=init_model)
        output_dir = phase_output_dir(phase_config, backend, phase_name)
        plan_agent_model = None
        if phase_cfg.get("reward_name") == "agentic_rag_rollout_reward":
            # dry-run/manifest 里的 continuation agent 必须来自 frozen search agent 配置，
            # 不能误用当前 phase 的 reranker init_model。否则审计结果会显示 reward 环境使用了 reranker base model。
            try:
                plan_agent_model = resolve_agent_model(phase_config)
            except Exception:
                plan_agent_model = None
        plan = build_verl_command_plan(phase_config, branch_manifest, output_dir, agent_model=plan_agent_model)
        phase_plans.append(
            {
                "phase_name": phase_name,
                "enabled": enabled,
                "reward_name": phase_cfg.get("reward_name"),
                "init_model": init_model,
                "output_dir": str(output_dir),
                "verl_command_plan": plan,
            }
        )
        if enabled:
            completed_for_plan[phase_name] = {"reranker_checkpoint": str(output_dir), "reranker_model": str(output_dir)}
            last_model = str(output_dir)
            last_checkpoint = str(output_dir)
    return phase_plans, last_model, last_checkpoint


def run_training_phases(
    *,
    config: dict[str, Any],
    branch_manifest_path: Path,
    branch_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    backend: str,
) -> dict[str, Any]:
    completed: dict[str, dict[str, Any]] = {}
    completed_phase_names: list[str] = []
    skipped_phase_names: list[str] = []
    phase_outputs: dict[str, Any] = {}
    final_model: str | None = None
    final_checkpoint: str | None = None

    for phase_name, phase_cfg in ordered_training_phases(config):
        if not phase_enabled(phase_cfg):
            skipped_phase_names.append(phase_name)
            phase_outputs[phase_name] = {
                "status": "skipped",
                "phase_name": phase_name,
                "enabled": False,
                "reward_name": phase_cfg.get("reward_name"),
            }
            continue
        init_model = resolve_phase_init_model(phase_name, phase_cfg, config["reranker_training"], completed)
        phase_config = effective_phase_config(config, phase_name, phase_cfg, init_model=init_model)
        output_dir = phase_output_dir(phase_config, backend, phase_name)
        if backend == "verl":
            outputs = run_verl_training(
                config=phase_config,
                branch_manifest_path=branch_manifest_path,
                branch_manifest=branch_manifest,
                rows=rows,
                output_dir=output_dir,
            )
        elif backend == "smoke":
            outputs = run_smoke_training(
                config=phase_config,
                branch_manifest_path=branch_manifest_path,
                branch_manifest=branch_manifest,
                rows=rows,
                output_dir=output_dir,
            )
        else:
            raise ValueError(f"unsupported reranker trainer backend: {backend}")
        phase_manifest = output_dir / "phase_manifest.json"
        write_json(
            phase_manifest,
            {
                "type": "air_llm_reranker_phase_manifest",
                "created_at": utc_now(),
                "phase_name": phase_name,
                "enabled": True,
                "init_model": init_model,
                "outputs": outputs,
            },
        )
        outputs["phase_manifest"] = str(phase_manifest)
        phase_outputs[phase_name] = outputs
        completed[phase_name] = outputs
        completed_phase_names.append(phase_name)
        final_model = str(outputs["reranker_model"])
        final_checkpoint = str(outputs["reranker_checkpoint"])

    if not final_model or not final_checkpoint:
        raise RuntimeError("no enabled reranker training phase completed")

    return {
        "status": "completed",
        "backend": backend,
        "completed_phases": completed_phase_names,
        "skipped_phases": skipped_phase_names,
        "phase_outputs": phase_outputs,
        "phase_manifests": {
            name: outputs.get("phase_manifest")
            for name, outputs in phase_outputs.items()
            if isinstance(outputs, dict) and outputs.get("phase_manifest")
        },
        "stage1_checkpoint": phase_outputs.get("stage1_format", {}).get("reranker_checkpoint"),
        "stage2_checkpoint": phase_outputs.get("stage2_agentic", {}).get("reranker_checkpoint"),
        "final_reranker_checkpoint": final_checkpoint,
        "reranker_model": final_model,
        "reranker_checkpoint": final_checkpoint,
        "branch_dataset_manifest": str(branch_manifest_path),
        "branch_dataset_version": branch_manifest.get("version"),
        "sample_count": len(rows),
        "config_hash": stable_config_hash(config["reranker_training"]),
    }


def run_from_config(config_path: Path, stage_manifest_path: Path, dry_run: bool = False) -> dict[str, Any]:
    config = read_yaml(config_path)
    stage_cfg = config["pipeline"]["stage_configs"]["train_llm_reranker"]
    rt_cfg = config["reranker_training"]
    trainer_cfg = rt_cfg["trainer"]
    backend = str(trainer_cfg["backend"])
    if backend not in {"smoke", "verl"}:
        raise ValueError(f"unsupported reranker trainer backend: {backend}")

    if dry_run:
        branch_manifest_path = resolve_branch_manifest(config, must_exist=False)
        if branch_manifest_path.exists():
            branch_manifest, _rows = load_branch_dataset(branch_manifest_path)
        else:
            branch_manifest = planned_branch_manifest_from_config(config)
        phase_plans, planned_model, planned_checkpoint = build_phase_plan_outputs(
            config=config,
            branch_manifest=branch_manifest,
            backend=backend,
        )
        outputs = {
            "status": "compiled",
            "backend": backend,
            "branch_dataset_manifest": str(branch_manifest_path),
            "branch_dataset_sample_count": branch_manifest.get("sample_count"),
            "reranker_model": planned_model,
            "reranker_checkpoint": planned_checkpoint,
            "phase_plans": phase_plans,
        }
        write_stage_manifest(stage_manifest_path, stage="train_llm_reranker", config=stage_cfg, outputs=outputs)
        return outputs

    branch_manifest_path = resolve_branch_manifest(config)
    branch_manifest, rows = load_branch_dataset(branch_manifest_path)
    outputs = run_training_phases(
        config=config,
        branch_manifest_path=branch_manifest_path,
        branch_manifest=branch_manifest,
        rows=rows,
        backend=backend,
    )
    write_stage_manifest(stage_manifest_path, stage="train_llm_reranker", config=stage_cfg, outputs=outputs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AIR LLM reranker for branch GRPO.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs = run_from_config(args.config, args.manifest, dry_run=args.dry_run)
    print(f"train_llm_reranker outputs: {outputs}")


if __name__ == "__main__":
    main()
