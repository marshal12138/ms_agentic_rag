"""Static validation for compiled launcher configuration.

这些校验仍然属于“配置编译”阶段，因为它们不启动训练、不启动服务，只确认：

- canonical full 模式不再接受历史 shell env 覆盖训练超参。
- async ranker training overlay / prompt / judge service config 可解析。
- 最终 Hydra 配置能 compose，并且关键路径存在。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .context import CompiledConfig, CompilerContext
from .final_config import select_config_value
from .runtime_env import safe_mkdir, truthy
from .yaml_utils import load_mapping

CANONICAL_DEPRECATED_ENV_OVERRIDES = [
    "TRAINER_LOGGER",
    "SAVE_FREQ",
    "TEST_FREQ",
    "RESUME_MODE",
    "MAX_ACTOR_CKPT_TO_KEEP",
    "DUMP_ROLLOUT_EVERY_STEP_NUM",
    "DUMP_ROLLOUT_NUM_EVERYTIME",
    "MAX_ROLLOUT_DUMP_NUM",
    "ROLLOUT_TRACE_MODE",
    "RECALL_TOP_K",
    "RANK_TOP_K",
    "ACTOR_BATCH_SIZE",
    "ACTOR_MICRO_BATCH_SIZE_PER_GPU",
    "LOG_PROB_MICRO_BATCH_SIZE_PER_GPU",
    "ACTOR_LR",
    "KL_LOSS_COEF",
    "TRAIN_BATCH_SIZE",
    "VAL_BATCH_SIZE",
    "TRAIN_MAX_SAMPLES",
    "VAL_MAX_SAMPLES",
    "N_ROLLOUTS",
    "MODEL_PATH",
    "TRAIN_DATA",
    "VAL_DATA",
    "LORA_RANK",
    "LORA_ALPHA",
    "TOTAL_STEPS",
    "ENABLE_ASYNC_RANKER_TRAINING",
]
"""canonical 模式禁止再用 shell env 管理的训练语义参数。

这些参数已经迁入 Hydra YAML 链路。保留这个拒绝列表是为了避免用户 export 一个旧变量
后，以为覆盖生效，但实际和 YAML 优先级冲突。
其中 ENABLE_ASYNC_RANKER_TRAINING 已被 run_mode 与最终 Hydra 配置推导取代。
"""


def reject_canonical_deprecated_env_overrides(environ: Mapping[str, str]) -> None:
    """拒绝 canonical 模式中的历史 shell 训练参数覆盖。"""

    bad = [name for name in CANONICAL_DEPRECATED_ENV_OVERRIDES if name in environ]
    if bad:
        raise ValueError(
            "These env vars are no longer accepted as canonical Hydra overrides: "
            + ", ".join(bad)
            + ". Move values to YAML/overlay, or pass explicit Hydra CLI overrides in the task."
        )


def _string_list(values: list[object]) -> str:
    """把审计用列表压成稳定字符串；空列表保留为空字符串。"""

    return ",".join(str(value) for value in values)


def _llm_judge_stages(final_config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """从最终 Hydra 配置中取出所有 LLM-as-judge stage。"""

    stages = select_config_value(dict(final_config), "ranker_training.async_ranker_training.stages") or []
    if not isinstance(stages, list):
        raise TypeError("final Hydra ranker_training.async_ranker_training.stages must be a list")
    judge_stages: list[Mapping[str, Any]] = []
    for index, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise TypeError(f"final Hydra async ranker stage[{index}] must be a mapping")
        if stage.get("type") == "llm_as_judge":
            judge_stages.append(stage)
    return judge_stages


def validate_llm_judge_stage_matches_service_config(
    *,
    final_config: Mapping[str, Any],
    env: dict[str, str],
    service_config_path: Path,
) -> None:
    """校验最终 Hydra LLM judge stage 和 service config 指向同一个 endpoint/model。

    full 模式下，训练侧请求地址来自最终 Hydra stage，服务侧启动参数来自 service YAML。
    两边必须完全一致，否则训练可能打到错误 endpoint，或用一个服务并未暴露的 model 名。
    """

    service_config = load_mapping(service_config_path, label="LLM judge service config")
    server = service_config.get("server") or {}
    model = service_config.get("model") or {}
    if not isinstance(server, dict):
        raise TypeError(f"LLM judge service config server must be a mapping: {service_config_path}")
    if not isinstance(model, dict):
        raise TypeError(f"LLM judge service config model must be a mapping: {service_config_path}")

    service_endpoint = str(server.get("endpoint") or "")
    service_model = str(model.get("served_model_name") or "")
    if not service_endpoint:
        raise ValueError(f"LLM judge service config is missing server.endpoint: {service_config_path}")
    if not service_model:
        raise ValueError(f"LLM judge service config is missing model.served_model_name: {service_config_path}")

    judge_stages = _llm_judge_stages(final_config)
    if not judge_stages:
        raise ValueError("LLM judge service is required but final Hydra config has no llm_as_judge stage")

    stage_endpoints: list[object] = []
    stage_models: list[object] = []
    mismatches: list[str] = []
    for index, stage in enumerate(judge_stages):
        endpoint = stage.get("endpoint")
        stage_model = stage.get("model")
        if endpoint is None:
            raise ValueError(f"final Hydra llm_as_judge stage[{index}] is missing endpoint")
        if stage_model is None:
            raise ValueError(f"final Hydra llm_as_judge stage[{index}] is missing model")
        endpoint_str = str(endpoint)
        model_str = str(stage_model)
        stage_endpoints.append(endpoint_str)
        stage_models.append(model_str)
        if endpoint_str != service_endpoint or model_str != service_model:
            mismatches.append(
                f"stage[{index}] endpoint={endpoint_str!r} model={model_str!r} "
                f"!= service endpoint={service_endpoint!r} served_model_name={service_model!r}"
            )

    env["LLM_JUDGE_HYDRA_STAGE_COUNT"] = str(len(judge_stages))
    env["LLM_JUDGE_HYDRA_STAGE_ENDPOINTS"] = _string_list(stage_endpoints)
    env["LLM_JUDGE_HYDRA_STAGE_MODELS"] = _string_list(stage_models)
    env["LLM_JUDGE_SERVICE_ENDPOINT"] = service_endpoint
    env["LLM_JUDGE_SERVICE_MODEL"] = service_model

    if mismatches:
        raise ValueError("LLM judge Hydra stage/service config mismatch:\n" + "\n".join(mismatches))


def validate_async_ranker_training_config(
    ctx: CompilerContext,
    env: dict[str, str],
    overlay_yamls: list[Path],
    final_config: Mapping[str, Any],
) -> None:
    """执行 async ranker training 的静态配置检查。

    这个函数只做 preflight：

    - 检查 overlay 和 LLM judge service config 文件存在。
    - 检查最终 Hydra stage endpoint/model 与 service config endpoint/served model 一致。
    - 用项目内 validator 检查 prompt path。
    - 用 judge launch script 的 `--dry-run` 检查服务配置可解析。

    它不会真正启动 judge 服务，也不会写训练数据。
    """

    if not truthy(env.get("NEEDS_LLM_JUDGE_SERVICE")):
        return
    if not overlay_yamls and not env.get("ASYNC_RANKER_TRAINING_YAML"):
        raise ValueError("LLM judge service is required but no async ranker training overlay YAML was provided")
    llm_judge_config = Path(env["LLM_JUDGE_SERVICE_CONFIG"])
    if not llm_judge_config.is_file():
        raise FileNotFoundError(f"LLM judge service config not found: {llm_judge_config}")
    validate_llm_judge_stage_matches_service_config(
        final_config=final_config,
        env=env,
        service_config_path=llm_judge_config,
    )
    for path in overlay_yamls:
        if not path.is_file():
            raise FileNotFoundError(f"async ranker training YAML not found: {path}")

    prompt_validator = r"""
import sys
from pathlib import Path
from omegaconf import OmegaConf

project_root = Path(sys.argv[1])
sys.path.insert(0, str(project_root))
from async_ranker_training.config import validate_prompt_path

cfg = OmegaConf.merge(*(OmegaConf.load(path) for path in sys.argv[2:]))
stages = OmegaConf.select(cfg, "ranker_training.async_ranker_training.stages") or []
for stage in stages:
    if stage.get("type") != "llm_as_judge":
        continue
    prompt_path = stage.get("prompt", {}).get("path")
    if not prompt_path:
        raise SystemExit("ERROR: llm_as_judge stage is missing prompt.path")
    validate_prompt_path(str(prompt_path), project_root=project_root)
"""
    if overlay_yamls:
        # prompt path 的解析逻辑已经存在于项目 Python 包中，这里通过短脚本复用项目逻辑，
        # 避免 launcher 自己重新实现一套路径规则。
        subprocess.run(
            [sys.executable, "-c", prompt_validator, str(ctx.project_root), *[str(p) for p in overlay_yamls]],
            check=True,
        )
    judge_env = os.environ.copy()
    judge_env["LLM_JUDGE_LOG_DIR"] = str(Path(env["ASYNC_RANKER_TRAINING_LOG_DIR"]) / "judge_server")
    # launch_llm_as_judge.sh --dry-run 只验证服务配置，不启动 vLLM 进程。
    subprocess.run(
        ["bash", str(ctx.project_root / "scripts" / "launch_llm_as_judge.sh"), "--config", str(llm_judge_config), "--dry-run"],
        check=True,
        stdout=subprocess.DEVNULL,
        env=judge_env,
    )
    safe_mkdir(Path(env["ASYNC_RANKER_TRAINING_LOG_DIR"]))
    safe_mkdir(Path(env["ASYNC_RANKER_TRAINING_LOG_DIR"]) / "judge_server")


def collect_canonical_config_paths(config_data: Mapping[str, Any]) -> list[Path]:
    """从最终 Hydra 配置中抽取训练前必须存在的路径。

    compiler 会先从 `hydra_args.txt` compose 出 resolved final config。这里只消费
    这份配置对象，避免为了不同检查重复 compose。
    """

    keys = [
        "actor_rollout_ref.model.path",
        "data.train_files",
        "data.val_files",
        "actor_rollout_ref.rollout.multi_turn.tool_config_path",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path",
    ]
    paths: list[Path] = []
    for key in keys:
        value = select_config_value(dict(config_data), key)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        paths.extend(Path(str(item)) for item in values if item)
    return paths


def infer_needs_llm_judge_service(config_data: Mapping[str, Any]) -> str:
    """从最终 Hydra 配置推导 launcher 是否需要准备 LLM judge 服务。

    这是运行时派生值，不是用户配置。规则是：

    - `ranker_training.async_ranker_training.enable == true`
    - 且 stages 中存在 `type: llm_as_judge`

    满足时返回 `"1"`，否则返回 `"0"`。
    """

    enabled = bool(select_config_value(dict(config_data), "ranker_training.async_ranker_training.enable"))
    stages = select_config_value(dict(config_data), "ranker_training.async_ranker_training.stages") or []
    has_judge = any(isinstance(stage, dict) and stage.get("type") == "llm_as_judge" for stage in stages)
    return "1" if enabled and has_judge else "0"


def check_required_paths(
    ctx: CompilerContext,
    compiled: CompiledConfig,
    final_config: Mapping[str, Any] | None = None,
) -> None:
    """在训练启动前检查关键文件/目录是否存在。

    canonical 模式下通过 Hydra compose 后抽取路径；legacy 模式下直接检查旧 env 中的
    TRAIN_DATA/VAL_DATA/MODEL_PATH 等字段。
    """

    env = compiled.env
    required = [ctx.project_root, Path(env["CORPUS_JSONL"]), Path(env["RECALL_MODEL_PATH"])]
    if compiled.canonical:
        if final_config is None:
            raise ValueError("final_config is required for canonical path checks")
        required.extend(collect_canonical_config_paths(final_config))
    else:
        required.extend([Path(env["TRAIN_DATA"]), Path(env["VAL_DATA"]), Path(env["MODEL_PATH"])])
    if env.get("RANKER_BASE_MODEL_PATH"):
        required.append(Path(env["RANKER_BASE_MODEL_PATH"]))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required path not found:\n" + "\n".join(missing))
