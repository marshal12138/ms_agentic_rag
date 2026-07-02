"""Resource environment merging for CoAgenticRetriever train tasks.

resource YAML 被定义为“task 运行环境默认值集合”，字段名直接使用环境变量名。
这里实现完整优先级：

`resource/base.yaml < resource/<selected>.yaml < 普通 OVERLAY_YAML < 显式外部 env`

注意：这里的 overlay 只抽取两类内容：

- 顶层环境变量名，例如 `AGENT_GPU_IDS`。
- 兼容字段 `resources.agent_gpu_ids` 等，它们会映射回环境变量。

抽取出的环境变量用于 Bash 运行态；这些顶层 env 名不会再写入 Hydra 配置树。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from .paths import normalize_config_group_value
from .yaml_utils import load_mapping

# resource YAML 当前允许管理的环境变量名。
#
# 字段名故意保持大写环境变量形式，不再映射成新的结构化命名。这样 task 脚本顶部的
# 显式配置、resource YAML、`.env` 审计文件三者能一一对应。
RESOURCE_KEYS = [
    "GROUP_NAME",
    "AGENT_GPU_IDS",
    "RANK_GPU_ID",
    "RECALL_GPU_ID",
    "LLM_JUDGE_GPU_IDS",
    "AUTO_START_RECALL_SERVICE",
    "AUTO_STOP_RECALL_SERVICE",
    "RECALL_SERVICE_WAIT_SECONDS",
    "AUTO_START_LLM_JUDGE",
    "AUTO_STOP_LLM_JUDGE",
    "LLM_JUDGE_PREFLIGHT",
    "LLM_JUDGE_WAIT_SECONDS",
    "WAIT_FOR_GPUS",
    "WAIT_FOR_GPU_RELEASE",
    "WAIT_FOR_GPU_INTERVAL_SECONDS",
    "WAIT_FOR_GPU_LABEL",
]

# canonical 模式必须提供的 resource 字段。
#
# `WAIT_FOR_GPUS` 没有放进 required，因为它可以由最终有效的 AGENT/RANK/RECALL/JUDGE
# GPU 列表自动拼出。
REQUIRED_CANONICAL_RESOURCE_KEYS = [
    "GROUP_NAME",
    "AGENT_GPU_IDS",
    "RANK_GPU_ID",
    "RECALL_GPU_ID",
    "LLM_JUDGE_GPU_IDS",
    "AUTO_START_RECALL_SERVICE",
    "AUTO_STOP_RECALL_SERVICE",
    "RECALL_SERVICE_WAIT_SECONDS",
    "AUTO_START_LLM_JUDGE",
    "AUTO_STOP_LLM_JUDGE",
    "LLM_JUDGE_PREFLIGHT",
    "LLM_JUDGE_WAIT_SECONDS",
    "WAIT_FOR_GPU_RELEASE",
    "WAIT_FOR_GPU_INTERVAL_SECONDS",
    "WAIT_FOR_GPU_LABEL",
]

# 兼容旧 overlay 中 `resources.*` 写法到环境变量名的映射。
RESOURCES_MAPPING = {
    "agent_gpu_ids": "AGENT_GPU_IDS",
    "rank_gpu_id": "RANK_GPU_ID",
    "recall_gpu_id": "RECALL_GPU_ID",
    "llm_judge_gpu_ids": "LLM_JUDGE_GPU_IDS",
}


def _merge_from_mapping(effective: dict[str, str], data: Mapping[str, Any], *, source: Path) -> None:
    """从一个 YAML mapping 中抽取 resource 环境变量。

    支持两种写法：

    - 顶层大写环境变量名，例如 `AGENT_GPU_IDS: "0,1,2,3"`。
    - 兼容旧写法 `resources.agent_gpu_ids`。

    后读入的 YAML 会覆盖先读入的 YAML，外层函数负责控制读取顺序。
    """

    for key in RESOURCE_KEYS:
        value = data.get(key)
        if value is not None:
            effective[key] = str(value)

    resources = data.get("resources")
    if resources is None:
        return
    if not isinstance(resources, dict):
        raise TypeError(f"resources must be a mapping in {source}")
    for key, env_name in RESOURCES_MAPPING.items():
        value = resources.get(key)
        if value is not None:
            effective[env_name] = str(value)


def load_canonical_resource_env(
    *,
    repo_root: Path,
    project_root: Path,
    resource_config: str,
    overlay_yamls: list[Path],
    environ: Mapping[str, str] | None = None,
) -> tuple[str, Path, Path, dict[str, str]]:
    """读取 canonical resource 配置并返回最终有效的环境变量。

    优先级在这里完整实现：

    `resource/base.yaml < resource/<selected>.yaml < 普通 OVERLAY_YAML < 显式外部 env`

    注意这里返回的是 Bash 运行态 env，不是 Hydra 配置树。后续 runtime override 会把
    最终 GPU 分配同步写入 `resources.*`，保证 shell runtime 和训练配置一致。
    """

    env = environ if environ is not None else os.environ
    base_path = project_root / "config" / "resource" / "base.yaml"
    selected_name = normalize_config_group_value(
        repo_root,
        project_root,
        option="--RESOURCE_CONFIG",
        group="resource",
        value=resource_config or "local_8gpu_0_7",
    )
    selected_path = project_root / "config" / "resource" / f"{selected_name}.yaml"
    effective: dict[str, str] = {}

    # base 提供字段集合，selected resource 提供机器默认值，普通 overlay 可以覆盖
    # 运行形态，最后外部 env 作为最高优先级的调试/临时覆盖。
    for path in [base_path, selected_path, *overlay_yamls]:
        data = load_mapping(path, label="resource/overlay YAML")
        _merge_from_mapping(effective, data, source=path)

    for key in RESOURCE_KEYS:
        if key in env:
            effective[key] = env[key]

    missing = [key for key in REQUIRED_CANONICAL_RESOURCE_KEYS if not effective.get(key)]
    if missing:
        raise ValueError(
            "canonical resource config did not provide required keys: " + ", ".join(missing)
        )
    return selected_name, base_path, selected_path, effective


def filter_resource_env_overlay_args(args: list[str]) -> list[str]:
    """从 Hydra dotlist 中移除顶层 resource 环境变量 override。

    普通 overlay 中允许出现 `AGENT_GPU_IDS` 这类顶层字段，用来覆盖 resource env；
    但这些字段不是 Hydra trainer 配置的一部分，不能继续传给 Hydra compose。
    """

    prefixes = tuple(f"++{key}=" for key in RESOURCE_KEYS)
    return [arg for arg in args if not arg.startswith(prefixes)]
