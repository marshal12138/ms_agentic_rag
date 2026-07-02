"""Launcher main-run config loading and merge rules.

`CoAgenticRetriever/config/main_run/coAgenticRetriever_main.yaml` 是 launcher 级
运行 manifest。它不是 Hydra defaults，也不会直接参与训练配置 compose；它只是给
launcher 一个可审计的默认选择集合，例如 trainer main config、data/model/rollout
group、run_mode、resource config、service config 和默认 overlay 列表。

合并规则在这里固定下来：

- main_run_config 提供默认基线。
- task 脚本显式传入的 launcher 参数覆盖 main_run_config。
- task 末尾的 Hydra CLI override 不在这里处理，它们会在 hydra_args.txt 最后出现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .cli import LauncherSelection
from .paths import MainRunConfigRef, normalize_main_run_config, resolve_repo_path
from .yaml_utils import load_mapping


@dataclass
class MainRunManifest:
    """main_run_config YAML 加载后的结构化 manifest。

    这个对象保存的是 launcher 级默认选择，不是最终训练配置。task 显式传入的
    launcher 参数仍然可以覆盖这里的字段。
    """

    ref: MainRunConfigRef | None = None
    trainer_main_hydra_config: str = ""
    data_config: str = ""
    model_config: str = ""
    rollout_config: str = ""
    ranker_base_config: str = ""
    async_ranker_training_base_config: str = ""
    run_mode: str = ""
    resource_config: str = ""
    llm_judge_service_config: Path | None = None
    tool_config: Path | None = None
    overlay_yamls: list[Path] = field(default_factory=list)
    trainer_cli_overrides: list[str] = field(default_factory=list)


def _scalar(data: dict[str, Any], key: str) -> str:
    """从 YAML mapping 中读取一个标量字段。

    main_run_config 中的选择项必须是短字符串或路径；如果误写成 list/dict，说明配置
    层级设计错了，应该直接报错。
    """

    value = data.get(key)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        raise TypeError(f"{key} in main_run_config must be a scalar")
    return str(value)


def load_main_run_manifest(repo_root: Path, project_root: Path, value: str) -> MainRunManifest:
    """读取并规范化 main_run_config manifest。

    这里只解析 launcher 需要理解的字段：

    - `trainer_main_hydra_config`
    - `trainer_config_groups`
    - `run_mode`
    - `resource_config`
    - `service_configs`
    - `runtime_configs`
    - `overlay_yamls`
    - `trainer_cli_overrides`

    未在这里建模的字段不会参与 launcher 编译，避免 main_run_config 变成新的隐式
    Hydra 配置树。
    """

    ref = normalize_main_run_config(repo_root, project_root, value)
    data = load_mapping(ref.path, label="main_run_config")
    groups = data.get("trainer_config_groups") or {}
    services = data.get("service_configs") or {}
    runtime = data.get("runtime_configs") or {}

    if not isinstance(groups, dict):
        raise TypeError("trainer_config_groups in main_run_config must be a mapping")
    if not isinstance(services, dict):
        raise TypeError("service_configs in main_run_config must be a mapping")
    if not isinstance(runtime, dict):
        raise TypeError("runtime_configs in main_run_config must be a mapping")

    manifest = MainRunManifest(
        ref=ref,
        trainer_main_hydra_config=_scalar(data, "trainer_main_hydra_config"),
        data_config=_scalar(groups, "data"),
        model_config=_scalar(groups, "model"),
        rollout_config=_scalar(groups, "rollout"),
        ranker_base_config=_scalar(groups, "ranker_base"),
        async_ranker_training_base_config=_scalar(groups, "async_ranker_training_base"),
        run_mode=_scalar(data, "run_mode"),
        resource_config=_scalar(data, "resource_config"),
    )

    llm_judge_service_config = services.get("llm_judge_service_config")
    if llm_judge_service_config:
        if isinstance(llm_judge_service_config, (dict, list)):
            raise TypeError("service_configs.llm_judge_service_config must be a scalar")
        manifest.llm_judge_service_config = resolve_repo_path(repo_root, str(llm_judge_service_config))

    tool_config = runtime.get("tool_config")
    if tool_config:
        if isinstance(tool_config, (dict, list)):
            raise TypeError("runtime_configs.tool_config must be a scalar")
        manifest.tool_config = resolve_repo_path(repo_root, str(tool_config))

    overlay_yamls = data.get("overlay_yamls") or []
    if not isinstance(overlay_yamls, list):
        raise TypeError("overlay_yamls in main_run_config must be a list")
    for value in overlay_yamls:
        if isinstance(value, (dict, list)):
            raise TypeError("overlay_yamls entries must be scalars")
        manifest.overlay_yamls.append(resolve_repo_path(repo_root, str(value)))

    trainer_cli_overrides = data.get("trainer_cli_overrides") or []
    if not isinstance(trainer_cli_overrides, list):
        raise TypeError("trainer_cli_overrides in main_run_config must be a list")
    for value in trainer_cli_overrides:
        if isinstance(value, (dict, list)):
            raise TypeError("trainer_cli_overrides entries must be scalars")
        manifest.trainer_cli_overrides.append(str(value))

    return manifest


def merge_main_run_selection(
    *,
    repo_root: Path,
    project_root: Path,
    selection: LauncherSelection,
) -> tuple[LauncherSelection, MainRunManifest]:
    """把 main_run_config 默认值合并到 CLI selection 中。

    合并规则很保守：main_run_config 只提供默认值，task 脚本显式传入的 launcher 参数
    一定覆盖它。这样 task 仍然保留“这次实验显式选择了什么”的可读性。
    """

    manifest = MainRunManifest()
    if selection.main_run_config:
        manifest = load_main_run_manifest(repo_root, project_root, selection.main_run_config)

    merged = LauncherSelection(
        main_run_config=manifest.ref.name if manifest.ref else "",
        trainer_main_hydra_config=selection.trainer_main_hydra_config or manifest.trainer_main_hydra_config,
        data_config=selection.data_config or manifest.data_config,
        model_config=selection.model_config or manifest.model_config,
        rollout_config=selection.rollout_config or manifest.rollout_config,
        ranker_base_config=selection.ranker_base_config or manifest.ranker_base_config,
        async_ranker_training_base_config=(
            selection.async_ranker_training_base_config or manifest.async_ranker_training_base_config
        ),
        run_mode=selection.run_mode or manifest.run_mode,
        resource_config=selection.resource_config or manifest.resource_config,
        overlay_yamls=selection.overlay_yamls or list(manifest.overlay_yamls),
        llm_judge_service_config=selection.llm_judge_service_config or manifest.llm_judge_service_config,
        trainer_cli_overrides=[],
    )
    for value in [*manifest.trainer_cli_overrides, *selection.trainer_cli_overrides]:
        if value not in merged.trainer_cli_overrides:
            merged.trainer_cli_overrides.append(value)
    return merged, manifest
