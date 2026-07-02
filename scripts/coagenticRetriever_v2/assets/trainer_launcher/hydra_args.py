"""Build canonical Hydra argument files for the v2 train launcher.

canonical 模式下，Bash launcher 不再拼接长串 Hydra CLI 参数。Python compiler 会
把最终参数顺序写到 `hydra_args.txt`：

1. Trainer Hydra main config。
2. data/model/rollout/ranker/async-ranker config group。
3. 普通 overlay YAML 展平后的参数。
4. launcher-only run_mode 编译出的具体训练语义 override。
5. 本次运行才知道的 runtime override YAML 展平后的参数。
6. task 末尾临时 Hydra CLI override。

这个顺序就是优先级；越靠后越能覆盖前面的值。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cli import LauncherSelection
from .paths import normalize_config_group_value, normalize_trainer_main_hydra_config
from .resource import filter_resource_env_overlay_args
from .run_mode import RUN_MODE_KEYS
from .yaml_utils import yaml_to_overrides


@dataclass(frozen=True)
class CanonicalHydraSelection:
    """canonical 模式下最终采用的 Hydra main config 和 config group。

    这里保存的是已经校验过的短名和 Hydra CLI 片段。后续 writer 只负责写文件，不再
    重新解析 config group。
    """

    trainer_main_hydra_config: str
    data_config: str
    model_config: str
    rollout_config: str
    ranker_base_config: str
    async_ranker_training_base_config: str
    trainer_main_arg: str
    group_args: list[str]


def normalize_canonical_selection(
    *,
    repo_root: Path,
    project_root: Path,
    selection: LauncherSelection,
) -> CanonicalHydraSelection:
    """校验并规范化所有 Hydra config 选择。

    data/model/rollout 是 canonical 模式必须显式给出的实验核心配置；ranker 和
    async_ranker_training 允许使用默认 base，因为它们是当前 launcher 的标准训练形态
    组成部分。
    """

    trainer_main = normalize_trainer_main_hydra_config(
        repo_root,
        project_root,
        selection.trainer_main_hydra_config or "coagentic_retriever_trainer",
    )
    if not selection.data_config:
        raise ValueError("--DATA_CONFIG is required in canonical config mode")
    if not selection.model_config:
        raise ValueError("--MODEL_CONFIG is required in canonical config mode")
    if not selection.rollout_config:
        raise ValueError("--ROLLOUT_CONFIG is required in canonical config mode")

    data = normalize_config_group_value(
        repo_root, project_root, option="--DATA_CONFIG", group="data", value=selection.data_config
    )
    model = normalize_config_group_value(
        repo_root, project_root, option="--MODEL_CONFIG", group="model", value=selection.model_config
    )
    rollout = normalize_config_group_value(
        repo_root,
        project_root,
        option="--ROLLOUT_CONFIG",
        group="rollout",
        value=selection.rollout_config,
    )
    ranker_base = normalize_config_group_value(
        repo_root,
        project_root,
        option="--RANKER_BASE_CONFIG",
        group="experimental/ranker_base",
        value=selection.ranker_base_config or "ranker_contrastive",
    )
    async_ranker_training_base = normalize_config_group_value(
        repo_root,
        project_root,
        option="--ASYNC_RANKER_TRAINING_BASE_CONFIG",
        group="experimental/async_ranker_training_base",
        value=selection.async_ranker_training_base_config or "async_ranker_training",
    )
    group_args = [
        f"data={data}",
        f"model@actor_rollout_ref.model={model}",
        f"rollout@actor_rollout_ref.rollout={rollout}",
        f"experimental/ranker_base@_global_={ranker_base}",
        f"experimental/async_ranker_training_base@_global_={async_ranker_training_base}",
    ]
    return CanonicalHydraSelection(
        trainer_main_hydra_config=trainer_main,
        data_config=data,
        model_config=model,
        rollout_config=rollout,
        ranker_base_config=ranker_base,
        async_ranker_training_base_config=async_ranker_training_base,
        trainer_main_arg=f"--config-name={trainer_main}",
        group_args=group_args,
    )


def build_hydra_args(
    *,
    hydra_selection: CanonicalHydraSelection,
    overlay_yamls: list[Path],
    run_mode_override_yaml: Path,
    runtime_override_yaml: Path,
    trainer_cli_overrides: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """生成最终 Hydra 参数列表，并返回中间 dotlist 便于审计。

    参数顺序不能随意改动：

    1. `--config-name`
    2. Hydra config group 选择
    3. 普通 overlay YAML 展平参数
    4. run_mode override YAML 展平参数
    5. runtime override YAML 展平参数
    6. task 末尾显式 Hydra CLI override

    这也是最终覆盖优先级，越靠后优先级越高。
    """

    for path in overlay_yamls:
        if not path.is_file():
            raise FileNotFoundError(f"canonical overlay YAML not found: {path}")
    # overlay 中的顶层 resource env 字段只参与 shell/resource 合并，不能作为 Hydra
    # 参数传入，否则 Hydra 会收到并不存在的顶层配置 key。
    overlay_args = filter_run_mode_overlay_args(filter_resource_env_overlay_args(yaml_to_overrides(overlay_yamls)))
    run_mode_args = yaml_to_overrides([run_mode_override_yaml]) if run_mode_override_yaml.is_file() else []
    runtime_args = yaml_to_overrides([runtime_override_yaml])
    hydra_args = [
        hydra_selection.trainer_main_arg,
        *hydra_selection.group_args,
        *overlay_args,
        *run_mode_args,
        *runtime_args,
        *trainer_cli_overrides,
    ]
    return hydra_args, overlay_args, runtime_args


def filter_run_mode_overlay_args(args: list[str]) -> list[str]:
    """从 Hydra dotlist 中移除 launcher-only run_mode 字段。

    `run_mode` 可以写在普通 overlay YAML 中，但它不是 trainer 配置字段。compiler 会先
    读取它并生成具体训练语义 override，因此这里不能再把 `++run_mode=...` 传给 Hydra。
    """

    prefixes = tuple(f"++{key}=" for key in RUN_MODE_KEYS)
    return [arg for arg in args if not arg.startswith(prefixes)]
