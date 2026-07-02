"""Path and config-name normalization for the train launcher compiler.

task 脚本为了可读性会显式写出 `--DATA_CONFIG=qwen3_4b`、`--OVERLAY_YAML=...`
这类参数；使用者也可能传完整 YAML 路径。这个模块统一负责把这些输入规整成
确定的仓库内路径或 Hydra group 短名。其它模块不再猜测“短名/文件名/路径”。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_UNSAFE_NAME_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]")


def slugify_name(raw: str | None, *, max_len: int = 0) -> str:
    """生成可用于目录名/文件名前缀的 run 名称片段。

    这里保持和旧 Bash `slugify_cosearch_name` 一致，避免迁移到 Python compiler 后
    `RUN_NAME`、`LOG_DIR`、checkpoint 目录命名发生隐式变化。
    """

    value = raw or "default"
    if max_len > 0:
        value = value[:max_len]
    value = _UNSAFE_NAME_CHARS_RE.sub("_", value)
    value = value.strip("._-")
    return value or "default"


def resolve_repo_path(repo_root: Path, value: str) -> Path:
    """把绝对路径或仓库相对路径规整为绝对 Path。

    task 脚本通常使用仓库相对路径以便阅读；用户调试时也可能直接传绝对路径。后续
    模块只处理规整后的 Path，不再重复判断路径形态。
    """

    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


@dataclass(frozen=True)
class MainRunConfigRef:
    """main_run_config 的规范化引用。

    `name` 用于写入审计文件，`path` 用于实际读取 YAML。
    """

    name: str
    path: Path


def normalize_main_run_config(repo_root: Path, project_root: Path, value: str) -> MainRunConfigRef:
    """解析 main_run_config 短名或路径。

    约定短名 `coAgenticRetriever_main` 对应
    `CoAgenticRetriever/config/main_run/coAgenticRetriever_main.yaml`。如果调用方
    传入路径，则允许读取仓库内任意 YAML，便于临时实验。
    """

    config_dir = project_root / "config" / "main_run"
    if "/" in value:
        path = resolve_repo_path(repo_root, value)
    elif value.endswith((".yaml", ".yml")):
        path = config_dir / value
    else:
        path = config_dir / f"{value}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"--main_run_config config file not found: {path}")
    try:
        rel = path.relative_to(config_dir)
        name = str(rel)
    except ValueError:
        name = path.name
    name = name.removesuffix(".yaml").removesuffix(".yml")
    return MainRunConfigRef(name=name, path=path)


def normalize_trainer_main_hydra_config(repo_root: Path, project_root: Path, value: str) -> str:
    """解析 trainer main Hydra config，并返回 Hydra `--config-name` 短名。

    trainer main config 必须位于 `CoAgenticRetriever/config/` 顶层。这个限制是刻意的：
    它区分“Hydra 主配置”和 data/model/rollout 这类 config group，避免 main config
    被误写成 group 文件。
    """

    config_dir = project_root / "config"
    if "/" in value:
        path = resolve_repo_path(repo_root, value)
    elif value.endswith((".yaml", ".yml")):
        path = config_dir / value
    else:
        path = config_dir / f"{value}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"--trainer_main_hydra_config config file not found: {path}")
    try:
        rel = path.relative_to(config_dir)
    except ValueError as exc:
        raise ValueError(f"--trainer_main_hydra_config path must be under {config_dir}: {path}") from exc
    if len(rel.parts) != 1:
        raise ValueError(
            "--trainer_main_hydra_config must point to a top-level Hydra main config "
            f"under {config_dir}: {path}"
        )
    if rel.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"--trainer_main_hydra_config must be a YAML file: {path}")
    return rel.name.removesuffix(".yaml").removesuffix(".yml")


def normalize_config_group_value(
    repo_root: Path,
    project_root: Path,
    *,
    option: str,
    group: str,
    value: str,
) -> str:
    """把 Hydra config group 选择规整成 group 内短名。

    支持三种输入：

    - `qwen3_4b`
    - `qwen3_4b.yaml`
    - `CoAgenticRetriever/config/model/qwen3_4b.yaml`

    返回值始终是 Hydra override 需要的 group-local 名称，例如 `qwen3_4b`。
    """

    group_dir = project_root / "config" / group
    if "/" in value:
        path = resolve_repo_path(repo_root, value)
    elif value.endswith((".yaml", ".yml")):
        path = group_dir / value
    else:
        path = group_dir / f"{value}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{option} config file not found: {path}")

    if "/" in value or value.endswith((".yaml", ".yml")):
        try:
            rel = path.relative_to(group_dir)
        except ValueError as exc:
            raise ValueError(f"{option} path must be under {group_dir}: {path}") from exc
        if rel.suffix not in {".yaml", ".yml"}:
            raise ValueError(f"{option} must be a YAML file: {path}")
        return str(rel).removesuffix(".yaml").removesuffix(".yml")
    return value
