"""YAML loading and Hydra dotlist conversion for launcher overlays.

这个模块承接原 Bash launcher 中两类容易出错的逻辑：

1. 读取 main_run/resource/overlay YAML，并确保顶层结构是 mapping。
2. 把普通 overlay YAML 展平成 Hydra dotlist override。

这里的 dotlist 转换规则刻意和 `src/hydra_overrides/yaml_to_dotlist.py` 保持一致：
每个叶子节点输出一条 `++a.b.c=value`。launcher compiler 会用这个结果写
`hydra_args.txt`，asset runner 最终只读取该文件，不再重新理解 overlay YAML。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on runtime image.
    yaml = None

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ModuleNotFoundError:  # pragma: no cover - depends on runtime image.
    DictConfig = ListConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

_HYDRA_DICT_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _to_plain(value: Any) -> Any:
    """把 OmegaConf 节点转成普通 Python 容器，便于统一处理。"""

    if OmegaConf is not None and isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def load_yaml(path: Path) -> Any:
    """读取 YAML 文件，优先用 PyYAML，缺失时回退到 OmegaConf。

    launcher compiler 运行环境可能不完全一致，因此这里提供两套读取路径。但读取后的
    结果都会转换成普通 Python 对象，避免后续模块依赖具体 YAML 库。
    """

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"YAML path is not a file: {path}")
    if yaml is not None:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if OmegaConf is not None:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    raise RuntimeError("PyYAML or OmegaConf is required to read YAML files")


def load_mapping(path: Path, *, label: str = "YAML") -> dict[str, Any]:
    """读取 YAML，并要求顶层必须是 mapping。

    main_run/resource/overlay 都按 partial mapping 处理；如果顶层是 list 或 scalar，
    说明该文件不适合作为 launcher 配置输入。
    """

    data = load_yaml(path) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{label} must contain a mapping at top level: {path}")
    return data


def dump_mapping(path: Path, data: dict[str, Any]) -> None:
    """把 mapping 写成 YAML。

    PyYAML 生成的文本更接近普通 YAML，优先使用；OmegaConf 只是运行镜像缺少 PyYAML
    时的兜底。
    """

    if yaml is not None:
        text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        path.write_text(text, encoding="utf-8")
        return
    if OmegaConf is not None:  # pragma: no cover - fallback only.
        OmegaConf.save(config=OmegaConf.create(data), f=str(path))
        return
    raise RuntimeError("PyYAML or OmegaConf is required to write YAML files")


def _format_string(value: str) -> str:
    """用 JSON 字符串规则格式化 Hydra 字符串值，保留空格和特殊字符。"""

    return json.dumps(value, ensure_ascii=False)


def _format_scalar(value: Any) -> str:
    """把 YAML 标量格式化成 Hydra dotlist 可接受的字面量。"""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _format_string(value)
    raise TypeError(f"unsupported scalar type: {type(value).__name__}")


def _format_hydra_value(value: Any) -> str:
    """格式化一个 YAML 值为 Hydra override 右侧表达式。

    dict/list 会被格式化为 Hydra inline container；scalar 走标量格式化。这里会检查
    dict key 是否安全，避免生成 Hydra 无法解析或含歧义的表达式。
    """

    value = _to_plain(value)
    if isinstance(value, dict):
        parts = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"YAML mapping keys must be strings, got {key!r}")
            if not _HYDRA_DICT_KEY_RE.match(key):
                raise ValueError(f"YAML mapping key is not safe for Hydra dict syntax: {key!r}")
            parts.append(f"{key}:{_format_hydra_value(child)}")
        return "{" + ",".join(parts) + "}"
    if isinstance(value, list):
        return "[" + ",".join(_format_hydra_value(item) for item in value) + "]"
    return _format_scalar(value)


def _flatten(prefix: str, value: Any) -> list[tuple[str, Any]]:
    """把嵌套 mapping 展平为 `(dot.path, leaf_value)` 列表。"""

    value = _to_plain(value)
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"YAML mapping keys must be strings, got {key!r}")
            child_key = f"{prefix}.{key}" if prefix else key
            items.extend(_flatten(child_key, child))
        return items
    return [(prefix, value)]


def yaml_to_overrides(paths: Iterable[Path], *, prefix: str = "++") -> list[str]:
    """把 partial overlay YAML 按输入顺序转换为 Hydra dotlist override。

    overlay YAML 不能包含 Hydra `defaults`，因为 defaults 属于 main config/group 组合
    语义；launcher overlay 只允许表达具体字段覆盖。
    """

    overrides: list[str] = []
    for path in paths:
        data = load_mapping(path, label="override YAML")
        if "defaults" in data:
            raise ValueError(
                f"{path} contains a Hydra defaults section. "
                "Launcher overlay YAML must be partial value overrides."
            )
        for key, value in _flatten("", data):
            if key:
                overrides.append(f"{prefix}{key}={_format_hydra_value(value)}")
    return overrides
