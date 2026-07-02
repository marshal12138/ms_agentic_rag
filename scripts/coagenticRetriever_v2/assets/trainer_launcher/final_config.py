"""Final Hydra config composition and export helpers.

这个模块只负责一件事：把 launcher compiler 已经写好的 `hydra_args.txt` 重新 compose
成最终完整 Hydra 配置，并把 resolved 后的同一份 Python mapping 同时导出为 YAML 和
JSON。

边界说明：

- 不解析 task CLI。
- 不合并 main_run/resource/overlay。
- 不生成新的 Hydra override。
- 不启动服务、不等待 GPU、不执行训练。

为什么要单独做这个模块：

- `hydra_args.txt` 是 canonical 训练入口真正使用的参数序列。
- 由这个文件 compose 出来的配置，才是训练进程最终看到的完整配置。
- YAML 和 JSON 必须来自同一个 resolved container，这样两个文件内容能一一对应，
  只存在序列化格式差异。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _read_hydra_args(hydra_args_file: Path) -> tuple[str, list[str]]:
    """读取 `hydra_args.txt`，拆出 Hydra main config 和 overrides。

    launcher 写入的第一类参数是 `--config-name=...`，其余都是 Hydra override。这里
    不重新理解 overlay 语义，只忠实消费最终参数文件。
    """

    config_name = "coagentic_retriever_trainer"
    overrides: list[str] = []
    for line in hydra_args_file.read_text(encoding="utf-8").splitlines():
        arg = line.strip()
        if not arg:
            continue
        if arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]
        else:
            overrides.append(arg)
    return config_name, overrides


def compose_final_config(project_root: Path, hydra_args_file: Path) -> dict[str, Any]:
    """compose 最终 Hydra 配置并返回 resolved Python dict。

    返回值会用于两个地方：

    - 写 `.final_config.yaml` 和 `.final_config.json`。
    - 做 compiler 阶段的静态检查和服务需求推导。
    """

    config_name, overrides = _read_hydra_args(hydra_args_file)
    with initialize_config_dir(config_dir=str(project_root / "config"), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise TypeError("final Hydra config must resolve to a mapping")
    return data


def select_config_value(config_data: dict[str, Any], dotted_key: str) -> Any:
    """从 resolved dict 中按 dot path 读取值。

    这里只支持普通 mapping/list traversal，足够覆盖 launcher preflight 需要的路径。
    如果中间节点不存在，返回 None。
    """

    value: Any = config_data
    for part in dotted_key.split("."):
        if isinstance(value, dict):
            value = value.get(part)
            continue
        return None
    return value
