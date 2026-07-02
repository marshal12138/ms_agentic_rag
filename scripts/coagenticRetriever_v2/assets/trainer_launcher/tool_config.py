"""Static tool-config reader used by the train launcher compiler.

CoAgenticRetriever 的检索工具配置中已有 retrieval service URL、top_n/top_m、
ranker_enabled 等信息。launcher 需要这些值来启动/预检服务，并把同一组值写进
审计文件。读取逻辑集中在这里，避免 Bash 通过内联 Python 片段重复解析 YAML。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from .yaml_utils import dump_mapping, load_mapping


@dataclass(frozen=True)
class StaticToolConfig:
    """launcher 从 tool config 中关心的静态字段。

    这不是完整 tool schema，只抽取启动服务和写审计文件需要的少数字段。
    """

    retrieval_service_url: str = ""
    retrieval_port: str = ""
    default_top_n: str = ""
    default_top_m: str = ""
    class_name: str = ""
    max_concurrent_per_worker: str = ""
    ranker_enabled: str = ""
    format_penalty: str = ""


def read_static_tool_config(path: Path) -> StaticToolConfig:
    """读取 `coagentic_retriever_tool_config.yaml` 的第一个 tool entry。

    当前训练任务只使用一个检索工具，因此 launcher 只抽取 `tools[0]`。如果未来支持多
    tool，这里应该改成按 tool name 查找，而不是在 launcher 其它地方散落解析逻辑。
    """

    data = load_mapping(path, label="tool config")
    tools = data.get("tools") or [{}]
    if not isinstance(tools, list) or not tools:
        raise TypeError(f"tool config must contain a non-empty tools list: {path}")
    tool = tools[0]
    if not isinstance(tool, dict):
        raise TypeError(f"first tool entry must be a mapping: {path}")
    config = tool.get("config") or {}
    if not isinstance(config, dict):
        raise TypeError(f"first tool config must be a mapping: {path}")

    retrieval_service_url = str(config.get("retrieval_service_url") or "")
    parsed_port = urlparse(retrieval_service_url).port
    return StaticToolConfig(
        retrieval_service_url=retrieval_service_url,
        retrieval_port=str(parsed_port or ""),
        default_top_n=str(config.get("default_top_n") or ""),
        default_top_m=str(config.get("default_top_m") or ""),
        class_name=str(tool.get("class_name") or ""),
        max_concurrent_per_worker=str(config.get("max_concurrent_per_worker") or ""),
        ranker_enabled=str(config.get("ranker_enabled")).lower()
        if config.get("ranker_enabled") is not None
        else "",
        format_penalty=str(config.get("format_penalty") or ""),
    )


def write_runtime_tool_config_from_hydra_actor(
    *,
    source_path: Path,
    output_path: Path,
    actor_name: object,
    actor_namespace: object,
) -> None:
    """写本次 run 专用 tool config，并用 Hydra shared ranker actor 字段覆盖 actor 标识。

    这样 full 模式下 actor_name/actor_namespace 的事实源只有最终 Hydra 配置中的
    `ranker_training.shared_inference_ranker`。静态 tool YAML 仍提供其它 tool 参数，
    但不再决定共享 ranker actor 名字。
    """

    data = load_mapping(source_path, label="tool config")
    tools = data.get("tools") or []
    if not isinstance(tools, list) or not tools:
        raise TypeError(f"tool config must contain a non-empty tools list: {source_path}")
    tool = tools[0]
    if not isinstance(tool, dict):
        raise TypeError(f"first tool entry must be a mapping: {source_path}")
    config = tool.get("config")
    if not isinstance(config, dict):
        raise TypeError(f"first tool config must be a mapping: {source_path}")
    ranker_config = config.get("ranker")
    if not isinstance(ranker_config, dict):
        raise TypeError(f"first tool ranker config must be a mapping: {source_path}")

    ranker_config["actor_name"] = "" if actor_name is None else str(actor_name)
    ranker_config["actor_namespace"] = actor_namespace
    dump_mapping(output_path, data)
