"""Static tool-config reader used by the train launcher compiler.

CoAgenticRetriever 的静态检索工具配置只维护 retrieval service URL、agent 可见
文档数、ranker_enabled 等工具侧信息。recall 候选池大小和 ranker 保留多少篇由
Hydra ranker base 管理，再由 launcher 生成进 runtime tool config。
读取逻辑集中在这里，避免 Bash 通过内联 Python 片段重复解析 YAML。
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
    search_tool_final_top_m: str = ""
    class_name: str = ""
    max_concurrent_per_worker: str = ""
    ranker_enabled: str = ""


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
    ranker_config = config.get("ranker") or {}
    if not isinstance(ranker_config, dict):
        raise TypeError(f"first tool ranker config must be a mapping: {path}")

    _reject_deprecated_static_tool_fields(
        source_path=path,
        config=config,
        tool_ranker_config=ranker_config,
    )

    retrieval_service_url = str(config.get("retrieval_service_url") or "")
    parsed_port = urlparse(retrieval_service_url).port
    return StaticToolConfig(
        retrieval_service_url=retrieval_service_url,
        retrieval_port=str(parsed_port or ""),
        search_tool_final_top_m=str(config.get("searchTool_final_top_m") or ""),
        class_name=str(tool.get("class_name") or ""),
        max_concurrent_per_worker=str(config.get("max_concurrent_per_worker") or ""),
        ranker_enabled=str(config.get("ranker_enabled")).lower()
        if config.get("ranker_enabled") is not None
        else "",
    )


def _reject_deprecated_static_tool_fields(
    *,
    source_path: Path,
    config: dict[str, object],
    tool_ranker_config: dict[str, object],
) -> None:
    deprecated_tool_fields = [key for key in ("default_top_n", "default_top_m", "format_penalty") if key in config]
    allowed_static_ranker_fields = {"backend", "required"}
    unsupported_ranker_fields = [key for key in tool_ranker_config if key not in allowed_static_ranker_fields]
    if not deprecated_tool_fields and not unsupported_ranker_fields:
        return

    problems: list[str] = []
    if deprecated_tool_fields:
        fields = ", ".join(f"config.{key}" for key in deprecated_tool_fields)
        problems.append(f"deprecated static tool top fields: {fields}")
    if unsupported_ranker_fields:
        fields = ", ".join(f"config.ranker.{key}" for key in unsupported_ranker_fields)
        problems.append(f"ranker fields that must be generated from Hydra, not static tool config: {fields}")
    raise ValueError(
        f"{source_path} still contains training-incompatible legacy tool fields: "
        + "; ".join(problems)
        + ". Use config.searchTool_final_top_m in the static tool config; "
        "use Hydra recall_retriever.recall_final_top_n, ranker.top_k, "
        "ranker token lengths, ranker_training.shared_inference_ranker, and "
        "custom_reward_function.reward_kwargs.format_penalty for generated/runtime fields."
    )


def write_runtime_tool_config_from_hydra_ranker(
    *,
    source_path: Path,
    output_path: Path,
    actor_name: object,
    actor_namespace: object,
    hydra_recall_retriever_config: dict[str, object],
    hydra_ranker_config: dict[str, object],
) -> None:
    """写本次 run 专用 tool config，并从最终 Hydra 配置注入 tool 调用参数。

    静态 tool YAML 只维护 retrieval、tool 并发、agent-visible top-M 和 ranker backend
    等工具语义。recall top-N、ranker final top-K、shared ranker actor 标识和 token
    length 由最终 Hydra 配置生成到 runtime tool config，避免静态 tool YAML 成为第
    二份 ranker/recall 语义配置。
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
    tool_ranker_config = config.get("ranker")
    if not isinstance(tool_ranker_config, dict):
        raise TypeError(f"first tool ranker config must be a mapping: {source_path}")

    _reject_deprecated_static_tool_fields(
        source_path=source_path,
        config=config,
        tool_ranker_config=tool_ranker_config,
    )

    recall_final_top_n = hydra_recall_retriever_config.get("recall_final_top_n")
    ranker_final_top_k = hydra_ranker_config.get("top_k")
    max_query_length = hydra_ranker_config.get("max_query_length")
    max_doc_length = hydra_ranker_config.get("max_doc_length")
    if recall_final_top_n is None or recall_final_top_n == "":
        raise ValueError("final Hydra recall_retriever.recall_final_top_n must be configured")
    if ranker_final_top_k is None or ranker_final_top_k == "":
        raise ValueError("final Hydra ranker.top_k must be configured")
    if max_query_length is None or max_query_length == "":
        raise ValueError("final Hydra ranker.max_query_length must be configured")
    if max_doc_length is None or max_doc_length == "":
        raise ValueError("final Hydra ranker.max_doc_length must be configured")

    config["recall_final_top_n"] = int(recall_final_top_n)
    tool_ranker_config["actor_name"] = "" if actor_name is None else str(actor_name)
    tool_ranker_config["actor_namespace"] = actor_namespace
    tool_ranker_config["final_top_k"] = int(ranker_final_top_k)
    tool_ranker_config["max_query_length"] = int(max_query_length)
    tool_ranker_config["max_doc_length"] = int(max_doc_length)
    dump_mapping(output_path, data)
