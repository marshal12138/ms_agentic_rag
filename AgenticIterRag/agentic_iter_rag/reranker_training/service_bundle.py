"""生成 AIR agentic RAG with LLM reranker 服务 bundle。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_iter_rag.utils.io import read_json, read_yaml, stable_config_hash, write_json, write_yaml


SERVICE_CONFIG_HEADER = """# AIR agentic RAG with LLM reranker 服务配置。
# 字段来源说明：
# - agent.model_path 来自训练/推理运行配置，部署侧可以按实际 agent checkpoint 覆盖。
# - retriever.endpoint 来自资源配置，部署侧必须替换成线上 retriever 服务地址。
# - llm_reranker.model_path 来自 train_llm_reranker stage 产物。
# - llm_reranker.expected_count 表示 reranker 输出 topM 数量，默认和 observation.visible_top_m 一致。
# - llm_reranker.max_index 表示 retriever topN 候选池最大编号，默认和 retriever.top_n 一致。
# - observation.visible_top_m 必须和训练时 agent observation top-M 保持一致。
"""


TOOL_CONFIG_HEADER = """# AIR search tool 配置模板。
# 这个文件由 build_service_bundle stage 生成，供服务启动脚本读取后再合并部署侧配置。
# retriever 字段来自资源配置；llm_reranker 字段来自训练产物；运行环境相关端口可由部署侧覆盖。
# ranker.required=true 表示 reranker 失败时 fail-fast，不回退到 retriever 原始顺序。
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_yaml_with_header(path: Path, data: dict[str, Any], header: str) -> None:
    """写带中文说明头的 YAML。

    write_yaml 负责结构化安全输出；这里额外补充注释头，避免生成配置变成无上下文的机器 YAML。
    """

    write_yaml(path, data)
    body = path.read_text(encoding="utf-8")
    path.write_text(header.rstrip() + "\n\n" + body, encoding="utf-8")


def resolve_agent_model(config: dict[str, Any]) -> str:
    """解析服务 bundle 使用的 agent 模型。

    这里和 dataproduce pipeline 对齐：优先使用 infer_runtime.models.trained_agent_model。
    如果 reranker 训练任务只给了已有 trajectory manifest，则从 trajectory.source_agent_checkpoint 继承。
    """

    stage_input = config["pipeline"]["stage_configs"]["build_service_bundle"].get("inputs", {}).get("agent_model")
    continuation_model = config.get("reranker_training", {}).get("continuation", {}).get("agent_model")
    trained_agent_model = config.get("infer_runtime", {}).get("models", {}).get("trained_agent_model")
    if stage_input:
        return str(stage_input)
    if continuation_model:
        return str(continuation_model)
    if trained_agent_model:
        return str(trained_agent_model)

    traj_manifest_value = (
        config.get("pipeline", {})
        .get("stage_configs", {})
        .get("build_reranker_branch_dataset", {})
        .get("inputs", {})
        .get("enhanced_trajectory_manifest")
        or config.get("reranker_training", {}).get("input", {}).get("enhanced_trajectory_manifest")
    )
    if traj_manifest_value and Path(str(traj_manifest_value)).exists():
        traj_manifest = read_json(Path(str(traj_manifest_value)))
        source_agent = traj_manifest.get("source_agent_checkpoint")
        if source_agent:
            return str(source_agent)

    origin_agent = config.get("infer_runtime", {}).get("models", {}).get("origin_agent_model")
    if origin_agent:
        return str(origin_agent)
    raise ValueError("agent model is required for service bundle")


def build_service_bundle(config: dict[str, Any], train_manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """写服务配置文件。

    字段来源必须清楚：reranker model 来自训练产物，agent model 来自运行配置，retriever endpoint 来自资源配置。
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    reranker_model = train_manifest.get("outputs", {}).get("reranker_model") or train_manifest.get("reranker_model")
    if not reranker_model:
        raise ValueError("train manifest does not contain reranker_model")
    agent_model = resolve_agent_model(config)
    train_resource = config["resource"]["stage_resources"]["train_llm_reranker"]
    phase_services = train_resource.get("phase_services")
    if isinstance(phase_services, dict) and "stage2_agentic" in phase_services:
        services = phase_services["stage2_agentic"].get("services", {})
    else:
        services = train_resource.get("services", {})
    recall = services["recall"]
    reranker_cfg = config["reranker_training"]

    service_config = {
        "schema_version": "air_service_bundle_v1",
        "service_type": "agentic_rag_with_llm_reranker",
        "agent": {
            "model_path": str(agent_model),
            "tokenizer_path": str(config["model"].get("path") or reranker_cfg["base_model"]),
        },
        "retriever": {
            "endpoint": str(recall["retrieval_service_url"]),
            "top_n": int(reranker_cfg["branch_dataset"]["candidate_top_n"]),
        },
        "llm_reranker": {
            "model_path": str(reranker_model),
            "base_model": str(reranker_cfg["base_model"]),
            "prompt_template_version": str(reranker_cfg["branch_dataset"]["prompt_template_version"]),
            "output_parser": "cosearch_rerank_tags_topm",
            "expected_count": int(reranker_cfg["branch_dataset"]["visible_top_m"]),
            "max_index": int(reranker_cfg["branch_dataset"]["candidate_top_n"]),
            "required": True,
        },
        "observation": {
            "visible_top_m": int(reranker_cfg["branch_dataset"]["visible_top_m"]),
            "tool_response_format_version": "air_search_tool_response_v1",
        },
    }
    tool_config = {
        "tools": [
            {
                "class_name": "verl.tools.agentic_iter_rag_retriever_tool.AgenticIterRagRetrieverTool",
                "config": {
                    "type": "native",
                    "retrieval_service_url": str(recall["retrieval_service_url"]),
                    "recall_final_top_n": int(reranker_cfg["branch_dataset"]["candidate_top_n"]),
                    "searchTool_final_top_m": int(reranker_cfg["branch_dataset"]["visible_top_m"]),
                    "ranker_enabled": True,
                    "ranker": {
                        "backend": "llm_reranker_service",
                        "required": True,
                        "model_path": str(reranker_model),
                        "prompt_template_version": str(reranker_cfg["branch_dataset"]["prompt_template_version"]),
                        "output_parser": "cosearch_rerank_tags_topm",
                        "expected_count": int(reranker_cfg["branch_dataset"]["visible_top_m"]),
                        "max_index": int(reranker_cfg["branch_dataset"]["candidate_top_n"]),
                    },
                },
            }
        ]
    }
    service_config_path = output_dir / "service_config.yaml"
    tool_config_path = output_dir / "tool_config.yaml"
    manifest_path = output_dir / "manifest.json"
    write_yaml_with_header(service_config_path, service_config, SERVICE_CONFIG_HEADER)
    write_yaml_with_header(tool_config_path, tool_config, TOOL_CONFIG_HEADER)
    manifest = {
        "type": "air_llm_reranker_service_bundle",
        "schema_version": "air_service_bundle_v1",
        "created_at": utc_now(),
        "service_config": str(service_config_path),
        "tool_config": str(tool_config_path),
        "source_train_stage_manifest": train_manifest.get("manifest_path"),
        "reranker_model": str(reranker_model),
        "agent_model": str(agent_model),
        "retriever_top_n": int(reranker_cfg["branch_dataset"]["candidate_top_n"]),
        "visible_top_m": int(reranker_cfg["branch_dataset"]["visible_top_m"]),
        "reranker_expected_count": int(reranker_cfg["branch_dataset"]["visible_top_m"]),
        "reranker_max_index": int(reranker_cfg["branch_dataset"]["candidate_top_n"]),
        "required": True,
        "config_hash": stable_config_hash(service_config),
    }
    write_json(manifest_path, manifest)
    validate_service_bundle(output_dir)
    return manifest


def validate_service_bundle(bundle_dir: Path) -> None:
    service_config = bundle_dir / "service_config.yaml"
    tool_config = bundle_dir / "tool_config.yaml"
    manifest = bundle_dir / "manifest.json"
    for path in (service_config, tool_config, manifest):
        if not path.exists():
            raise FileNotFoundError(f"missing service bundle file: {path}")
    payload = read_json(manifest)
    if int(payload["visible_top_m"]) > int(payload["retriever_top_n"]):
        raise ValueError("service bundle visible_top_m must be <= retriever_top_n")
    if not payload.get("reranker_model"):
        raise ValueError("service bundle reranker_model is empty")


def run_from_config(config_path: Path, stage_manifest_path: Path, dry_run: bool = False) -> dict[str, Any]:
    config = read_yaml(config_path)
    artifact_root = Path(str(config["runtime_compiled"]["ARTIFACT_ROOT"]))
    train_manifest_path = Path(str(config["pipeline"]["stage_configs"]["train_llm_reranker"]["outputs"]["manifest"]))
    bundle_dir = artifact_root / "service_bundle"
    if dry_run:
        outputs = {"status": "compiled", "service_bundle_dir": str(bundle_dir)}
        write_json(stage_manifest_path, {"type": "agentic_iter_rag_stage_manifest", "stage": "build_service_bundle", "created_at": utc_now(), "config": config["pipeline"]["stage_configs"]["build_service_bundle"], "outputs": outputs})
        return outputs
    train_manifest = read_json(train_manifest_path)
    train_manifest["manifest_path"] = str(train_manifest_path)
    bundle_manifest = build_service_bundle(config, train_manifest, bundle_dir)
    outputs = {"status": "completed", "service_bundle_dir": str(bundle_dir), "manifest": str(bundle_dir / "manifest.json"), **bundle_manifest}
    write_json(stage_manifest_path, {"type": "agentic_iter_rag_stage_manifest", "stage": "build_service_bundle", "created_at": utc_now(), "config": config["pipeline"]["stage_configs"]["build_service_bundle"], "outputs": outputs})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIR service bundle with LLM reranker.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs = run_from_config(args.config, args.stage_manifest, dry_run=args.dry_run)
    print(f"built service bundle: {outputs}")


if __name__ == "__main__":
    main()
