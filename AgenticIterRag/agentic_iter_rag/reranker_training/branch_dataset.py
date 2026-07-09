"""从增强轨迹构造 AIR LLM reranker branch GRPO 数据集。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from agentic_iter_rag.llm_reranker.format import render_air_rerank_tags_prompt
from agentic_iter_rag.trajectory.enhanced import (
    CONTEXT_FORMAT_VERSION,
    ENHANCED_TRAJECTORY_SCHEMA_VERSION,
    TOOL_RESPONSE_FORMAT_VERSION,
    validate_enhanced_record,
)
from agentic_iter_rag.utils.io import (
    copy_file,
    iter_jsonl,
    read_json,
    read_yaml,
    stable_config_hash,
    write_example,
    write_json,
    write_jsonl,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def write_dataset_readme(dataset_dir: Path) -> Path:
    """写 branch dataset README。

    通用 README 生成器目前只覆盖静态 reranker dataset；branch dataset 多了 continuation 上下文，
    这里单独写清楚字段用途，避免执行期依赖旧类型探测逻辑。
    """

    readme = dataset_dir / "README.md"
    manifest_path = dataset_dir / "manifest.json"
    content = [
        "# AIR LLM Reranker Branch Dataset",
        "",
        "这个数据集用于 AIR LLM reranker branch GRPO 训练。",
        "每条样本都对应增强轨迹中的一个 search step，并保留 continuation rollout 需要的历史上下文。",
        "",
        "## Files",
        "",
        "- `dataset.jsonl`: 训练样本，每行一个 branch sample。",
        "- `dataset.parquet`: 可选 Parquet 镜像，写入失败时只保留 JSONL。",
        "- `example.json`: 单条样本示例。",
        "- `manifest.json`: 数据集元信息。",
        "- `source_enhanced_trajectory.manifest.json`: 来源增强轨迹 manifest 快照。",
        "- `final_config.yaml`: 构造该数据集时使用的 final config 快照。",
        "",
        "## Important Fields",
        "",
        "- `prompt`: CoSearch 对齐的 topM reranker prompt，输入 top50，要求模型只输出 top5 候选编号。",
        "- `extra_info.messages_before_tool_response`: frozen agent continuation 拼接历史上下文时使用。",
        "- `extra_info.candidate_docs`: reranker 候选 top50 doc 内容。",
        "- `extra_info.candidate_index_to_doc_id`: reranker 输出编号到真实 doc_id 的映射。",
        "- `extra_info.baseline_reward`: delta reward 策略使用的 baseline 分数。",
        "",
        f"Manifest path: `{manifest_path}`",
        "",
    ]
    readme.write_text("\n".join(content), encoding="utf-8")
    return readme


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists and overwrite=false: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def resolve_version_dir(base_dir: Path, version: str, overwrite: bool) -> tuple[str, Path]:
    if overwrite or not (base_dir / version).exists():
        return version, base_dir / version
    for idx in range(1, 26):
        candidate = f"{version}_{chr(ord('a') + idx)}"
        if not (base_dir / candidate).exists():
            return candidate, base_dir / candidate
    raise FileExistsError(f"cannot find free version dir for {version}")


def load_enhanced_trajectory_manifest(path: Path) -> dict[str, Any]:
    manifest = read_json(path)
    if manifest.get("enhanced_schema_version") != ENHANCED_TRAJECTORY_SCHEMA_VERSION:
        raise ValueError("enhanced trajectory manifest schema version is invalid")
    if manifest.get("context_format_version") != CONTEXT_FORMAT_VERSION:
        raise ValueError("enhanced trajectory context format version is invalid")
    if manifest.get("tool_response_format_version") != TOOL_RESPONSE_FORMAT_VERSION:
        raise ValueError("enhanced trajectory tool response format version is invalid")
    jsonl = manifest.get("enhanced_trajectory_jsonl")
    if not jsonl or not Path(str(jsonl)).exists():
        raise FileNotFoundError(f"enhanced_trajectory_jsonl does not exist: {jsonl}")
    return manifest


def iter_enhanced_trajectories(manifest: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield from iter_jsonl(manifest["enhanced_trajectory_jsonl"])


def stable_step_index(seed: int, trajectory_id: str, num_steps: int) -> int:
    raw = f"{seed}:{trajectory_id}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16) % num_steps


def select_step(trajectory: dict[str, Any], policy: str, seed: int, allow_no_search: bool) -> dict[str, Any] | None:
    """按配置选择 reranker 介入点。

    random_point 使用稳定 hash，而不是进程级 random，保证并发分片和重跑时选择结果一致。
    """

    steps = list(trajectory.get("steps") or [])
    if not steps:
        if allow_no_search:
            return None
        raise ValueError(f"trajectory {trajectory.get('trajectory_id')} has no search steps")
    if policy == "first_point":
        return steps[0]
    if policy == "end_point":
        return steps[-1]
    if policy == "random_point":
        return steps[stable_step_index(seed, str(trajectory.get("trajectory_id")), len(steps))]
    raise ValueError(f"unsupported step_policy={policy!r}")


def normalize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    doc_id = str(out.get("doc_id") or out.get("id") or "")
    if not doc_id:
        raise ValueError("candidate doc is missing doc_id")
    text = out.get("text") or out.get("contents") or out.get("passage") or ""
    out["doc_id"] = doc_id
    out["id"] = str(out.get("id") or doc_id)
    out["text"] = str(text)
    out["contents"] = str(out.get("contents") or text)
    return out


def validate_selected_step(trajectory: dict[str, Any], step: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """强校验 query、doc 顺序和上下文，防止把轨迹 A 的 observation 拼到轨迹 B。"""

    candidate_top_n = int(cfg["candidate_top_n"])
    visible_top_m = int(cfg["visible_top_m"])
    if visible_top_m > candidate_top_n:
        raise ValueError("visible_top_m must be <= candidate_top_n")
    tool_query = str(step.get("tool_call", {}).get("arguments", {}).get("query") or "").strip()
    if str(step.get("sub_query") or "").strip() != tool_query:
        raise ValueError("step.sub_query must match tool_call.arguments.query")
    docs = [normalize_doc(doc) for doc in list(step.get("recall_topn_docs") or [])[:candidate_top_n]]
    if len(docs) != candidate_top_n:
        raise ValueError(f"step recall_topn_docs must contain {candidate_top_n} docs")
    doc_ids = [str(doc["doc_id"]) for doc in docs]
    if len(doc_ids) != len(set(doc_ids)):
        raise ValueError("candidate docs contain duplicated doc_id")
    if doc_ids != [str(item) for item in step.get("doc_id_order", [])[:candidate_top_n]]:
        raise ValueError("candidate doc_id order must match enhanced step doc_id_order")
    messages = step.get("messages_before_tool_response")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages_before_tool_response must be a non-empty list")
    if messages[-1].get("role") != "assistant" or "<tool_call>" not in str(messages[-1].get("content") or ""):
        raise ValueError("messages_before_tool_response must end with assistant tool_call")
    return docs


def build_branch_sample(
    trajectory: dict[str, Any],
    step: dict[str, Any],
    docs: list[dict[str, Any]],
    cfg: dict[str, Any],
    source_ref: dict[str, Any],
) -> dict[str, Any]:
    """组装 branch sample。

    extra_info 中一部分字段给 continuation 用，一部分给 reward 用；这里显式写全，避免训练时再猜来源。
    """

    # branch dataset 仍保存 top50 候选池，但 prompt 行为对齐 CoSearch：模型只需要输出 topM。
    # 这样 reward 直接评估“给 agent 的 5 篇 observation 是否可执行”，不再训练无用的 full50 全排序。
    prompt, index_to_doc_id = render_air_rerank_tags_prompt(
        initial_query=str(trajectory.get("question") or ""),
        sub_query=str(step.get("sub_query") or ""),
        docs=docs,
        top_m=int(cfg["visible_top_m"]),
        max_doc_chars=int(cfg["max_doc_chars"]),
    )
    trajectory_id = str(trajectory["trajectory_id"])
    step_index = int(step["step_index"])
    candidate_doc_ids = [str(doc["doc_id"]) for doc in docs]
    return {
        "sample_id": f"{trajectory_id}:step:{step_index}",
        "data_source": "agentic_iter_rag.llm_reranker.branch_grpo",
        "ability": "llm_reranker",
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": {"target": list(trajectory.get("gold_answers") or [])}},
        "prompt_template_version": cfg["prompt_template_version"],
        "formatter": cfg["formatter"],
        "target_text": None,
        "extra_info": {
            "trajectory_id": trajectory_id,
            "sample_id": str(trajectory.get("sample_id") or ""),
            "source_index": trajectory.get("source_index"),
            "step_index": step_index,
            "turn_index": step.get("turn_index"),
            "step_policy": cfg["step_policy"],
            "question": trajectory.get("question"),
            "sub_query": step.get("sub_query"),
            "candidate_doc_ids": candidate_doc_ids,
            "candidate_index_to_doc_id": index_to_doc_id,
            "candidate_docs": docs,
            "candidate_top_n": int(cfg["candidate_top_n"]),
            "visible_top_m": int(cfg["visible_top_m"]),
            "prompt_template_version": cfg["prompt_template_version"],
            "messages_before_tool_response": step.get("messages_before_tool_response"),
            "original_visible_doc_ids": step.get("original_visible_doc_ids"),
            "baseline_final_answer": trajectory.get("baseline_final_answer"),
            "baseline_reward": trajectory.get("baseline_reward"),
            "baseline_metrics": trajectory.get("baseline_metrics", {}),
            "context_format_version": trajectory.get("context_format_version"),
            "tool_response_format_version": trajectory.get("tool_response_format_version"),
            "reward_strategy": cfg.get("reward_strategy", "answer_reward"),
            "source_enhanced_trajectory_ref": source_ref,
        },
    }


def auto_version(trajectory_version: str, cfg: dict[str, Any]) -> str:
    return (
        f"{trajectory_version}__branch_{cfg['step_policy']}_top{cfg['candidate_top_n']}"
        f"_top{cfg['visible_top_m']}_{cfg['prompt_template_version']}"
    )


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except Exception as exc:
        print(f"warning: failed to write branch parquet, keeping JSONL only: {exc}")
        return False


def run_from_config(config_path: Path, stage_manifest_path: Path, dry_run: bool = False) -> dict[str, Any]:
    config = read_yaml(config_path)
    stage_cfg = config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"]
    rt_cfg = config["reranker_training"]
    branch_cfg = dict(rt_cfg["branch_dataset"])
    branch_cfg["reward_strategy"] = rt_cfg.get("reward", {}).get("strategy", "answer_reward")

    manifest_value = (
        stage_cfg.get("inputs", {}).get("enhanced_trajectory_manifest")
        or rt_cfg.get("input", {}).get("enhanced_trajectory_manifest")
    )
    if not manifest_value:
        raise ValueError("enhanced_trajectory_manifest is required for build_reranker_branch_dataset")
    source_manifest_path = Path(str(manifest_value))
    if dry_run and not source_manifest_path.exists():
        root = Path(str(config["main_run"]["data_artifacts"]["root"])) / "llm_reranker_branch_train_set"
        artifact_cfg = config["main_run"]["data_artifacts"]["llm_reranker_branch_train_set"]
        requested_version = branch_cfg.get("version") or artifact_cfg.get("version")
        source_version = source_manifest_path.parent.name or "planned_trajectory"
        version_base = str(requested_version or auto_version(source_version, branch_cfg))
        version, out_dir = resolve_version_dir(root, version_base, bool(branch_cfg.get("overwrite") or artifact_cfg.get("overwrite")))
        manifest_path = out_dir / "manifest.json"
        static_manifest = (
            stage_cfg.get("inputs", {}).get("reranker_train_set_manifest")
            or rt_cfg.get("input", {}).get("reranker_train_set_manifest")
        )
        outputs = {
            "status": "compiled",
            "branch_dataset_version": version,
            "branch_dataset_dir": str(out_dir),
            "branch_dataset_manifest": str(manifest_path),
            "source_enhanced_trajectory_manifest": str(source_manifest_path),
            "source_static_reranker_train_set_manifest": static_manifest,
            "note": "dry-run source manifest does not exist yet; using planned trajectory manifest path",
        }
        write_json(stage_manifest_path, {"type": "agentic_iter_rag_stage_manifest", "stage": "build_reranker_branch_dataset", "created_at": utc_now(), "config": stage_cfg, "outputs": outputs})
        return outputs
    source_manifest = load_enhanced_trajectory_manifest(source_manifest_path)

    root = Path(str(config["main_run"]["data_artifacts"]["root"])) / "llm_reranker_branch_train_set"
    artifact_cfg = config["main_run"]["data_artifacts"]["llm_reranker_branch_train_set"]
    requested_version = branch_cfg.get("version") or artifact_cfg.get("version")
    version_base = str(requested_version or auto_version(str(source_manifest["version"]), branch_cfg))
    version, out_dir = resolve_version_dir(root, version_base, bool(branch_cfg.get("overwrite") or artifact_cfg.get("overwrite")))

    dataset_jsonl = out_dir / "dataset.jsonl"
    dataset_parquet = out_dir / "dataset.parquet"
    example_json = out_dir / "example.json"
    manifest_path = out_dir / "manifest.json"
    final_config_yaml = out_dir / "final_config.yaml"
    source_snapshot = out_dir / "source_enhanced_trajectory.manifest.json"
    static_manifest = (
        stage_cfg.get("inputs", {}).get("reranker_train_set_manifest")
        or rt_cfg.get("input", {}).get("reranker_train_set_manifest")
    )

    if dry_run:
        outputs = {
            "status": "compiled",
            "branch_dataset_version": version,
            "branch_dataset_dir": str(out_dir),
            "branch_dataset_manifest": str(manifest_path),
            "source_enhanced_trajectory_manifest": str(source_manifest_path),
            "source_static_reranker_train_set_manifest": static_manifest,
        }
        write_json(stage_manifest_path, {"type": "agentic_iter_rag_stage_manifest", "stage": "build_reranker_branch_dataset", "created_at": utc_now(), "config": stage_cfg, "outputs": outputs})
        return outputs

    ensure_output_dir(out_dir, bool(branch_cfg.get("overwrite") or artifact_cfg.get("overwrite")))
    rows: list[dict[str, Any]] = []
    skipped_no_search = 0
    smoke_max = branch_cfg.get("smoke_max_samples")
    for line_index, trajectory in enumerate(iter_enhanced_trajectories(source_manifest)):
        validate_enhanced_record(trajectory, no_ranker=True, top_m=int(branch_cfg["visible_top_m"]))
        step = select_step(
            trajectory,
            str(branch_cfg["step_policy"]),
            int(branch_cfg["random_seed"]),
            bool(branch_cfg["allow_no_search"]),
        )
        if step is None:
            skipped_no_search += 1
            continue
        docs = validate_selected_step(trajectory, step, branch_cfg)
        rows.append(
            build_branch_sample(
                trajectory,
                step,
                docs,
                branch_cfg,
                {"path": str(source_manifest["enhanced_trajectory_jsonl"]), "line_index": line_index},
            )
        )
        if smoke_max is not None and int(smoke_max) > 0 and len(rows) >= int(smoke_max):
            break
    if not rows:
        raise ValueError("build_reranker_branch_dataset produced zero rows")

    count = write_jsonl(dataset_jsonl, rows)
    parquet_written = write_parquet(dataset_parquet, rows)
    write_example(example_json, rows[0])
    copy_file(source_manifest_path, source_snapshot)
    copy_file(config_path, final_config_yaml)
    readme = write_dataset_readme(out_dir)
    manifest = {
        "dataset_type": "llm_reranker_branch_train_set",
        "schema_version": "air_reranker_branch_dataset_v1",
        "version": version,
        "version_dir": str(out_dir),
        "created_at": utc_now(),
        "source_enhanced_trajectory_manifest": str(source_manifest_path),
        "source_enhanced_trajectory_jsonl": source_manifest["enhanced_trajectory_jsonl"],
        "source_static_reranker_train_set_manifest": static_manifest,
        "step_policy": branch_cfg["step_policy"],
        "random_seed": branch_cfg["random_seed"],
        "candidate_top_n": branch_cfg["candidate_top_n"],
        "visible_top_m": branch_cfg["visible_top_m"],
        "prompt_template_version": branch_cfg["prompt_template_version"],
        "formatter": branch_cfg["formatter"],
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_parquet": str(dataset_parquet) if parquet_written else None,
        "example_json": str(example_json),
        "readme": str(readme),
        "sample_count": count,
        "skipped_no_search_count": skipped_no_search,
        "final_config_yaml": str(final_config_yaml),
        "config_hash": stable_config_hash(branch_cfg),
    }
    write_json(manifest_path, manifest)
    outputs = {
        "status": "completed",
        "branch_dataset_manifest": str(manifest_path),
        "branch_dataset_jsonl": str(dataset_jsonl),
        "branch_dataset_version": version,
        "sample_count": count,
        "readme": str(readme),
    }
    write_json(stage_manifest_path, {"type": "agentic_iter_rag_stage_manifest", "stage": "build_reranker_branch_dataset", "created_at": utc_now(), "config": stage_cfg, "outputs": outputs})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIR LLM reranker branch dataset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    outputs = run_from_config(args.config, args.stage_manifest, dry_run=args.dry_run)
    print(f"built branch dataset stage: {outputs}")


if __name__ == "__main__":
    main()
