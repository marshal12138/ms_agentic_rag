"""构造 AgenticIterRag v1 的 LLM reranker 数据集。"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from agentic_iter_rag.llm_reranker.format import render_air_rerank_tags_prompt
from agentic_iter_rag.reranker_dataset.schema import RerankerSample, validate_reranker_sample
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


# ---------------------------------------------------------------------------
# 通用工具：路径、版本、Parquet 镜像
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: Any) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in str(text or ""))
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or "unknown"


def _ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output directory already exists and overwrite=false: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _local_date_prefix() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%y%m%d")


def _suffixed_version(version: str, suffix: str) -> str:
    match = re.match(r"^(\d{6})(_.*)$", version)
    if match:
        return f"{match.group(1)}{suffix}{match.group(2)}"
    return f"{version}_{suffix}"


def _resolve_version_dir(base_dir: Path, version: str, *, overwrite: bool, manual_version: bool) -> tuple[str, Path]:
    """解析版本目录；overwrite=false 时同名目录自动生成 b/c/d 新版本。"""

    candidate_version = version
    candidate_dir = base_dir / candidate_version
    if overwrite or not candidate_dir.exists():
        return candidate_version, candidate_dir
    for suffix_idx in range(1, 26):
        retry_version = _suffixed_version(version, chr(ord("a") + suffix_idx))
        retry_dir = base_dir / retry_version
        if not retry_dir.exists():
            return retry_version, retry_dir
    raise FileExistsError(f"cannot find free auto version under {base_dir} for base version={version}")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> bool:
    try:
        import pandas as pd

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        return True
    except Exception as exc:
        print(f"warning: failed to write parquet, keeping JSONL only: {exc}")
        return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_dataset_readme(dataset_dir: Path) -> Path:
    script = _repo_root() / "scripts" / "agenticIterRag_v1" / "assets" / "build_dataset_readme.py"
    subprocess.check_call([sys.executable, str(script), "--dataset-dir", str(dataset_dir)])
    return dataset_dir / "README.md"


def _resolve_root(config: dict[str, Any]) -> Path:
    root = config["main_run"]["data_artifacts"]["root"]
    return Path(str(root))


def _auto_input_version(trajectory_version: str, input_cfg: dict[str, Any]) -> str:
    top_n = input_cfg["candidate_top_n"]
    label = "emptylabel" if input_cfg.get("label_policy") is None else _slug(input_cfg["label_policy"])
    if input_cfg.get("derive_version_from_trajectory"):
        return f"{trajectory_version}__input_recall{top_n}_{label}"
    stamp = _local_date_prefix()
    return f"{stamp}__input_recall{top_n}_{label}"


def _auto_train_version(train_cfg: dict[str, Any]) -> str:
    stamp = _local_date_prefix()
    fmt = _slug(train_cfg["format"])
    prompt = _slug(train_cfg["prompt_template_version"])
    formatter = _slug(train_cfg["formatter"])
    gt = _slug(train_cfg["ground_truth_policy"]).replace("_", "-")
    reward = _slug(train_cfg["reward_policy"]).replace("_", "-")
    return f"{stamp}_{fmt}_{prompt}_{formatter}_gt-{gt}_reward-{reward}"


# ---------------------------------------------------------------------------
# 子阶段一：trajectory -> input_dataset
# ---------------------------------------------------------------------------


def _dedupe_docs(docs: list[dict[str, Any]], *, limit: int, policy: str) -> list[dict[str, Any]]:
    if policy != "keep_first":
        raise ValueError(f"unsupported dedupe_policy={policy!r}; only keep_first is implemented")
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc.get("doc_id") or doc.get("id") or "")
        if not doc_id or doc_id in seen:
            continue
        seen.add(doc_id)
        normalized = dict(doc)
        normalized["doc_id"] = doc_id
        out.append(normalized)
        if len(out) >= limit:
            break
    return out


def sample_from_trace(trace: dict[str, Any], input_cfg: dict[str, Any], trajectory_version: str) -> dict[str, Any]:
    candidate_source = str(input_cfg["candidate_source"])
    if candidate_source not in trace:
        raise ValueError(f"trace missing candidate_source={candidate_source!r}: {trace.get('trace_id')}")
    candidates = _dedupe_docs(
        trace.get(candidate_source) or [],
        limit=int(input_cfg["candidate_top_n"]),
        policy=str(input_cfg["dedupe_policy"]),
    )
    sample = RerankerSample(
        query_id=str(trace["trace_id"]),
        question=str(trace.get("question") or ""),
        sub_query=str(trace.get("sub_query") or ""),
        candidate_docs=candidates,
        label_policy=input_cfg.get("label_policy"),
        target_ranking=[],
        positive_doc_ids=[],
        source_trace_id=str(trace["trace_id"]),
        metadata={
            "sample_id": trace.get("sample_id"),
            "reward": trace.get("reward"),
            "metrics": trace.get("metrics", {}),
            "trajectory_version": trajectory_version,
            "candidate_source": candidate_source,
            "candidate_top_n": input_cfg["candidate_top_n"],
            "positive_policy": input_cfg.get("positive_policy"),
            "target_ranking_policy": input_cfg.get("target_ranking_policy"),
        },
    ).to_dict()
    validate_reranker_sample(sample)
    return sample


def build_input_dataset(
    *,
    config: dict[str, Any],
    trajectory_jsonl: Path,
    trajectory_manifest: Path | None,
    trajectory_version: str,
    output_root: Path,
    input_cfg: dict[str, Any],
    final_config_yaml: Path,
) -> dict[str, Any]:
    _ensure_output_dir(output_root / "input_dataset", bool(input_cfg["overwrite"]))
    rows = [sample_from_trace(trace, input_cfg, trajectory_version) for trace in iter_jsonl(trajectory_jsonl)]
    if not rows:
        raise ValueError("build_input_dataset produced zero rows")

    dataset_jsonl = output_root / "input_dataset" / "dataset.jsonl"
    dataset_parquet = output_root / "input_dataset" / "dataset.parquet"
    example_json = output_root / "input_dataset" / "example.json"
    manifest_path = output_root / "input_dataset" / "manifest.json"
    count = write_jsonl(dataset_jsonl, rows)
    parquet_written = _write_parquet(dataset_parquet, rows)
    write_example(example_json, rows[0] if rows else None)

    manifest = {
        "dataset_type": "llm_reranker_input_dataset",
        "version": output_root.name,
        "version_dir": str(output_root),
        "created_at": _utc_now(),
        "source_trajectory_version": trajectory_version,
        "source_trajectory_jsonl": str(trajectory_jsonl),
        "source_trajectory_manifest": str(trajectory_manifest) if trajectory_manifest else None,
        "schema_version": input_cfg["schema_version"],
        "builder_policy": input_cfg["builder_policy"],
        "candidate_source": input_cfg["candidate_source"],
        "candidate_top_n": input_cfg["candidate_top_n"],
        "dedupe_policy": input_cfg["dedupe_policy"],
        "label_policy": input_cfg.get("label_policy"),
        "positive_policy": input_cfg.get("positive_policy"),
        "target_ranking_policy": input_cfg.get("target_ranking_policy"),
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_parquet": str(dataset_parquet) if parquet_written else None,
        "example_json": str(example_json),
        "readme": str(output_root / "input_dataset" / "README.md"),
        "sample_count": count,
        "final_config_yaml": str(final_config_yaml),
        "config_hash": stable_config_hash(input_cfg),
    }
    write_json(manifest_path, manifest)
    _write_dataset_readme(output_root / "input_dataset")
    return manifest


# ---------------------------------------------------------------------------
# 子阶段二：input_dataset -> train_dataset
# ---------------------------------------------------------------------------


def _ground_truth(policy: str) -> dict[str, Any]:
    if policy != "empty":
        raise ValueError(f"unsupported ground_truth_policy={policy!r}; only empty is implemented")
    return {"target": None}


def train_sample_from_input(sample: dict[str, Any], train_cfg: dict[str, Any], input_manifest: dict[str, Any], top_m: int) -> dict[str, Any]:
    if train_cfg["format"] != "grpo":
        raise ValueError(f"unsupported train_dataset format={train_cfg['format']!r}; only grpo is implemented")
    if train_cfg["prompt_template_version"] != "air_rerank_tags_v1":
        raise ValueError("only prompt_template_version=air_rerank_tags_v1 is implemented")
    if train_cfg["formatter"] != "verl_chat":
        raise ValueError("only formatter=verl_chat is implemented")
    if train_cfg["reward_policy"] != "none":
        raise ValueError("only reward_policy=none is implemented in data produce v1")

    prompt, index_to_doc_id = render_air_rerank_tags_prompt(
        initial_query=str(sample.get("question") or ""),
        sub_query=str(sample.get("sub_query") or ""),
        docs=list(sample.get("candidate_docs") or []),
        top_m=top_m,
        max_doc_chars=int(train_cfg["max_doc_chars"]),
    )
    candidate_doc_ids = [str(doc.get("doc_id")) for doc in sample.get("candidate_docs", [])]
    return {
        "sample_id": str(sample["query_id"]),
        "source_query_id": str(sample["query_id"]),
        "data_source": "agentic_iter_rag.llm_reranker.grpo",
        "ability": "llm_reranker",
        "prompt": prompt,
        "reward_model": {
            "style": "rule",
            "ground_truth": _ground_truth(str(train_cfg["ground_truth_policy"])),
        },
        "prompt_template_version": train_cfg["prompt_template_version"],
        "formatter": train_cfg["formatter"],
        "target_text": None,
        "extra_info": {
            "source_query_id": str(sample["query_id"]),
            "source_trace_id": sample.get("source_trace_id"),
            "input_dataset_version": input_manifest["version"],
            "trajectory_version": input_manifest.get("source_trajectory_version"),
            "candidate_doc_ids": candidate_doc_ids,
            "candidate_index_to_doc_id": index_to_doc_id,
            "ground_truth_policy": train_cfg["ground_truth_policy"],
            "reward_policy": train_cfg["reward_policy"],
            "output_schema": train_cfg["output_schema"],
        },
    }


def build_train_dataset(
    *,
    input_manifest: dict[str, Any],
    input_manifest_path: Path,
    train_root: Path,
    train_cfg: dict[str, Any],
    top_m: int,
    final_config_yaml: Path,
) -> dict[str, Any]:
    _ensure_output_dir(train_root, bool(train_cfg["overwrite"]))
    rows = [train_sample_from_input(sample, train_cfg, input_manifest, top_m) for sample in iter_jsonl(input_manifest["dataset_jsonl"])]
    if not rows:
        raise ValueError("build_train_dataset produced zero rows")

    dataset_jsonl = train_root / "dataset.jsonl"
    dataset_parquet = train_root / "dataset.parquet"
    example_json = train_root / "example.json"
    manifest_path = train_root / "manifest.json"
    count = write_jsonl(dataset_jsonl, rows)
    parquet_written = _write_parquet(dataset_parquet, rows)
    write_example(example_json, rows[0] if rows else None)

    manifest = {
        "dataset_type": "llm_reranker_train_dataset",
        "version": train_root.name,
        "version_dir": str(train_root),
        "created_at": _utc_now(),
        "source_input_dataset_manifest": str(input_manifest_path),
        "prompt_template_version": train_cfg["prompt_template_version"],
        "formatter": train_cfg["formatter"],
        "format": train_cfg["format"],
        "ground_truth_policy": train_cfg["ground_truth_policy"],
        "reward_policy": train_cfg["reward_policy"],
        "output_schema": train_cfg["output_schema"],
        "reranker_top_m": top_m,
        "max_doc_chars": train_cfg["max_doc_chars"],
        "dataset_jsonl": str(dataset_jsonl),
        "dataset_parquet": str(dataset_parquet) if parquet_written else None,
        "example_json": str(example_json),
        "readme": str(train_root / "README.md"),
        "sample_count": count,
        "final_config_yaml": str(final_config_yaml),
        "config_hash": stable_config_hash(train_cfg),
    }
    write_json(manifest_path, manifest)
    _write_dataset_readme(train_root)
    return manifest


# ---------------------------------------------------------------------------
# 父 stage：读取 final config，执行两个子阶段并汇总 manifest
# ---------------------------------------------------------------------------


def run_from_config(config_path: Path, stage_manifest_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    stage_cfg = config["pipeline"]["stage_configs"]["build_reranker_dataset"]
    input_cfg = stage_cfg["sub_stages"]["build_input_dataset"]
    train_cfg = stage_cfg["sub_stages"]["build_train_dataset"]
    input_enabled = bool(input_cfg["enabled"])
    train_enabled = bool(train_cfg["enabled"])
    if not input_enabled and not train_enabled:
        raise ValueError("build_reranker_dataset has no enabled sub stages")

    artifacts_root = _resolve_root(config)
    trajectory_manifest_path = stage_cfg["inputs"].get("trajectory_manifest")
    trajectory_manifest = read_json(trajectory_manifest_path) if trajectory_manifest_path else {}
    trajectory_jsonl_value = stage_cfg["inputs"].get("canonical_trace_jsonl") or trajectory_manifest.get("trajectory_jsonl")
    if input_enabled and not trajectory_jsonl_value:
        raise ValueError("canonical_trace_jsonl is required when build_input_dataset.enabled=true")
    trajectory_jsonl = Path(str(trajectory_jsonl_value)) if trajectory_jsonl_value else Path("")
    trajectory_version = str(trajectory_manifest.get("version") or trajectory_jsonl.parent.name)

    if input_enabled:
        requested_version = input_cfg.get("version") or config["main_run"]["data_artifacts"]["llm_reranker_train_set"].get("version")
        input_version_base = str(requested_version or _auto_input_version(trajectory_version, input_cfg))
        input_version, output_root = _resolve_version_dir(
            artifacts_root / "llm_reranker_train_set",
            input_version_base,
            overwrite=bool(input_cfg["overwrite"]),
            manual_version=bool(requested_version),
        )
        input_manifest = build_input_dataset(
            config=config,
            trajectory_jsonl=trajectory_jsonl,
            trajectory_manifest=Path(trajectory_manifest_path) if trajectory_manifest_path else None,
            trajectory_version=trajectory_version,
            output_root=output_root,
            input_cfg=input_cfg,
            final_config_yaml=config_path,
        )
        input_manifest_path = output_root / "input_dataset" / "manifest.json"
        input_cfg.setdefault("outputs", {}).update(
            {
                "dataset_jsonl": input_manifest["dataset_jsonl"],
                "dataset_parquet": input_manifest.get("dataset_parquet"),
                "example_json": input_manifest["example_json"],
                "manifest": str(input_manifest_path),
            }
        )
    else:
        input_manifest_path_value = stage_cfg["inputs"].get("input_dataset_manifest")
        if not input_manifest_path_value:
            raise ValueError("input_dataset_manifest is required when build_input_dataset.enabled=false")
        input_manifest_path = Path(str(input_manifest_path_value))
        input_manifest = read_json(input_manifest_path)
        output_root = Path(str(input_manifest["version_dir"]))

    train_manifest: dict[str, Any] | None = None
    if train_enabled:
        top_m = train_cfg.get("reranker_top_m") or config["infer_runtime"]["retrieval"]["visible_top_m"]
        requested_train_version = train_cfg.get("version")
        train_version_base = str(requested_train_version or _auto_train_version(train_cfg))
        train_version, train_root = _resolve_version_dir(
            output_root / "train_dataset",
            train_version_base,
            overwrite=bool(train_cfg["overwrite"]),
            manual_version=bool(requested_train_version),
        )
        train_manifest = build_train_dataset(
            input_manifest=input_manifest,
            input_manifest_path=input_manifest_path,
            train_root=train_root,
            train_cfg=train_cfg,
            top_m=int(top_m),
            final_config_yaml=config_path,
        )
        train_cfg.setdefault("outputs", {}).update(
            {
                "dataset_jsonl": train_manifest["dataset_jsonl"],
                "dataset_parquet": train_manifest.get("dataset_parquet"),
                "example_json": train_manifest["example_json"],
                "manifest": str(Path(train_manifest["version_dir"]) / "manifest.json"),
            }
        )

    if input_enabled:
        write_example(output_root / "example.json", next(iter_jsonl(input_manifest["dataset_jsonl"]), None))
    if trajectory_manifest_path:
        copy_file(trajectory_manifest_path, output_root / "source_trajectory.manifest.json")
    copy_file(config_path, output_root / "final_config.yaml")

    parent_manifest = {
        "dataset_type": "llm_reranker_train_set",
        "version": output_root.name,
        "version_dir": str(output_root),
        "created_at": _utc_now(),
        "source_trajectory_version": trajectory_version,
        "source_trajectory_manifest": str(trajectory_manifest_path) if trajectory_manifest_path else None,
        "input_dataset_manifest": str(input_manifest_path),
        "train_dataset_versions": [train_manifest["version"]] if train_manifest else [],
        "input_dataset_jsonl": input_manifest.get("dataset_jsonl"),
        "input_dataset_example_json": input_manifest.get("example_json"),
        "train_dataset_manifest": str(Path(train_manifest["version_dir"]) / "manifest.json") if train_manifest else None,
        "example_json": str(output_root / "example.json"),
        "readme": str(output_root / "README.md"),
        "sample_count": input_manifest.get("sample_count"),
        "final_config_yaml": str(output_root / "final_config.yaml"),
        "config_hash": stable_config_hash(stage_cfg),
    }
    parent_manifest_path = output_root / "manifest.json"
    write_json(parent_manifest_path, parent_manifest)
    parent_readme = _write_dataset_readme(output_root)

    # ---- 子阶段 manifest：真实执行时也保留独立审计文件，便于验收和断点排查 ----
    sub_stage_manifests: dict[str, str] = {}
    sub_manifest_root = stage_manifest_path.parent
    if input_enabled:
        input_sub_manifest = sub_manifest_root / "build_input_dataset" / "manifest.json"
        write_json(
            input_sub_manifest,
            {
                "type": "agentic_iter_rag_sub_stage_manifest",
                "stage": "build_reranker_dataset.build_input_dataset",
                "created_at": _utc_now(),
                "config": input_cfg,
                "outputs": {
                    "status": "completed",
                    "input_dataset_manifest": str(input_manifest_path),
                    "dataset_jsonl": input_manifest["dataset_jsonl"],
                    "example_json": input_manifest["example_json"],
                    "readme": input_manifest["readme"],
                    "sample_count": input_manifest["sample_count"],
                },
            },
        )
        sub_stage_manifests["build_input_dataset"] = str(input_sub_manifest)
    if train_enabled and train_manifest:
        train_sub_manifest = sub_manifest_root / "build_train_dataset" / "manifest.json"
        write_json(
            train_sub_manifest,
            {
                "type": "agentic_iter_rag_sub_stage_manifest",
                "stage": "build_reranker_dataset.build_train_dataset",
                "created_at": _utc_now(),
                "config": train_cfg,
                "outputs": {
                    "status": "completed",
                    "train_dataset_manifest": str(Path(train_manifest["version_dir"]) / "manifest.json"),
                    "dataset_jsonl": train_manifest["dataset_jsonl"],
                    "example_json": train_manifest["example_json"],
                    "readme": train_manifest["readme"],
                    "sample_count": train_manifest["sample_count"],
                },
            },
        )
        sub_stage_manifests["build_train_dataset"] = str(train_sub_manifest)

    stage_cfg.setdefault("outputs", {}).update(
        {
            "input_dataset_manifest": str(input_manifest_path),
            "train_dataset_manifest": str(Path(train_manifest["version_dir"]) / "manifest.json") if train_manifest else None,
            "reranker_train_set_manifest": str(parent_manifest_path),
            "manifest": str(stage_manifest_path),
        }
    )
    stage_outputs = {
        "status": "completed",
        "input_dataset_enabled": input_enabled,
        "train_dataset_enabled": train_enabled,
        "input_dataset_manifest": str(input_manifest_path),
        "train_dataset_manifest": str(Path(train_manifest["version_dir"]) / "manifest.json") if train_manifest else None,
        "reranker_train_set_manifest": str(parent_manifest_path),
        "reranker_train_set_readme": str(parent_readme),
        "sub_stage_manifests": sub_stage_manifests,
    }
    write_json(
        stage_manifest_path,
        {
            "type": "agentic_iter_rag_stage_manifest",
            "stage": "build_reranker_dataset",
            "created_at": _utc_now(),
            "config": stage_cfg,
            "outputs": stage_outputs,
        },
    )
    return stage_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AgenticIterRag v1 reranker input/train datasets.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage-manifest", required=True, type=Path)
    args = parser.parse_args()
    outputs = run_from_config(args.config, args.stage_manifest)
    print(f"built reranker datasets: {outputs}")


if __name__ == "__main__":
    main()
