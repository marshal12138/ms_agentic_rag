"""Filter AIR branch reranker datasets into training slices."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agentic_iter_rag.reranker_training.branch_dataset import write_dataset_readme, write_parquet
from agentic_iter_rag.reranker_training.reward_bound_diagnosis import (
    contains_any_answer,
    doc_text,
    sample_passes_filter,
)
from agentic_iter_rag.utils.io import copy_file, iter_jsonl, read_json, read_yaml, stable_config_hash, write_example, write_json, write_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def baseline_zero(sample: dict[str, Any]) -> bool:
    return float((sample.get("extra_info") or {}).get("baseline_reward") or 0.0) == 0.0


def sample_filter(sample: dict[str, Any], name: str) -> bool:
    if name in {"all", "top50_hit", "top50_miss", "top50_hit_top5_miss", "top50_hit_top5_miss_baseline0"}:
        return sample_passes_filter(sample, name)
    if name == "baseline0":
        return baseline_zero(sample)
    if name == "top50_hit_baseline0":
        return sample_passes_filter(sample, "top50_hit") and baseline_zero(sample)
    if name == "top50_miss_baseline0":
        return sample_passes_filter(sample, "top50_miss") and baseline_zero(sample)
    if name == "top5_hit":
        extra = sample.get("extra_info") or {}
        docs = list(extra.get("candidate_docs") or [])
        targets = list((sample.get("reward_model") or {}).get("ground_truth", {}).get("target") or [])
        return contains_any_answer("\n".join(doc_text(doc) for doc in docs[:5]), targets)
    raise ValueError(f"unsupported filter={name!r}")


def dotted_get(data: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = data
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def load_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError("python callable strategy must use module:function syntax")
    module_name, func_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"strategy callable is not callable: {spec}")
    return func


def strategy_name(strategy: dict[str, Any], fallback: str) -> str:
    return str(strategy.get("name") or strategy.get("builtin_name") or fallback)


def builtin_keep(sample: dict[str, Any], strategy: dict[str, Any]) -> bool:
    builtin_name = str(strategy.get("builtin_name") or strategy.get("name") or "all")
    kwargs = dict(strategy.get("kwargs") or {})
    if not kwargs:
        return sample_filter(sample, builtin_name)

    target_key = str(kwargs.get("target_key") or "reward_model.ground_truth.target")
    docs_key = str(kwargs.get("candidate_docs_key") or "extra_info.candidate_docs")
    baseline_key = str(kwargs.get("baseline_reward_key") or "extra_info.baseline_reward")
    visible_top_m = int(kwargs.get("visible_top_m") or (sample.get("extra_info") or {}).get("visible_top_m") or 5)

    targets = list(dotted_get(sample, target_key, []) or [])
    docs = list(dotted_get(sample, docs_key, []) or [])
    top50_hit = contains_any_answer("\n".join(doc_text(doc) for doc in docs), targets)
    top5_hit = contains_any_answer("\n".join(doc_text(doc) for doc in docs[:visible_top_m]), targets)
    baseline_reward = float(dotted_get(sample, baseline_key, 0.0) or 0.0)

    checks = [
        (not kwargs.get("require_top50_hit", False)) or top50_hit,
        (not kwargs.get("require_top50_miss", False)) or (not top50_hit),
        (not kwargs.get("require_top5_hit", False)) or top5_hit,
        (not kwargs.get("require_top5_miss", False)) or (not top5_hit),
        (not kwargs.get("require_baseline_zero", False)) or baseline_reward == 0.0,
    ]
    if any(key in kwargs for key in ("min_baseline_reward", "max_baseline_reward")):
        min_reward = float(kwargs.get("min_baseline_reward", "-inf"))
        max_reward = float(kwargs.get("max_baseline_reward", "inf"))
        checks.append(min_reward <= baseline_reward <= max_reward)
    if builtin_name != "custom_predicates":
        checks.append(sample_filter(sample, builtin_name))
    return all(checks)


def callable_keep(sample: dict[str, Any], manifest: dict[str, Any], strategy: dict[str, Any]) -> bool:
    spec = str(strategy.get("callable") or "")
    if not spec:
        raise ValueError("python_callable strategy requires strategy.callable")
    result = load_callable(spec)(sample, manifest=manifest, kwargs=dict(strategy.get("kwargs") or {}))
    if isinstance(result, dict):
        return bool(result.get("keep"))
    return bool(result)


def script_rows(
    *,
    source_manifest_path: Path,
    out_dir: Path,
    strategy: dict[str, Any],
) -> list[dict[str, Any]]:
    script_path = Path(str(strategy.get("script_path") or ""))
    if not script_path.exists():
        raise FileNotFoundError(f"filter strategy script does not exist: {script_path}")
    selected_jsonl = out_dir / "script_selected_rows.jsonl"
    cmd = [
        sys.executable,
        str(script_path),
        "--source-manifest",
        str(source_manifest_path),
        "--output-jsonl",
        str(selected_jsonl),
        "--kwargs-json",
        json.dumps(dict(strategy.get("kwargs") or {}), ensure_ascii=True, sort_keys=True),
    ]
    subprocess.check_call(cmd)
    return list(iter_jsonl(selected_jsonl))


def apply_sample_mode(rows: list[dict[str, Any]], *, max_samples: int, sample_mode: str, seed: int) -> list[dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows
    if sample_mode in {"none", "first"}:
        return rows[:max_samples]
    if sample_mode == "random":
        rng = random.Random(seed)
        selected = list(rows)
        rng.shuffle(selected)
        return selected[:max_samples]
    raise ValueError(f"unsupported branch filter sample_mode={sample_mode!r}")


def build_subset(
    *,
    source_manifest_path: Path,
    out_root: Path | None,
    out_version: str | None,
    filter_name: str,
    strategy: dict[str, Any] | None = None,
    max_samples: int,
    sample_mode: str = "first",
    random_seed: int = 20260708,
    overwrite: bool,
) -> dict[str, Any]:
    source_manifest = read_json(source_manifest_path)
    data_path = Path(str(source_manifest["dataset_jsonl"]))
    base_dir = out_root or source_manifest_path.parent.parent
    strategy_cfg = dict(strategy or {"kind": "builtin", "name": filter_name, "builtin_name": filter_name, "kwargs": {}})
    selected_strategy_name = strategy_name(strategy_cfg, filter_name)
    version_base = out_version or f"{source_manifest['version']}__filter_{selected_strategy_name}"
    version, out_dir = resolve_version_dir(base_dir, str(version_base), overwrite)
    ensure_output_dir(out_dir, overwrite)

    rows: list[dict[str, Any]] = []
    sample_count_before = 0
    kind = str(strategy_cfg.get("kind") or "builtin")
    if kind == "script":
        all_rows = script_rows(source_manifest_path=source_manifest_path, out_dir=out_dir, strategy=strategy_cfg)
        sample_count_before = len(all_rows)
        rows = apply_sample_mode(all_rows, max_samples=max_samples, sample_mode=sample_mode, seed=random_seed)
    else:
        for row in iter_jsonl(data_path):
            sample_count_before += 1
            if kind == "builtin":
                keep = builtin_keep(row, strategy_cfg)
            elif kind == "python_callable":
                keep = callable_keep(row, source_manifest, strategy_cfg)
            else:
                raise ValueError(f"unsupported branch filter strategy kind={kind!r}")
            if keep:
                rows.append(row)
        rows = apply_sample_mode(rows, max_samples=max_samples, sample_mode=sample_mode, seed=random_seed)
    if not rows:
        raise ValueError(f"filter {filter_name!r} selected zero rows from {source_manifest_path}")

    dataset_jsonl = out_dir / "dataset.jsonl"
    dataset_parquet = out_dir / "dataset.parquet"
    manifest_path = out_dir / "manifest.json"
    example_json = out_dir / "example.json"
    source_snapshot = out_dir / "source_branch_dataset.manifest.json"
    count = write_jsonl(dataset_jsonl, rows)
    parquet_written = write_parquet(dataset_parquet, rows)
    write_example(example_json, rows[0])
    copy_file(source_manifest_path, source_snapshot)
    readme = write_dataset_readme(out_dir)
    filter_cfg = {
        "filter": filter_name,
        "strategy": strategy_cfg,
        "max_samples": max_samples,
        "sample_mode": sample_mode,
        "random_seed": random_seed,
    }
    manifest = dict(source_manifest)
    manifest.update(
        {
            "version": version,
            "version_dir": str(out_dir),
            "created_at": utc_now(),
            "source_branch_dataset_manifest": str(source_manifest_path),
            "source_branch_dataset_version": source_manifest.get("version"),
            "subset_filter": filter_cfg,
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_parquet": str(dataset_parquet) if parquet_written else None,
            "example_json": str(example_json),
            "readme": str(readme),
            "sample_count": count,
            "sample_count_before_filter": sample_count_before,
            "source_step_policy": source_manifest.get("step_policy"),
            "config_hash": stable_config_hash({"source": str(source_manifest_path), **filter_cfg}),
        }
    )
    write_json(manifest_path, manifest)
    return {
        "status": "completed",
        "filtered_branch_dataset_manifest": str(manifest_path),
        "filtered_branch_dataset_jsonl": str(dataset_jsonl),
        "filtered_branch_dataset_parquet": str(dataset_parquet) if parquet_written else None,
        "filtered_branch_dataset_version": version,
        "subset_manifest": str(manifest_path),
        "subset_jsonl": str(dataset_jsonl),
        "subset_parquet": str(dataset_parquet) if parquet_written else None,
        "subset_version": version,
        "source_branch_dataset_manifest": str(source_manifest_path),
        "source_branch_dataset_version": source_manifest.get("version"),
        "source_step_policy": source_manifest.get("step_policy"),
        "sample_count_before_filter": sample_count_before,
        "sample_count": count,
        "filter": selected_strategy_name,
        "strategy": strategy_cfg,
        "sample_mode": sample_mode,
    }


def run_from_config(config_path: Path, stage_manifest_path: Path, dry_run: bool = False) -> dict[str, Any]:
    config = read_yaml(config_path)
    stage_cfg = config["pipeline"]["stage_configs"]["filter_reranker_branch_dataset"]
    rt_cfg = config["reranker_training"]
    filter_cfg = dict(rt_cfg.get("branch_filter") or {})
    strategy = dict(filter_cfg.get("strategy") or {})
    filter_name = strategy_name(strategy, str(filter_cfg.get("filter") or "all"))

    source_manifest_value = (
        stage_cfg.get("inputs", {}).get("source_branch_dataset_manifest")
        or filter_cfg.get("source_branch_dataset_manifest")
        or rt_cfg.get("input", {}).get("branch_dataset_manifest")
        or config["pipeline"]["stage_configs"]["build_reranker_branch_dataset"].get("outputs", {}).get("branch_dataset_manifest")
    )
    if not source_manifest_value:
        raise ValueError("filter_reranker_branch_dataset requires source_branch_dataset_manifest or previous branch dataset output")
    source_manifest_path = Path(str(source_manifest_value))
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"source branch dataset manifest does not exist: {source_manifest_path}")

    out_root = Path(str(filter_cfg["out_root"])) if filter_cfg.get("out_root") else None
    out_version = filter_cfg.get("version")
    max_samples = int(filter_cfg.get("max_samples", -1))
    sample_mode = str(filter_cfg.get("sample_mode") or "none")
    random_seed = int(filter_cfg.get("random_seed") or 20260708)
    overwrite = bool(filter_cfg.get("overwrite", False))

    if dry_run:
        source_manifest = read_json(source_manifest_path)
        version = str(out_version or f"{source_manifest['version']}__filter_{filter_name}")
        base_dir = out_root or source_manifest_path.parent.parent
        out_dir = base_dir / version
        manifest_path = out_dir / "manifest.json"
        outputs = {
            "status": "compiled",
            "filtered_branch_dataset_manifest": str(manifest_path),
            "filtered_branch_dataset_version": version,
            "source_branch_dataset_manifest": str(source_manifest_path),
            "source_branch_dataset_version": source_manifest.get("version"),
            "source_step_policy": source_manifest.get("step_policy"),
            "filter": filter_name,
            "strategy": strategy,
            "max_samples": max_samples,
            "sample_mode": sample_mode,
        }
        write_json(
            stage_manifest_path,
            {
                "type": "agentic_iter_rag_stage_manifest",
                "stage": "filter_reranker_branch_dataset",
                "created_at": utc_now(),
                "config": stage_cfg,
                "outputs": outputs,
            },
        )
        return outputs

    outputs = build_subset(
        source_manifest_path=source_manifest_path,
        out_root=out_root,
        out_version=str(out_version) if out_version else None,
        filter_name=filter_name,
        strategy=strategy,
        max_samples=max_samples,
        sample_mode=sample_mode,
        random_seed=random_seed,
        overwrite=overwrite,
    )
    write_json(
        stage_manifest_path,
        {
            "type": "agentic_iter_rag_stage_manifest",
            "stage": "filter_reranker_branch_dataset",
            "created_at": utc_now(),
            "config": stage_cfg,
            "outputs": outputs,
        },
    )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter an AIR branch dataset into a hard/improvable training subset.")
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out-root", type=Path, default=None)
    parser.add_argument("--out-version", default=None)
    parser.add_argument("--filter", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--sample-mode", default="first", choices=["none", "first", "random"])
    parser.add_argument("--random-seed", type=int, default=20260708)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_subset(
        source_manifest_path=args.source_manifest,
        out_root=args.out_root,
        out_version=args.out_version,
        filter_name=args.filter,
        strategy={"kind": "builtin", "name": args.filter, "builtin_name": args.filter, "kwargs": {}},
        max_samples=args.max_samples,
        sample_mode=args.sample_mode,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(outputs)


if __name__ == "__main__":
    main()
