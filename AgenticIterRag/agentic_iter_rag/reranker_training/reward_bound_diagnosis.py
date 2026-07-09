"""AIR reranker stage2 reward-bound diagnosis.

This script does not train the reranker. It evaluates fixed, legal reranker
actions through the same continuation reward path used by stage2 GRPO.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import string
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentic_iter_rag.reranker_training.continuation_reward import (
    compute_air_branch_continuation_reward_details,
    load_tokenizer,
)
from agentic_iter_rag.reranker_training.service_manager import TrainingServiceManager
from agentic_iter_rag.reranker_training.trainer_entry import (
    build_verl_env_vars,
    default_verl_root,
    phase_services_for_config,
    project_root,
    repo_root,
    resolve_agent_model,
)
from agentic_iter_rag.utils.io import iter_jsonl, read_json, read_yaml, write_json, write_jsonl


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_answer_text(text: Any) -> str:
    raw = str(text or "").lower()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    return " ".join(raw.translate(table).split())


def doc_text(doc: dict[str, Any]) -> str:
    return str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")


def contains_any_answer(text: str, targets: list[Any]) -> bool:
    normalized = normalize_answer_text(text)
    for target in targets:
        normalized_target = normalize_answer_text(target)
        if normalized_target and normalized_target in normalized:
            return True
    return False


def answer_hit_indices(sample: dict[str, Any]) -> list[int]:
    targets = list((sample.get("reward_model") or {}).get("ground_truth", {}).get("target") or [])
    docs = list((sample.get("extra_info") or {}).get("candidate_docs") or [])
    return [idx for idx, doc in enumerate(docs, start=1) if contains_any_answer(doc_text(doc), targets)]


def response_from_indices(indices: list[int], *, strategy: str) -> str:
    chain = " > ".join(f"[{idx}]" for idx in indices)
    return (
        f"<reason>{strategy} selects a legal top-5 document action for reward-bound diagnosis.</reason>\n"
        f"<rerank>{chain}</rerank>"
    )


def identity_indices(sample: dict[str, Any], visible_top_m: int) -> list[int]:
    del sample
    return list(range(1, visible_top_m + 1))


def random_indices(sample: dict[str, Any], visible_top_m: int, candidate_top_n: int, seed: int) -> list[int]:
    key = str(sample.get("sample_id") or sample.get("extra_info", {}).get("trajectory_id") or "")
    rng = random.Random(f"{seed}:{key}")
    values = list(range(1, candidate_top_n + 1))
    rng.shuffle(values)
    return values[:visible_top_m]


def oracle_indices(sample: dict[str, Any], visible_top_m: int, candidate_top_n: int) -> list[int]:
    hits = answer_hit_indices(sample)
    selected: list[int] = []
    for idx in hits:
        if 1 <= idx <= candidate_top_n and idx not in selected:
            selected.append(idx)
        if len(selected) >= visible_top_m:
            return selected[:visible_top_m]
    for idx in range(1, candidate_top_n + 1):
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= visible_top_m:
            break
    return selected[:visible_top_m]


def sample_passes_filter(sample: dict[str, Any], sample_filter: str) -> bool:
    if sample_filter == "all":
        return True
    extra = sample.get("extra_info") or {}
    docs = list(extra.get("candidate_docs") or [])
    targets = list((sample.get("reward_model") or {}).get("ground_truth", {}).get("target") or [])
    top50_hit = contains_any_answer("\n".join(doc_text(doc) for doc in docs), targets)
    top5_hit = contains_any_answer("\n".join(doc_text(doc) for doc in docs[:5]), targets)
    baseline = float(extra.get("baseline_reward") or 0.0)
    if sample_filter == "top50_hit_top5_miss":
        return top50_hit and not top5_hit
    if sample_filter == "top50_hit_top5_miss_baseline0":
        return top50_hit and not top5_hit and baseline == 0.0
    if sample_filter == "top50_hit":
        return top50_hit
    if sample_filter == "top50_miss":
        return not top50_hit
    raise ValueError(f"unsupported sample_filter={sample_filter!r}")


def load_samples(
    branch_manifest_path: Path,
    *,
    max_samples: int,
    sample_filter: str,
    sample_mode: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(branch_manifest_path)
    data_path = Path(str(manifest["dataset_jsonl"]))
    if sample_mode == "random":
        rows = [row for row in iter_jsonl(data_path) if sample_passes_filter(row, sample_filter)]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if max_samples > 0:
            rows = rows[:max_samples]
    elif sample_mode == "first":
        rows = []
        for row in iter_jsonl(data_path):
            if not sample_passes_filter(row, sample_filter):
                continue
            rows.append(row)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    elif sample_mode != "first":
        raise ValueError(f"unsupported sample_mode={sample_mode!r}")
    if not rows:
        raise ValueError(f"no samples selected from {branch_manifest_path} with filter={sample_filter}")
    return manifest, rows


def prepare_stage2_config(config: dict[str, Any]) -> dict[str, Any]:
    rt_cfg = config["reranker_training"]
    phase_cfg = dict(rt_cfg.get("training_phases", {}).get("stage2_agentic") or {})
    if not phase_cfg:
        raise ValueError("config is missing reranker_training.training_phases.stage2_agentic")
    rt_cfg["_active_phase_name"] = "stage2_agentic"
    rt_cfg["_active_phase_config"] = phase_cfg
    return config


def set_continuation_env(config: dict[str, Any], agent_model: Path) -> dict[str, str]:
    env_vars = build_verl_env_vars(config, agent_model)
    os.environ.update(env_vars)
    return env_vars


def maybe_start_services(
    *,
    config: dict[str, Any],
    runtime_dir: Path,
    start_services: bool,
) -> tuple[TrainingServiceManager | None, dict[str, Any], Path]:
    agent_model = resolve_agent_model(config)
    if not start_services:
        set_continuation_env(config, agent_model)
        return None, {"status": "external"}, agent_model

    services = phase_services_for_config(config, "stage2_agentic")
    manager = TrainingServiceManager(
        repo_root=repo_root(),
        project_root=project_root(),
        verl_root=Path(str(config["reranker_training"]["trainer"].get("verl_root") or default_verl_root())),
        runtime_dir=runtime_dir,
        config=config,
    )
    outputs: dict[str, Any] = {}
    try:
        outputs["recall"] = manager.start_recall(services["recall"])
        outputs["frozen_agent"] = manager.start_frozen_agent(services["frozen_agent_vllm"], agent_model=agent_model)
        env_vars = set_continuation_env(config, agent_model)
        outputs["env_vars"] = env_vars
        return manager, outputs, agent_model
    except BaseException:
        manager.stop_all()
        raise


def evaluate_one(
    *,
    sample: dict[str, Any],
    strategy: str,
    solution: str,
    expected_count: int,
    max_index: int,
    format_penalty: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    extra = sample.get("extra_info") or {}
    ground_truth = sample.get("reward_model", {}).get("ground_truth", {})
    try:
        details = compute_air_branch_continuation_reward_details(
            data_source=str(sample.get("data_source") or "agentic_iter_rag.llm_reranker.branch_grpo"),
            solution_str=solution,
            ground_truth=ground_truth,
            extra_info=extra,
            expected_count=expected_count,
            max_index=max_index,
            format_penalty=format_penalty,
            reward_strategy="answer_reward",
        )
        error = None
    except Exception as exc:  # Keep the run inspectable even if one continuation fails.
        details = {"score": None, "valid": False, "format_valid": False, "continuation_status": "exception"}
        error = repr(exc)
    elapsed_s = time.perf_counter() - started
    baseline = extra.get("baseline_reward")
    score = details.get("score")
    return {
        "sample_id": sample.get("sample_id"),
        "trajectory_id": extra.get("trajectory_id"),
        "step_index": extra.get("step_index"),
        "strategy": strategy,
        "score": score,
        "baseline_reward": baseline,
        "delta_vs_baseline": (float(score) - float(baseline)) if score is not None and baseline is not None else None,
        "format_valid": details.get("format_valid"),
        "format_error_code": details.get("format_error_code"),
        "continuation_status": details.get("continuation_status"),
        "answer": details.get("answer"),
        "visible_doc_ids": details.get("visible_doc_ids"),
        "assistant_turns": details.get("assistant_turns"),
        "user_turns": details.get("user_turns"),
        "search_count": details.get("search_count"),
        "elapsed_s": elapsed_s,
        "error": error,
        "solution": solution,
        "gold_answers": list(ground_truth.get("target") or []),
        "answer_hit_indices": answer_hit_indices(sample),
    }


def build_jobs(
    samples: list[dict[str, Any]],
    *,
    strategies: list[str],
    visible_top_m: int,
    candidate_top_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for sample in samples:
        for strategy in strategies:
            if strategy == "identity":
                indices = identity_indices(sample, visible_top_m)
            elif strategy == "random":
                indices = random_indices(sample, visible_top_m, candidate_top_n, seed)
            elif strategy == "oracle":
                indices = oracle_indices(sample, visible_top_m, candidate_top_n)
            else:
                raise ValueError(f"unsupported fixed strategy={strategy!r}")
            jobs.append(
                {
                    "sample": sample,
                    "strategy": strategy,
                    "solution": response_from_indices(indices, strategy=strategy),
                }
            )
    return jobs


def summarize_results(results: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_strategy[str(row["strategy"])].append(row)

    sample_baselines = [
        float((sample.get("extra_info") or {}).get("baseline_reward"))
        for sample in samples
        if (sample.get("extra_info") or {}).get("baseline_reward") is not None
    ]
    summary: dict[str, Any] = {
        "created_at": utc_now(),
        "sample_count": len(samples),
        "baseline": metric_summary(sample_baselines),
        "strategies": {},
    }
    for strategy, rows in sorted(by_strategy.items()):
        scores = [float(row["score"]) for row in rows if row.get("score") is not None]
        deltas = [float(row["delta_vs_baseline"]) for row in rows if row.get("delta_vs_baseline") is not None]
        summary["strategies"][strategy] = {
            "count": len(rows),
            "score": metric_summary(scores),
            "delta_vs_baseline": metric_summary(deltas),
            "format_valid_rate": sum(1 for row in rows if row.get("format_valid")) / max(1, len(rows)),
            "exception_count": sum(1 for row in rows if row.get("error")),
            "continuation_status": dict(Counter(str(row.get("continuation_status")) for row in rows)),
            "search_count": dict(Counter(str(row.get("search_count")) for row in rows)),
        }
    return summary


def metric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "max": ordered[-1],
        "stdev": statistics.pstdev(ordered) if len(ordered) > 1 else 0.0,
    }


def write_markdown_report(path: Path, *, args: argparse.Namespace, summary: dict[str, Any], outputs: dict[str, Any]) -> None:
    lines = [
        "# AIR Stage2 Reward Bound Diagnosis",
        "",
        f"- created_at: `{summary['created_at']}`",
        f"- config_yaml: `{args.config_yaml}`",
        f"- branch_manifest: `{args.branch_manifest}`",
        f"- sample_filter: `{args.sample_filter}`",
        f"- sample_mode: `{args.sample_mode}`",
        f"- sample_count: `{summary['sample_count']}`",
        f"- services: `{outputs.get('status', 'started')}`",
        "",
        "## Baseline",
        "",
        metric_markdown_table({"baseline": summary["baseline"]}),
        "",
        "## Strategies",
        "",
        metric_markdown_table({name: item["score"] for name, item in summary["strategies"].items()}),
        "",
        "## Delta vs Baseline",
        "",
        metric_markdown_table({name: item["delta_vs_baseline"] for name, item in summary["strategies"].items()}),
        "",
        "## Status",
        "",
    ]
    for name, item in summary["strategies"].items():
        lines.append(f"- `{name}`: format_valid_rate={item['format_valid_rate']:.3f}, statuses={item['continuation_status']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_markdown_table(items: dict[str, dict[str, Any]]) -> str:
    lines = ["| name | count | mean | min | p50 | max | stdev |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, metric in items.items():
        if not metric or metric.get("count", 0) == 0:
            lines.append(f"| {name} | 0 | N/A | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            "| {name} | {count} | {mean:.6f} | {min:.6f} | {p50:.6f} | {max:.6f} | {stdev:.6f} |".format(
                name=name,
                count=metric["count"],
                mean=metric["mean"],
                min=metric["min"],
                p50=metric["p50"],
                max=metric["max"],
                stdev=metric["stdev"],
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AIR stage2 fixed-policy reward-bound diagnosis.")
    parser.add_argument("--config-yaml", required=True, type=Path)
    parser.add_argument("--branch-manifest", type=Path, default=None)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--sample-filter", default="all", choices=["all", "top50_hit", "top50_miss", "top50_hit_top5_miss", "top50_hit_top5_miss_baseline0"])
    parser.add_argument("--sample-mode", default="first", choices=["first", "random"])
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--strategies", nargs="+", default=["identity", "random", "oracle"], choices=["identity", "random", "oracle"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-start-services", action="store_true")
    parser.add_argument("--keep-services", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = args.out_dir / "runtime_services"
    config = prepare_stage2_config(read_yaml(args.config_yaml))
    branch_manifest = args.branch_manifest
    if branch_manifest is None:
        branch_manifest = Path(str(config["reranker_training"]["input"]["branch_dataset_manifest"]))
    args.branch_manifest = branch_manifest
    manifest, samples = load_samples(
        branch_manifest,
        max_samples=args.max_samples,
        sample_filter=args.sample_filter,
        sample_mode=args.sample_mode,
        seed=args.seed,
    )
    visible_top_m = int(manifest.get("visible_top_m") or config["reranker_training"]["branch_dataset"]["visible_top_m"])
    candidate_top_n = int(manifest.get("candidate_top_n") or config["reranker_training"]["branch_dataset"]["candidate_top_n"])
    format_penalty = float(
        config["reranker_training"]["training_phases"]["stage2_agentic"].get(
            "format_invalid_score",
            config["reranker_training"].get("reward", {}).get("format_penalty", -0.5),
        )
    )

    manager: TrainingServiceManager | None = None
    service_outputs: dict[str, Any] = {}
    try:
        manager, service_outputs, agent_model = maybe_start_services(
            config=config,
            runtime_dir=runtime_dir,
            start_services=not args.no_start_services,
        )
        # Avoid concurrent lazy imports/tokenizer initialization inside worker threads.
        load_tokenizer()
        jobs = build_jobs(
            samples,
            strategies=list(args.strategies),
            visible_top_m=visible_top_m,
            candidate_top_n=candidate_top_n,
            seed=args.seed,
        )
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [
                executor.submit(
                    evaluate_one,
                    sample=job["sample"],
                    strategy=job["strategy"],
                    solution=job["solution"],
                    expected_count=visible_top_m,
                    max_index=candidate_top_n,
                    format_penalty=format_penalty,
                )
                for job in jobs
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[reward-bound] {result['strategy']} {result['sample_id']} "
                    f"score={result['score']} status={result['continuation_status']} elapsed_s={result['elapsed_s']:.2f}",
                    flush=True,
                )
        results.sort(key=lambda row: (str(row["sample_id"]), str(row["strategy"])))
        summary = summarize_results(results, samples)
        summary.update(
            {
                "branch_manifest": str(branch_manifest),
                "config_yaml": str(args.config_yaml),
                "sample_filter": args.sample_filter,
                "sample_mode": args.sample_mode,
                "strategies_requested": list(args.strategies),
                "agent_model": str(agent_model),
                "service_outputs": service_outputs,
            }
        )
        write_jsonl(args.out_dir / "reward_bound_results.jsonl", results)
        write_json(args.out_dir / "reward_bound_summary.json", summary)
        write_markdown_report(args.out_dir / "reward_bound_report.md", args=args, summary=summary, outputs=service_outputs)
        print(f"[reward-bound] wrote {args.out_dir / 'reward_bound_report.md'}", flush=True)
    finally:
        if manager is not None and not args.keep_services:
            manager.stop_all()


if __name__ == "__main__":
    main()
