#!/usr/bin/env python3
"""Aggregate independent new-data AIR evaluations and paired comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SINGLE_HOP = {"nq", "popqa", "triviaqa"}
MULTI_HOP = {"2wikimultihopqa", "bamboogle", "hotpotqa", "musique"}
SCALAR_METRICS = (
    "em",
    "f1",
    "structured_em",
    "answer_group_f1",
    "answer_group_recall",
    "valid_complete_answer_rate",
    "first_search_rate",
    "search_count",
    "unique_query_count",
    "duplicate_query_rate",
    "max_turns_rate",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(rows: Iterable[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return statistics.fmean(values) if values else 0.0


def normalize_query(query: Any) -> str:
    return " ".join(str(query or "").lower().split())


def metric_row(
    trace: dict[str, Any],
    expected_source: str,
    expected_question: str,
) -> dict[str, Any]:
    metrics = trace.get("metrics") or {}
    source = str(trace.get("data_source") or metrics.get("data_source") or "")
    if source != expected_source:
        raise ValueError(
            f"data source mismatch at index {trace.get('index')}: {source!r} != {expected_source!r}"
        )
    trace_question = str((trace.get("enhanced_trajectory") or {}).get("question") or "")
    if trace_question != expected_question:
        raise ValueError(
            f"question mismatch at index {trace.get('index')}: "
            f"{trace_question!r} != {expected_question!r}"
        )
    queries = [normalize_query(query) for query in trace.get("sub_queries") or []]
    queries = [query for query in queries if query]
    unique_queries = set(queries)
    status = str(trace.get("status") or metrics.get("status") or "")
    tool_calls = int(metrics.get("tool_calls", len(queries)))
    return {
        "index": int(trace["index"]),
        "data_source": source,
        "em": float(metrics.get("em", metrics.get("legacy_em", 0.0))),
        "f1": float(metrics.get("f1", metrics.get("legacy_f1", 0.0))),
        "structured_em": float(metrics.get("structured_em", metrics.get("em", 0.0))),
        "answer_group_f1": float(metrics.get("answer_group_f1", metrics.get("f1", 0.0))),
        "answer_group_recall": float(
            metrics.get("answer_group_recall", metrics.get("em", 0.0))
        ),
        "valid_complete_answer_rate": float(bool(str(trace.get("final_answer") or "").strip())),
        "first_search_rate": float(tool_calls > 0),
        "search_count": float(tool_calls),
        "unique_query_count": float(len(unique_queries)),
        "duplicate_query_rate": float(len(queries) > len(unique_queries)),
        "max_turns_rate": float(status in {"max_turns", "max_user_turns"}),
        "status": status,
    }


def scalar_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"n": len(rows)}
    result.update({key: mean(rows, key) for key in SCALAR_METRICS})
    result["status_counts"] = dict(sorted(Counter(row["status"] for row in rows).items()))
    buckets = Counter(
        "5+" if row["search_count"] >= 5 else str(int(row["search_count"])) for row in rows
    )
    result["search_count_buckets"] = {
        name: {"count": buckets.get(name, 0), "rate": buckets.get(name, 0) / len(rows) if rows else 0.0}
        for name in ("0", "1", "2", "3", "4", "5+")
    }
    return result


def macro_metric(by_source: dict[str, dict[str, Any]], sources: set[str], key: str) -> float:
    values = [float(by_source[source][key]) for source in sorted(sources) if source in by_source]
    return statistics.fmean(values) if values else 0.0


def run_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["data_source"]].append(row)
    by_source = {source: scalar_summary(items) for source, items in sorted(grouped.items())}
    return {
        "overall": scalar_summary(rows),
        "by_data_source": by_source,
        "single_hop_macro": {
            key: macro_metric(by_source, SINGLE_HOP, key) for key in SCALAR_METRICS
        },
        "multi_hop_macro": {
            key: macro_metric(by_source, MULTI_HOP, key) for key in SCALAR_METRICS
        },
    }


def aggregate_run_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"run_count": len(summaries)}
    for scope in ("overall", "single_hop_macro", "multi_hop_macro"):
        result[scope] = {}
        for key in SCALAR_METRICS:
            values = [float(item[scope][key]) for item in summaries]
            result[scope][key] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            }
    sources = sorted({source for item in summaries for source in item["by_data_source"]})
    result["by_data_source"] = {}
    for source in sources:
        result["by_data_source"][source] = {}
        for key in SCALAR_METRICS:
            values = [float(item["by_data_source"][source][key]) for item in summaries]
            result["by_data_source"][source][key] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    result["search_count_buckets"] = {}
    for bucket in ("0", "1", "2", "3", "4", "5+"):
        values = [float(item["overall"]["search_count_buckets"][bucket]["rate"]) for item in summaries]
        result["search_count_buckets"][bucket] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return result


def average_per_question(runs: list[list[dict[str, Any]]], key: str) -> np.ndarray:
    return np.asarray(
        [statistics.fmean(float(run[index][key]) for run in runs) for index in range(len(runs[0]))],
        dtype=np.float64,
    )


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError(f"paired arrays differ: {left.shape} != {right.shape}")
    delta = right - left
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap_means = delta[draws].mean(axis=1)
    return {
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "delta_right_minus_left": float(delta.mean()),
        "ci95_low": float(np.percentile(bootstrap_means, 2.5)),
        "ci95_high": float(np.percentile(bootstrap_means, 97.5)),
        "bootstrap_samples": samples,
    }


def resolve_run(task_name: str) -> tuple[Path, Path, Path]:
    base = ROOT / "log/eval/agenticIterRag" / task_name
    return (
        base / "trace/traces.jsonl",
        base / "trace/metrics.jsonl",
        base / "runtime_logs/eval_run_manifest.json",
    )


def load_model_runs(
    model_name: str,
    run_names: list[str],
    expected_sources: list[str],
    expected_questions: list[str],
    data_sha256: str,
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not run_names:
        raise ValueError(f"{model_name} must have at least one independent run")
    all_rows: list[list[dict[str, Any]]] = []
    run_metadata: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for repeat_id, task_name in enumerate(run_names, start=1):
        trace_path, metrics_path, manifest_path = resolve_run(task_name)
        manifest = read_json(manifest_path)
        if manifest.get("task_name") != task_name:
            raise ValueError(f"{task_name} manifest task_name mismatch")
        if manifest["data"]["sha256"] != data_sha256:
            raise ValueError(f"{task_name} used unexpected eval data")
        if manifest.get("repeat_id") != repeat_id:
            raise ValueError(f"{task_name} repeat_id must be {repeat_id}")
        if manifest.get("output_reuse") is not False:
            raise ValueError(f"{task_name} does not prove fresh output isolation")
        fingerprints.add(str(manifest["model"]["fingerprint"]))
        traces = read_jsonl(trace_path)
        if len(traces) != len(expected_sources):
            raise ValueError(f"{task_name} has {len(traces)} traces, expected {len(expected_sources)}")
        by_index = {int(trace["index"]): trace for trace in traces}
        if sorted(by_index) != list(range(len(expected_sources))):
            raise ValueError(f"{task_name} trace indices are incomplete or duplicated")
        rows = [
            metric_row(by_index[index], expected_sources[index], expected_questions[index])
            for index in range(len(expected_sources))
        ]
        all_rows.append(rows)
        summary = run_summary(rows)
        summaries.append(summary)
        run_metadata.append(
            {
                "repeat_id": repeat_id,
                "task_name": task_name,
                "trace_path": str(trace_path),
                "trace_sha256": sha256_file(trace_path),
                "metrics_sha256": sha256_file(metrics_path),
                "manifest_path": str(manifest_path),
                "model": manifest["model"],
                "summary": summary,
            }
        )
    if len(fingerprints) != 1:
        raise ValueError(f"{model_name} repeats used different model fingerprints: {fingerprints}")
    return all_rows, run_metadata, summaries


def render_markdown(summary: dict[str, Any]) -> str:
    run_counts = {model["aggregate"]["run_count"] for model in summary["models"].values()}
    run_description = (
        f"Each model has {next(iter(run_counts))} isolated inference run(s)."
        if len(run_counts) == 1
        else "Models have the isolated inference run counts recorded below."
    )
    lines = [
        "# New-Data Model Evaluation",
        "",
        f"- Dataset: `{summary['data_path']}`",
        f"- Dataset SHA256: `{summary['data_sha256']}`",
        f"- {run_description} Repeats are not pooled as independent examples.",
        "- Paired bootstrap averages repeats per question before resampling questions.",
        "",
        "## Overall",
        "",
        "| Model | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in summary["models"].items():
        values = model["aggregate"]["overall"]
        fmt = lambda key: f"{values[key]['mean']:.4f} +/- {values[key]['std']:.4f}"
        lines.append(
            f"| {name} | {fmt('em')} | {fmt('f1')} | {fmt('structured_em')} | "
            f"{fmt('answer_group_f1')} | {fmt('answer_group_recall')} | "
            f"{fmt('valid_complete_answer_rate')} | "
            f"{fmt('first_search_rate')} | {fmt('search_count')} | {fmt('duplicate_query_rate')} | "
            f"{fmt('max_turns_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Per Run",
            "",
            "| Model | Repeat | EM | F1 | Structured EM | Group F1 | Group recall | Valid answer | Search rate | Searches | Duplicate query | Max turns |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, model in summary["models"].items():
        for run in model["runs"]:
            values = run["summary"]["overall"]
            lines.append(
                f"| {name} | {run['repeat_id']} | {values['em']:.4f} | {values['f1']:.4f} | "
                f"{values['structured_em']:.4f} | {values['answer_group_f1']:.4f} | "
                f"{values['answer_group_recall']:.4f} | "
                f"{values['valid_complete_answer_rate']:.4f} | {values['first_search_rate']:.4f} | "
                f"{values['search_count']:.4f} | {values['duplicate_query_rate']:.4f} | "
                f"{values['max_turns_rate']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Search Count Buckets",
            "",
            "| Model | 0 | 1 | 2 | 3 | 4 | 5+ |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, model in summary["models"].items():
        buckets = model["aggregate"]["search_count_buckets"]
        cells = [f"{buckets[key]['mean']:.4f} +/- {buckets[key]['std']:.4f}" for key in ("0", "1", "2", "3", "4", "5+")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Per Data Source",
            "",
            "| Model | Data source | EM | F1 | Structured EM | Group F1 | Group recall |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, model in summary["models"].items():
        for source, values in model["aggregate"]["by_data_source"].items():
            lines.append(
                f"| {name} | {source} | {values['em']['mean']:.4f} +/- {values['em']['std']:.4f} | "
                f"{values['f1']['mean']:.4f} +/- {values['f1']['std']:.4f} | "
                f"{values['structured_em']['mean']:.4f} +/- {values['structured_em']['std']:.4f} | "
                f"{values['answer_group_f1']['mean']:.4f} +/- {values['answer_group_f1']['std']:.4f} | "
                f"{values['answer_group_recall']['mean']:.4f} +/- {values['answer_group_recall']['std']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Paired Comparisons",
            "",
            "| Comparison | Metric | Delta (right-left) | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for comparison, metrics in summary["paired_comparisons"].items():
        for metric, item in metrics.items():
            lines.append(
                f"| {comparison} | {metric} | {item['delta_right_minus_left']:.4f} | "
                f"[{item['ci95_low']:.4f}, {item['ci95_high']:.4f}] |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-spec", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec = read_json(args.run_spec)
    data_path = args.data.resolve()
    data_sha256 = sha256_file(data_path)
    data = pd.read_parquet(data_path).reset_index(drop=True)
    expected_sources = [str(source) for source in data["data_source"]]
    expected_questions = [str(extra["question"]) for extra in data["extra_info"]]
    if not expected_sources:
        raise ValueError("new-data comparison requires a non-empty eval dataset")
    expected_source_set = SINGLE_HOP | MULTI_HOP
    if set(expected_sources) != expected_source_set:
        raise ValueError(f"unexpected eval sources: {sorted(set(expected_sources))}")

    model_rows: dict[str, list[list[dict[str, Any]]]] = {}
    models: dict[str, Any] = {}
    for model_name, model_spec in spec["models"].items():
        rows, run_metadata, run_summaries = load_model_runs(
            model_name,
            list(model_spec["runs"]),
            expected_sources,
            expected_questions,
            data_sha256,
        )
        model_rows[model_name] = rows
        models[model_name] = {
            "role": model_spec.get("role"),
            "scale": model_spec.get("scale"),
            "stage": model_spec.get("stage"),
            "runs": run_metadata,
            "aggregate": aggregate_run_summaries(run_summaries),
        }

    paired: dict[str, Any] = {}
    for comparison in spec["comparisons"]:
        left, right = comparison
        if left not in model_rows or right not in model_rows:
            raise ValueError(f"unknown comparison: {comparison}")
        paired[f"{left} -> {right}"] = {
            key: paired_bootstrap(
                average_per_question(model_rows[left], key),
                average_per_question(model_rows[right], key),
                samples=args.bootstrap_samples,
                seed=args.seed,
            )
            for key in (
                "em",
                "f1",
                "structured_em",
                "answer_group_f1",
                "answer_group_recall",
                "valid_complete_answer_rate",
            )
        }

    summary = {
        "version": "newdata-model-eval-v1",
        "data_path": str(data_path),
        "data_sha256": data_sha256,
        "repeat_handling": "average_per_question_then_paired_bootstrap",
        "models": models,
        "paired_comparisons": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    report_path = args.output_dir / "report.md"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(summary), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
