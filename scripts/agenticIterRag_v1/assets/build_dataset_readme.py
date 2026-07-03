#!/usr/bin/env python3
"""Generate README.md files for AgenticIterRag produced data directories."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


MAX_SCHEMA_ROWS = 100


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {"value": data}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                yield obj
            else:
                yield {"value": obj}


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def md_escape(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def fmt_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, ensure_ascii=False)}`"
    return md_escape(value)


def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def summarize_schema(path: Path, *, limit: int = MAX_SCHEMA_ROWS) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "present": 0,
            "types": Counter(),
            "list_lens": [],
            "dict_keys": Counter(),
            "examples": [],
        }
    )
    rows = 0
    for row in iter_jsonl(path):
        rows += 1
        for key, value in row.items():
            item = fields[key]
            item["present"] += 1
            item["types"][type_name(value)] += 1
            if isinstance(value, list):
                item["list_lens"].append(len(value))
            elif isinstance(value, dict):
                item["dict_keys"].update(str(k) for k in value.keys())
            if len(item["examples"]) < 1:
                item["examples"].append(example_preview(value))
        if rows >= limit:
            break
    return {"rows_scanned": rows, "fields": dict(fields)}


def example_preview(value: Any) -> str:
    if isinstance(value, str):
        text = value[:80]
        return json.dumps(text, ensure_ascii=False)
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        keys = list(value.keys())[:8]
        return "dict keys=" + json.dumps(keys, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def schema_markdown(path: Path, title: str) -> list[str]:
    count = count_jsonl(path)
    schema = summarize_schema(path)
    lines = [f"### {title}", "", f"- Records: `{count}`", f"- Schema rows scanned: `{schema['rows_scanned']}`", ""]
    fields = schema["fields"]
    if not fields:
        lines += ["No fields detected.", ""]
        return lines
    lines += ["| field | presence | types | shape / keys | example |", "|---|---:|---|---|---|"]
    for field in sorted(fields):
        info = fields[field]
        types = ", ".join(f"{k}:{v}" for k, v in sorted(info["types"].items()))
        shapes: list[str] = []
        if info["list_lens"]:
            shapes.append(f"list_len={min(info['list_lens'])}..{max(info['list_lens'])}")
        if info["dict_keys"]:
            keys = [k for k, _ in info["dict_keys"].most_common(8)]
            shapes.append("dict_keys=" + ",".join(keys))
        example = info["examples"][0] if info["examples"] else ""
        lines.append(
            f"| `{md_escape(field)}` | {info['present']}/{schema['rows_scanned']} | "
            f"{md_escape(types)} | {md_escape('; '.join(shapes))} | {md_escape(example)} |"
        )
    lines.append("")
    return lines


def file_table(files: list[tuple[str, Path, str]], base: Path) -> list[str]:
    lines = ["| file | records | bytes | purpose |", "|---|---:|---:|---|"]
    for label, path, purpose in files:
        records = count_jsonl(path) if path.suffix == ".jsonl" else "-"
        size = file_size(path) if path.exists() else 0
        target = rel(path, base)
        if path.exists():
            lines.append(f"| `{md_escape(target)}` | {records} | {size} | {md_escape(purpose)} |")
        else:
            lines.append(f"| `{md_escape(target)}` | missing | 0 | {md_escape(purpose)} |")
    lines.append("")
    return lines


def counter_from_jsonl(path: Path, key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in iter_jsonl(path):
        value = row.get(key)
        if value is None:
            continue
        counts[str(value)] += 1
    return counts


def counter_table(title: str, counts: Counter[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if not counts:
        return lines + ["No values detected.", ""]
    lines += ["| value | count |", "|---|---:|"]
    for value, count in counts.most_common():
        lines.append(f"| `{md_escape(value)}` | {count} |")
    lines.append("")
    return lines


def key_value_section(title: str, rows: list[tuple[str, Any]]) -> list[str]:
    lines = [f"## {title}", "", "| key | value |", "|---|---|"]
    for key, value in rows:
        lines.append(f"| `{md_escape(key)}` | {fmt_value(value)} |")
    lines.append("")
    return lines


def jsonl_mismatch_note(label: str, manifest_count: Any, actual_count: int) -> str:
    if manifest_count is None:
        return ""
    try:
        expected = int(manifest_count)
    except (TypeError, ValueError):
        return ""
    if expected == actual_count:
        return ""
    return f"WARNING: manifest {label}={expected}, actual records={actual_count}."


def load_source_summary(dataset_dir: Path) -> dict[str, Any]:
    summary = read_json(dataset_dir / "summary.json")
    if summary.get("source_summary"):
        return summary.get("source_summary") or {}
    return summary


def generate_trajectory_readme(dataset_dir: Path) -> str:
    manifest = read_json(dataset_dir / "manifest.json")
    summary = load_source_summary(dataset_dir)
    trajectory_jsonl = dataset_dir / "trajectory.jsonl"
    raw_trace_jsonl = dataset_dir / "raw_traces.jsonl"
    metrics_jsonl = dataset_dir / "metrics.jsonl"
    enhanced_trajectory_jsonl = dataset_dir / "enhanced_trajectory.jsonl"
    readme = dataset_dir / "README.md"

    trajectory_count = count_jsonl(trajectory_jsonl)
    raw_count = count_jsonl(raw_trace_jsonl)
    metric_count = count_jsonl(metrics_jsonl)
    enhanced_count = count_jsonl(enhanced_trajectory_jsonl)
    enhanced_summary = read_json(dataset_dir / "enhanced_summary.json")
    status_counts = Counter({str(k): int(v) for k, v in (summary.get("status_counts") or {}).items()})
    if not status_counts:
        status_counts = counter_from_jsonl(metrics_jsonl, "status")
    data_source_counts = counter_from_jsonl(metrics_jsonl, "data_source")

    lines = [
        f"# Trajectory Dataset: {manifest.get('version', dataset_dir.name)}",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
    ]
    lines += key_value_section(
        "Summary",
        [
            ("dataset_type", manifest.get("dataset_type", "trajectory")),
            ("version", manifest.get("version", dataset_dir.name)),
            ("created_at", manifest.get("created_at")),
            ("source_mode", manifest.get("source_mode")),
            ("run_mode", manifest.get("run_mode")),
            ("reranker", manifest.get("reranker")),
            ("trace_max_samples", manifest.get("trace_max_samples")),
            ("raw_trace_count_actual", raw_count),
            ("metric_count_actual", metric_count),
            ("trajectory_record_count_actual", trajectory_count),
            ("enhanced_record_count_actual", enhanced_count),
            ("enhanced_search_step_count", enhanced_summary.get("search_step_count", manifest.get("enhanced_search_step_count"))),
            ("raw_trace_count_manifest", manifest.get("raw_trace_count")),
            ("record_count_manifest", manifest.get("record_count")),
            ("enhanced_record_count_manifest", manifest.get("enhanced_record_count")),
            ("config_hash", manifest.get("config_hash")),
        ],
    )
    warnings = [
        jsonl_mismatch_note("raw_trace_count", manifest.get("raw_trace_count"), raw_count),
        jsonl_mismatch_note("record_count", manifest.get("record_count"), trajectory_count),
        jsonl_mismatch_note("enhanced_record_count", manifest.get("enhanced_record_count"), enhanced_count),
    ]
    warnings = [item for item in warnings if item]
    if warnings:
        lines += ["## Consistency Warnings", ""]
        lines += [f"- {item}" for item in warnings]
        lines.append("")

    lines += key_value_section(
        "Source",
        [
            ("source_agent_checkpoint", manifest.get("source_agent_checkpoint")),
            ("source_data_files", manifest.get("source_data_files")),
            ("final_config_yaml", manifest.get("final_config_yaml")),
            ("manifest_json", str(dataset_dir / "manifest.json")),
        ],
    )
    lines += ["## Files", ""]
    lines += file_table(
        [
            ("raw_traces", raw_trace_jsonl, "per-source-sample raw AIR inference output"),
            ("metrics", metrics_jsonl, "per-source-sample evaluation/status metrics"),
            ("trajectory", trajectory_jsonl, "canonical per-search-query trajectory records"),
            ("enhanced_trajectory", enhanced_trajectory_jsonl, "per-source-sample continuation-ready trajectory records"),
            ("enhanced_summary", dataset_dir / "enhanced_summary.json", "aggregate enhanced trajectory statistics"),
            ("enhanced_example", dataset_dir / "enhanced_example.json", "one enhanced trajectory example"),
            ("summary", dataset_dir / "summary.json", "aggregate metrics"),
            ("manifest", dataset_dir / "manifest.json", "dataset metadata"),
            ("example", dataset_dir / "example.json", "one example record"),
            ("final_config", dataset_dir / "final_config.yaml", "final merged production config"),
        ],
        dataset_dir,
    )
    lines += counter_table("Status Counts", status_counts)
    lines += counter_table("Data Source Counts", data_source_counts)
    micro = summary.get("micro") or {}
    if micro:
        lines += key_value_section(
            "Micro Metrics",
            [
                ("n", micro.get("n")),
                ("em", micro.get("em")),
                ("f1", micro.get("f1")),
                ("tool_calls", micro.get("tool_calls")),
                ("agent_decision_avg_s", micro.get("agent_decision_avg_s")),
                ("recall_avg_s", micro.get("recall_avg_s")),
                ("total_s", micro.get("total_s")),
                ("num_recall_docs", micro.get("num_recall_docs")),
                ("num_ranked_docs", micro.get("num_ranked_docs")),
                ("num_agent_visible_docs", micro.get("num_agent_visible_docs")),
            ],
        )
    lines += ["## Schemas", ""]
    lines += schema_markdown(raw_trace_jsonl, "raw_traces.jsonl")
    lines += schema_markdown(metrics_jsonl, "metrics.jsonl")
    lines += schema_markdown(trajectory_jsonl, "trajectory.jsonl")
    lines += schema_markdown(enhanced_trajectory_jsonl, "enhanced_trajectory.jsonl")
    lines += ["## Example", "", "See `example.json` for one full example record.", ""]
    return "\n".join(lines)


def find_train_manifests(dataset_dir: Path) -> list[Path]:
    train_root = dataset_dir / "train_dataset"
    if not train_root.exists():
        return []
    return sorted(train_root.glob("*/manifest.json"))


def generate_reranker_set_readme(dataset_dir: Path) -> str:
    manifest = read_json(dataset_dir / "manifest.json")
    input_manifest_path = dataset_dir / "input_dataset" / "manifest.json"
    input_manifest = read_json(input_manifest_path)
    train_manifests = [read_json(path) for path in find_train_manifests(dataset_dir)]
    input_jsonl = dataset_dir / "input_dataset" / "dataset.jsonl"

    lines = [
        f"# LLM Reranker Train Set: {manifest.get('version', dataset_dir.name)}",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
    ]
    train_counts = []
    for item in train_manifests:
        if item.get("dataset_jsonl"):
            train_counts.append((item.get("version"), count_jsonl(Path(str(item["dataset_jsonl"])))))
    lines += key_value_section(
        "Summary",
        [
            ("dataset_type", manifest.get("dataset_type", "llm_reranker_train_set")),
            ("version", manifest.get("version", dataset_dir.name)),
            ("created_at", manifest.get("created_at")),
            ("source_trajectory_version", manifest.get("source_trajectory_version")),
            ("sample_count_manifest", manifest.get("sample_count")),
            ("input_dataset_count_actual", count_jsonl(input_jsonl)),
            ("train_dataset_counts_actual", train_counts),
            ("train_dataset_versions", manifest.get("train_dataset_versions")),
            ("config_hash", manifest.get("config_hash")),
        ],
    )
    lines += key_value_section(
        "Source",
        [
            ("source_trajectory_manifest", manifest.get("source_trajectory_manifest")),
            ("input_dataset_manifest", manifest.get("input_dataset_manifest")),
            ("train_dataset_manifest", manifest.get("train_dataset_manifest")),
            ("final_config_yaml", manifest.get("final_config_yaml")),
        ],
    )
    file_rows: list[tuple[str, Path, str]] = [
        ("input_jsonl", input_jsonl, "intermediate candidate reranker input"),
        ("input_parquet", dataset_dir / "input_dataset" / "dataset.parquet", "parquet copy of input dataset"),
        ("input_manifest", input_manifest_path, "input dataset metadata"),
        ("parent_manifest", dataset_dir / "manifest.json", "reranker train set metadata"),
        ("example", dataset_dir / "example.json", "one input example"),
        ("final_config", dataset_dir / "final_config.yaml", "final merged production config"),
    ]
    for train_manifest in train_manifests:
        train_dir = Path(str(train_manifest.get("version_dir", "")))
        if train_dir:
            file_rows += [
                ("train_jsonl", train_dir / "dataset.jsonl", "final formatted training data"),
                ("train_parquet", train_dir / "dataset.parquet", "parquet copy of training data"),
                ("train_manifest", train_dir / "manifest.json", "train dataset metadata"),
            ]
    lines += ["## Files", ""]
    lines += file_table(file_rows, dataset_dir)
    lines += key_value_section(
        "Input Dataset Config",
        [
            ("schema_version", input_manifest.get("schema_version")),
            ("builder_policy", input_manifest.get("builder_policy")),
            ("candidate_source", input_manifest.get("candidate_source")),
            ("candidate_top_n", input_manifest.get("candidate_top_n")),
            ("dedupe_policy", input_manifest.get("dedupe_policy")),
            ("label_policy", input_manifest.get("label_policy")),
            ("positive_policy", input_manifest.get("positive_policy")),
            ("target_ranking_policy", input_manifest.get("target_ranking_policy")),
        ],
    )
    if train_manifests:
        train = train_manifests[0]
        lines += key_value_section(
            "Train Dataset Config",
            [
                ("format", train.get("format")),
                ("prompt_template_version", train.get("prompt_template_version")),
                ("formatter", train.get("formatter")),
                ("ground_truth_policy", train.get("ground_truth_policy")),
                ("reward_policy", train.get("reward_policy")),
                ("output_schema", train.get("output_schema")),
                ("reranker_top_m", train.get("reranker_top_m")),
                ("max_doc_chars", train.get("max_doc_chars")),
            ],
        )
    lines += ["## Schemas", ""]
    lines += schema_markdown(input_jsonl, "input_dataset/dataset.jsonl")
    for train_manifest in train_manifests:
        train_dir = Path(str(train_manifest.get("version_dir", "")))
        lines += schema_markdown(train_dir / "dataset.jsonl", f"train_dataset/{train_dir.name}/dataset.jsonl")
    lines += ["## Example", "", "See `example.json` and subdirectory `example.json` files for full examples.", ""]
    return "\n".join(lines)


def generate_input_dataset_readme(dataset_dir: Path) -> str:
    manifest = read_json(dataset_dir / "manifest.json")
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    lines = [
        f"# LLM Reranker Input Dataset: {manifest.get('version', dataset_dir.name)}",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
    ]
    lines += key_value_section(
        "Summary",
        [
            ("dataset_type", manifest.get("dataset_type", "llm_reranker_input_dataset")),
            ("version", manifest.get("version")),
            ("created_at", manifest.get("created_at")),
            ("source_trajectory_version", manifest.get("source_trajectory_version")),
            ("sample_count_manifest", manifest.get("sample_count")),
            ("sample_count_actual", count_jsonl(dataset_jsonl)),
            ("config_hash", manifest.get("config_hash")),
        ],
    )
    lines += key_value_section(
        "Config",
        [
            ("schema_version", manifest.get("schema_version")),
            ("builder_policy", manifest.get("builder_policy")),
            ("candidate_source", manifest.get("candidate_source")),
            ("candidate_top_n", manifest.get("candidate_top_n")),
            ("dedupe_policy", manifest.get("dedupe_policy")),
            ("label_policy", manifest.get("label_policy")),
            ("positive_policy", manifest.get("positive_policy")),
            ("target_ranking_policy", manifest.get("target_ranking_policy")),
        ],
    )
    lines += ["## Files", ""]
    lines += file_table(
        [
            ("dataset_jsonl", dataset_jsonl, "intermediate candidate reranker input"),
            ("dataset_parquet", dataset_dir / "dataset.parquet", "parquet copy"),
            ("manifest", dataset_dir / "manifest.json", "input dataset metadata"),
            ("example", dataset_dir / "example.json", "one example record"),
        ],
        dataset_dir,
    )
    lines += ["## Schemas", ""]
    lines += schema_markdown(dataset_jsonl, "dataset.jsonl")
    return "\n".join(lines)


def generate_train_dataset_readme(dataset_dir: Path) -> str:
    manifest = read_json(dataset_dir / "manifest.json")
    dataset_jsonl = dataset_dir / "dataset.jsonl"
    lines = [
        f"# LLM Reranker Train Dataset: {manifest.get('version', dataset_dir.name)}",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
    ]
    lines += key_value_section(
        "Summary",
        [
            ("dataset_type", manifest.get("dataset_type", "llm_reranker_train_dataset")),
            ("version", manifest.get("version")),
            ("created_at", manifest.get("created_at")),
            ("source_input_dataset_manifest", manifest.get("source_input_dataset_manifest")),
            ("sample_count_manifest", manifest.get("sample_count")),
            ("sample_count_actual", count_jsonl(dataset_jsonl)),
            ("config_hash", manifest.get("config_hash")),
        ],
    )
    lines += key_value_section(
        "Config",
        [
            ("format", manifest.get("format")),
            ("prompt_template_version", manifest.get("prompt_template_version")),
            ("formatter", manifest.get("formatter")),
            ("ground_truth_policy", manifest.get("ground_truth_policy")),
            ("reward_policy", manifest.get("reward_policy")),
            ("output_schema", manifest.get("output_schema")),
            ("reranker_top_m", manifest.get("reranker_top_m")),
            ("max_doc_chars", manifest.get("max_doc_chars")),
        ],
    )
    lines += ["## Files", ""]
    lines += file_table(
        [
            ("dataset_jsonl", dataset_jsonl, "final formatted training data"),
            ("dataset_parquet", dataset_dir / "dataset.parquet", "parquet copy"),
            ("manifest", dataset_dir / "manifest.json", "train dataset metadata"),
            ("example", dataset_dir / "example.json", "one example record"),
        ],
        dataset_dir,
    )
    lines += ["## Schemas", ""]
    lines += schema_markdown(dataset_jsonl, "dataset.jsonl")
    return "\n".join(lines)


def detect_dataset_kind(dataset_dir: Path) -> str:
    if (dataset_dir / "raw_traces.jsonl").exists() and (dataset_dir / "trajectory.jsonl").exists():
        return "trajectory"
    manifest = read_json(dataset_dir / "manifest.json")
    dataset_type = manifest.get("dataset_type")
    if dataset_type == "llm_reranker_input_dataset":
        return "reranker_input"
    if dataset_type == "llm_reranker_train_dataset":
        return "reranker_train"
    if dataset_type == "llm_reranker_train_set":
        return "reranker_set"
    if (dataset_dir / "input_dataset" / "manifest.json").exists() and (dataset_dir / "train_dataset").exists():
        return "reranker_set"
    raise ValueError(f"cannot detect supported AgenticIterRag dataset type for {dataset_dir}")


def generate_readme(dataset_dir: Path) -> Path:
    dataset_dir = dataset_dir.resolve()
    kind = detect_dataset_kind(dataset_dir)
    if kind == "trajectory":
        content = generate_trajectory_readme(dataset_dir)
    elif kind == "reranker_set":
        content = generate_reranker_set_readme(dataset_dir)
    elif kind == "reranker_input":
        content = generate_input_dataset_readme(dataset_dir)
    elif kind == "reranker_train":
        content = generate_train_dataset_readme(dataset_dir)
    else:
        raise AssertionError(kind)
    output = dataset_dir / "README.md"
    output.write_text(content, encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_readme(args.dataset_dir)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
