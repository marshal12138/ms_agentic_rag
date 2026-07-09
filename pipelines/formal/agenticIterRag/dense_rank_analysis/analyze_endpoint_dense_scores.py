#!/usr/bin/env python3
"""Analyze E5 dense recall score properties for AIR end-point reranker data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPO_ROOT / (
    "data/AgenticIterRag/llm_reranker_branch_train_set/"
    "260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/"
    "manifest.json"
)


def normalize_answer_text(text: Any) -> str:
    import string

    raw = str(text or "").lower()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    return " ".join(raw.translate(table).split())


def doc_text(doc: dict[str, Any]) -> str:
    return str(doc.get("contents") or doc.get("text") or doc.get("passage") or "")


def contains_any_answer(text: str, targets: list[Any]) -> bool:
    # Match the CoSearch public answer_in_docs rule: normalize, pad both sides
    # with spaces, then perform exact phrase containment. This avoids accidental
    # substring hits such as "yes" in "Yeshiva" while keeping wiki-18 convention.
    normalized = f" {normalize_answer_text(text)} "
    for target in targets:
        normalized_target = normalize_answer_text(target)
        if normalized_target and f" {normalized_target} " in normalized:
            return True
    return False


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def score(doc: dict[str, Any]) -> float:
    return float(doc.get("recall_score", doc.get("score", 0.0)) or 0.0)


def rank(doc: dict[str, Any], index: int) -> int:
    return int(doc.get("recall_rank") or doc.get("rank") or (index + 1))


def quantiles(values: list[float]) -> dict[str, float | int | None]:
    clean = sorted(float(v) for v in values if v is not None and not math.isnan(float(v)))
    if not clean:
        return {"n": 0}
    n = len(clean)

    def at(p: float) -> float:
        idx = min(n - 1, max(0, round((n - 1) * p)))
        return clean[idx]

    return {
        "n": n,
        "mean": sum(clean) / n,
        "std": stats.pstdev(clean) if n > 1 else 0.0,
        "min": clean[0],
        "p10": at(0.10),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "max": clean[-1],
    }


def targets(row: dict[str, Any]) -> list[Any]:
    target = ((row.get("reward_model") or {}).get("ground_truth") or {}).get("target")
    if target is None:
        return []
    if isinstance(target, list):
        return target
    return [target]


def is_yes_no_sample(row: dict[str, Any]) -> bool:
    normalized = [normalize_answer_text(target) for target in targets(row)]
    normalized = [target for target in normalized if target]
    return bool(normalized) and all(target in {"yes", "no"} for target in normalized)


def sorted_docs(row: dict[str, Any]) -> list[dict[str, Any]]:
    docs = list((row.get("extra_info") or {}).get("candidate_docs") or [])
    return sorted(docs, key=lambda d: int(d.get("recall_rank") or d.get("rank") or 999999))


def sample_id(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") or {}
    return str(row.get("sample_id") or extra.get("sample_id") or extra.get("trajectory_id") or "")


def analyze_row(row: dict[str, Any], visible_top_m: int) -> dict[str, Any]:
    docs = sorted_docs(row)
    targs = targets(row)
    extra = row.get("extra_info") or {}
    hit_docs: list[dict[str, Any]] = []
    nonhit_topm: list[dict[str, Any]] = []
    for idx, doc in enumerate(docs):
        is_hit = contains_any_answer(doc_text(doc), targs)
        enriched = {
            "rank": rank(doc, idx),
            "score": score(doc),
            "doc_id": str(doc.get("doc_id") or doc.get("id") or ""),
            "title": str(doc.get("title") or ""),
            "text": doc_text(doc),
            "is_hit": is_hit,
        }
        if is_hit:
            hit_docs.append(enriched)
        elif idx < visible_top_m:
            nonhit_topm.append(enriched)

    top_scores = [score(doc) for doc in docs]
    topm_scores = top_scores[:visible_top_m]
    top50_hit = bool(hit_docs)
    topm_hit = any(doc["rank"] <= visible_top_m for doc in hit_docs)
    best_hit = max(hit_docs, key=lambda d: d["score"]) if hit_docs else None
    best_nonhit_topm = max(nonhit_topm, key=lambda d: d["score"]) if nonhit_topm else None
    group = "top50_hit_top5_miss" if top50_hit and not topm_hit else "top5_hit" if topm_hit else "top50_miss"

    return {
        "sample_id": sample_id(row),
        "trajectory_id": str(extra.get("trajectory_id") or ""),
        "source_index": extra.get("source_index"),
        "step_index": extra.get("step_index"),
        "turn_index": extra.get("turn_index"),
        "baseline_reward": float(extra.get("baseline_reward") or 0.0),
        "question": str(extra.get("question") or ""),
        "sub_query": str(extra.get("sub_query") or ""),
        "targets": targs,
        "group": group,
        "top50_hit": top50_hit,
        "top5_hit": topm_hit,
        "top1_score": top_scores[0] if top_scores else None,
        "top5_mean_score": sum(topm_scores) / len(topm_scores) if topm_scores else None,
        "top5_min_score": min(topm_scores) if topm_scores else None,
        "top50_mean_score": sum(top_scores) / len(top_scores) if top_scores else None,
        "top50_score_range": (max(top_scores) - min(top_scores)) if top_scores else None,
        "answer_doc_count": len(hit_docs),
        "best_answer_rank": best_hit["rank"] if best_hit else None,
        "best_answer_score": best_hit["score"] if best_hit else None,
        "best_answer_doc_id": best_hit["doc_id"] if best_hit else "",
        "best_answer_title": best_hit["title"] if best_hit else "",
        "best_top5_nonanswer_score": best_nonhit_topm["score"] if best_nonhit_topm else None,
        "best_top5_nonanswer_doc_id": best_nonhit_topm["doc_id"] if best_nonhit_topm else "",
        "score_gap_top5_nonanswer_minus_answer": (
            best_nonhit_topm["score"] - best_hit["score"] if best_nonhit_topm and best_hit else None
        ),
    }


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "baseline_reward_counts": dict(Counter(row["baseline_reward"] for row in rows)),
        "step_index_counts": dict(sorted(Counter(row["step_index"] for row in rows).items())),
        "top1_score": quantiles([row["top1_score"] for row in rows if row["top1_score"] is not None]),
        "top5_mean_score": quantiles([row["top5_mean_score"] for row in rows if row["top5_mean_score"] is not None]),
        "top5_min_score": quantiles([row["top5_min_score"] for row in rows if row["top5_min_score"] is not None]),
        "top50_mean_score": quantiles([row["top50_mean_score"] for row in rows if row["top50_mean_score"] is not None]),
        "top50_score_range": quantiles([row["top50_score_range"] for row in rows if row["top50_score_range"] is not None]),
        "answer_doc_count": quantiles([row["answer_doc_count"] for row in rows]),
        "best_answer_score": quantiles([row["best_answer_score"] for row in rows if row["best_answer_score"] is not None]),
        "best_answer_rank": quantiles([row["best_answer_rank"] for row in rows if row["best_answer_rank"] is not None]),
        "best_top5_nonanswer_score": quantiles(
            [row["best_top5_nonanswer_score"] for row in rows if row["best_top5_nonanswer_score"] is not None]
        ),
        "score_gap_top5_nonanswer_minus_answer": quantiles(
            [row["score_gap_top5_nonanswer_minus_answer"] for row in rows if row["score_gap_top5_nonanswer_minus_answer"] is not None]
        ),
        "best_answer_rank_counts_top20": Counter(
            row["best_answer_rank"] for row in rows if row["best_answer_rank"] is not None
        ).most_common(20),
    }


def fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


def quant_table(metric: dict[str, Any], keys: list[str]) -> str:
    headers = ["metric", "n", "mean", "std", "min", "p25", "p50", "p75", "p90", "p95", "max"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for key in keys:
        q = metric.get(key) or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    fmt(q.get("n")),
                    fmt(q.get("mean")),
                    fmt(q.get("std")),
                    fmt(q.get("min")),
                    fmt(q.get("p25")),
                    fmt(q.get("p50")),
                    fmt(q.get("p75")),
                    fmt(q.get("p90")),
                    fmt(q.get("p95")),
                    fmt(q.get("max")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "trajectory_id",
        "source_index",
        "step_index",
        "turn_index",
        "baseline_reward",
        "group",
        "top1_score",
        "top5_mean_score",
        "top5_min_score",
        "top50_mean_score",
        "answer_doc_count",
        "best_answer_rank",
        "best_answer_score",
        "best_top5_nonanswer_score",
        "score_gap_top5_nonanswer_minus_answer",
        "question",
        "sub_query",
        "targets",
        "best_answer_doc_id",
        "best_top5_nonanswer_doc_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field) for field in fields}
            out["targets"] = json.dumps(out.get("targets"), ensure_ascii=False)
            writer.writerow(out)


def example_block(rows: list[dict[str, Any]], title: str, limit: int = 5) -> str:
    lines = [f"### {title}", ""]
    for row in rows[:limit]:
        lines.extend(
            [
                f"- sample_id: `{row['sample_id']}`",
                f"  step_index={row['step_index']} baseline_reward={row['baseline_reward']} "
                f"best_answer_rank={row['best_answer_rank']} best_answer_score={fmt(row['best_answer_score'])} "
                f"top5_nonanswer_score={fmt(row['best_top5_nonanswer_score'])} "
                f"gap={fmt(row['score_gap_top5_nonanswer_minus_answer'])}",
                f"  query: {row['sub_query'][:240]}",
                f"  answer_targets: {row['targets']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_report(
    *,
    run_id: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
    report_json: Path,
    row_csv: Path,
) -> str:
    group_a = summary["groups"]["top50_hit_top5_miss"]
    group_b = summary["groups"]["top5_hit"]
    group_c = summary["groups"]["top50_miss"]
    keys = [
        "top1_score",
        "top5_mean_score",
        "top5_min_score",
        "top50_mean_score",
        "top50_score_range",
        "answer_doc_count",
        "best_answer_score",
        "best_answer_rank",
        "best_top5_nonanswer_score",
        "score_gap_top5_nonanswer_minus_answer",
    ]

    def rows_for_group(group: str) -> list[dict[str, Any]]:
        return [row for row in summary["rows"] if row["group"] == group]

    hard_rows = rows_for_group("top50_hit_top5_miss")
    hard_largest_gap = sorted(
        hard_rows,
        key=lambda r: (r["score_gap_top5_nonanswer_minus_answer"] is not None, r["score_gap_top5_nonanswer_minus_answer"] or -999),
        reverse=True,
    )
    hard_near_miss = sorted(
        hard_rows,
        key=lambda r: abs(r["score_gap_top5_nonanswer_minus_answer"] or 999),
    )
    hit_rows = rows_for_group("top5_hit")
    hit_low_margin = sorted(
        hit_rows,
        key=lambda r: r["best_answer_score"] or 999,
    )

    lines = [
        "# AIR End-Point Dense Rank Score Analysis",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at: `{summary['generated_at']}`",
        f"- source_manifest: `{manifest_path}`",
        f"- source_dataset_jsonl: `{manifest.get('dataset_jsonl')}`",
        f"- prompt_template_version: `{manifest.get('prompt_template_version')}`",
        f"- candidate_top_n: `{manifest.get('candidate_top_n')}`",
        f"- visible_top_m: `{manifest.get('visible_top_m')}`",
        f"- answer_hit_rule: `{summary.get('answer_hit_rule')}`",
        f"- excluded_answer_types: `{summary.get('excluded_answer_types')}`",
        f"- excluded_yes_no_count: `{summary.get('excluded_yes_no_count')}`",
        f"- json_summary: `{report_json}`",
        f"- row_metrics_csv: `{row_csv}`",
        "",
        "## Analysis Objective",
        "",
        "Compare dense E5 recall score properties for end-point search queries between:",
        "",
        "- Group A: `top50_hit_top5_miss`, where top50 contains answer evidence but original dense top5 does not.",
        "- Group B: `top5_hit`, where original dense top5 already contains answer evidence.",
        "- Group C: `top50_miss`, where top50 contains no answer evidence.",
        "",
        "The query object is the final search query of each AIR trajectory (`step_policy=end_point`).",
        "",
        "## Group Counts",
        "",
        "| group | count | ratio |",
        "| --- | ---: | ---: |",
        f"| A top50_hit_top5_miss | {group_a['count']} | {group_a['count'] / summary['total_count']:.4f} |",
        f"| B top5_hit | {group_b['count']} | {group_b['count'] / summary['total_count']:.4f} |",
        f"| C top50_miss | {group_c['count']} | {group_c['count'] / summary['total_count']:.4f} |",
        f"| total | {summary['total_count']} | 1.0000 |",
        "",
        "## Group A: top50_hit_top5_miss",
        "",
        quant_table(group_a, keys),
        "",
        "## Group B: top5_hit",
        "",
        quant_table(group_b, keys),
        "",
        "## Group C: top50_miss",
        "",
        quant_table(group_c, keys),
        "",
        "## Key Comparisons",
        "",
        "| metric | Group A top50_hit_top5_miss | Group B top5_hit | delta A-B |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ["top1_score", "top5_mean_score", "best_answer_score", "best_answer_rank", "answer_doc_count"]:
        av = (group_a.get(key) or {}).get("mean")
        bv = (group_b.get(key) or {}).get("mean")
        delta = av - bv if av is not None and bv is not None else None
        lines.append(f"| {key} mean | {fmt(av)} | {fmt(bv)} | {fmt(delta)} |")
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- Group A is not a recall-miss group; answer evidence is in top50 but dense score ranks it below top5.",
            "- Group A's top1/top5 dense scores are close to Group B, but its best answer-evidence score is much lower.",
            "- The key failure mode is a positive gap between top5 non-answer dense score and best answer-evidence dense score.",
            "- For Group A, this gap quantifies how much reranker must overcome dense retriever ordering.",
            "",
            "## Rank Count Snapshots",
            "",
            "### Group A best answer rank top20",
            "",
            "```json",
            json.dumps(group_a["best_answer_rank_counts_top20"], ensure_ascii=False, indent=2),
            "```",
            "",
            "### Group B best answer rank top20",
            "",
            "```json",
            json.dumps(group_b["best_answer_rank_counts_top20"], ensure_ascii=False, indent=2),
            "```",
            "",
            example_block(hard_largest_gap, "Group A examples with largest dense score gap"),
            "",
            example_block(hard_near_miss, "Group A near-boundary examples"),
            "",
            example_block(hit_low_margin, "Group B examples with low answer-evidence score"),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "reports")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = read_json(manifest_path)
    dataset_jsonl = Path(str(manifest["dataset_jsonl"]))
    visible_top_m = int(manifest.get("visible_top_m") or 5)
    run_id = args.run_id or datetime.now().strftime("%y%m%d-%H%M%S-dense_rank_endpoint_score_analysis")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped_yes_no = 0
    rows = []
    for row in iter_jsonl(dataset_jsonl):
        if is_yes_no_sample(row):
            skipped_yes_no += 1
            continue
        rows.append(analyze_row(row, visible_top_m))
    groups = {
        group: group_metrics([row for row in rows if row["group"] == group])
        for group in ["top50_hit_top5_miss", "top5_hit", "top50_miss"]
    }
    summary = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": str(manifest_path),
        "source_dataset_jsonl": str(dataset_jsonl),
        "total_count": len(rows),
        "excluded_yes_no_count": skipped_yes_no,
        "answer_hit_rule": "cosearch_boundary_normalized_phrase_match",
        "excluded_answer_types": ["yes", "no"],
        "groups": groups,
        "rows": rows,
    }

    json_summary = out_dir / f"{run_id}.summary.json"
    row_csv = out_dir / f"{run_id}.row_metrics.csv"
    report_md = out_dir / f"{run_id}.report.md"
    json_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(row_csv, rows)
    report_md.write_text(
        render_report(
            run_id=run_id,
            manifest_path=manifest_path,
            manifest=manifest,
            summary=summary,
            report_json=json_summary,
            row_csv=row_csv,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"report": str(report_md), "summary": str(json_summary), "csv": str(row_csv)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
