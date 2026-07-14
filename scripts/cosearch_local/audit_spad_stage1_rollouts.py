#!/usr/bin/env python3
"""Validate and summarize SPAD Stage1 group rewards from full rollout shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_shard(path: Path, *, group_size: int, partial_weight: float) -> dict[str, Any]:
    rows = read_jsonl(path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    formula_errors: list[str] = []
    for row_index, row in enumerate(rows):
        uid = str(row.get("uid") or row.get("group_uid") or "")
        if not uid:
            formula_errors.append(f"row {row_index}: missing uid")
            continue
        groups[uid].append(row)
        for key in ("score", "em_reward", "teacher_status_reward"):
            if not math.isfinite(float(row.get(key, 0.0))):
                formula_errors.append(f"uid {uid}: non-finite {key}")

    all_zero_groups = 0
    partial_nonconstant_groups = 0
    partial_constant_groups = 0
    final_nonconstant_groups = 0
    teacher_called_groups = 0
    for uid, group in groups.items():
        if len(group) != group_size:
            formula_errors.append(f"uid {uid}: group size {len(group)} != {group_size}")
            continue
        em_values = [float(row.get("em_reward", 0.0)) for row in group]
        all_zero = not any(value >= 1.0 for value in em_values)
        final_values = [float(row.get("score", 0.0)) for row in group]
        teacher_values = [float(row.get("teacher_status_reward", 0.0)) for row in group]
        if len(set(final_values)) > 1:
            final_nonconstant_groups += 1
        if any(bool(row.get("teacher_called")) for row in group):
            teacher_called_groups += 1
        if all_zero:
            all_zero_groups += 1
            if len(set(teacher_values)) > 1:
                partial_nonconstant_groups += 1
            else:
                partial_constant_groups += 1
        for row_index, row in enumerate(group):
            if bool(row.get("group_all_em_zero")) != all_zero:
                formula_errors.append(f"uid {uid} row {row_index}: group_all_em_zero mismatch")
            expected = partial_weight * teacher_values[row_index] if all_zero else em_values[row_index]
            if abs(final_values[row_index] - expected) > 1e-8:
                formula_errors.append(
                    f"uid {uid} row {row_index}: score={final_values[row_index]} expected={expected}"
                )
            has_evidence = int(row.get("search_count") or 0) > 0
            if has_evidence and not row.get("teacher_messages"):
                formula_errors.append(f"uid {uid} row {row_index}: evidence without teacher_messages")
            if not all_zero and bool(row.get("teacher_called")):
                formula_errors.append(f"uid {uid} row {row_index}: teacher called despite positive EM")
            if all_zero and has_evidence and not bool(row.get("teacher_called")):
                formula_errors.append(f"uid {uid} row {row_index}: teacher not called for all-zero evidence")

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "record_count": len(rows),
        "group_count": len(groups),
        "em_positive_rollout_count": sum(float(row.get("em_reward", 0.0)) >= 1.0 for row in rows),
        "complete_answer_count": sum(row.get("actor_answer_parse_status") == "parsed" for row in rows),
        "evidence_rollout_count": sum(int(row.get("search_count") or 0) > 0 for row in rows),
        "teacher_called_rollout_count": sum(bool(row.get("teacher_called")) for row in rows),
        "teacher_called_group_count": teacher_called_groups,
        "teacher_status_counts": dict(
            sorted(Counter(str(row.get("teacher_evidence_status") or "not_called") for row in rows).items())
        ),
        "teacher_parse_status_counts": dict(
            sorted(Counter(str(row.get("teacher_parse_status") or "") for row in rows).items())
        ),
        "teacher_format_error_count": sum(bool(row.get("teacher_format_error")) for row in rows),
        "all_zero_group_count": all_zero_groups,
        "partial_nonconstant_group_count": partial_nonconstant_groups,
        "partial_constant_group_count": partial_constant_groups,
        "final_nonconstant_group_count": final_nonconstant_groups,
        "formula_errors": formula_errors,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    total = summary["total"]
    lines = [
        "# SPAD Stage1 Rollout Audit",
        "",
        f"- Rollout manifest: `{summary['manifest_path']}`",
        f"- Manifest SHA256: `{summary['manifest_sha256']}`",
        f"- Validation: **{'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "| Scope | Rollouts | Groups | EM=1 | Complete answer | Evidence | Teacher calls | All-zero groups | Backoff nonconstant | Final nonconstant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| total | {total['record_count']} | {total['group_count']} | "
            f"{total['em_positive_rollout_count']} | {total['complete_answer_count']} | "
            f"{total['evidence_rollout_count']} | {total['teacher_called_rollout_count']} | "
            f"{total['all_zero_group_count']} | {total['partial_nonconstant_group_count']} | "
            f"{total['final_nonconstant_group_count']} |"
        ),
    ]
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partial-weight", type=float, default=0.1)
    args = parser.parse_args()

    manifest_path = args.rollout_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group_size = int(manifest["n_samples_per_prompt"])
    errors: list[str] = []
    shards = []
    for shard_manifest in manifest["shards"]:
        path = Path(str(shard_manifest["path"]))
        shard = audit_shard(path, group_size=group_size, partial_weight=args.partial_weight)
        if shard["sha256"] != str(shard_manifest["sha256"]):
            errors.append(f"hash mismatch: {path}")
        errors.extend(shard["formula_errors"])
        shards.append(shard)
    additive_keys = (
        "record_count",
        "group_count",
        "em_positive_rollout_count",
        "complete_answer_count",
        "evidence_rollout_count",
        "teacher_called_rollout_count",
        "teacher_called_group_count",
        "teacher_format_error_count",
        "all_zero_group_count",
        "partial_nonconstant_group_count",
        "partial_constant_group_count",
        "final_nonconstant_group_count",
    )
    total = {key: sum(int(shard[key]) for shard in shards) for key in additive_keys}
    if total["record_count"] != int(manifest["expected_rollout_count"]):
        errors.append("total rollout count does not match manifest expectation")
    if total["group_count"] != int(manifest["expected_group_count"]):
        errors.append("total group count does not match manifest expectation")
    if not bool(manifest.get("completed")):
        errors.append("rollout manifest is not completed")
    summary = {
        "version": "spad-stage1-rollout-audit-v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "passed": not errors,
        "errors": errors,
        "total": total,
        "shards": shards,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(render_markdown(summary), encoding="utf-8")
    print(args.output_dir / "report.md")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
