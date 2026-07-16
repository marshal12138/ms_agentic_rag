#!/usr/bin/env python3
"""Build a deterministic 512-case Teacher-PE sample from a SPAD training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from run_search_r1_reward_replay import parse_docs_tolerant


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
DEFAULT_RUN_NAME = (
    "260715-005906-987696-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_"
    "newdata_5100_gold_token_f1_v3_postnorm03_stage1"
)
DEFAULT_ROLLOUT_DIR = (
    REPO_ROOT
    / "log/agenticIterRag"
    / DEFAULT_RUN_NAME
    / "outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data"
)
DEFAULT_OUTPUT = HERE / "benchmark_newdata_512.jsonl"
DEFAULT_MANIFEST = HERE / "benchmark_newdata_512.manifest.json"
LAYER_RANGES = {
    "L1_steps_01_20": (1, 20),
    "L2_steps_21_40": (21, 40),
    "L3_steps_41_60": (41, 60),
    "L4_steps_61_79": (61, 79),
}
STATUS_TO_LABEL = {
    "supported_answer": "S",
    "insufficient_evidence": "I",
    "ambiguous_evidence": "A",
}
QUOTED_TITLE_RE = re.compile(r'^"([^"]+)"\s*\n(.*)$', re.DOTALL)


@dataclass(frozen=True)
class RowRef:
    path: Path
    source_line: int
    byte_offset: int
    rollout_index: int
    teacher_called: bool
    selection_key: str


@dataclass
class GroupRecord:
    group_uid: str
    step: int
    question: str
    row_count: int = 0
    teacher_called_count: int = 0
    best_ref: RowRef | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_key(seed: int, *parts: Any) -> str:
    return sha256_text("|".join([str(seed), *(str(part) for part in parts)]))


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def step_layer(step: int) -> str:
    for name, (start, end) in LAYER_RANGES.items():
        if start <= step <= end:
            return name
    raise ValueError(f"Step {step} is outside the configured layers")


def compact_doc(doc: dict[str, Any]) -> dict[str, str]:
    title = str(doc.get("title") or "").strip()
    contents = str(doc.get("contents") or "").strip()
    if not title:
        match = QUOTED_TITLE_RE.match(contents)
        if match:
            title = match.group(1).strip()
            contents = match.group(2).strip()
    return {"title": title, "contents": contents}


def evidence_from_row(row: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    evidence_steps: list[dict[str, Any]] = []
    warnings: list[str] = []
    details = sorted(
        row.get("tool_call_details") or [],
        key=lambda detail: int(detail.get("step_index") or 0),
    )
    for fallback_index, detail in enumerate(details, start=1):
        raw_observation = str(
            detail.get("tool_observation")
            or detail.get("tool_observation_before_truncation")
            or ""
        )
        docs, parse_warnings = parse_docs_tolerant(raw_observation)
        if not docs:
            rank_docs = detail.get("rank_top5_docs") or []
            if isinstance(rank_docs, list):
                docs = [compact_doc(doc) for doc in rank_docs[:5] if isinstance(doc, dict)]
                if docs:
                    parse_warnings.append("used_rank_top5_docs_fallback")
        docs = [compact_doc(doc) for doc in docs[:5]]
        round_index = int(detail.get("step_index") or fallback_index)
        warnings.extend(f"round_{round_index}:{warning}" for warning in parse_warnings)
        evidence_steps.append(
            {
                "round_index": round_index,
                "sub_query": str(
                    detail.get("sub_query")
                    or detail.get("executed_query")
                    or detail.get("attempted_query")
                    or ""
                ).strip(),
                "docs": docs,
            }
        )
    return evidence_steps, warnings


def discover_paths(rollout_dir: Path) -> list[Path]:
    paths = sorted(rollout_dir.glob("*.jsonl"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"No rollout JSONL files found in {rollout_dir}")
    return paths


def infer_run_name(rollout_dir: Path) -> str:
    relative = rollout_dir.resolve().relative_to(REPO_ROOT.resolve())
    if len(relative.parts) >= 3 and relative.parts[:2] == ("log", "agenticIterRag"):
        return relative.parts[2]
    raise ValueError(f"Cannot infer run name from rollout directory {rollout_dir}")


def scan_groups(
    paths: list[Path], seed: int
) -> tuple[dict[str, GroupRecord], list[dict[str, Any]], int]:
    groups: dict[str, GroupRecord] = {}
    source_files: list[dict[str, Any]] = []
    total_rows = 0
    for path in paths:
        file_hash = hashlib.sha256()
        file_rows = 0
        with path.open("rb") as handle:
            while True:
                byte_offset = handle.tell()
                raw_line = handle.readline()
                if not raw_line:
                    break
                file_hash.update(raw_line)
                file_rows += 1
                total_rows += 1
                row = json.loads(raw_line.decode("utf-8"))
                group_uid = str(row.get("group_uid") or row.get("uid") or "").strip()
                if not group_uid:
                    raise ValueError(f"Missing group_uid at {path}:{file_rows}")
                step = int(row.get("step") or int(path.stem))
                question = str(
                    row.get("initial_query")
                    or (row.get("extra_info") or {}).get("question")
                    or ""
                ).strip()
                group = groups.get(group_uid)
                if group is None:
                    group = GroupRecord(group_uid=group_uid, step=step, question=question)
                    groups[group_uid] = group
                elif group.step != step or group.question != question:
                    raise ValueError(f"Inconsistent group metadata for {group_uid}")
                group.row_count += 1
                teacher_called = bool(row.get("teacher_called"))
                group.teacher_called_count += int(teacher_called)
                rollout_index = int(row.get("rollout_index") or 0)
                ref = RowRef(
                    path=path,
                    source_line=file_rows,
                    byte_offset=byte_offset,
                    rollout_index=rollout_index,
                    teacher_called=teacher_called,
                    selection_key=stable_key(
                        seed,
                        "trajectory",
                        group_uid,
                        rollout_index,
                        row.get("request_id") or "",
                    ),
                )
                ref_rank = (0 if ref.teacher_called else 1, ref.selection_key)
                if group.best_ref is None:
                    group.best_ref = ref
                else:
                    best_rank = (
                        0 if group.best_ref.teacher_called else 1,
                        group.best_ref.selection_key,
                    )
                    if ref_rank < best_rank:
                        group.best_ref = ref
        source_files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "rows": file_rows,
                "sha256": file_hash.hexdigest(),
            }
        )
    return groups, source_files, total_rows


def select_groups(
    groups: dict[str, GroupRecord], seed: int, quota_per_layer: int
) -> list[tuple[str, GroupRecord]]:
    by_layer: dict[str, list[GroupRecord]] = defaultdict(list)
    for group in groups.values():
        if group.row_count != 8:
            raise ValueError(f"Group {group.group_uid} has {group.row_count} rows instead of 8")
        if group.best_ref is None:
            raise ValueError(f"Group {group.group_uid} has no representative row")
        by_layer[step_layer(group.step)].append(group)

    selected: list[tuple[str, GroupRecord]] = []
    for layer in LAYER_RANGES:
        candidates = by_layer[layer]
        if len(candidates) < quota_per_layer:
            raise ValueError(
                f"Layer {layer} has only {len(candidates)} groups; quota is {quota_per_layer}"
            )
        ranked = sorted(
            candidates,
            key=lambda group: stable_key(seed, "group", layer, group.group_uid),
        )
        selected.extend((layer, group) for group in ranked[:quota_per_layer])
    selected.sort(key=lambda item: (item[1].step, item[1].group_uid))
    return selected


def read_row(ref: RowRef) -> tuple[dict[str, Any], bytes]:
    with ref.path.open("rb") as handle:
        handle.seek(ref.byte_offset)
        raw_line = handle.readline()
    if not raw_line:
        raise ValueError(f"Cannot read selected row at {ref.path}:{ref.source_line}")
    return json.loads(raw_line.decode("utf-8")), raw_line


def make_case(
    layer: str,
    group: GroupRecord,
    row: dict[str, Any],
    raw_line: bytes,
    source_ref: RowRef,
    run_name: str,
) -> dict[str, Any]:
    question = str(
        row.get("initial_query")
        or (row.get("extra_info") or {}).get("question")
        or ""
    ).strip()
    gold_answers = [str(value) for value in (row.get("gts") or {}).get("target") or []]
    evidence_steps, evidence_warnings = evidence_from_row(row)
    historical_status = str(row.get("teacher_evidence_status") or "")
    source_path = source_ref.path.relative_to(REPO_ROOT)
    case_id = (
        f"new5100-s{group.step:02d}-{group.group_uid}-"
        f"r{source_ref.rollout_index}"
    )
    return {
        "case_id": case_id,
        "source_run": run_name,
        "source_file": str(source_path),
        "source_line": source_ref.source_line,
        "source_row_sha256": hashlib.sha256(raw_line).hexdigest(),
        "step": group.step,
        "step_layer": layer,
        "group_uid": group.group_uid,
        "uid": str(row.get("uid") or ""),
        "rollout_index": source_ref.rollout_index,
        "group_size": int(row.get("group_size") or group.row_count),
        "group_teacher_called_count": group.teacher_called_count,
        "representative_pool": (
            "teacher_called" if source_ref.teacher_called else "teacher_not_called_control"
        ),
        "question_group": normalize_question(question),
        "question": question,
        "gold_answers": gold_answers,
        "data_source": str(row.get("data_source") or ""),
        "source_id": str((row.get("extra_info") or {}).get("source_id") or ""),
        "actor_checkpoint": str(row.get("actor_checkpoint") or ""),
        "actor_answer": str(
            row.get("actor_answer") or row.get("search_r1_extracted_answer") or ""
        ),
        "actor_answer_parse_status": str(row.get("actor_answer_parse_status") or ""),
        "actor_em": float(row.get("search_r1_answer_em") or 0.0),
        "actor_f1": float(row.get("legacy_f1") or 0.0),
        "format_status": str(row.get("format_status") or ""),
        "stop_status": str(row.get("stop_status") or ""),
        "search_count": int(row.get("search_count") or 0),
        "duplicate_query_count": int(row.get("duplicate_query_count") or 0),
        "group_all_em_zero": bool(row.get("group_all_em_zero")),
        "teacher_called": bool(row.get("teacher_called")),
        "teacher_skip_reason": str(row.get("teacher_skip_reason") or ""),
        "historical_teacher_prompt_version": str(row.get("teacher_prompt_version") or ""),
        "historical_teacher_status": historical_status,
        "historical_teacher_label": STATUS_TO_LABEL.get(historical_status, ""),
        "historical_teacher_answer": str(row.get("teacher_answer") or ""),
        "historical_teacher_parse_status": str(row.get("teacher_parse_status") or ""),
        "historical_teacher_raw_content": str(row.get("teacher_raw_content") or ""),
        "historical_teacher_f1": float(row.get("teacher_f1") or 0.0),
        "teacher_gold_token_f1": float(row.get("teacher_gold_token_f1") or 0.0),
        "teacher_status_reward": float(row.get("teacher_status_reward") or 0.0),
        "base_reward": float(row.get("base_reward") or 0.0),
        "final_reward": float(row.get("final_reward") or row.get("score") or 0.0),
        "advantage_source": str(row.get("advantage_source") or ""),
        "advantage_postnorm_scale": float(row.get("advantage_postnorm_scale") or 0.0),
        "bad_stop_applied": bool(row.get("bad_stop_applied")),
        "bad_stop_reason": str(row.get("bad_stop_reason") or ""),
        "evidence_steps": evidence_steps,
        "evidence_parse_warnings": evidence_warnings,
        "evidence_sha256": sha256_text(canonical_json(evidence_steps)),
    }


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quota-per-layer", type=int, default=128)
    args = parser.parse_args()

    paths = discover_paths(args.rollout_dir)
    groups, source_files, total_rows = scan_groups(paths, args.seed)
    selected = select_groups(groups, args.seed, args.quota_per_layer)

    cases: list[dict[str, Any]] = []
    run_name = infer_run_name(args.rollout_dir)
    for layer, group in selected:
        assert group.best_ref is not None
        row, raw_line = read_row(group.best_ref)
        cases.append(
            make_case(layer, group, row, raw_line, group.best_ref, run_name)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")

    layer_population = Counter(step_layer(group.step) for group in groups.values())
    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_run": run_name,
        "source_rollout_dir": str(args.rollout_dir.relative_to(REPO_ROOT)),
        "source_file_count": len(paths),
        "source_row_count": total_rows,
        "source_group_count": len(groups),
        "source_group_sizes": dict(Counter(group.row_count for group in groups.values())),
        "source_files": source_files,
        "sampling_seed": args.seed,
        "quota_per_layer": args.quota_per_layer,
        "step_layers": {
            layer: {
                "start": bounds[0],
                "end": bounds[1],
                "population_groups": layer_population[layer],
                "sampled_groups": sum(case["step_layer"] == layer for case in cases),
            }
            for layer, bounds in LAYER_RANGES.items()
        },
        "group_selection": "lowest_sha256(seed, group, layer, group_uid)",
        "trajectory_selection": (
            "prefer teacher_called rows, then lowest "
            "sha256(seed, trajectory, group_uid, rollout_index, request_id)"
        ),
        "case_count": len(cases),
        "representative_pool_counts": dict(
            Counter(case["representative_pool"] for case in cases)
        ),
        "data_source_counts": dict(Counter(case["data_source"] for case in cases)),
        "group_all_em_zero_counts": dict(
            Counter(str(case["group_all_em_zero"]).lower() for case in cases)
        ),
        "teacher_called_counts": dict(
            Counter(str(case["teacher_called"]).lower() for case in cases)
        ),
        "format_status_counts": dict(Counter(case["format_status"] for case in cases)),
        "benchmark_sha256": sha256_text(canonical_json(cases)),
        "output_file": args.output.name,
    }
    write_json_atomic(args.manifest, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifest": str(args.manifest),
                "case_count": len(cases),
                "layers": {
                    layer: sum(case["step_layer"] == layer for case in cases)
                    for layer in LAYER_RANGES
                },
                "representative_pool_counts": manifest["representative_pool_counts"],
                "benchmark_sha256": manifest["benchmark_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
