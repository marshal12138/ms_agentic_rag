#!/usr/bin/env python3
"""Apply a deterministic composite selection policy to persisted two-stage outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from composite_prompt_variants import COMPOSITE_PROMPT_VARIANTS, composite_strategy_spec
from run_ablation import (
    answer_exact_match,
    answer_token_f1,
    load_cases,
    render_report,
    score_by_split,
    sha256_json,
    write_answer_audit,
    write_errors,
    write_json_atomic,
)
from run_composite_ablation import load_jsonl, select_composite_output


HERE = Path(__file__).resolve().parent


def resolve_from_here(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else HERE / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(COMPOSITE_PROMPT_VARIANTS))
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    variant = COMPOSITE_PROMPT_VARIANTS[args.variant]
    source_dir = resolve_from_here(args.source_dir)
    output_dir = resolve_from_here(args.output_dir)
    source_run = json.loads((source_dir / "run.json").read_text(encoding="utf-8"))
    if int(source_run.get("cache_hits") or 0) or int(source_run.get("request_errors") or 0):
        raise ValueError("Policy derivation requires a cache-free, error-free source run")

    benchmark = resolve_from_here(source_run["benchmark"])
    cases = load_cases(str(source_run["split"]), benchmark)
    cases_by_id = {str(case["case_id"]): case for case in cases}
    source_by_id = {
        str(row["case_id"]): row for row in load_jsonl(source_dir / "predictions.jsonl")
    }
    stage_a_dir = resolve_from_here(source_run["stage_a_dir"])
    stage_a_by_id = {
        str(row["case_id"]): row for row in load_jsonl(stage_a_dir / "predictions.jsonl")
    }
    stage_b_by_id = {
        str(row["case_id"]): row for row in load_jsonl(source_dir / "stage_b_predictions.jsonl")
    }
    expected = set(cases_by_id)
    for name, rows in (("source", source_by_id), ("stage_a", stage_a_by_id)):
        if set(rows) != expected:
            raise ValueError(f"{name} case IDs do not match the source benchmark split")
    expected_stage_b = (
        expected
        if source_run.get("stage_b_scope") == "all"
        else {
            case_id
            for case_id, row in stage_a_by_id.items()
            if row.get("parsed") and row.get("predicted_label") in {"S", "A"}
        }
    )
    if set(stage_b_by_id) != expected_stage_b:
        raise ValueError("stage_b case IDs do not match the source strategy's eligible cases")

    prompt_sha256 = sha256_json(composite_strategy_spec(variant))
    final_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = str(case["case_id"])
        source = source_by_id[case_id]
        stage_a = stage_a_by_id[case_id]
        stage_b = stage_b_by_id.get(case_id)
        selection = select_composite_output(case, stage_a, stage_b, variant)
        selected = selection["selected"]
        answer_supported = bool(selected.get("parsed") and selected.get("predicted_label") == "S")
        answer = str(selection["answer"])
        final_rows.append(
            {
                **source,
                "index": index,
                "variant": variant.name,
                "prompt_sha256": prompt_sha256,
                "request_sha256": str(selected.get("request_sha256") or ""),
                "messages": selected.get("messages") or [],
                "endpoint": str(selected.get("endpoint") or ""),
                "raw_content": str(selected.get("raw_content") or ""),
                "raw_reasoning_content": str(selected.get("raw_reasoning_content") or ""),
                "parsed": bool(selected.get("parsed")),
                "parse_status": str(selected.get("parse_status") or ""),
                "predicted_status": str(selected.get("predicted_status") or ""),
                "predicted_label": str(selected.get("predicted_label") or "E"),
                "reason": str(selected.get("reason") or ""),
                "answer": answer,
                "teacher_gold_token_f1": (
                    answer_token_f1(answer, case.get("gold_answers") or [])
                    if answer_supported
                    else 0.0
                ),
                "teacher_gold_exact_match": (
                    answer_exact_match(answer, case.get("gold_answers") or [])
                    if answer_supported
                    else 0.0
                ),
                "teacher_manual_answer_token_f1": (
                    answer_token_f1(answer, [case.get("manual_answer")])
                    if answer_supported and case.get("manual_answer")
                    else 0.0
                ),
                "stage_b_used": bool(selection["stage_b_used"]),
                "canonical_gold": selection["canonical_gold"],
                "selection_reason": selection["selection_reason"],
            }
        )

    derived_at = datetime.now().astimezone().isoformat()
    run_metadata = {
        **source_run,
        "variant": variant.name,
        "description": variant.description,
        "prompt_sha256": prompt_sha256,
        "stage_b_scope": variant.stage_b_scope,
        "i_override_min_gold_f1": variant.i_override_min_gold_f1,
        "prefer_higher_gold_f1_between_supported_stages": (
            variant.prefer_higher_gold_f1_between_supported_stages
        ),
        "canonicalize_evidence_literal_gold": variant.canonicalize_evidence_literal_gold,
        "derived_from": str(source_dir),
        "derived_at": derived_at,
        "model_requests_this_derivation": 0,
        "combination_postprocess_only": True,
    }
    metrics = score_by_split(final_rows)
    canonicalized = sum(bool(row.get("canonical_gold")) for row in final_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_prompt.txt").write_text(variant.system_prompt + "\n", encoding="utf-8")
    write_json_atomic(
        output_dir / "variant.json",
        {
            **composite_strategy_spec(variant),
            "family": "hard_gate_composite",
            "description": variant.description,
            "stage_a_variant": source_run["stage_a_variant"],
            "prompt_sha256": prompt_sha256,
        },
    )
    with (output_dir / "stage_b_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in sorted(stage_b_by_id.values(), key=lambda item: int(item["index"])):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in final_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json_atomic(output_dir / "run.json", run_metadata)
    write_json_atomic(output_dir / "metrics.json", metrics)
    write_errors(output_dir / "errors.tsv", final_rows)
    write_answer_audit(output_dir / "answer_audit.tsv", final_rows)
    derivation_lines = [
        "# Deterministic Composite Derivation",
        "",
        f"- Derived at: `{derived_at}`",
        f"- Source two-stage run: `{source_dir}`",
        "- Additional model requests: `0`",
        f"- Evidence-literal canonicalizations: `{canonicalized}`",
        f"- Mean elapsed ratio versus Stage A: `{source_run['budget']['mean_elapsed_ratio_vs_stage_a']:.4f}`",
        f"- Within 2x budget: `{str(source_run['budget']['within_two_x_budget']).lower()}`",
        "",
    ]
    (output_dir / "report.md").write_text(
        "\n".join(derivation_lines) + render_report(run_metadata, metrics), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "derived_from": str(source_dir),
                "canonicalized": canonicalized,
                "budget": source_run["budget"],
                "metrics": metrics["selected"],
                "teacher_called": metrics["teacher_called"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
