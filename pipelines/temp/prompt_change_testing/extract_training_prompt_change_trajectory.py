#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
DOC_RE = re.compile(r"\[(\d+)\]\s+\"([^\"]+)\"\n(.*?)(?=\n\[\d+\]\s+\"|\Z)", re.DOTALL)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_subset_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def parse_tool_call(text: str) -> dict[str, Any]:
    match = TOOL_CALL_RE.search(text or "")
    if not match:
        return {}
    raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "json_valid": False}
    return {"raw": raw, "json_valid": True, "parsed": parsed}


def parse_tool_response_docs(text: str) -> list[dict[str, Any]]:
    match = TOOL_RESPONSE_RE.search(text or "")
    if not match:
        return []
    block = match.group(1).strip()
    docs: list[dict[str, Any]] = []
    for idx, title, snippet in DOC_RE.findall(block):
        docs.append(
            {
                "rank": int(idx),
                "title": title.strip(),
                "snippet": " ".join(snippet.strip().split()),
            }
        )
    return docs


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip()


def select_rollout_trajectory(rollout_dir: Path) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    for path in sorted((rollout_dir / "main").glob("*.jsonl")):
        samples.extend(read_jsonl(path))
    if not samples:
        return {}

    preferred = None
    for sample in samples:
        output = sample.get("output", "")
        if "<tool_call>" in output and "<tool_response>" in output and "<answer>" in output:
            preferred = sample
            break
    if preferred is None:
        preferred = samples[0]
    return preferred


def select_matching_subset_row(rows: list[dict[str, Any]], text: str) -> dict[str, Any]:
    for row in rows:
        prompt = row.get("prompt") or []
        if not prompt:
            continue
        user_content = prompt[-1].get("content", "")
        if user_content and user_content in text:
            return row
    return rows[0] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-subset", type=Path, required=True)
    parser.add_argument("--train-llm-io", type=Path, required=True)
    parser.add_argument("--train-search-timing", type=Path, required=True)
    parser.add_argument("--train-rollout-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    subset_rows = load_subset_rows(args.train_subset)
    llm_io_rows = read_jsonl(args.train_llm_io)
    search_rows = read_jsonl(args.train_search_timing)
    rollout = select_rollout_trajectory(args.train_rollout_dir)

    first_generation = llm_io_rows[0] if llm_io_rows else {}
    second_generation = llm_io_rows[1] if len(llm_io_rows) > 1 else {}
    matching_text = "\n".join(
        [
            first_generation.get("prompt_text", "") or "",
            first_generation.get("output_text", "") or "",
            rollout.get("input", "") or "",
            rollout.get("output", "") or "",
        ]
    )
    subset_row = select_matching_subset_row(subset_rows, matching_text)
    tool_call = parse_tool_call(first_generation.get("output_text", "") or rollout.get("output", ""))
    tool_response_docs = parse_tool_response_docs(rollout.get("output", ""))
    final_answer = extract_answer(rollout.get("output", ""))

    payload = {
        "artifacts": {
            "train_subset": str(args.train_subset),
            "train_llm_io": str(args.train_llm_io),
            "train_search_timing": str(args.train_search_timing),
            "train_rollout_dir": str(args.train_rollout_dir),
        },
        "verification": {
            "has_first_turn_think": "<think>" in (first_generation.get("output_text", "") or ""),
            "has_first_turn_tool_call": "<tool_call>" in (first_generation.get("output_text", "") or ""),
            "tool_call_json_valid": tool_call.get("json_valid", False),
            "has_tool_response": "<tool_response>" in (rollout.get("output", "") or ""),
            "has_final_answer": bool(final_answer),
            "num_llm_io_records": len(llm_io_rows),
            "num_search_records": len(search_rows),
        },
        "sample": {
            "data_source": subset_row.get("data_source"),
            "prompt": subset_row.get("prompt"),
            "ground_truth": (subset_row.get("reward_model") or {}).get("ground_truth", {}),
            "extra_info": subset_row.get("extra_info"),
        },
        "trajectory": {
            "uid": rollout.get("uid"),
            "first_generation": first_generation,
            "parsed_tool_call": tool_call,
            "search_timing_records": search_rows[:4],
            "retrieved_documents": tool_response_docs[:5],
            "second_generation": second_generation,
            "final_answer": final_answer,
            "rollout_record": rollout,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
