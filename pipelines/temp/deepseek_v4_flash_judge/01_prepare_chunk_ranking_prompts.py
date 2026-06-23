#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """You are a strict passage ranking judge. Rank passages only by evidence relevance, not by their original order or retriever score."""


def passage_text(passage: dict[str, Any]) -> str:
    title = str(passage.get("title") or "").strip()
    contents = str(passage.get("contents") or "").strip()
    if title and title not in contents[:200]:
        return f"{title}\n{contents}"
    return contents


def build_user_prompt(obj: dict[str, Any], *, think: bool) -> str:
    origin_query = str(obj["origin_query"]).strip()
    sub_query = str(obj["sub_query"]).strip()
    passages = obj["passage_list_top50"]

    lines = [
        "Task: Rank the 50 passages by usefulness for a retrieval-augmented QA agent.",
        "",
        "Ranking criteria, in priority order:",
        "1. Direct relevance to the sub_query.",
        "2. Indirect usefulness for answering the origin_query.",
        "3. Prefer passages that contain concrete answer evidence over passages that only mention related entities.",
        "",
        f"origin_query: {origin_query}",
        f"sub_query: {sub_query}",
        "",
        "Passages:",
    ]
    for rank, passage in enumerate(passages, start=1):
        pid = str(passage.get("id", f"missing_{rank}"))
        text = passage_text(passage).replace("\n", " ").strip()
        lines.append(f"[{pid}] {text}")

    lines.extend(
        [
            "",
            "Output requirements:",
            "- Return all 50 passage ids exactly once.",
            "- Return ids from most relevant to least relevant.",
            "- Do not invent ids.",
        ]
    )
    if think:
        lines.extend(
            [
                "- You may think internally, but the final answer must end with one JSON object.",
                "- JSON schema: {\"ranked_ids\": [\"id1\", \"id2\", ...]}",
            ]
        )
    else:
        lines.extend(
            [
                "- Do not include reasoning.",
                "- Output only one JSON object with schema: {\"ranked_ids\": [\"id1\", \"id2\", ...]}",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            obj = json.loads(line)
            rows.append(obj)

    for mode, think in (("think", True), ("no_think", False)):
        prompt_path = args.output_dir / f"prompts_{mode}.jsonl"
        data_path = args.output_dir / f"requests_{mode}.jsonl"
        with prompt_path.open("w", encoding="utf-8") as pf, data_path.open("w", encoding="utf-8") as df:
            for i, obj in enumerate(rows):
                passage_ids = [str(p.get("id")) for p in obj["passage_list_top50"]]
                user_prompt = build_user_prompt(obj, think=think)
                prompt_record = {
                    "example_id": i,
                    "mode": mode,
                    "origin_query": obj["origin_query"],
                    "sub_query": obj["sub_query"],
                    "passage_ids": passage_ids,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                request_record = {
                    "example_id": i,
                    "mode": mode,
                    "passage_ids": passage_ids,
                    "payload": {
                        "messages": prompt_record["messages"],
                        "temperature": 0.0,
                        "max_tokens": 1024,
                    },
                }
                pf.write(json.dumps(prompt_record, ensure_ascii=False) + "\n")
                df.write(json.dumps(request_record, ensure_ascii=False) + "\n")

    manifest = {
        "input": str(args.input),
        "rows": len(rows),
        "outputs": {
            "think_prompts": str(args.output_dir / "prompts_think.jsonl"),
            "no_think_prompts": str(args.output_dir / "prompts_no_think.jsonl"),
            "think_requests": str(args.output_dir / "requests_think.jsonl"),
            "no_think_requests": str(args.output_dir / "requests_no_think.jsonl"),
        },
    }
    (args.output_dir / "prepare_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
