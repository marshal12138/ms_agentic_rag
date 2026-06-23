#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    think = load(args.result_dir / "summary_think.json")
    no_think = load(args.result_dir / "summary_no_think.json")
    lines = [
        "# DeepSeek-V4-Flash Chunk Ranking Judge Benchmark",
        "",
        "## Setup",
        "",
        "- Model: `/data01/ms_wksp/agent_up_to_date/models/llm/DeepSeek-V4-Flash`",
        "- vLLM GPUs: `6,7`",
        "- Task: rank 50 passages for each `(origin_query, sub_query)` input.",
        "- Data: `chunk_ranking_judge_examples_100.jsonl`",
        "",
        "## Throughput",
        "",
        "| mode | ok/total | elapsed_s | qpm | qps | mean_s | p50_s | p95_s | total_tokens | tokens/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in (think, no_think):
        lat = item["latency_s"]
        tok = item["tokens"]
        lines.append(
            f"| {item['mode']} | {item['ok_count']}/{item['request_count']} | "
            f"{item['total_elapsed_s']:.3f} | {item['qpm']:.3f} | {item['qps']:.3f} | "
            f"{(lat['mean'] or 0):.3f} | {(lat['p50'] or 0):.3f} | {(lat['p95'] or 0):.3f} | "
            f"{tok['total_tokens']} | {tok['tokens_per_s']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Prompts: `prompts_think.jsonl`, `prompts_no_think.jsonl`",
            "- Request payloads: `requests_think.jsonl`, `requests_no_think.jsonl`",
            "- Raw outputs: `outputs_think.jsonl`, `outputs_no_think.jsonl`",
            "- JSON summaries: `summary_think.json`, `summary_no_think.json`",
            "- vLLM log: `logs/vllm_gpu06_07_8067.log`",
            "",
            "## Notes",
            "",
            "- `think` permits internal reasoning but still asks final JSON.",
            "- `no_think` explicitly asks for JSON only and no reasoning.",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
