#!/usr/bin/env python3
"""Offline batched SPAD teacher generation worker.

The parent process shards eligible Stage 2 trajectories and launches one worker
per GLM4.7 vLLM replica. This script only performs batched generation and writes
raw teacher responses; parsing/filtering stays in the main Stage 2 process so
the Stage 3 dataset schema is identical to the HTTP path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from agentic_iter_rag.agent_training.spad.prompts import (
    DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
    build_teacher_messages,
    resolve_teacher_prompt,
)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"bad JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def batched(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    size = max(1, int(batch_size))
    return [items[start : start + size] for start in range(0, len(items), size)]


def messages_to_prompt(tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
    try:
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    except TypeError:
        return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=32000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enable-chunked-prefill", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompt-version", default=DEFAULT_TEACHER_STATUS_PROMPT_VERSION)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--shard-id", default="0")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.prompt_version, _ = resolve_teacher_prompt(args.prompt_version, include_status=True)
    args.output_jsonl.unlink(missing_ok=True)
    rows = iter_jsonl(args.input_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts: list[dict[str, Any]] = []
    for row in rows:
        messages = build_teacher_messages(
            question=str(row.get("question") or ""),
            evidence_steps=row.get("evidence_steps") or [],
            include_status=True,
            prompt_version=args.prompt_version,
        )
        prompts.append(
            {
                "index": int(row["index"]),
                "prompt": messages_to_prompt(tokenizer, messages, enable_thinking=bool(args.enable_thinking)),
            }
        )

    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "dtype": str(args.dtype),
        "trust_remote_code": True,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_model_len": int(args.max_model_len),
        "max_num_seqs": int(args.max_num_seqs),
        "max_num_batched_tokens": int(args.max_num_batched_tokens),
        "enable_prefix_caching": bool(args.enable_prefix_caching),
        "enable_chunked_prefill": bool(args.enable_chunked_prefill),
        "enforce_eager": bool(args.enforce_eager),
        "disable_custom_all_reduce": bool(args.disable_custom_all_reduce),
    }
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        max_tokens=int(args.max_tokens),
    )

    total_started = time.perf_counter()
    for batch in batched(prompts, int(args.batch_size)):
        batch_started = time.perf_counter()
        outputs = llm.generate([item["prompt"] for item in batch], sampling_params)
        batch_elapsed = time.perf_counter() - batch_started
        per_sample_elapsed = batch_elapsed / max(1, len(batch))
        records: list[dict[str, Any]] = []
        for item, output in zip(batch, outputs):
            text = output.outputs[0].text if output.outputs else ""
            records.append(
                {
                    "index": int(item["index"]),
                    "teacher_raw_content": text,
                    "teacher_elapsed_s": per_sample_elapsed,
                    "batch_elapsed_s": batch_elapsed,
                    "batch_size": len(batch),
                    "shard_id": str(args.shard_id),
                    "teacher_prompt_version": str(args.prompt_version),
                }
            )
        append_jsonl(args.output_jsonl, records)
        print(
            f"SPAD offline teacher shard={args.shard_id} generated={len(records)} "
            f"batch_elapsed_s={batch_elapsed:.3f}",
            flush=True,
        )
    print(
        f"SPAD offline teacher shard={args.shard_id} completed total={len(rows)} "
        f"elapsed_s={time.perf_counter() - total_started:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
