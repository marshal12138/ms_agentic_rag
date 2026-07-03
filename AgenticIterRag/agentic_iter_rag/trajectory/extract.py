"""Extract canonical AgenticIterRag trace records from AIR raw infer traces."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from agentic_iter_rag.trajectory.schema import TraceRecord, normalize_doc, validate_trace_record
from agentic_iter_rag.utils.io import iter_jsonl, write_json, write_jsonl


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def extract_records(raw: dict[str, Any], source_path: Path, index: int) -> list[dict[str, Any]]:
    # ---- 样本级字段：面向 AIR v1 infer engine 的 traces.jsonl 输出 ----
    trace_id = _first_nonempty(raw.get("trace_id"), raw.get("uid"), raw.get("id"), f"{source_path.stem}-{index}")
    sample_id = _first_nonempty(raw.get("sample_id"), raw.get("uid"), raw.get("id"), raw.get("index"), trace_id)
    prompt_value = raw.get("prompt")
    prompt_text = ""
    if isinstance(prompt_value, list) and prompt_value:
        prompt_text = _first_nonempty(*(item.get("content") for item in prompt_value if isinstance(item, dict)))
    question = _first_nonempty(raw.get("question"), raw.get("initial_query"), raw.get("query"), prompt_text)
    gold_answers = [str(x) for x in _as_list(raw.get("gold_answers") or raw.get("ground_truth_answer") or raw.get("answers"))]
    final_answer = raw.get("final_answer")
    metrics = raw.get("metrics") if isinstance(raw.get("metrics"), dict) else {}
    reward = metrics.get("f1")

    # ---- turn 级字段：AIR infer 将每次搜索拆成并行数组 ----
    sub_queries = [str(x) for x in _as_list(raw.get("sub_queries")) if str(x).strip()]
    recall_by_call = _as_list(raw.get("retrieved_top50_chunks") or raw.get("recall_top50_chunks"))
    ranked_by_call = _as_list(raw.get("ranked_top50_chunks"))
    visible_by_call = _as_list(raw.get("final_top5_chunks") or raw.get("ranked_top5_chunks"))
    records: list[dict[str, Any]] = []
    for call_idx, sub_query in enumerate(sub_queries):
        recall_docs = _as_list(recall_by_call[call_idx] if call_idx < len(recall_by_call) else [])
        ranked_docs = _as_list(ranked_by_call[call_idx] if call_idx < len(ranked_by_call) else recall_docs)
        visible_docs = _as_list(visible_by_call[call_idx] if call_idx < len(visible_by_call) else ranked_docs[:5])
        record = TraceRecord(
            trace_id=f"{trace_id}:search:{call_idx}",
            sample_id=sample_id,
            question=question,
            gold_answers=gold_answers,
            sub_query=sub_query,
            recall_topn_docs=[normalize_doc(x, rank=i + 1) for i, x in enumerate(_as_list(recall_docs)) if isinstance(x, dict)],
            ranked_docs=[normalize_doc(x, rank=i + 1) for i, x in enumerate(_as_list(ranked_docs)) if isinstance(x, dict)],
            visible_docs=[normalize_doc(x, rank=i + 1) for i, x in enumerate(_as_list(visible_docs)) if isinstance(x, dict)],
            final_answer=str(final_answer) if final_answer is not None else None,
            reward=float(reward) if isinstance(reward, (int, float)) else None,
            metrics=metrics,
            source={"path": str(source_path), "line_index": index, "tool_call_index": call_idx},
            raw_trace_ref={"path": str(source_path), "line_index": index},
        ).to_dict()
        validate_trace_record(record)
        records.append(record)
    return records


def extract_file(input_path: Path, output_jsonl: Path, manifest_path: Path | None = None) -> int:
    all_records: list[dict[str, Any]] = []
    for index, raw in enumerate(iter_jsonl(input_path)):
        all_records.extend(extract_records(raw, input_path, index))
    count = write_jsonl(output_jsonl, all_records)
    if manifest_path:
        write_json(
            manifest_path,
            {
                "type": "agentic_iter_rag_trace_manifest",
                "source_trace": str(input_path),
                "output_jsonl": str(output_jsonl),
                "records": count,
            },
        )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract canonical AgenticIterRag trace records.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    count = extract_file(args.input, args.output_jsonl, args.manifest)
    print(f"wrote {count} trace records to {args.output_jsonl}")


if __name__ == "__main__":
    main()
