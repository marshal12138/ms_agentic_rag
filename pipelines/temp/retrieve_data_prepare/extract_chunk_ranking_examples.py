#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def normalize_doc(doc: Any) -> Any:
    if isinstance(doc, dict):
        return doc
    return {"text": str(doc)}


def iter_validation_rows(paths: list[Path]):
    for path in sorted(paths):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield path, line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"skip invalid json: {path}:{line_no}: {exc}")


def usable_examples(row: dict[str, Any]):
    origin_query = row.get("initial_query") or row.get("origin_query") or ""
    details = row.get("tool_call_details") or []
    if not isinstance(details, list):
        return

    for detail in details:
        if not isinstance(detail, dict):
            continue
        sub_query = detail.get("sub_query") or ""
        passages = detail.get("top_50_documents") or detail.get("top_n_documents") or []
        if not origin_query or not sub_query or not isinstance(passages, list):
            continue
        if len(passages) < 50:
            continue
        yield {
            "origin_query": origin_query,
            "sub_query": sub_query,
            "passage_list_top50": [normalize_doc(x) for x in passages[:50]],
        }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--shard-size", type=int, default=10)
    parser.add_argument("--final-name", default="chunk_ranking_judge_examples_100.jsonl")
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    paths = list(validation_dir.glob("*.jsonl"))
    for path, line_no, row in iter_validation_rows(paths):
        for example in usable_examples(row):
            key = (example["origin_query"], example["sub_query"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(example)
            if len(rows) % args.shard_size == 0:
                shard = output_dir / f"part_{len(rows):06d}.jsonl"
                write_jsonl(shard, rows[-args.shard_size :])
                print(f"wrote shard {shard} from latest usable examples")
            if len(rows) >= args.target:
                break
        if len(rows) >= args.target:
            break

    if len(rows) < args.target:
        print(f"usable examples: {len(rows)} < target {args.target}")
        return

    final_path = output_dir / args.final_name
    write_jsonl(final_path, rows[: args.target])

    for shard in output_dir.glob("part_*.jsonl"):
        shard.unlink()

    print(f"wrote final dataset: {final_path}")
    print(f"rows: {args.target}")


if __name__ == "__main__":
    main()
