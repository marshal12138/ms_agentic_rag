#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "/data01/ms_wksp/agent_up_to_date/CoSearch_derevitives/src/env_manage/compatible_python.sh"
DATA_DIR="${DATA_DIR:-${ROOT}/data/coAgenticRetriever/albation_1}"
FILES=(
  "${DATA_DIR}/co_search_ablation.train.parquet"
  "${DATA_DIR}/co_search_ablation.eval.parquet"
)

for path in "${FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: parquet file not found: ${path}" >&2
    exit 2
  fi
done

"${PY}" - "${FILES[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = """You are a tool-augmented research agent for wiki-based factoid question answering.

Answer questions drawn from Wikipedia-style datasets. The final answer is evaluated using exact match (EM) or token-level F1, so it must be short and precise.

For every assistant turn:
1. First reason inside <think>...</think>.
2. Then output exactly one block:
   - <tool_call>...</tool_call> when you need retrieval, following the provided function schema exactly.
   - <answer>...</answer> when you have enough evidence to answer.

On the first assistant turn, call the search tool before answering.
Do not output <answer> until after a tool response has been provided.
Do not output both <tool_call> and <answer> in the same turn.
Inside <answer>, output only the final short answer string."""


QUESTION_RE = re.compile(r"(?:^|\n)Question:\s*(.+?)\s*$", re.DOTALL)


def extract_question(row: dict, prompt_messages: list[dict]) -> str:
    extra_info = row.get("extra_info") or {}
    question = extra_info.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()

    for message in reversed(prompt_messages):
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = QUESTION_RE.search(content)
        if match:
            return match.group(1).strip()

    raise ValueError("failed to extract question from row")


def already_converted(prompt_messages: list[dict]) -> bool:
    if len(prompt_messages) != 2:
        return False
    first, second = prompt_messages
    return (
        first.get("role") == "system"
        and second.get("role") == "user"
        and isinstance(first.get("content"), str)
        and isinstance(second.get("content"), str)
    )


def transform_prompt(row: dict) -> list[dict]:
    prompt_messages = row["prompt"]
    if already_converted(prompt_messages):
        return prompt_messages

    question = extract_question(row, prompt_messages)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def write_example(path: Path, row: dict) -> Path:
    example_path = path.with_suffix(".example.json")
    example_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return example_path


def rewrite_parquet(path_str: str) -> None:
    path = Path(path_str)
    table = pq.read_table(path)
    rows = table.to_pylist()
    prompt_type = table.schema.field("prompt").type

    converted = [transform_prompt(row) for row in rows]
    prompt_array = pa.array(converted, type=prompt_type)

    prompt_index = table.schema.get_field_index("prompt")
    updated = table.set_column(prompt_index, "prompt", prompt_array)
    updated = updated.replace_schema_metadata(table.schema.metadata)

    backup_path = path.with_suffix(path.suffix + ".bak")
    if not backup_path.exists():
        path.replace(backup_path)
    else:
        path.unlink()

    pq.write_table(updated, path)

    first_prompt = converted[0]
    example_row = dict(rows[0])
    example_row["prompt"] = first_prompt
    example_path = write_example(path, example_row)
    print(f"[converted] {path}")
    print(f"  rows={len(converted)}")
    print(f"  first_prompt_roles={[msg['role'] for msg in first_prompt]}")
    print(f"  first_user={first_prompt[-1]['content']!r}")
    print(f"  example={example_path}")


for parquet_path in sys.argv[1:]:
    rewrite_parquet(parquet_path)
PY
