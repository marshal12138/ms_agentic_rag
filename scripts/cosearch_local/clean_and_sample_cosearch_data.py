#!/usr/bin/env python3
"""Clean CoSearch QA answer semantics and build deterministic data subsets.

The output keeps the exact nested parquet schema used by the current 5,100
train / 350 eval datasets. Audit provenance is written to sidecar JSONL files
instead of being added to the training schema.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL_TRAIN = REPO_ROOT / "data/co_search/local_flashrag/co_search_rl_51k.train.parquet"
DEFAULT_FULL_EVAL = REPO_ROOT / "data/co_search/local_flashrag/co_search_7bench.eval.parquet"
DEFAULT_SCHEMA_REFERENCE = REPO_ROOT / "data/coAgenticRetriever/albation_1/co_search_ablation.train.parquet"
DEFAULT_PROCESSED_ROOT = REPO_ROOT / "data/processed_sets"
DEFAULT_SAMPLE_ROOT = REPO_ROOT / "data/global_train_eval_data"

TRAIN_SOURCE_COUNTS = {
    "nq": 20_480,
    "hotpotqa": 14_220,
    "musique": 9_000,
    "2wikimultihopqa": 7_500,
}
TRAIN_SAMPLE_SIZES = (512, 5_100, 12_000, 25_000)
EVAL_SAMPLE_SIZES = (350, 3_500)
TRAIN_SOURCE_ORDER = tuple(TRAIN_SOURCE_COUNTS)
EVAL_SOURCE_ORDER = (
    "popqa",
    "2wikimultihopqa",
    "triviaqa",
    "hotpotqa",
    "nq",
    "musique",
    "bamboogle",
)
TRAIN_SEED = 26_041_755
EVAL_SEED = 42
RULESET_VERSION = "single-or-v2"


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def dedupe_answers(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        answer = str(value).strip()
        normalized = normalize_text(answer)
        if not answer or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(answer)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if math.isnan(value) else value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def get_targets(row: Any) -> list[str]:
    return dedupe_answers(row.reward_model["ground_truth"]["target"])


def object_id(metadata: dict[str, Any]) -> str:
    value = metadata.get("obj_id")
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
    return str(value)


def parse_aliases(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return dedupe_answers(parsed if isinstance(parsed, list) else [])
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
        values = value.tolist() if hasattr(value, "tolist") else value
        return dedupe_answers(values)
    return []


def build_popqa_question_objects(frame: pd.DataFrame) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for row in frame.itertuples(index=False):
        metadata = row.extra_info.get("metadata") or {}
        mapping[normalize_text(row.extra_info["question"])].add(object_id(metadata))
    return mapping


def musique_final_answer(metadata: dict[str, Any]) -> str | None:
    decomposition = metadata.get("question_decomposition")
    if decomposition is None:
        return None
    items = decomposition.tolist() if hasattr(decomposition, "tolist") else list(decomposition)
    if not items:
        return None
    final = items[-1]
    answer = final.get("answer") if isinstance(final, dict) else None
    if answer is None or not str(answer).strip():
        return None
    return str(answer).strip()


def classify_row(
    row: Any,
    popqa_question_objects: dict[str, set[str]],
) -> tuple[str, str, list[str] | None]:
    source = str(row.data_source)
    before = get_targets(row)
    metadata = row.extra_info.get("metadata") or {}

    if not before:
        return "DROP_UNRESOLVED", "empty target list", None

    if source == "nq":
        if len(before) == 1:
            return "KEEP_SINGLE", "single DPR/FlashRAG answer", before
        return (
            "DROP_REQUIRED_SET",
            "NQ annotation boundaries were flattened; multi-answer OR semantics cannot be proven",
            None,
        )

    if source == "musique":
        final_answer = musique_final_answer(metadata)
        if final_answer is None:
            return "DROP_UNRESOLVED", "missing final question-decomposition answer", None
        after = [final_answer]
        if len(before) == 1 and normalize_text(before[0]) == normalize_text(final_answer):
            return "KEEP_SINGLE", "matches final question-decomposition answer", after
        return (
            "REPAIR_TO_SINGLE",
            "canonicalized to final question-decomposition answer; aliases and bridge answers removed",
            after,
        )

    if source == "popqa":
        question_key = normalize_text(row.extra_info["question"])
        ids = {value for value in popqa_question_objects.get(question_key, set()) if value}
        if len(ids) != 1:
            return (
                "DROP_AMBIGUOUS",
                f"PopQA question maps to {len(ids)} distinct object IDs",
                None,
            )
        official = dedupe_answers([metadata.get("obj"), *parse_aliases(metadata.get("o_aliases"))])
        if not official:
            return "DROP_UNRESOLVED", "missing PopQA object and object aliases", None
        before_norm = {normalize_text(value) for value in before}
        official_norm = {normalize_text(value) for value in official}
        if before_norm == official_norm:
            decision = "KEEP_SINGLE" if len(official) == 1 else "KEEP_ALIAS_OR"
            return decision, "single PopQA object ID with official object aliases", official
        decision = "REPAIR_TO_SINGLE" if len(official) == 1 else "REPAIR_ALIAS_OR"
        return decision, "restricted target to one PopQA object ID and its official aliases", official

    if source == "2wikimultihopqa":
        if len(before) == 1:
            return "KEEP_SINGLE", "2Wiki official single answer", before
        return (
            "REPAIR_TO_SINGLE",
            "kept the leading 2Wiki canonical answer; removed unverified expanded aliases",
            [before[0]],
        )

    if source == "triviaqa":
        if len(before) == 1:
            return "KEEP_SINGLE", "TriviaQA single answer", before
        return (
            "DROP_UNVERIFIED_ALIAS",
            "official TriviaQA Answer.Value/Aliases boundary is unavailable in the flattened row",
            None,
        )

    if source in {"hotpotqa", "bamboogle"}:
        if len(before) == 1:
            return "KEEP_SINGLE", f"{source} official single answer", before
        return "DROP_UNRESOLVED", f"unexpected multi-answer {source} row", None

    return "DROP_UNRESOLVED", f"unsupported data source: {source}", None


def reference_prompt_prefix(reference_path: Path) -> str:
    frame = pd.read_parquet(reference_path, columns=["prompt"])
    content = str(frame.iloc[0].prompt[0]["content"])
    if "Question:" not in content:
        raise ValueError(f"Reference prompt has no Question marker: {reference_path}")
    return content.rsplit("Question:", 1)[0]


def output_row(row: Any, answers: list[str], prompt_prefix: str) -> dict[str, Any]:
    question = str(row.extra_info["question"])
    return {
        "data_source": str(row.data_source),
        "prompt": [{"content": f"{prompt_prefix}Question: {question}\n", "role": "user"}],
        "ability": str(row.ability),
        "reward_model": {
            "ground_truth": {"target": answers},
            "style": str(row.reward_model["style"]),
        },
        "extra_info": {
            "index": int(row.extra_info["index"]),
            "question": question,
            "source_id": str(row.extra_info["source_id"]),
            "split": str(row.extra_info["split"]),
        },
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path, compression="snappy")


def clean_dataset(
    input_path: Path,
    output_dir: Path,
    output_name: str,
    schema: pa.Schema,
    prompt_prefix: str,
) -> tuple[Path, dict[str, Any]]:
    frame = pd.read_parquet(input_path)
    popqa = frame[frame["data_source"] == "popqa"]
    popqa_question_objects = build_popqa_question_objects(popqa)

    kept_rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    source_before: Counter[str] = Counter(str(value) for value in frame["data_source"])
    source_after: Counter[str] = Counter()
    source_decisions: dict[str, Counter[str]] = defaultdict(Counter)

    for row in frame.itertuples(index=False):
        before = get_targets(row)
        decision, reason, after = classify_row(row, popqa_question_objects)
        source = str(row.data_source)
        record = {
            "ruleset": RULESET_VERSION,
            "data_source": source,
            "source_id": str(row.extra_info["source_id"]),
            "split": str(row.extra_info["split"]),
            "question": str(row.extra_info["question"]),
            "decision": decision,
            "reason": reason,
            "before_target": before,
            "after_target": after,
        }
        audit.append(record)
        decision_counts[decision] += 1
        source_decisions[source][decision] += 1
        if after is None:
            rejected.append(record)
            continue
        if not after:
            raise AssertionError(f"Kept row has no target: {source}/{record['source_id']}")
        kept_rows.append(output_row(row, after, prompt_prefix))
        source_after[source] += 1

    output_path = output_dir / output_name
    write_parquet(output_path, kept_rows, schema)
    write_jsonl(output_dir / "audit.jsonl", audit)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    summary = {
        "ruleset": RULESET_VERSION,
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "input_rows": len(frame),
        "output_rows": len(kept_rows),
        "rejected_rows": len(rejected),
        "source_before": dict(sorted(source_before.items())),
        "source_after": dict(sorted(source_after.items())),
        "decision_counts": dict(sorted(decision_counts.items())),
        "source_decisions": {
            source: dict(sorted(counts.items())) for source, counts in sorted(source_decisions.items())
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, summary


def largest_remainder_quotas(size: int) -> dict[str, int]:
    total = sum(TRAIN_SOURCE_COUNTS.values())
    exact = {
        source: Fraction(size * count, total) for source, count in TRAIN_SOURCE_COUNTS.items()
    }
    quotas = {source: int(value) for source, value in exact.items()}
    remainder = size - sum(quotas.values())
    priority = sorted(
        TRAIN_SOURCE_ORDER,
        key=lambda source: (exact[source] - quotas[source], -TRAIN_SOURCE_ORDER.index(source)),
        reverse=True,
    )
    for source in priority[:remainder]:
        quotas[source] += 1
    return quotas


def capped_equal_quotas(
    size: int,
    available: dict[str, int],
    source_order: tuple[str, ...],
) -> dict[str, int]:
    if size > sum(available.values()):
        raise ValueError(f"Requested {size} rows, only {sum(available.values())} available")
    quotas = {source: 0 for source in source_order}
    active = list(source_order)
    remaining = size
    while active:
        base, extra = divmod(remaining, len(active))
        capped = [source for source in active if available[source] < base]
        if capped:
            for source in capped:
                quotas[source] = available[source]
                remaining -= quotas[source]
                active.remove(source)
            continue
        for position, source in enumerate(active):
            quotas[source] = base + (1 if position < extra else 0)
        remaining = 0
        break
    if remaining or sum(quotas.values()) != size:
        raise AssertionError(f"Could not allocate {size} rows: {quotas}, remaining={remaining}")
    return quotas


def stable_rank(seed: int, source: str, source_id: str) -> str:
    value = f"{seed}\0{source}\0{source_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def sample_rows(
    frame: pd.DataFrame,
    quotas: dict[str, int],
    seed: int,
    source_order: tuple[str, ...],
    shuffle_output: bool,
) -> list[dict[str, Any]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    for source in source_order:
        group = frame[frame["data_source"] == source]
        candidates: list[tuple[str, dict[str, Any]]] = []
        for row in group.to_dict(orient="records"):
            source_id = str(row["extra_info"]["source_id"])
            candidates.append((stable_rank(seed, source, source_id), row))
        candidates.sort(key=lambda item: item[0])
        quota = quotas.get(source, 0)
        if quota > len(candidates):
            raise ValueError(f"Need {quota} {source} rows, only {len(candidates)} available")
        selected.extend(candidates[:quota])
    if shuffle_output:
        selected.sort(
            key=lambda item: hashlib.sha256(f"output\0{seed}\0{item[0]}".encode()).hexdigest()
        )
    return [row for _, row in selected]


def create_samples(
    cleaned_train_path: Path,
    cleaned_eval_path: Path,
    sample_root: Path,
    schema: pa.Schema,
) -> dict[str, Any]:
    train = pd.read_parquet(cleaned_train_path)
    eval_frame = pd.read_parquet(cleaned_eval_path)
    output: dict[str, Any] = {"train": {}, "eval": {}}

    for size in TRAIN_SAMPLE_SIZES:
        quotas = largest_remainder_quotas(size)
        rows = sample_rows(train, quotas, TRAIN_SEED, TRAIN_SOURCE_ORDER, shuffle_output=True)
        directory = sample_root / f"{size}t"
        path = directory / "co_search_ablation.train.parquet"
        write_parquet(path, rows, schema)
        manifest = {
            "kind": "train",
            "size": size,
            "seed": TRAIN_SEED,
            "sampling": "stable SHA-256 rank per source; CoSearch largest-remainder source quotas",
            "source_quotas": quotas,
            "source_cleaned_path": str(cleaned_train_path),
            "source_cleaned_sha256": sha256_file(cleaned_train_path),
            "output_path": str(path),
            "output_sha256": sha256_file(path),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output["train"][str(size)] = manifest

    available = Counter(str(value) for value in eval_frame["data_source"])
    for size in EVAL_SAMPLE_SIZES:
        quotas = capped_equal_quotas(size, dict(available), EVAL_SOURCE_ORDER)
        rows = sample_rows(eval_frame, quotas, EVAL_SEED, EVAL_SOURCE_ORDER, shuffle_output=False)
        directory = sample_root / f"{size}e"
        path = directory / "co_search_ablation.eval.parquet"
        write_parquet(path, rows, schema)
        manifest = {
            "kind": "eval",
            "size": size,
            "seed": EVAL_SEED,
            "sampling": "stable SHA-256 rank per source; capped equal benchmark quotas",
            "source_quotas": quotas,
            "source_cleaned_path": str(cleaned_eval_path),
            "source_cleaned_sha256": sha256_file(cleaned_eval_path),
            "output_path": str(path),
            "output_sha256": sha256_file(path),
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output["eval"][str(size)] = manifest
    return output


def verify_nested_samples(sample_root: Path) -> None:
    def keys(path: Path) -> set[tuple[str, str]]:
        frame = pd.read_parquet(path, columns=["data_source", "extra_info"])
        return {
            (str(row.data_source), str(row.extra_info["source_id"]))
            for row in frame.itertuples(index=False)
        }

    train_512 = keys(sample_root / "512t/co_search_ablation.train.parquet")
    train_5100 = keys(sample_root / "5100t/co_search_ablation.train.parquet")
    train_12000 = keys(sample_root / "12000t/co_search_ablation.train.parquet")
    train_25000 = keys(sample_root / "25000t/co_search_ablation.train.parquet")
    eval_350 = keys(sample_root / "350e/co_search_ablation.eval.parquet")
    eval_3500 = keys(sample_root / "3500e/co_search_ablation.eval.parquet")
    if not train_512 < train_5100 < train_12000 < train_25000:
        raise AssertionError("Train sample nesting invariant failed")
    if not eval_350 < eval_3500:
        raise AssertionError("Eval sample nesting invariant failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-train", type=Path, default=DEFAULT_FULL_TRAIN)
    parser.add_argument("--full-eval", type=Path, default=DEFAULT_FULL_EVAL)
    parser.add_argument("--schema-reference", type=Path, default=DEFAULT_SCHEMA_REFERENCE)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = pq.read_schema(args.schema_reference)
    prompt_prefix = reference_prompt_prefix(args.schema_reference)
    train_path, train_summary = clean_dataset(
        args.full_train,
        args.processed_root / "train",
        "co_search_single_or.train.parquet",
        schema,
        prompt_prefix,
    )
    eval_path, eval_summary = clean_dataset(
        args.full_eval,
        args.processed_root / "eval",
        "co_search_single_or.eval.parquet",
        schema,
        prompt_prefix,
    )
    samples = create_samples(train_path, eval_path, args.sample_root, schema)
    verify_nested_samples(args.sample_root)

    manifest = {
        "ruleset": RULESET_VERSION,
        "schema_reference": str(args.schema_reference),
        "schema_reference_sha256": sha256_file(args.schema_reference),
        "train_summary": train_summary,
        "eval_summary": eval_summary,
        "samples": samples,
    }
    manifest_path = args.processed_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
