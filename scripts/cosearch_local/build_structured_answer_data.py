#!/usr/bin/env python3
"""Build the frozen 512/350 structured-answer experiment datasets.

The default CoSearch parquet files were cleaned after the multi-Gold audit. This
builder deliberately reads the timestamped pre-cleaning backup and never
replaces a default data path. The manual taxonomy below is keyed by immutable
source IDs; raw benchmark metadata is used only to recover valid aliases and
MuSiQue final-hop answers.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BACKUP_ROOT = ROOT / "data/global_train_eval_data/replaced_backups/20260710-231218/data/coAgenticRetriever/albation_1"
DEFAULT_TRAIN = BACKUP_ROOT / "co_search_ablation.train.parquet"
DEFAULT_EVAL = BACKUP_ROOT / "co_search_ablation.eval.parquet"
DEFAULT_ROLLOUT_DIR = ROOT / (
    "log/agenticIterRag/260710-113003-543853-pipeline-"
    "agentic_iter_rag_v1_search_r1_original_qwen3_1_7b_formal/outputs/"
    "stages/train_agent/spad_rag/search_policy_rl/rollout_data"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/AgenticIterRag/structured_answer/260711a_search_r1_512_350"
TAXONOMY = {"alias_or", "required_set", "multi_slot", "ambiguous", "contaminated"}


# Every one of the 29 multi-Gold prompts used by the frozen Search-R1 run was
# read against the question wording and, for MuSiQue, its decomposition chain.
TRAIN_LABELS = {
    "train_3530": "alias_or",
    "train_16007": "contaminated",
    "train_18779": "alias_or",
    "train_3390": "required_set",
    "train_74968": "contaminated",
    "train_6639": "ambiguous",
    "train_17645": "required_set",
    "train_10379": "alias_or",
    "train_27192": "required_set",
    "train_8914": "alias_or",
    "train_27820": "ambiguous",
    "train_18307": "alias_or",
    "train_41004": "required_set",
    "train_7824": "alias_or",
    "train_11517": "alias_or",
    "train_19265": "alias_or",
    "train_11306": "alias_or",
    "train_67047": "contaminated",
    "train_33855": "required_set",
    "train_4270": "contaminated",
    "train_17584": "alias_or",
    "train_8853": "alias_or",
    "train_54773": "required_set",
    "train_9664": "alias_or",
    "train_60713": "required_set",
    "train_8172": "alias_or",
    "train_22467": "required_set",
    "train_6073": "alias_or",
    "train_53054": "required_set",
}

TRAIN_INELIGIBLE = {"train_16007", "train_74968", "train_6639", "train_27820", "train_67047"}


# TriviaQA's flattened knowledge-base aliases contain related pages, people and
# even opposite concepts. These IDs were manually reviewed; only the listed
# aliases remain eligible for structured scoring.
TRIVIA_SAFE_GROUPS = {
    "test_4793": ["Abolitionism", "Abolition of slavery", "Anti-slavery movement"],
    "test_3813": ["Edward the Seventh"],
    "test_1608": ["Andrew Bonar Law", "Andrew Bonar-Law", "A Bonar Law", "Bonar Law"],
    "test_7998": ["Persuasion"],
    "test_11127": ["McCormick"],
    "test_3886": ["Rodrigo y Gabriela", "Rodrigo and Gabriela", "Rodrigo & Gabriela", "Rod y Gab"],
    "test_10107": ["Michael Winner"],
    "test_3400": ["Allergic reaction", "Type I hypersensitivity reaction", "Allergy"],
    "test_2576": ["Kate Winslet", "Kate Elizabeth Winslet", "Winslet"],
    "test_2820": ["Carmen Miranda", "Maria do Carmo Miranda da Cunha"],
    "test_1718": ["Mexico", "Mexico (country)", "México", "United Mexican States"],
    "test_2189": ["Orwell"],
    "test_1160": ["Popeye", "Popeye the Sailor", "Popeye the Sailor Man"],
    "test_107": ["Perry Mason"],
    "test_2855": ["The Rocky Horror Show", "Rocky Horror Show", "Richard O'Brien's The Rocky Horror Show"],
    "test_4301": ["Turkey", "Türkiye", "Republic of Turkey"],
    "test_4339": ["Sacred Island"],
    "test_7425": ["Mexico", "Mexico (country)", "México", "United Mexican States"],
    "test_3329": ["Charles Taylor", "Charles Taylor (politician)", "Taylor, Charles"],
    "test_908": ["Cheryl Cole", "Cheryl Tweedy", "Cheryl Fernandez-Versini", "Cheryl Ann Cole"],
    "test_4867": ["Iggy Pop", "Iggy Stooge", "The Godfather of Punk"],
    "test_2389": ["Floating"],
    "test_7268": ["Massachusetts", "Commonwealth of Massachusetts", "Massachusetts, United States"],
    "test_8557": ["Songs of Experience", "'SONGS OF EXPERIENCE'"],
    "test_10605": ["Pictures at an Exhibition", "Pictures from an Exhibition", "Bilder einer Ausstellung"],
    "test_7207": ["DV"],
    "test_9581": ["Claude Monet", "Oscar-Claude Monet", "Claude Oscar Monet", "Monet"],
    "test_10494": ["Crackerjack", "Crackerjack!", "CrackerJack"],
}

TRIVIA_AMBIGUOUS = {"test_5516"}


NQ_LABELS = {
    "test_3480": "multi_slot",
    "test_759": "alias_or",
    "test_32": "ambiguous",
    "test_1117": "multi_slot",
    "test_1108": "required_set",
    "test_1080": "ambiguous",
    "test_752": "alias_or",
    "test_1001": "ambiguous",
    "test_3329": "alias_or",
    "test_2218": "contaminated",
    "test_2981": "required_set",
    "test_527": "multi_slot",
    "test_2487": "multi_slot",
    "test_26": "required_set",
    "test_3556": "alias_or",
    "test_3082": "ambiguous",
    "test_1206": "ambiguous",
    "test_2854": "ambiguous",
    "test_1582": "ambiguous",
    "test_3109": "ambiguous",
    "test_3526": "alias_or",
}

NQ_GROUPS = {
    "test_3480": [["Lighthouse Cove"], ["Near Tavistock"]],
    "test_1117": [["in the 7th century"], ["700-1000 AD", "700–1000 AD"]],
    "test_1108": [["Ridley Park"], ["Lansdowne"], ["just outside Philadelphia, Pennsylvania"], ["Upper Darby"]],
    "test_2218": [["Clare Torry"]],
    "test_2981": [["Clarence Anglin"], ["John Anglin"], ["Frank Morris"]],
    "test_527": [["Mendel", "Gregor Mendel"], ["the common edible pea", "pea plants", "variation in plants"]],
    "test_2487": [["Welsh"], ["Mervyn", "the Welsh name Mervyn"]],
    "test_26": [
        ["Yakima (Washington)", "Washington"],
        ["western Canyon County, Idaho", "Idaho"],
        ["Willamette (Oregon)", "Oregon"],
    ],
}

NQ_INELIGIBLE = {
    "test_32",
    "test_1080",
    "test_1001",
    "test_1206",
    "test_2854",
    "test_1582",
    "test_3109",
}


def normalize(value: Any) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value).casefold()).split())


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value if isinstance(value, list) else [value]


def targets(row: Any) -> list[str]:
    return [str(item).strip() for item in to_list(row.reward_model["ground_truth"]["target"]) if str(item).strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_raw(source: str, split: str) -> dict[str, dict[str, Any]]:
    path = ROOT / f"data/raw_sets/{source}/{split}.jsonl"
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            result[str(row["id"])] = row
    return result


def parse_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    return [str(item).strip() for item in to_list(value) if str(item).strip()]


def final_musique_group(raw: dict[str, Any], answers: list[str]) -> list[str]:
    decomposition = raw.get("metadata", {}).get("question_decomposition") or []
    final_answer = str(decomposition[-1].get("answer") or "").strip() if decomposition else ""
    if not final_answer:
        return [answers[0]] if answers else []
    final_norm = normalize(final_answer)
    aliases = [
        answer
        for answer in answers
        if normalize(answer) == final_norm or normalize(answer) in final_norm or final_norm in normalize(answer)
    ]
    return aliases or [final_answer]


def annotation_for_train(source_id: str, answers: list[str]) -> tuple[str, list[list[str]], bool, str]:
    label = TRAIN_LABELS[source_id]
    eligible = source_id not in TRAIN_INELIGIBLE
    if label == "required_set":
        groups = [[answer] for answer in answers if normalize(answer) != "others"]
    elif source_id == "train_4270":
        groups = [["Texas Tech"]]
    elif source_id == "train_74968":
        groups = []
    else:
        groups = [answers]
    note = "manual review of frozen Search-R1 training prompt"
    if not eligible:
        note += "; excluded because the Gold list cannot define a complete, unique target"
    return label, groups, eligible, note


def annotation_for_eval(
    source: str,
    source_id: str,
    answers: list[str],
    raw: dict[str, Any],
) -> tuple[str, list[list[str]], bool, str]:
    if source == "popqa":
        metadata = raw.get("metadata") or {}
        official = [str(metadata.get("obj") or "").strip(), *parse_aliases(metadata.get("o_aliases"))]
        official = [item for item in official if item]
        contaminated = {normalize(item) for item in answers} != {normalize(item) for item in official}
        label = "contaminated" if contaminated else "alias_or"
        return label, [official], True, "validated against one PopQA object ID and its declared aliases"

    if source == "2wikimultihopqa":
        if source_id == "dev_8394":
            group = [answer for answer in answers if normalize(answer) != "sunday of life"]
            return "contaminated", [group], True, "removed an unrelated expanded alias"
        return "alias_or", [answers], True, "manual review: spelling, title, place or entity aliases"

    if source == "triviaqa":
        if source_id in TRIVIA_AMBIGUOUS:
            return "ambiguous", [], False, "question is incomplete and does not identify the alleged sequel"
        if source_id in TRIVIA_SAFE_GROUPS:
            return "contaminated", [TRIVIA_SAFE_GROUPS[source_id]], True, "removed related or unrelated KB expansions"
        return "alias_or", [answers], True, "manual review: answer aliases and formatting variants"

    if source == "nq":
        label = NQ_LABELS[source_id]
        eligible = source_id not in NQ_INELIGIBLE
        if source_id in NQ_GROUPS:
            groups = NQ_GROUPS[source_id]
        elif label == "required_set":
            groups = [[answer] for answer in answers]
        else:
            groups = [answers]
        return label, groups, eligible, "manual review of question cardinality, slots and time dependence"

    if source == "musique":
        group = final_musique_group(raw, answers)
        kept = {normalize(item) for item in group}
        contaminated = any(normalize(item) not in kept for item in answers)
        label = "contaminated" if contaminated else "alias_or"
        return label, [group], True, "validated against the final MuSiQue decomposition answer"

    raise ValueError(f"unsupported multi-Gold source: {source}")


def rollout_questions(rollout_dir: Path) -> set[str]:
    questions: set[str] = set()
    pattern = re.compile(r"Question: (.*?)\n(?:\nassistant|$)", re.S)
    for path in sorted(rollout_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                match = pattern.search(json.loads(line)["input"])
                if match:
                    questions.add(match.group(1).strip())
    return questions


def annotate_frame(
    frame: pd.DataFrame,
    *,
    split: str,
    raw_by_source: dict[str, dict[str, dict[str, Any]]],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    output_rows: list[dict[str, Any]] = []
    classification_rows: list[dict[str, Any]] = []
    for source_index, row in frame.iterrows():
        record = row.to_dict()
        reward_model = dict(record["reward_model"])
        ground_truth = dict(reward_model["ground_truth"])
        answer_list = targets(row)
        extra_info = dict(record["extra_info"])
        source = str(record["data_source"])
        source_id = str(extra_info.get("source_id") or "")
        if len(answer_list) == 1:
            label, groups, eligible, note = "single", [answer_list], True, "single legacy target"
        elif split == "train":
            label, groups, eligible, note = annotation_for_train(source_id, answer_list)
        else:
            raw = raw_by_source.get(source, {}).get(source_id, {})
            label, groups, eligible, note = annotation_for_eval(source, source_id, answer_list, raw)

        ground_truth.update(
            {
                "required_answer_groups": groups,
                "answer_semantics": label,
                "structured_reward_eligible": eligible,
            }
        )
        reward_model["ground_truth"] = ground_truth
        record["reward_model"] = reward_model
        output_rows.append(record)
        if len(answer_list) > 1:
            if label not in TAXONOMY:
                raise AssertionError(f"invalid label for {source_id}: {label}")
            classification_rows.append(
                {
                    "split": split,
                    "source_row_index": int(source_index),
                    "data_source": source,
                    "source_id": source_id,
                    "question": str(extra_info.get("question") or ""),
                    "legacy_targets": answer_list,
                    "classification": label,
                    "required_answer_groups": groups,
                    "structured_reward_eligible": eligible,
                    "review_note": note,
                }
            )
    return pd.DataFrame(output_rows), classification_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "source_row_index",
        "data_source",
        "source_id",
        "question",
        "legacy_targets",
        "classification",
        "required_answer_groups",
        "structured_reward_eligible",
        "review_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            rendered = dict(row)
            rendered["legacy_targets"] = json.dumps(row["legacy_targets"], ensure_ascii=False)
            rendered["required_answer_groups"] = json.dumps(row["required_answer_groups"], ensure_ascii=False)
            writer.writerow(rendered)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--rollout-dir", type=Path, default=DEFAULT_ROLLOUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(args.train)
    eval_frame = pd.read_parquet(args.eval)
    questions = rollout_questions(args.rollout_dir)
    train_512 = train[train["extra_info"].map(lambda item: str(item.get("question") or "") in questions)].copy()
    if len(questions) != 512 or len(train_512) != 512:
        raise AssertionError(f"expected 512 frozen prompts, got questions={len(questions)} rows={len(train_512)}")

    raw_by_source = {
        "popqa": read_raw("popqa", "test"),
        "2wikimultihopqa": read_raw("2wikimultihopqa", "dev"),
        "triviaqa": read_raw("triviaqa", "test"),
        "nq": read_raw("nq", "test"),
        "musique": read_raw("musique", "dev"),
    }
    structured_train, train_classifications = annotate_frame(train_512, split="train", raw_by_source={})
    structured_eval, eval_classifications = annotate_frame(eval_frame, split="eval", raw_by_source=raw_by_source)
    if len(train_classifications) != 29 or len(eval_classifications) != 150:
        raise AssertionError(
            f"classification coverage mismatch: train={len(train_classifications)} eval={len(eval_classifications)}"
        )

    train_path = args.output_dir / "search_r1_structured.train.parquet"
    eval_path = args.output_dir / "search_r1_structured.eval.parquet"
    structured_train.to_parquet(train_path, index=False)
    structured_eval.to_parquet(eval_path, index=False)
    classifications = train_classifications + eval_classifications
    jsonl_path = args.output_dir / "multi_gold_classification.jsonl"
    tsv_path = args.output_dir / "multi_gold_classification.tsv"
    write_jsonl(jsonl_path, classifications)
    write_tsv(tsv_path, classifications)

    manifest = {
        "version": "search-r1-structured-answer-v1",
        "source_train": str(args.train),
        "source_train_sha256": sha256_file(args.train),
        "source_eval": str(args.eval),
        "source_eval_sha256": sha256_file(args.eval),
        "source_rollout_dir": str(args.rollout_dir),
        "train_rows": len(structured_train),
        "eval_rows": len(structured_eval),
        "train_multi_gold_rows": len(train_classifications),
        "eval_multi_gold_rows": len(eval_classifications),
        "classification_counts": dict(sorted(Counter(row["classification"] for row in classifications).items())),
        "structured_eligible_counts": dict(
            sorted(Counter(str(row["structured_reward_eligible"]).lower() for row in classifications).items())
        ),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "eval_path": str(eval_path),
        "eval_sha256": sha256_file(eval_path),
        "classification_jsonl": str(jsonl_path),
        "classification_jsonl_sha256": sha256_file(jsonl_path),
        "classification_tsv": str(tsv_path),
        "classification_tsv_sha256": sha256_file(tsv_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
