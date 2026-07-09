#!/usr/bin/env python3
"""Sample AIR endpoint data and analyze gold-answer text types."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = REPO_ROOT / (
    "data/AgenticIterRag/llm_reranker_branch_train_set/"
    "260704e_AIR_v1_traj_co_search_ablation.train_global_step_79__branch_end_point_top50_top5_short_reason/"
    "manifest.json"
)
DEFAULT_REPORT_DIR = Path(__file__).resolve().parent / "reports"

MONTH_WORDS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}
ORDINAL_WORDS = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
}
COMMON_PROFESSION_WORDS = {
    "actor",
    "actress",
    "artist",
    "author",
    "composer",
    "director",
    "footballer",
    "journalist",
    "musician",
    "novelist",
    "painter",
    "player",
    "poet",
    "politician",
    "producer",
    "singer",
    "songwriter",
    "tennis",
    "writer",
}
GEO_OR_NATIONALITY_WORDS = {
    "africa",
    "african",
    "america",
    "american",
    "australia",
    "australian",
    "britain",
    "british",
    "california",
    "canada",
    "canadian",
    "china",
    "chinese",
    "england",
    "english",
    "europe",
    "european",
    "france",
    "french",
    "germany",
    "german",
    "india",
    "indian",
    "ireland",
    "irish",
    "italian",
    "italy",
    "japan",
    "japanese",
    "kentucky",
    "london",
    "mexico",
    "new york",
    "russia",
    "russian",
    "scotland",
    "scottish",
    "spain",
    "spanish",
    "texas",
    "united kingdom",
    "united states",
    "wales",
    "welsh",
}
ORG_WORK_EVENT_MARKERS = {
    "academy",
    "album",
    "association",
    "attack",
    "attacks",
    "battle",
    "book",
    "church",
    "college",
    "company",
    "corporation",
    "entertainment",
    "cup",
    "film",
    "group",
    "institute",
    "league",
    "party",
    "pictures",
    "school",
    "series",
    "song",
    "stadium",
    "studios",
    "university",
    "war",
}
LEADING_FRAGMENT_WORDS = {
    "a",
    "an",
    "at",
    "by",
    "for",
    "from",
    "his",
    "in",
    "on",
    "the",
    "their",
    "to",
}


@dataclass(frozen=True)
class SpanType:
    primary: str
    tags: tuple[str, ...]
    normalized: str
    token_count: int
    char_count: int


def normalize_answer_text(text: Any) -> str:
    raw = str(text or "").lower()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    return " ".join(raw.translate(table).split())


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def targets(row: dict[str, Any]) -> list[str]:
    target = ((row.get("reward_model") or {}).get("ground_truth") or {}).get("target")
    if target is None:
        return []
    if isinstance(target, list):
        return [str(x) for x in target]
    return [str(target)]


def sample_id(row: dict[str, Any]) -> str:
    extra = row.get("extra_info") or {}
    return str(row.get("sample_id") or extra.get("sample_id") or extra.get("trajectory_id") or "")


def is_title_like(raw: str) -> bool:
    words = [w.strip("\"'()[]{}") for w in raw.split() if w.strip("\"'()[]{}")]
    if len(words) < 2 or len(words) > 5:
        return False
    capitalized = 0
    for word in words:
        if word[:1].isupper() or re.fullmatch(r"[A-Z][A-Z0-9&.-]+", word):
            capitalized += 1
    return capitalized >= max(2, len(words) - 1)


def is_acronym(raw: str, norm: str) -> bool:
    compact = re.sub(r"[^A-Za-z0-9]", "", raw)
    if len(compact) < 2 or len(compact) > 8:
        return False
    return compact.upper() == compact and any(ch.isalpha() for ch in compact)


def classify_span(answer: str) -> SpanType:
    raw = str(answer or "").strip()
    norm = normalize_answer_text(raw)
    tokens = norm.split()
    tags: set[str] = set()

    if not norm:
        return SpanType("empty_answer", ("empty",), norm, 0, 0)

    token_count = len(tokens)
    char_count = len(norm)
    has_digit = any(ch.isdigit() for ch in raw)
    if token_count <= 2 or char_count <= 15:
        tags.add("short_answer")
    if has_digit:
        tags.add("contains_number")
    if token_count > 1 and tokens[0] in LEADING_FRAGMENT_WORDS:
        tags.add("leading_fragment")
    if len(tokens) > 4:
        tags.add("longer_phrase")

    primary = "entity_or_phrase_other"
    if norm in {"yes", "no"}:
        primary = "yes_no"
        tags.add("boolean_answer")
    elif re.fullmatch(r"(?:in|by|since|around|circa|c)?\s*\d{3,4}", norm) and 100 <= int(tokens[-1]) <= 2100:
        primary = "year"
        tags.add("year_like")
    elif (
        any(tok in MONTH_WORDS for tok in tokens)
        or re.search(r"\d{1,2}\s+\d{1,2}", norm)
        or re.fullmatch(r"\d{3,4}s", norm)
        or "century" in tokens
    ):
        primary = "date_or_time_expression"
        tags.add("time_like")
    elif (
        re.fullmatch(r"\d+(?:\s+\d+)?", norm)
        or re.fullmatch(r"\d+(?:st|nd|rd|th)", norm)
        or re.fullmatch(r"\d+\s+(?:hundred|thousand|million|billion|percent|per cent)", norm)
        or re.fullmatch(r"\d+\s+(?:to|-)\s+\d+", norm)
        or norm in ORDINAL_WORDS
    ):
        primary = "number_or_ordinal"
        tags.add("numeric_or_ordinal")
    elif is_acronym(raw, norm):
        primary = "acronym_or_code"
        tags.add("short_entity")
    elif norm in GEO_OR_NATIONALITY_WORDS:
        primary = "geo_or_nationality"
        tags.add("short_entity")
    elif any(tok in COMMON_PROFESSION_WORDS for tok in tokens) and raw[:1].islower():
        primary = "common_noun_phrase"
        tags.add("common_phrase")
    elif any(tok in ORG_WORK_EVENT_MARKERS for tok in tokens):
        primary = "organization_work_or_event"
    elif is_title_like(raw) and not any(tok in GEO_OR_NATIONALITY_WORDS for tok in tokens):
        primary = "proper_name_entity"
        tags.add("proper_name")
    elif raw[:1].islower() and token_count >= 2:
        primary = "common_noun_phrase"
        tags.add("common_phrase")
    elif token_count <= 2:
        primary = "short_entity_or_short_phrase"
        tags.add("short_entity")
    elif token_count >= 5:
        primary = "long_descriptive_phrase"

    return SpanType(primary, tuple(sorted(tags)), norm, token_count, char_count)


def pair_has_containment(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    return f" {left} " in f" {right} " or f" {right} " in f" {left} "


def sample_type(span_types: list[SpanType]) -> tuple[str, list[str]]:
    if not span_types:
        return "no_gold_answer", ["empty"]

    tags = set(tag for span in span_types for tag in span.tags)
    primary_counts = Counter(span.primary for span in span_types)
    normalized = sorted({span.normalized for span in span_types if span.normalized})
    if len(normalized) > 1:
        tags.add("multi_target")
        has_alias = any(
            pair_has_containment(left, right)
            for i, left in enumerate(normalized)
            for right in normalized[i + 1 :]
        )
        if has_alias:
            tags.add("contains_alias_variant")
            return "multi_answer_set_with_aliases", sorted(tags)
        return "multi_answer_list", sorted(tags)

    return primary_counts.most_common(1)[0][0], sorted(tags)


TYPE_EXPLANATIONS = {
    "yes_no": "这类答案只有 yes/no。它对字符串命中奖励最危险，因为 yes/no 在普通段落里太常见；即便边界匹配也只是减少 Yeshiva 这种子串误伤，不能保证语义正确。",
    "year": "这类答案主要是年份，比如 1974 或 In 1642。年份在百科文本里出现频率很高，命中年份不等于命中问题证据，容易把背景时间误当正例。",
    "date_or_time_expression": "这类是更完整的时间表达，比如月份日期、9/11、世纪、年代。它比纯年份更具体一些，但如果问题问的是事件关系，单纯出现这个时间仍然不一定是答案证据。",
    "number_or_ordinal": "这类是数字、数量或序数，比如 7、eighth。它通常比年份还脆，因为数字本身缺少实体约束，很多无关 passage 都可能包含同一个数字。",
    "acronym_or_code": "这类是缩写或代码。短缩写如果很独特还好，但像 US、UK、TV 这种会非常泛，必须结合问题实体看。",
    "geo_or_nationality": "这类是地点、国家、州名、民族或语言，比如 Kentucky、French。单词越短越容易歧义，French 既可能是语言、国籍，也可能只是形容词。",
    "common_noun_phrase": "这类是普通名词短语或职业身份，比如 film director、professional tennis player。它们不是唯一实体，很多人物页面都会出现，所以字符串命中通常只能说明类型对，不一定说明答案对。",
    "organization_work_or_event": "这类是组织、作品、赛事、事件等名字。通常比纯数字稳定，但如果名字里有通用词，比如 attack、party、college，也可能命中背景介绍而不是答案关系。",
    "proper_name_entity": "这类是首字母大写的专名实体，包括人名、地名、机构名、作品名或页面标题。它通常比数字/年份稳定，但仍然不能直接等同 true evidence，因为列表页、演员表、消歧义页也会提到这些名字。",
    "short_entity_or_short_phrase": "这类是短实体或短短语，不一定能判断是人、地、组织还是概念。短答案最大的问题是上下文约束弱，字符串命中不能代表语义命中。",
    "long_descriptive_phrase": "这类是较长描述短语。它的误命中率一般低一些，但 exact phrase 匹配会更脆，换一种说法就可能漏掉。",
    "entity_or_phrase_other": "这类是启发式规则没法稳定归到上面类型的答案。它不一定有问题，只是需要人工看上下文后再决定是否适合字符串证据规则。",
    "multi_answer_list": "这类样本有多个不同 gold answer，通常是列表题或多跳结果。contains-any 规则会把只包含其中一个答案的 doc 当正例，这对训练 reranker 可能偏宽。",
    "multi_answer_set_with_aliases": "这类样本有多个答案字符串，其中一些是同一个答案的别名或带修饰版本。它有利于召回，但也可能让统计时看起来像多答案。",
    "no_gold_answer": "这类样本没有 gold target，正常情况下不应该参与基于答案字符串的证据判断。",
    "empty_answer": "这类是空字符串答案，通常应该过滤或单独处理。",
}


def example_line(example: dict[str, Any]) -> str:
    question = example["question"].replace("\n", " ").strip()
    sub_query = example["sub_query"].replace("\n", " ").strip()
    if len(question) > 120:
        question = question[:117] + "..."
    if len(sub_query) > 100:
        sub_query = sub_query[:97] + "..."
    return (
        f"- `{example['answer']}` | sample_id=`{example['sample_id']}` | "
        f"question: {question} | sub_query: {sub_query}"
    )


def add_example(bucket: list[dict[str, Any]], example: dict[str, Any], limit: int = 10) -> None:
    key = (example.get("answer"), example.get("sample_id"))
    if any((item.get("answer"), item.get("sample_id")) == key for item in bucket):
        return
    if len(bucket) < limit:
        bucket.append(example)


def pct(count: int, total: int) -> str:
    if not total:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def counter_table(counter: Counter[str], total: int) -> str:
    lines = ["| type | count | pct |", "| --- | ---: | ---: |"]
    for key, count in counter.most_common():
        lines.append(f"| `{key}` | {count} | {pct(count, total)} |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_order",
        "source_index",
        "sample_id",
        "question",
        "sub_query",
        "targets",
        "sample_primary",
        "sample_tags",
        "span_primary_counts",
        "span_details",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("targets", "sample_tags", "span_primary_counts", "span_details"):
                out[key] = json.dumps(out[key], ensure_ascii=False)
            writer.writerow({key: out.get(key, "") for key in fields})


def build_report(
    *,
    manifest_path: Path,
    dataset_path: Path,
    report_path: Path,
    summary: dict[str, Any],
    span_examples: dict[str, list[dict[str, Any]]],
    sample_examples: dict[str, list[dict[str, Any]]],
    tag_examples: dict[str, list[dict[str, Any]]],
) -> str:
    sample_total = summary["sample_size"]
    span_total = summary["span_count"]
    lines: list[str] = [
        "# AIR endpoint 5100 样本 gold answer 类型抽样分析",
        "",
        "## 基本信息",
        f"- created_at: `{summary['created_at']}`",
        f"- manifest: `{manifest_path}`",
        f"- dataset_jsonl: `{dataset_path}`",
        f"- full_sample_count: `{summary['full_sample_count']}`",
        f"- sampled_sample_count: `{sample_total}`",
        f"- random_seed: `{summary['random_seed']}`",
        f"- gold_answer_field: `reward_model.ground_truth.target`",
        f"- report_path: `{report_path}`",
        "",
        "## 结论先说",
        "- 这 200 条里不只是 yes/no、年份、数字、短实体、常见词短语；还明显存在多答案列表、专名实体、地点/国籍/语言、组织/作品/事件名、日期/事件时间表达、别名/修饰版本答案、较长描述短语。",
        "- 对 reranker 训练最需要警惕的不是某一个类型，而是“短答案 + contains-any 证据规则”的组合。它会把只沾到一个词、一个数字、一个年份的 passage 当成正例。",
        "- 多答案列表也要单独看。一个 doc 只包含列表中的一个 answer，也会被 contains-any 判成 hit；如果问题本来要求多个答案，这种正例偏宽。",
        "- 人名和较完整实体通常比数字/年份稳定，但仍然不是纯净 gold doc。演员表、列表页、消歧义页都可能只提到名字，不真正回答当前问题。",
        "",
        "## 样本级主类型分布",
        "",
        counter_table(Counter(summary["sample_primary_counts"]), sample_total),
        "",
        "说明：样本级会优先识别多答案样本，所以一个包含多个演员名的样本会进入 `multi_answer_list`，不会被简单算成人名。",
        "",
        "## 单个 answer span 主类型分布",
        "",
        counter_table(Counter(summary["span_primary_counts"]), span_total),
        "",
        "说明：span 级是把每个 gold answer 字符串单独拆开看，所以同一个样本有 3 个 gold answers，就会贡献 3 个 span。",
        "",
        "## 风险标签分布",
        "",
        counter_table(Counter(summary["risk_tag_counts"]), span_total),
        "",
        "说明：风险标签是可重叠的。比如 `French` 既是 `short_answer`，也可能是 `short_entity`；`In 1642` 既是年份，也带有 `leading_fragment`。",
        "",
        "## 类型解释与例子",
    ]

    type_order = list(dict.fromkeys(list(summary["sample_primary_counts"]) + list(summary["span_primary_counts"])))
    for type_name in type_order:
        lines.extend(
            [
                "",
                f"### `{type_name}`",
                "",
                TYPE_EXPLANATIONS.get(type_name, "这个类型没有单独解释，建议看明细 CSV 后人工复核。"),
                "",
                "例子：",
            ]
        )
        examples = span_examples.get(type_name) or sample_examples.get(type_name) or []
        if not examples:
            lines.append("- 暂无抽样例子。")
        else:
            lines.extend(example_line(example) for example in examples[:8])

    lines.extend(
        [
            "",
            "## 重点风险标签例子",
        ]
    )
    for tag in ["short_answer", "contains_number", "common_phrase", "leading_fragment", "multi_target", "contains_alias_variant"]:
        examples = tag_examples.get(tag) or []
        lines.extend(["", f"### `{tag}`", ""])
        if not examples:
            lines.append("- 抽样中没有收集到例子。")
        else:
            lines.extend(example_line(example) for example in examples[:8])

    lines.extend(
        [
            "",
            "## 口语化解读",
            "",
            "如果我们继续用 answer string 去找 true answer doc，最危险的是那种“看起来命中了，其实只是碰巧出现”的情况。yes/no 是最明显的，年份和数字也类似；它们本身信息量太低，不能证明这个 doc 真的回答了问题。",
            "",
            "短实体也要小心。像一个州名、一个语言名、一个姓氏，命中以后只能说明 passage 提到了这个词，不能说明它处在正确关系里。对于 reranker 来说，这会把一些语义不够正的文档推成正例，训练信号会变软。",
            "",
            "多答案样本是另一个问题。比如问题要求三个人或三个分支，gold answer 里有多个字符串。当前 contains-any 逻辑只要命中其中一个就算 evidence hit，这对召回分析是宽松的，但对训练 reranker 未必理想，因为 reranker 学到的可能是“包含任意一个答案片段就够了”。",
            "",
            "比较可靠的类型一般是完整人名、独特组织名、独特作品名、较长事件名。但这里也不能完全放心，因为 Wikipedia passage 可能只是列表式提到实体，没有提供问题需要的关系证据。",
            "",
            "所以后续如果要清洗 hard subset，我建议不要只做 yes/no 排除。更稳的路线是分层：先排除 yes/no；再对年份、纯数字、极短 answer 单独降权或要求 query/entity 共现；最后对多答案样本考虑 all/partial hit 的区别，而不是简单 contains-any。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    dataset_path = Path(manifest["dataset_jsonl"])
    rows = list(iter_jsonl(dataset_path))
    if args.sample_size > len(rows):
        raise ValueError(f"sample_size={args.sample_size} exceeds dataset size={len(rows)}")

    rng = random.Random(args.seed)
    sampled_indices = sorted(rng.sample(range(len(rows)), args.sample_size))
    sampled_rows = [(idx, rows[idx]) for idx in sampled_indices]

    sample_primary_counter: Counter[str] = Counter()
    span_primary_counter: Counter[str] = Counter()
    risk_tag_counter: Counter[str] = Counter()
    span_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tag_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detail_rows: list[dict[str, Any]] = []

    for order, (source_index, row) in enumerate(sampled_rows, start=1):
        extra = row.get("extra_info") or {}
        targs = targets(row)
        span_details = []
        span_types = []
        for answer in targs:
            span_type = classify_span(answer)
            span_types.append(span_type)
            span_primary_counter[span_type.primary] += 1
            risk_tag_counter.update(span_type.tags)
            example = {
                "answer": answer,
                "sample_id": sample_id(row),
                "question": str(extra.get("question") or ""),
                "sub_query": str(extra.get("sub_query") or ""),
            }
            add_example(span_examples[span_type.primary], example)
            for tag in span_type.tags:
                add_example(tag_examples[tag], example)
            span_details.append(
                {
                    "answer": answer,
                    "primary": span_type.primary,
                    "tags": list(span_type.tags),
                    "normalized": span_type.normalized,
                    "token_count": span_type.token_count,
                    "char_count": span_type.char_count,
                }
            )

        primary, sample_tags = sample_type(span_types)
        sample_primary_counter[primary] += 1
        sample_example = {
            "answer": " | ".join(targs),
            "sample_id": sample_id(row),
            "question": str(extra.get("question") or ""),
            "sub_query": str(extra.get("sub_query") or ""),
        }
        add_example(sample_examples[primary], sample_example)
        for tag in sample_tags:
            add_example(tag_examples[tag], sample_example)

        detail_rows.append(
            {
                "sample_order": order,
                "source_index": source_index,
                "sample_id": sample_id(row),
                "question": str(extra.get("question") or ""),
                "sub_query": str(extra.get("sub_query") or ""),
                "targets": targs,
                "sample_primary": primary,
                "sample_tags": sample_tags,
                "span_primary_counts": dict(Counter(item["primary"] for item in span_details)),
                "span_details": span_details,
            }
        )

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%y%m%d-%H%M%S")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.report_dir / f"{stamp}-answer_type_sample200"
    report_path = prefix.with_suffix(".report.md")
    summary_path = prefix.with_suffix(".summary.json")
    csv_path = prefix.with_suffix(".samples.csv")

    summary = {
        "created_at": created_at,
        "manifest": str(args.manifest),
        "dataset_jsonl": str(dataset_path),
        "full_sample_count": len(rows),
        "sample_size": len(sampled_rows),
        "span_count": sum(span_primary_counter.values()),
        "random_seed": args.seed,
        "sampled_source_indices": sampled_indices,
        "sample_primary_counts": dict(sample_primary_counter),
        "span_primary_counts": dict(span_primary_counter),
        "risk_tag_counts": dict(risk_tag_counter),
        "report_path": str(report_path),
        "summary_path": str(summary_path),
        "csv_path": str(csv_path),
    }

    report = build_report(
        manifest_path=args.manifest,
        dataset_path=dataset_path,
        report_path=report_path,
        summary=summary,
        span_examples=span_examples,
        sample_examples=sample_examples,
        tag_examples=tag_examples,
    )
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, detail_rows)

    print(f"wrote report: {report_path}")
    print(f"wrote summary: {summary_path}")
    print(f"wrote samples: {csv_path}")
    print(json.dumps({
        "sample_primary_counts": dict(sample_primary_counter),
        "span_primary_counts": dict(span_primary_counter),
        "risk_tag_counts": dict(risk_tag_counter),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
