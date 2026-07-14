#!/usr/bin/env python3
"""Rebuild the fixed 237-case SPAD teacher manual-judgment table."""

from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
ROLLOUT_REL = Path(
    "log/agenticIterRag/"
    "260710-021433-474200-pipeline-agentic_iter_rag_v1_spad_qwen3_1_7b_glm47_formal_500_offlinebatch_260710/"
    "outputs/stages/train_agent/spad_rag/search_policy_rl/rollout_data"
)
ROLLOUT_DIR = REPO_ROOT / ROLLOUT_REL
OUTPUT_PATH = Path(__file__).with_name("manual_judgments_237.tsv")
SEED = 260710100

STATUS_TO_LABEL = {
    "supported_answer": "S",
    "insufficient_evidence": "I",
    "ambiguous_evidence": "A",
}
LABEL_TO_STATUS = {value: key for key, value in STATUS_TO_LABEL.items()}
QUESTION_RE = re.compile(r"Question: (.*?)\n\nassistant", re.DOTALL)
REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL)


# These are the disagreements recorded during the manual review. Every other
# sampled row retains the teacher bucket as its manual status.
SUPPORTED_OVERRIDES: dict[int, tuple[str, str, str]] = {
    9: (
        "I",
        "entity_mismatch",
        "问题是 Tyler Spencer，证据和 teacher 理由却改成了 Tim Spencer，无法判断原问题中的两人是否都是美国歌手。",
    ),
    17: (
        "I",
        "predicate_mismatch",
        "证据证明 Jeff Kinney 是作者，没有证明谁是出版方；teacher 用作者回答 published。",
    ),
    20: (
        "I",
        "missing_bridge",
        "证据给出死亡城市 Stamford，但可见 passage 未建立 Stamford 到 Fairfield County 的关系。",
    ),
    24: (
        "I",
        "scope_mismatch",
        "证据中的 Rene 和 Sandra Pelt 来自不同情节范围，没有支持第一季中试图杀死 Sookie 的所问对象。",
    ),
    25: (
        "I",
        "predicate_mismatch",
        "证据只能确定两首歌的演唱者是 Mariah Carey，没有说明 We Belong Together 是关于谁。",
    ),
    35: (
        "I",
        "referent_mismatch",
        "问题指向 Meyerland Plaza 的位置，teacher 改答了 Gulfgate Mall 所在的 East End。",
    ),
    37: (
        "I",
        "entity_mismatch",
        "证据没有建立 Forces of Nature (2004) 导演及其国籍，理由引用了其他泰国电影和导演。",
    ),
    38: (
        "A",
        "underspecified_title",
        "近似标题对应 Technotronic、Avicii 和 Wham! 等不同表演者，问题中的误引不足以唯一定位歌曲。",
    ),
    40: (
        "I",
        "entity_mismatch",
        "证据中的法国第55步兵师没有被证明是题目通过 Dalton 所在行政区所指的部队。",
    ),
    41: (
        "A",
        "underspecified_referent",
        "检索结果中多个州或地区都有 Badlands，问题缺少能唯一确定州的先行实体。",
    ),
    52: (
        "A",
        "version_ambiguity",
        "Sidekick 有原始机型、Slide、LX 等多个版本和不同发布日期，问题未限定版本。",
    ),
    59: (
        "I",
        "missing_bridge",
        "证据只给出 Lydia Pinkham 出生于 Lynn，没有给出 Lynn 所属 county。",
    ),
    60: (
        "I",
        "missing_bridge",
        "证据只能确定 Unitech Group 总部在 New Delhi，没有给出该城市有多少 districts。",
    ),
    62: (
        "I",
        "unsupported_inference",
        "电影是美国电影不等于导演具有美国国籍，证据未分别给出两位导演的国籍。",
    ),
    63: (
        "I",
        "entity_mismatch",
        "证据中的 Tom 与 Emily Stewart 并未被可靠限定为 Desperate Housewives 所问的 Tom 和对应情节。",
    ),
    65: (
        "I",
        "predicate_mismatch",
        "证据识别出 Animal Liberation 的作者 Peter Singer，却没有给出谁雇用他。",
    ),
    66: (
        "I",
        "predicate_mismatch",
        "问题问死亡地点，teacher 使用死亡日期作答，且证据未提供所需地点。",
    ),
    68: (
        "I",
        "missing_attribute",
        "证据给出导演 Stuart Rosenberg 的姓名，没有给出他的国家或国籍。",
    ),
    82: (
        "I",
        "unsupported_alias",
        "证据没有出现 Ek Saudagar；把它直接视为 1991 年 Saudagar 的别名属于未由 passage 支持的假设。",
    ),
    90: (
        "I",
        "predicate_mismatch",
        "证据说 Santa Clarita 是拍摄地，没有证明它是 MythBusters 的叙事地点，后续市长答案因此失去依据。",
    ),
}

INSUFFICIENT_OVERRIDES: dict[int, tuple[str, str, str]] = {
    11: (
        "S",
        "simple_numeric_inference",
        "50-32 的赛季记录直接表示 50 胜；这是无需外部知识的简单数值读取。",
    ),
    16: (
        "A",
        "version_ambiguity",
        "证据包含多个 Sidekick 型号及日期，但问题没有限定原始机型或具体版本。",
    ),
    26: (
        "S",
        "direct_evidence_overlooked",
        "passage 明确说 Jessie 被其主人 Emily 抛弃；后来归 Al 所有不妨碍回答所问的原主人。",
    ),
    37: (
        "S",
        "temporal_endpoint_inference",
        "剧集播出区间明确截止于 2017-08-03，结合 final season 足以回答最后一集播出日期。",
    ),
    39: (
        "S",
        "negative_comparison",
        "证据分别把 Lynda Thomas 定义为个人歌手、Say Anything 定义为乐队，因此二者是否都是乐队可唯一回答 No。",
    ),
    40: (
        "S",
        "direct_evidence_overlooked",
        "理由本身已经读取 Jean-Paul Riopelle 的日期 7 October 1923 - 12 March 2002，其中首项就是出生日期。",
    ),
    44: (
        "S",
        "ordinal_temporal_inference",
        "证据列出 Tito 与 Haile Selassie 在 1954 和 1956 年互访，较早的 1954 年即可回答 first visit。",
    ),
    48: (
        "S",
        "ownership_bridge",
        "证据明确 Houston's 连锁品牌由 Hillstone Restaurant Group 所有，可用于回答其 Atlanta 门店的品牌所有者。",
    ),
    57: (
        "S",
        "direct_relation_overlooked",
        "No. 29 Squadron RFC 中的 RFC 已明确给出 Royal Flying Corps，正是问题所需答案。",
    ),
    73: (
        "S",
        "compositional_relation",
        "Aristaeus 的父亲是 Apollo，检索到的 Apollo of Mantua 条目给出该雕像以 Apollo Citharoedus 为基础。",
    ),
    76: (
        "S",
        "answer_granularity",
        "证据明确说明约30名非墨西哥定居者发动叛乱；who 不强制答案必须是单个人名。",
    ),
    77: (
        "A",
        "entity_ambiguity",
        "证据含 Missouri、Kentucky、Washington D.C. 等多个 Crestwood，问题没有州限定。",
    ),
    80: (
        "S",
        "direct_evidence_overlooked",
        "问题需要组织名，长期 RCA 合同已足以唯一识别 RCA Victor；无需再次证明题干给出的 44 年。",
    ),
    84: (
        "A",
        "title_ambiguity",
        "误引标题同时接近 Wake Me Up 和 Wake Me Up Before You Go-Go，现有证据不能唯一选择表演者。",
    ),
    86: (
        "S",
        "shared_medium_inference",
        "Lip Service 被明确描述为 television serial，Ruta Gedmintas 又出演该剧，二者共同媒介可确定为 television。",
    ),
    95: (
        "S",
        "temporal_inference",
        "证据通过 2015 语境中的 previous year's edition 指向 2014，可确定最近一次夺冠年份。",
    ),
}

AMBIGUOUS_OVERRIDES: dict[int, tuple[str, str, str]] = {
    1: (
        "S",
        "compatible_specificity",
        "political documentary 与 visual nonstory documentary 都是 documentary 的细分类，不是互斥候选答案。",
    ),
    2: (
        "I",
        "missing_fact",
        "证据只描述若干跳舞场景，没有给出问题所需的唯一舞种或特定事件限定。",
    ),
    6: (
        "S",
        "predicate_specificity",
        "3.5 billion years 明确对应已知最早的化石化原核生物；约40亿年描述的是更宽泛的最早生命祖先。",
    ),
    8: (
        "S",
        "temporal_granularity",
        "1997 是首次引入，2002 是成为固定节目；两个日期对应不同谓词，所问 start 可由 1997 唯一回答。",
    ),
    9: (
        "S",
        "predicate_mismatch",
        "Islamic Jihad Organization claimed responsibility；Hezbollah carried out 是不同关系，不构成 claimed 的竞争答案。",
    ),
    10: (
        "S",
        "event_disambiguation",
        "证据把所问推翻墨西哥统治的行动对应到约30名非墨西哥定居者；Fremont 后来抵达、Alvarado 是另一事件。",
    ),
    11: (
        "S",
        "predicate_mismatch",
        "Granville Woods 的 Multiplex Telegraph 明确用于车站与行驶列车通信；其他发明回答的是铁路信号或一般电报。",
    ),
    12: (
        "S",
        "temporal_scope",
        "最后一次 Firefall 和停止命令都明确在 1968 年 1 月；1969 是更宽泛的行政叙述，不是同一事件的竞争日期。",
    ),
    13: (
        "I",
        "missing_entity_evidence",
        "该轮可见 passage 未可靠提供 Say Anything 的实体事实；缺一侧比较依据属于证据不足，不是多个答案。",
    ),
    14: (
        "S",
        "temporal_granularity",
        "986、约1000年和 late 10th century 是兼容的时间粒度，共同支持十世纪末这一答案。",
    ),
    15: (
        "I",
        "undefined_measure",
        "证据给出消费量和多数吸烟者所在地区，但没有定义或直接测量 most popular，不能据此制造多个候选答案。",
    ),
    18: (
        "I",
        "entity_mismatch",
        "问题所指丈夫是 Alexander Hamilton，证据却围绕 William S. Hamilton；其多个死因不属于目标实体。",
    ),
    20: (
        "S",
        "predicate_mismatch",
        "nucleus 是 cell control centre；centrosome 是 microtubule organizing center，后者不是同一谓词下的候选答案。",
    ),
    23: (
        "S",
        "version_specified",
        "问题明确限定 book，原著中的凶手 Leo 可唯一回答；电视改编中的 Kirsten 不构成冲突。",
    ),
    26: (
        "S",
        "canonical_answer",
        "证据明确把无 mens rea 的犯罪称为 strict liability crimes；相关法域中的窄分类不推翻这一规范答案。",
    ),
}


def extract_question(raw_input: str) -> str:
    match = QUESTION_RE.search(raw_input)
    if not match:
        raise ValueError("Cannot extract question from rollout input")
    return match.group(1).strip()


def extract_teacher_reason(raw_content: str) -> str:
    match = REASON_RE.search(raw_content)
    if not match:
        return ""
    return " ".join(match.group(1).split())


def load_teacher_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(ROLLOUT_DIR.glob("*.jsonl"), key=lambda path: int(path.stem))
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if not record.get("teacher_called"):
                    continue
                record["question"] = extract_question(str(record.get("input") or ""))
                record["source_file"] = str(ROLLOUT_REL / path.name)
                record["source_line"] = line_number
                rows.append(record)
    return rows


def sample_rows(rows: list[dict[str, Any]], status: str) -> list[dict[str, Any]]:
    unique_by_question: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("teacher_evidence_status") != status:
            continue
        unique_by_question.setdefault(row["question"], row)
    unique_rows = list(unique_by_question.values())
    if status == "ambiguous_evidence":
        return unique_rows
    return random.Random(SEED).sample(unique_rows, 100)


def default_reason(label: str) -> tuple[str, str]:
    if label == "S":
        return "aligned_supported", "可见证据足以支持同一问题谓词下的唯一短答案。"
    if label == "I":
        return "aligned_insufficient", "可见证据缺少回答问题所必需的事实或关系。"
    return "aligned_ambiguous", "可见证据在同一问题谓词下支持多个互不兼容的候选答案。"


def build_output_rows() -> list[dict[str, Any]]:
    source_rows = load_teacher_rows()
    output_rows: list[dict[str, Any]] = []
    specifications = [
        ("supported_answer", "S", SUPPORTED_OVERRIDES),
        ("insufficient_evidence", "I", INSUFFICIENT_OVERRIDES),
        ("ambiguous_evidence", "A", AMBIGUOUS_OVERRIDES),
    ]
    for teacher_status, prefix, overrides in specifications:
        for sample_index, row in enumerate(sample_rows(source_rows, teacher_status), start=1):
            teacher_label = STATUS_TO_LABEL[teacher_status]
            manual_label = teacher_label
            error_type, manual_reason = default_reason(manual_label)
            if sample_index in overrides:
                manual_label, error_type, manual_reason = overrides[sample_index]
            output_rows.append(
                {
                    "audit_id": f"{prefix}-{sample_index:03d}",
                    "teacher_bucket": teacher_status,
                    "teacher_label": teacher_label,
                    "manual_status": LABEL_TO_STATUS[manual_label],
                    "manual_label": manual_label,
                    "agreement": int(teacher_label == manual_label),
                    "error_type": error_type,
                    "manual_reason": manual_reason,
                    "source_file": row["source_file"],
                    "source_line": row["source_line"],
                    "uid": row.get("uid") or "",
                    "question": row["question"],
                    "teacher_answer": row.get("teacher_answer") or "",
                    "teacher_reason": extract_teacher_reason(str(row.get("teacher_raw_content") or "")),
                    "gold_targets_reference_only": json.dumps(
                        (row.get("gts") or {}).get("target") or [], ensure_ascii=False
                    ),
                }
            )
    return output_rows


def validate(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 237:
        raise AssertionError(f"Expected 237 rows, got {len(rows)}")
    if len({row["audit_id"] for row in rows}) != 237:
        raise AssertionError("audit_id must be unique")
    expected = {
        "supported_answer": Counter({"S": 80, "I": 17, "A": 3}),
        "insufficient_evidence": Counter({"I": 84, "S": 13, "A": 3}),
        "ambiguous_evidence": Counter({"A": 22, "S": 11, "I": 4}),
    }
    for teacher_status, expected_counts in expected.items():
        actual = Counter(
            row["manual_label"] for row in rows if row["teacher_bucket"] == teacher_status
        )
        if actual != expected_counts:
            raise AssertionError(f"Unexpected counts for {teacher_status}: {actual}")
    agreements = sum(int(row["agreement"]) for row in rows)
    if agreements != 186:
        raise AssertionError(f"Expected 186 agreements, got {agreements}")


def main() -> None:
    rows = build_output_rows()
    validate(rows)
    fieldnames = list(rows[0])
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
