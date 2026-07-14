"""Legacy OR-reference and structured AND-of-OR answer metrics."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


def normalize_answer(text: Any) -> str:
    """Use the normalization historically used by Search-R1 and AIR eval."""

    value = str(text or "").lower()
    value = "".join(ch for ch in value if ch not in set(string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1_pair(prediction: Any, answer: Any) -> float:
    pred_tokens = normalize_answer(prediction).split()
    answer_tokens = normalize_answer(answer).split()
    if not pred_tokens or not answer_tokens:
        return 0.0
    num_same = sum((Counter(pred_tokens) & Counter(answer_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def legacy_exact_match(prediction: Any, answers: Sequence[Any]) -> float:
    normalized_prediction = normalize_answer(prediction)
    if not normalized_prediction:
        return 0.0
    return float(any(normalized_prediction == normalize_answer(answer) for answer in answers))


def legacy_token_f1(prediction: Any, answers: Sequence[Any]) -> float:
    return max((token_f1_pair(prediction, answer) for answer in answers), default=0.0)


def coerce_answer_groups(value: Any, fallback_answers: Sequence[Any] = ()) -> list[list[str]]:
    """Convert parquet/OmegaConf containers into non-empty answer groups."""

    if value is None:
        raw_groups: list[Any] = []
    elif hasattr(value, "tolist"):
        converted = value.tolist()
        raw_groups = converted if isinstance(converted, list) else [converted]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_groups = list(value)
    else:
        raw_groups = [value]

    groups: list[list[str]] = []
    for raw_group in raw_groups:
        if hasattr(raw_group, "tolist"):
            raw_group = raw_group.tolist()
        if isinstance(raw_group, Sequence) and not isinstance(raw_group, (str, bytes)):
            aliases = [str(item).strip() for item in raw_group if str(item).strip()]
        else:
            aliases = [str(raw_group).strip()] if str(raw_group).strip() else []
        if aliases:
            groups.append(aliases)
    if groups:
        return groups
    fallback = [str(item).strip() for item in fallback_answers if str(item).strip()]
    return [fallback] if fallback else []


def groups_from_ground_truth(ground_truth: Any) -> tuple[list[str], list[list[str]], bool, str]:
    """Read legacy targets and optional structured fields from ground truth."""

    if isinstance(ground_truth, Mapping):
        raw_answers = ground_truth.get("target")
        if raw_answers is None:
            raw_answers = ground_truth.get("answers")
        if raw_answers is None:
            raw_answers = ground_truth.get("answer")
        if raw_answers is None:
            raw_answers = []
        raw_groups = ground_truth.get("required_answer_groups")
        eligible = bool(ground_truth.get("structured_reward_eligible", True))
        semantics = str(ground_truth.get("answer_semantics") or "single_or")
    else:
        raw_answers = ground_truth
        raw_groups = None
        eligible = True
        semantics = "single_or"
    if hasattr(raw_answers, "tolist"):
        raw_answers = raw_answers.tolist()
    if not isinstance(raw_answers, Sequence) or isinstance(raw_answers, (str, bytes)):
        raw_answers = [raw_answers]
    answers = [str(item).strip() for item in raw_answers if str(item).strip()]
    return answers, coerce_answer_groups(raw_groups, answers), eligible, semantics


def _contains_alias(prediction_tokens: list[str], alias: Any) -> bool:
    alias_tokens = normalize_answer(alias).split()
    if not prediction_tokens or not alias_tokens or len(alias_tokens) > len(prediction_tokens):
        return False
    width = len(alias_tokens)
    return any(prediction_tokens[start : start + width] == alias_tokens for start in range(len(prediction_tokens) - width + 1))


@dataclass(frozen=True)
class AnswerGroupMetrics:
    legacy_em: float
    legacy_f1: float
    structured_em: float
    answer_group_f1: float
    answer_group_recall: float
    matched_group_count: int
    required_group_count: int
    structured_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def answer_group_metrics(
    prediction: Any,
    answers: Sequence[Any],
    required_answer_groups: Any = None,
    *,
    structured_eligible: bool = True,
) -> AnswerGroupMetrics:
    """Score all required groups independent of group and answer order.

    For one OR group, structured EM intentionally equals legacy EM. For two or
    more required groups, each group must contribute one contiguous normalized
    alias span. The partial group F1 combines group coverage with token
    precision, so complete answers outrank one-member answers and verbose text.
    """

    legacy_answers = [str(item) for item in answers]
    groups = coerce_answer_groups(required_answer_groups, legacy_answers)
    legacy_em = legacy_exact_match(prediction, legacy_answers)
    legacy_f1 = legacy_token_f1(prediction, legacy_answers)
    if not structured_eligible or not groups:
        return AnswerGroupMetrics(legacy_em, legacy_f1, 0.0, 0.0, 0.0, 0, len(groups), False)

    if len(groups) == 1:
        structured_em = legacy_exact_match(prediction, groups[0])
        group_f1 = legacy_token_f1(prediction, groups[0])
        return AnswerGroupMetrics(
            legacy_em,
            legacy_f1,
            structured_em,
            group_f1,
            structured_em,
            int(structured_em),
            1,
            True,
        )

    prediction_tokens = normalize_answer(prediction).split()
    matched_groups: list[list[str]] = []
    for group in groups:
        matching_aliases = [alias for alias in group if _contains_alias(prediction_tokens, alias)]
        if matching_aliases:
            matched_groups.append(matching_aliases)

    matched_count = len(matched_groups)
    group_recall = matched_count / len(groups)
    covered_tokens: Counter[str] = Counter()
    for aliases in matched_groups:
        alias_tokens = max((normalize_answer(alias).split() for alias in aliases), key=len)
        covered_tokens.update(alias_tokens)
    token_precision = (
        sum((Counter(prediction_tokens) & covered_tokens).values()) / len(prediction_tokens)
        if prediction_tokens
        else 0.0
    )
    group_f1 = (
        2 * token_precision * group_recall / (token_precision + group_recall)
        if token_precision + group_recall > 0
        else 0.0
    )
    return AnswerGroupMetrics(
        legacy_em,
        legacy_f1,
        float(matched_count == len(groups)),
        group_f1,
        group_recall,
        matched_count,
        len(groups),
        True,
    )
