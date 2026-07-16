"""Independent V3 reward using the frozen Hard-Gate v2 Teacher strategy.

The existing stable and Gold Token-F1 reward modules are not modified. Stage A
is delegated to the frozen stable batch reward. This module conditionally runs
Stage B, merges only the Teacher answer path, then applies the existing V2
bonus eligibility and V3 post-normalization group scale.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward import (
    _call_teacher,
    _gold_answers,
    _question,
    _search_evidence,
    _to_plain,
    compute_spad_em_teacher_backoff_batch,
)
from agentic_iter_rag.agent_training.spad.rewards.search_policy_teacher_reward_gold_match_bonus_v3 import (
    ADVANTAGE_SCALE_KEY,
    ADVANTAGE_SCALE_VERSION,
    apply_teacher_gold_token_f1_bonus_v3,
)
from agentic_iter_rag.agent_training.spad.teacher_strategies import (
    HARD_GATE_R5_LITERAL_CANONICAL_V2,
    HARD_GATE_STAGE_A_PROMPT_VERSION,
    HARD_GATE_STAGE_B_PROMPT_VERSION,
    build_gold_support_evidence_only_messages,
    select_hard_gate_output,
)


REWARD_VERSION = "spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2"
BASE_REWARD_VERSION = "spad_em_teacher_backoff"


def _request_audit(
    *,
    messages: list[dict[str, str]],
    request_cfg: dict[str, Any],
    prompt_version: str,
) -> dict[str, Any]:
    identity = {
        "model": str(request_cfg.get("model") or os.environ.get("SPAD_TEACHER_MODEL") or ""),
        "endpoint": str(
            request_cfg.get("endpoint") or os.environ.get("SPAD_TEACHER_ENDPOINT") or ""
        ),
        "messages": messages,
        "temperature": float(request_cfg.get("temperature", 0.0)),
        "top_p": float(request_cfg.get("top_p", 1.0)),
        "max_tokens": int(request_cfg.get("max_tokens", 512)),
        "chat_template_kwargs": _to_plain(request_cfg.get("chat_template_kwargs") or {}),
        "prompt_version": prompt_version,
    }
    request_hash = hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "teacher_messages": messages,
        "teacher_request_hash": request_hash,
        "teacher_model": identity["model"],
        "teacher_endpoint": identity["endpoint"],
        "teacher_temperature": identity["temperature"],
        "teacher_top_p": identity["top_p"],
        "teacher_max_tokens": identity["max_tokens"],
    }


def _stage_fields(prefix: str, detail: Mapping[str, Any]) -> dict[str, Any]:
    return {
        f"teacher_{prefix}_answer": str(detail.get("teacher_answer") or ""),
        f"teacher_{prefix}_evidence_status": str(
            detail.get("teacher_evidence_status") or ""
        ),
        f"teacher_{prefix}_raw_content": str(detail.get("teacher_raw_content") or ""),
        f"teacher_{prefix}_parse_status": str(detail.get("teacher_parse_status") or ""),
        f"teacher_{prefix}_format_error": bool(
            detail.get("teacher_format_error", False)
        ),
        f"teacher_{prefix}_elapsed_s": float(detail.get("teacher_elapsed_s") or 0.0),
        f"teacher_{prefix}_messages": _to_plain(detail.get("teacher_messages") or []),
        f"teacher_{prefix}_request_hash": str(
            detail.get("teacher_request_hash") or ""
        ),
    }


def _call_stage_b(
    *,
    question: str,
    gold_answers: list[str],
    evidence: list[dict[str, Any]],
    request_cfg: dict[str, Any],
) -> dict[str, Any]:
    messages = build_gold_support_evidence_only_messages(
        question=question,
        gold_answers=gold_answers,
        evidence_steps=evidence,
    )
    audit = _request_audit(
        messages=messages,
        request_cfg=request_cfg,
        prompt_version=HARD_GATE_STAGE_B_PROMPT_VERSION,
    )
    try:
        (
            answer,
            status,
            format_error,
            elapsed_s,
            raw_content,
            parse_status,
            enable_thinking,
        ) = _call_teacher(
            question=question,
            evidence=evidence,
            request_cfg=request_cfg,
            prompt_version=HARD_GATE_STAGE_B_PROMPT_VERSION,
            messages=messages,
        )
        return {
            **audit,
            "teacher_called": True,
            "teacher_answer": answer,
            "teacher_evidence_status": status,
            "teacher_format_error": format_error,
            "teacher_elapsed_s": elapsed_s,
            "teacher_raw_content": raw_content,
            "teacher_parse_status": parse_status,
            "teacher_enable_thinking": enable_thinking,
        }
    except Exception as exc:
        return {
            **audit,
            "teacher_called": True,
            "teacher_answer": "",
            "teacher_evidence_status": "",
            "teacher_format_error": True,
            "teacher_elapsed_s": 0.0,
            "teacher_raw_content": "",
            "teacher_parse_status": f"teacher_error:{type(exc).__name__}",
            "teacher_enable_thinking": "",
        }


def _merge_hard_gate_results(
    *,
    stage_a_results: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
    extra_infos: list[dict[str, Any]],
    request_cfg: dict[str, Any],
    visible_top_m: int,
    batch_workers: int,
) -> list[dict[str, Any]]:
    evidence_by_index = [
        _search_evidence(extra_info, visible_top_m=visible_top_m)
        for extra_info in extra_infos
    ]
    eligible = [
        index
        for index, stage_a in enumerate(stage_a_results)
        if stage_a.get("teacher_called")
        and not stage_a.get("teacher_format_error")
        and stage_a.get("teacher_parse_status") == "parsed"
        and stage_a.get("teacher_evidence_status")
        in {"supported_answer", "ambiguous_evidence"}
    ]
    stage_b_by_index: dict[int, dict[str, Any]] = {}
    workers = max(1, min(batch_workers, max(1, len(eligible))))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _call_stage_b,
                question=_question(extra_infos[index]),
                gold_answers=_gold_answers(ground_truths[index]),
                evidence=evidence_by_index[index],
                request_cfg=request_cfg,
            ): index
            for index in eligible
        }
        for future in as_completed(futures):
            stage_b_by_index[futures[future]] = dict(future.result())

    merged: list[dict[str, Any]] = []
    for index, stage_a in enumerate(stage_a_results):
        stage_b = stage_b_by_index.get(index)
        if stage_b is None:
            selection = {
                "selected": dict(stage_a),
                "stage_b_used": False,
                "canonical_gold": "",
                "selection_reason": "stage_a_i_or_not_called_or_format",
            }
        else:
            selection = select_hard_gate_output(
                stage_a=stage_a,
                stage_b=stage_b,
                gold_answers=_gold_answers(ground_truths[index]),
                evidence_steps=evidence_by_index[index],
            )
        selected = dict(selection["selected"])
        stage_a_is_i = bool(
            stage_a.get("teacher_parse_status") == "parsed"
            and not stage_a.get("teacher_format_error")
            and stage_a.get("teacher_evidence_status") == "insufficient_evidence"
        )
        final_is_i = bool(
            selected.get("teacher_parse_status") == "parsed"
            and not selected.get("teacher_format_error")
            and selected.get("teacher_evidence_status") == "insufficient_evidence"
        )
        updated = {
            **stage_a,
            **{
                key: selected.get(key, stage_a.get(key))
                for key in (
                    "teacher_answer",
                    "teacher_evidence_status",
                    "teacher_raw_content",
                    "teacher_parse_status",
                    "teacher_format_error",
                    "teacher_enable_thinking",
                    "teacher_messages",
                    "teacher_request_hash",
                )
            },
            **_stage_fields("stage_a", stage_a),
            "teacher_strategy_id": HARD_GATE_R5_LITERAL_CANONICAL_V2,
            "teacher_total_call_count": int(bool(stage_a.get("teacher_called")))
            + int(stage_b is not None),
            "teacher_stage_b_called": stage_b is not None,
            "teacher_stage_b_used": bool(selection["stage_b_used"]),
            "teacher_stage_b_skip_reason": (
                ""
                if stage_b is not None and selection["stage_b_used"]
                else str(selection["selection_reason"])
            ),
            "teacher_selection_reason": str(selection["selection_reason"]),
            "teacher_canonical_gold": str(selection["canonical_gold"]),
            "teacher_i_boundary_preserved": stage_a_is_i == final_is_i,
            "teacher_elapsed_s": float(stage_a.get("teacher_elapsed_s") or 0.0)
            + float((stage_b or {}).get("teacher_elapsed_s") or 0.0),
        }
        if stage_b is not None:
            updated.update(_stage_fields("stage_b", stage_b))
        else:
            updated.update(_stage_fields("stage_b", {}))
        final_status = str(updated.get("teacher_evidence_status") or "")
        updated.update(
            {
                "supported_answer_count": int(final_status == "supported_answer"),
                "insufficient_evidence_count": int(
                    final_status == "insufficient_evidence"
                ),
                "ambiguous_evidence_count": int(final_status == "ambiguous_evidence"),
            }
        )
        merged.append(updated)
    return merged


def compute_spad_em_teacher_backoff_gold_token_f1_bonus_v3_hard_gate_v2_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict[str, Any]],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run frozen Stage A, Hard-Gate v2 Stage B, then V3 bonus/scaling."""

    reward_cfg = dict(kwargs.get("reward_cfg") or {})
    configured_type = str(reward_cfg.get("type") or REWARD_VERSION)
    if configured_type != REWARD_VERSION:
        raise ValueError(
            f"{REWARD_VERSION} entry point received reward type {configured_type!r}"
        )
    variant_cfg = reward_cfg.get(REWARD_VERSION)
    if not isinstance(variant_cfg, Mapping):
        variant_cfg = {}
    partial_reward = float(variant_cfg.get("partial_reward", 0.1))
    bonus_weight = float(variant_cfg.get("gold_token_f1_bonus", 0.1))
    teacher_group_postnorm_scale = float(
        variant_cfg.get("teacher_group_postnorm_scale", 0.1)
    )
    if bonus_weight < 0.0:
        raise ValueError("gold_token_f1_bonus must be non-negative")
    if not 0.0 < teacher_group_postnorm_scale <= 1.0:
        raise ValueError("teacher_group_postnorm_scale must be in (0, 1]")
    teacher_prompt_version = str(
        kwargs.get("teacher_prompt_version") or HARD_GATE_STAGE_A_PROMPT_VERSION
    )
    if teacher_prompt_version != HARD_GATE_STAGE_A_PROMPT_VERSION:
        raise ValueError(
            f"{REWARD_VERSION} requires teacher_prompt_version="
            f"{HARD_GATE_STAGE_A_PROMPT_VERSION}"
        )
    strategy_id = str(kwargs.get("teacher_strategy_id") or "")
    if strategy_id != HARD_GATE_R5_LITERAL_CANONICAL_V2:
        raise ValueError(
            f"{REWARD_VERSION} requires teacher_strategy_id="
            f"{HARD_GATE_R5_LITERAL_CANONICAL_V2}"
        )

    base_reward_cfg = dict(reward_cfg)
    base_reward_cfg["type"] = BASE_REWARD_VERSION
    base_reward_cfg[BASE_REWARD_VERSION] = {"partial_reward": partial_reward}
    stage_a_results = compute_spad_em_teacher_backoff_batch(
        data_sources,
        solution_strs,
        ground_truths,
        extra_infos,
        **{**kwargs, "reward_cfg": base_reward_cfg},
    )
    merged_results = _merge_hard_gate_results(
        stage_a_results=stage_a_results,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        request_cfg=dict(kwargs.get("teacher_request") or {}),
        visible_top_m=int(kwargs.get("visible_top_m") or 5),
        batch_workers=int(
            kwargs.get("batch_workers")
            or os.environ.get("SPAD_TEACHER_BATCH_WORKERS")
            or 16
        ),
    )
    results = []
    for index, result in enumerate(merged_results):
        updated = apply_teacher_gold_token_f1_bonus_v3(
            result,
            ground_truths[index],
            bonus_weight=bonus_weight,
            teacher_group_postnorm_scale=teacher_group_postnorm_scale,
        )
        updated.update(
            {
                "reward_type": REWARD_VERSION,
                ADVANTAGE_SCALE_KEY: float(updated[ADVANTAGE_SCALE_KEY]),
                "advantage_postnorm_scale_version": ADVANTAGE_SCALE_VERSION,
            }
        )
        results.append(updated)
    return results
