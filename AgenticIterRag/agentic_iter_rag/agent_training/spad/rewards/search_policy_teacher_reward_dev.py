"""Speculative teacher-prefetch scheduler for SPAD Stage 1.

The stable ``spad_em_teacher_backoff`` implementation remains untouched.  This
module preserves its group-level reward formula while moving teacher requests
to the per-rollout reward loop so they overlap with other actor rollouts.
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from concurrent.futures import Future
from collections.abc import Mapping
from typing import Any

from agentic_iter_rag.agent_training.spad.prompts import (
    DEFAULT_TEACHER_STATUS_PROMPT_VERSION,
    resolve_teacher_prompt,
)
from agentic_iter_rag.agent_training.spad.rewards import search_policy_teacher_reward as stable
from agentic_iter_rag.metrics.answer_metrics import answer_group_metrics


REWARD_VERSION = "spad_em_teacher_backoff_dev"
PREFETCH_DETAIL_KEY = "spad_dev_prefetched_teacher_detail"
PREFETCH_ONLY_KEY = "spad_dev_prefetch_only"
_MAX_PREFETCH_CACHE_ENTRIES = 2048
_PREFETCH_CACHE: OrderedDict[str, Future[dict[str, Any]]] = OrderedDict()
_PREFETCH_CACHE_LOCK = threading.Lock()


def _reward_config(kwargs: dict[str, Any]) -> dict[str, Any]:
    reward_cfg = dict(kwargs.get("reward_cfg") or {})
    configured_type = str(reward_cfg.get("type") or REWARD_VERSION)
    if configured_type != REWARD_VERSION:
        raise ValueError(f"{REWARD_VERSION} received reward type {configured_type!r}")
    return reward_cfg


def _teacher_context(kwargs: dict[str, Any]) -> tuple[dict[str, Any], str, int]:
    request_cfg = dict(kwargs.get("teacher_request") or {})
    prompt_version, _ = resolve_teacher_prompt(
        str(kwargs.get("teacher_prompt_version") or DEFAULT_TEACHER_STATUS_PROMPT_VERSION),
        include_status=True,
    )
    return request_cfg, prompt_version, int(kwargs.get("visible_top_m") or 5)


def _coalesced_teacher_details(
    *,
    base: dict[str, Any],
    request_cfg: dict[str, Any],
    teacher_prompt_version: str,
) -> tuple[dict[str, Any], bool]:
    audit = stable._teacher_audit_details(
        base=base,
        request_cfg=request_cfg,
        teacher_prompt_version=teacher_prompt_version,
    )
    request_hash = str(audit["teacher_request_hash"])
    owner = False
    with _PREFETCH_CACHE_LOCK:
        future = _PREFETCH_CACHE.get(request_hash)
        if future is None:
            future = Future()
            _PREFETCH_CACHE[request_hash] = future
            owner = True
            while len(_PREFETCH_CACHE) > _MAX_PREFETCH_CACHE_ENTRIES:
                oldest_hash, oldest_future = next(iter(_PREFETCH_CACHE.items()))
                if not oldest_future.done():
                    break
                _PREFETCH_CACHE.pop(oldest_hash)
        else:
            _PREFETCH_CACHE.move_to_end(request_hash)

    if owner:
        try:
            detail = stable._spad_teacher_status_details(
                base=base,
                request_cfg=request_cfg,
                teacher_prompt_version=teacher_prompt_version,
            )
        except BaseException as exc:
            future.set_exception(exc)
            with _PREFETCH_CACHE_LOCK:
                _PREFETCH_CACHE.pop(request_hash, None)
            raise
        future.set_result(copy.deepcopy(detail))

    return copy.deepcopy(future.result()), not owner


def _compute_prefetch_details(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    del data_source
    _reward_config(kwargs)
    request_cfg, prompt_version, visible_top_m = _teacher_context(kwargs)
    base = stable._spad_group_base_details(
        solution_str=str(solution_str),
        ground_truth=ground_truth,
        extra_info=dict(extra_info or {}),
        visible_top_m=visible_top_m,
    )
    detail, cache_hit = _coalesced_teacher_details(
        base=base,
        request_cfg=request_cfg,
        teacher_prompt_version=prompt_version,
    )
    return {
        "score": 0.0,
        "reward_type": REWARD_VERSION,
        PREFETCH_ONLY_KEY: True,
        PREFETCH_DETAIL_KEY: detail,
        "spad_dev_prefetch_submitted": bool(base["evidence"]),
        "spad_dev_prefetch_cache_hit": cache_hit,
        "spad_dev_prefetch_request_hash": str(detail.get("teacher_request_hash") or ""),
    }


def _compute_group_rewards(
    *,
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[Any],
    extra_infos: list[dict[str, Any]],
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    item_count = len(solution_strs)
    if not (len(data_sources) == len(ground_truths) == len(extra_infos) == item_count):
        raise ValueError("SPAD dev batch reward inputs must have the same length")
    if item_count == 0:
        return []

    reward_cfg = _reward_config(kwargs)
    request_cfg, prompt_version, visible_top_m = _teacher_context(kwargs)
    expected_group_size = int(kwargs.get("n_samples_per_prompt") or 8)
    dev_cfg = reward_cfg.get(REWARD_VERSION)
    if not isinstance(dev_cfg, Mapping):
        dev_cfg = reward_cfg.get("spad_em_teacher_backoff")
    if not isinstance(dev_cfg, Mapping):
        dev_cfg = {}
    partial_weight = float(dev_cfg.get("partial_reward", 0.1))

    bases = [
        stable._spad_group_base_details(
            solution_str=str(solution_strs[index]),
            ground_truth=ground_truths[index],
            extra_info=extra_infos[index],
            visible_top_m=visible_top_m,
        )
        for index in range(item_count)
    ]
    grouped_indices: dict[str, list[int]] = {}
    for index, base in enumerate(bases):
        uid = str(base["group_uid"])
        if not uid:
            raise ValueError(f"SPAD dev batch reward item {index} is missing uid")
        grouped_indices.setdefault(uid, []).append(index)
    invalid_groups = {
        uid: len(indices)
        for uid, indices in grouped_indices.items()
        if len(indices) != expected_group_size
    }
    if invalid_groups:
        raise ValueError(
            f"SPAD dev batch reward requires exactly {expected_group_size} rollouts per uid; "
            f"got {invalid_groups}"
        )

    all_zero_by_uid = {
        uid: not any(bases[index]["em_reward"] >= 1.0 for index in indices)
        for uid, indices in grouped_indices.items()
    }
    teacher_details: dict[int, dict[str, Any]] = {}
    for index, base in enumerate(bases):
        uid = str(base["group_uid"])
        if not all_zero_by_uid[uid] or not base["evidence"]:
            continue
        raw_detail = extra_infos[index].get(PREFETCH_DETAIL_KEY)
        if isinstance(raw_detail, Mapping):
            teacher_details[index] = copy.deepcopy(dict(raw_detail))
        else:
            teacher_details[index], _ = _coalesced_teacher_details(
                base=base,
                request_cfg=request_cfg,
                teacher_prompt_version=prompt_version,
            )

    results: list[dict[str, Any]] = []
    for index, base in enumerate(bases):
        uid = str(base["group_uid"])
        all_zero = all_zero_by_uid[uid]
        teacher_audit = stable._teacher_audit_details(
            base=base,
            request_cfg=request_cfg,
            teacher_prompt_version=prompt_version,
        )
        detail = teacher_details.get(
            index,
            {
                **teacher_audit,
                "teacher_called": False,
                "teacher_skip_reason": "group_has_positive_em" if not all_zero else "no_search_evidence",
                "teacher_parse_status": "not_called",
                "teacher_status_reward": 0.0,
            },
        )
        final_reward = (
            partial_weight * float(detail["teacher_status_reward"])
            if all_zero
            else float(base["em_reward"])
        )
        answers = stable._gold_answers(ground_truths[index])
        results.append(
            stable._sanitize(
                {
                    **detail,
                    "final_reward": final_reward,
                    "reward_type": REWARD_VERSION,
                    "teacher_f1": final_reward,
                    "format_status": "valid" if base["actor_answer_parse_status"] == "parsed" else "invalid",
                    "stop_status": (
                        "answer_closed"
                        if base["actor_answer_parse_status"] == "parsed"
                        else base["actor_answer_parse_status"]
                    ),
                    "search_count": base["search_count"],
                    "duplicate_query_count": base["duplicate_query_count"],
                    "teacher_prompt_version": prompt_version,
                    "actor_answer": base["actor_answer"],
                    "actor_answer_parse_status": base["actor_answer_parse_status"],
                    "em_reward": base["em_reward"],
                    "search_r1_answer_em": base["em_reward"],
                    "search_r1_extracted_answer": base["actor_answer"],
                    "legacy_em": base["em_reward"],
                    "legacy_f1": answer_group_metrics(base["actor_answer"], answers).legacy_f1,
                    "group_uid": uid,
                    "group_size": len(grouped_indices[uid]),
                    "group_all_em_zero": all_zero,
                    "partial_reward_applied": all_zero,
                }
            )
        )
    return results


def compute_spad_em_teacher_backoff_dev(*args: Any, **kwargs: Any) -> Any:
    """Dispatch single-rollout prefetch and complete UID-group reward calls."""

    if "data_sources" in kwargs or (args and isinstance(args[0], (list, tuple))):
        if args:
            data_sources, solution_strs, ground_truths, extra_infos, *rest = args
            if rest:
                raise TypeError("unexpected positional arguments for SPAD dev batch reward")
        else:
            data_sources = kwargs.pop("data_sources")
            solution_strs = kwargs.pop("solution_strs")
            ground_truths = kwargs.pop("ground_truths")
            extra_infos = kwargs.pop("extra_infos")
        return _compute_group_rewards(
            data_sources=list(data_sources),
            solution_strs=list(solution_strs),
            ground_truths=list(ground_truths),
            extra_infos=[dict(item or {}) for item in extra_infos],
            kwargs=kwargs,
        )

    if args:
        data_source, solution_str, ground_truth, extra_info, *rest = args
        if rest:
            raise TypeError("unexpected positional arguments for SPAD dev prefetch reward")
    else:
        data_source = kwargs.pop("data_source")
        solution_str = kwargs.pop("solution_str")
        ground_truth = kwargs.pop("ground_truth")
        extra_info = kwargs.pop("extra_info")
    return _compute_prefetch_details(
        data_source=str(data_source),
        solution_str=str(solution_str),
        ground_truth=ground_truth,
        extra_info=dict(extra_info or {}),
        kwargs=kwargs,
    )


def _clear_prefetch_cache_for_tests() -> None:
    with _PREFETCH_CACHE_LOCK:
        _PREFETCH_CACHE.clear()
