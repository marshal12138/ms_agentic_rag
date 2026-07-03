"""Validation helpers for AgenticIterRag v1."""

from __future__ import annotations

import os
from typing import Iterable, Mapping


ALLOWED_ENV_CONFIG_KEYS = {
    "EXP_NAME",
    "GROUP_NAME",
    "PY",
    "PYTHONPATH",
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "PWD",
    "OLDPWD",
    "LANG",
    "LC_ALL",
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "AIR_ACCELERATOR",
    "COAGENTIC_PROJECT_ROOT",
    "AGENTIC_ITER_RAG_PROJECT_ROOT",
}


BUSINESS_ENV_PREFIXES = (
    "AGENT_",
    "RANKER_",
    "RERANKER_",
    "RECALL_",
    "RETRIEVAL_",
    "LLM_JUDGE_",
    "DATA_",
    "MODEL_",
    "MAX_",
    "TOP_",
    "RUN_MODE",
    "INFER_",
    "TRAIN_",
    "BATCH_",
)


def find_shell_only_business_env(
    environ: Mapping[str, str] | None = None,
    allowed: Iterable[str] = ALLOWED_ENV_CONFIG_KEYS,
) -> list[str]:
    """Return env keys that look like business config but are not allow-listed."""

    env = dict(os.environ if environ is None else environ)
    allowed_set = set(allowed)
    offenders: list[str] = []
    for key in env:
        if key in allowed_set:
            continue
        if key.startswith(BUSINESS_ENV_PREFIXES):
            offenders.append(key)
    return sorted(offenders)


def require_no_shell_only_business_env(environ: Mapping[str, str] | None = None) -> None:
    offenders = find_shell_only_business_env(environ)
    if offenders:
        joined = ", ".join(offenders)
        raise ValueError(
            "shell-only business configuration is forbidden for AgenticIterRag v1; "
            f"move these values into YAML/overlay/CLI dotlist: {joined}"
        )
