#!/usr/bin/env python3
"""CoAgenticRetriever training report schema."""

PROJECT_NAME = "CoAgenticRetriever"

METRIC_GROUPS = {
    "agent_scores": [
        "main_agent/score_mean",
        "main_agent/f1_mean",
        "main_agent/valid_rate",
    ],
    "agent_losses": [
        "main_agent_actor/pg_loss",
        "main_agent_actor/kl_loss",
        "main_agent_actor/entropy_loss",
    ],
    "agent_optimization": [
        "main_agent_actor/grad_norm",
        "main_agent_actor/lr",
        "main_agent_actor/MFU",
    ],
    "agent_lengths": [
        "main_agent_response_length/mean",
        "main_agent_prompt_length/mean",
        "main_agent_num_turns/mean",
    ],
    "ranker_quality": [
        "ranker/acc@1",
        "ranker/mrr",
        "ranker/score_margin",
        "ranker/pos_score_mean",
        "ranker/neg_score_mean",
    ],
    "ranker_losses": [
        "ranker/loss",
        "ranker/loss_step_0",
        "ranker/loss_step_1",
    ],
    "ranker_optimization": [
        "ranker/lr",
        "ranker/grad_norm",
    ],
    "performance": [
        "perf/time_per_step",
        "perf/throughput",
    ],
}

PLOT_GROUPS = {
    "agent_scores": {
        "agent": [
            "main_agent/score_mean",
            "main_agent/f1_mean",
            "main_agent/valid_rate",
        ],
    },
    "agent_losses": {
        "agent": [
            "main_agent_actor/pg_loss",
            "main_agent_actor/kl_loss",
            "main_agent_actor/entropy_loss",
        ],
    },
    "agent_lengths": {
        "agent": [
            "main_agent_response_length/mean",
            "main_agent_prompt_length/mean",
            "main_agent_num_turns/mean",
        ],
    },
    "ranker_quality": {
        "ranker": [
            "ranker/acc@1",
            "ranker/mrr",
            "ranker/score_margin",
        ],
    },
    "ranker_losses": {
        "ranker": [
            "ranker/loss",
            "ranker/loss_step_0",
            "ranker/loss_step_1",
        ],
    },
}

DETAILED_METRIC_KEYS = [
    "agent_rollout_num",
    "main_agent/score_mean",
    "main_agent/f1_mean",
    "main_agent/valid_rate",
    "main_agent_actor/pg_loss",
    "main_agent_actor/kl_loss",
    "main_agent_response_length/mean",
    "main_agent_prompt_length/mean",
    "main_agent_num_turns/mean",
    "main_agent_actor/lr",
    "ranker/loss",
    "ranker/acc@1",
    "ranker/mrr",
    "ranker/score_margin",
    "ranker/pos_score_mean",
    "ranker/neg_score_mean",
    "ranker/lr",
]

ROLLOUT_ROLE_DIRS = ["main_agent", "main"]

GPU_GROUPS = {
    "main_agent": "MAIN_GPU_IDS",
    "ranker": "RANKER_GPU_IDS",
}

TIMING_ALIASES = {
    "agent/update_actor": ["timing_s/main_agent_update_actor"],
    "ranker/contrastive_total": ["timing_s/ranker_contrastive_total"],
}


def _env_value(env, key):
    value = env.get(key, "")
    return "" if value is None else str(value)


def _first_env_value(env, *keys):
    for key in keys:
        value = _env_value(env, key)
        if value:
            return value
    return ""


def build_extra_markdown_sections(context):
    env = context.get("env") or {}
    lines = [
        "## Retrieval Cutoffs",
        "",
        "| meaning | field | value |",
        "| --- | --- | ---: |",
        (
            "| recall candidate pool | `HYDRA_RECALL_FINAL_TOP_N` / `RECALL_FINAL_TOP_N` | "
            f"`{_first_env_value(env, 'HYDRA_RECALL_FINAL_TOP_N', 'RECALL_FINAL_TOP_N')}` |"
        ),
        (
            "| recall candidate pool written to runtime tool config | "
            "`RUNTIME_TOOL_RECALL_FINAL_TOP_N` | "
            f"`{_env_value(env, 'RUNTIME_TOOL_RECALL_FINAL_TOP_N')}` |"
        ),
        (
            "| ranker keeps after rerank | `HYDRA_RANKER_FINAL_TOP_K` / `RANKER_FINAL_TOP_K` | "
            f"`{_first_env_value(env, 'HYDRA_RANKER_FINAL_TOP_K', 'RANKER_FINAL_TOP_K')}` |"
        ),
        (
            "| ranker keeps after rerank written to runtime tool config | "
            "`RUNTIME_TOOL_RANKER_FINAL_TOP_K` | "
            f"`{_env_value(env, 'RUNTIME_TOOL_RANKER_FINAL_TOP_K')}` |"
        ),
        (
            "| agent-visible docs from static tool config | "
            "`SEARCH_TOOL_FINAL_TOP_M` | "
            f"`{_env_value(env, 'SEARCH_TOOL_FINAL_TOP_M')}` |"
        ),
        (
            "| agent-visible docs written to runtime tool config | "
            "`RUNTIME_TOOL_SEARCH_TOOL_FINAL_TOP_M` | "
            f"`{_env_value(env, 'RUNTIME_TOOL_SEARCH_TOOL_FINAL_TOP_M')}` |"
        ),
        "",
        "## Retrieval Cutoff Runtime Aliases",
        "",
        "| alias | value | note |",
        "| --- | ---: | --- |",
        (
            "| `RECALL_TOP_K` | "
            f"`{_env_value(env, 'RECALL_TOP_K')}` | Runtime/preflight alias for recall final top-N. |"
        ),
        (
            "| `TOP_N` | "
            f"`{_env_value(env, 'TOP_N')}` | Runtime/preflight alias for recall final top-N. |"
        ),
        (
            "| `TOP_M` | "
            f"`{_env_value(env, 'TOP_M')}` | Runtime/preflight alias for searchTool final top-M. |"
        ),
    ]
    return "\n".join(lines)
