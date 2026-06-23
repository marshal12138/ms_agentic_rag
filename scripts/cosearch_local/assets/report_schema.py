#!/usr/bin/env python3
"""CoSearch training report schema."""

PROJECT_NAME = "CoSearch"

METRIC_GROUPS = {
    "reward": [
        "main/score_mean",
        "reranker/score_mean",
        "main_critic/rewards/mean",
        "reranker_critic/rewards/mean",
    ],
    "loss": [
        "main_actor/pg_loss",
        "main_actor/entropy_loss",
        "main_actor/kl_loss",
        "reranker_actor/pg_loss",
        "reranker_actor/entropy_loss",
        "reranker_actor/kl_loss",
    ],
    "optimization": [
        "main_actor/grad_norm",
        "main_actor/lr",
        "reranker_actor/grad_norm",
        "reranker_actor/lr",
    ],
    "length": [
        "main_response_length/mean",
        "main_prompt_length/mean",
        "reranker_response_length/mean",
        "reranker_prompt_length/mean",
    ],
    "performance": [
        "perf/time_per_step",
        "perf/throughput",
        "main_actor/MFU",
        "reranker_actor/MFU",
    ],
}

PLOT_GROUPS = {
    "scores": {
        "main": ["main/score_mean"],
        "reranker": ["reranker/score_mean"],
    },
    "losses": {
        "main": ["main_actor/pg_loss"],
        "reranker": ["reranker_actor/pg_loss"],
    },
    "lengths": {
        "main": ["main_response_length/mean", "main_prompt_length/mean"],
        "reranker": ["reranker_response_length/mean", "reranker_prompt_length/mean"],
    },
}

DETAILED_METRIC_KEYS = [
    "agent_rollout_num",
    "main/score_mean",
    "reranker/score_mean",
    "main_actor/pg_loss",
    "main_actor/kl_loss",
    "reranker_actor/pg_loss",
    "reranker_actor/kl_loss",
    "main_response_length/mean",
    "main_prompt_length/mean",
    "reranker_response_length/mean",
    "reranker_prompt_length/mean",
    "main_actor/lr",
    "reranker_actor/lr",
]

ROLLOUT_ROLE_DIRS = ["main", "reranker"]

GPU_GROUPS = {
    "main_actor": "MAIN_GPU_IDS",
    "reranker": "RERANKER_GPU_IDS",
}

