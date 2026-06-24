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

