#!/usr/bin/env python3
"""AIR LLM reranker training report schema.

这份 schema 只描述 AIR LLM reranker 训练曲线，不复用 CAR 的 schema。
公共 report_system 会读取这里的分组配置，然后生成 markdown 报告和 latest 曲线图。
"""

PROJECT_NAME = "AgenticIterRag LLM Reranker"

METRIC_GROUPS = {
    "reranker_rewards": [
        "critic/rewards/mean",
        "critic/rewards/max",
        "critic/rewards/min",
        "critic/score/mean",
        "critic/score/max",
        "critic/score/min",
        "critic/advantages/mean",
    ],
    "reranker_losses": [
        "actor/pg_loss",
        "actor/kl_loss",
        "actor/ppo_kl",
        "actor/entropy",
        "actor/grad_norm",
        "actor/lr",
    ],
    "reranker_lengths": [
        "response_length/mean",
        "response_length/max",
        "response_length/clip_ratio",
        "prompt_length/mean",
        "prompt_length/max",
        "prompt_length/clip_ratio",
        "num_turns/mean",
    ],
    "reranker_performance": [
        "perf/time_per_step",
        "perf/throughput",
        "perf/total_num_tokens",
        "timing_s/gen",
        "timing_s/reward",
        "timing_s/update_actor",
        "timing_s/ref",
        "timing_s/old_log_prob",
    ],
}

PLOT_GROUPS = {
    "reranker_rewards": {
        "reward": [
            "critic/rewards/mean",
            "critic/rewards/max",
            "critic/rewards/min",
            "critic/score/mean",
            "critic/score/max",
            "critic/score/min",
        ],
    },
    "reranker_losses": {
        "loss": [
            "actor/pg_loss",
            "actor/kl_loss",
            "actor/ppo_kl",
            "actor/entropy",
        ],
    },
    "reranker_lengths": {
        "length": [
            "response_length/mean",
            "response_length/max",
            "prompt_length/mean",
            "prompt_length/max",
            "num_turns/mean",
        ],
        "clip_ratio": [
            "response_length/clip_ratio",
            "prompt_length/clip_ratio",
        ],
    },
    "reranker_performance": {
        "seconds": [
            "perf/time_per_step",
            "timing_s/gen",
            "timing_s/reward",
            "timing_s/update_actor",
            "timing_s/ref",
            "timing_s/old_log_prob",
        ],
        "throughput": [
            "perf/throughput",
        ],
    },
}

DETAILED_METRIC_KEYS = [
    "critic/rewards/mean",
    "critic/rewards/max",
    "critic/rewards/min",
    "critic/score/mean",
    "critic/score/max",
    "critic/score/min",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/ppo_kl",
    "actor/entropy",
    "actor/grad_norm",
    "actor/lr",
    "response_length/mean",
    "response_length/max",
    "response_length/clip_ratio",
    "prompt_length/mean",
    "prompt_length/max",
    "prompt_length/clip_ratio",
    "num_turns/mean",
    "perf/time_per_step",
    "perf/throughput",
    "timing_s/gen",
    "timing_s/reward",
    "timing_s/update_actor",
]

ROLLOUT_ROLE_DIRS = ["reranker", "main"]

GPU_GROUPS = {
    "reranker": "RERANKER_GPU_IDS",
    "frozen_agent": "FROZEN_AGENT_GPU_IDS",
    "retriever": "RETRIEVER_GPU_IDS",
}

TIMING_ALIASES = {
    "reranker/generate": ["timing_s/gen"],
    "reranker/reward": ["timing_s/reward"],
    "reranker/update_actor": ["timing_s/update_actor"],
}


def build_extra_markdown_sections(context):
    """给 AIR LLM reranker 报告补一小段运行语义说明。"""

    env = context.get("env") or {}
    lines = [
        "## AIR LLM Reranker Context",
        "",
        "| item | value |",
        "| --- | --- |",
        f"| reranker gpus | `{env.get('RERANKER_GPU_IDS', '')}` |",
        f"| frozen agent gpus | `{env.get('FROZEN_AGENT_GPU_IDS', '')}` |",
        f"| retriever gpus | `{env.get('RETRIEVER_GPU_IDS', '')}` |",
        f"| rollout n | `{env.get('RERANKER_ROLLOUT_N', '')}` |",
        f"| branch policy | `{env.get('RERANKER_BRANCH_STEP_POLICY', '')}` |",
        "",
    ]
    return "\n".join(lines)
