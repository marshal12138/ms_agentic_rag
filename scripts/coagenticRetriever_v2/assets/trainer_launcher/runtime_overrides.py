"""Runtime Hydra override YAML generation.

普通 overlay 表达实验语义，runtime override 表达“这次运行才知道”的信息：
run 目录、设备前缀、GPU 分配、service URL、async ranker 日志目录等。

这个模块只写 `*.runtime_env_overrides.yaml`，不负责把 YAML 展平成 Hydra dotlist。
展平顺序由 `hydra_args.py` 和主编译流程控制。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .context import CompilerContext
from .yaml_utils import dump_mapping


def _yaml_bool(value: str) -> bool | str:
    """把字符串 true/false 转为 YAML bool，其它值保持原样。"""

    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return value


def build_runtime_override_yaml(ctx: CompilerContext, env: Mapping[str, str], path: Path) -> None:
    """写入只属于本次运行的 Hydra override YAML。

    这些值不适合放进实验 overlay：它们依赖当前 run 目录、最终 GPU 分配、外部 service
    URL 或当前 launcher 生成的日志路径。主流程会把这个 YAML 展平成 dotlist，并放在
    普通 overlay 后面。
    """

    # ranker model path/encoder path 允许为空；为空时不写入，避免覆盖 base config 中
    # 可能已经存在的值。
    ranker: dict[str, Any] = {"device": ctx.device_spec(env["RANK_GPU_ID"])}
    if env.get("RANKER_BASE_MODEL_PATH"):
        ranker["model_path"] = env["RANKER_BASE_MODEL_PATH"]
    if env.get("RANKER_ENCODER_PATH"):
        ranker["encoder_path"] = env["RANKER_ENCODER_PATH"]

    data = {
        "trainer": {
            "experiment_name": env["EXP_NAME"],
            "default_local_dir": env["OUT_DIR"],
            "device": ctx.device_prefix,
            "n_gpus_per_node": int(env["AGENT_N_GPUS_PER_NODE"]),
            "nnodes": int(env["NNODES"]),
            "rollout_data_dir": env["ROLLOUT_DATA_DIR"],
            "validation_data_dir": env["VALIDATION_DATA_DIR"],
        },
        "actor_rollout_ref": {
            "nccl_timeout": int(env.get("NCCL_TIMEOUT") or "600"),
            "actor": {
                "use_torch_compile": _yaml_bool(env["ACTOR_USE_TORCH_COMPILE"]),
            },
            "rollout": {
                "multi_turn": {
                    # tool config 是训练 rollout 的真实运行输入。full/no-ranker 会在
                    # compile_config.py 中先确定最终 TOOL_CONFIG，这里再同步写回 Hydra，
                    # 保证训练进程和 Bash runtime 使用同一份 tool YAML。
                    "tool_config_path": env["TOOL_CONFIG"],
                },
            },
        },
        "recall_retriever": {
            "model_path": env["RECALL_MODEL_PATH"],
            "device": ctx.device_spec(env["RECALL_GPU_ID"]),
            "service_url": env["RETRIEVAL_SERVICE_URL"],
        },
        "ranker": ranker,
        "ranker_training": {
            "construction_log_jsonl": str(Path(env["LOG_DIR"]) / f"{env['RUN_NAME']}.contrastive_construction.jsonl"),
            "async_ranker_training": {
                "logging": {
                    "log_dir": env["ASYNC_RANKER_TRAINING_LOG_DIR"],
                },
            },
        },
        "resources": {
            # resources.* 是训练配置中的审计视图，值来自最终有效 env。真正的优先级合并
            # 已在 resource.py 中完成。
            "agent_gpu_ids": env["AGENT_GPU_IDS"],
            "rank_gpu_id": env["RANK_GPU_ID"],
            "recall_gpu_id": env["RECALL_GPU_ID"],
            "llm_judge_gpu_ids": env["LLM_JUDGE_GPU_IDS"],
        },
    }
    dump_mapping(path, data)
